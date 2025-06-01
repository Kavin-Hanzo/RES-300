
import ee
import geemap
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
import torch
import os
import pandas as pd
from scipy.ndimage import gaussian_filter, rotate
import rasterio
from rasterio.transform import from_bounds
from torchvision.transforms.functional import resize

# ---- Parameters ----
# Corrected bands based on the paper [cite: 42]
BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B8', 'B8A', 'B9', 'B11', 'B12']
IMG_SIZE = 128  # Pixels
SCALE = 20      # Meters per pixel [cite: 32, 151]
NUM_TILES = 4  # Number of tiles to split the image into
COLLECTION = 'COPERNICUS/S2_HARMONIZED' # Or S2_SR_HARMONIZED if using Surface Reflectance
CLOUD_COVER_THRESHOLD = 25 # Max cloud cover % [cite: 27, 153]

# ---- Cloud Masking (Simplified Example - Adapt based on S2 L1C/L2A specifics) ----
def mask_s2_clouds(image):
    """Masks clouds in Sentinel-2 L1C TOA data using QA60 band."""
    qa = image.select('QA60')
    # Bits 10 and 11 are cloud and cirrus masks, respectively
    cloud_mask = (qa.bitwiseAnd(1 << 10).eq(0)).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(cloud_mask).copyProperties(image, ["system:time_start"])

# ---- CSV Reading Function ----
def read_coordinates_from_csv(csv_path):
    """Reads rectangle coordinates and timestamps from a CSV file."""
    df = pd.read_csv(csv_path)
    # Check if required columns exist
    required_cols = ['min_lon', 'min_lat', 'max_lon', 'max_lat', 's2_timestamp_ms']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    # Extract coordinates and timestamps
    coordinates = []
    for _, row in df.iterrows():
        coordinates.append({
            'min_lon': row['min_lon'],
            'min_lat': row['min_lat'],
            'max_lon': row['max_lon'],
            'max_lat': row['max_lat'],
            'timestamp_ms': row['s2_timestamp_ms']
        })

    return coordinates[814:]

# ---- Sentinel-2 Fetch ----
def get_s2_image(rect_coords, timestamp_ms):
    """Fetches a Sentinel-2 image for the given rectangular area and timestamp."""

    # Convert timestamp from milliseconds to date and create a search window
    date = ee.Date(timestamp_ms)
    start_date = date.advance(-15, 'day')  # Increased to 15 days before timestamp for better chances
    end_date = date.advance(15, 'day')     # Increased to 15 days after timestamp

    # Create rectangle geometry
    rect = ee.Geometry.Rectangle(
        [rect_coords['min_lon'], rect_coords['min_lat'],
         rect_coords['max_lon'], rect_coords['max_lat']]
    )

    # Filter collection by date, bounds, and cloud cover
    collection = ee.ImageCollection(COLLECTION) \
                   .filterBounds(rect) \
                   .filterDate(start_date, end_date) \
                   .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', CLOUD_COVER_THRESHOLD) \
                   .select(BANDS + ['QA60'])  # Include QA60 for masking

    # Get collection size
    collection_size = collection.size().getInfo()
    if collection_size == 0:
        print(f"No images found for the given date range and cloud cover threshold")
        return None, rect

    # Apply cloud masking and select median image
    image = collection.map(mask_s2_clouds) \
                     .select(BANDS) \
                     .median()  # Use median if multiple images found

    # Verify that the image has all bands
    band_count = image.bandNames().size().getInfo()
    if band_count < len(BANDS):
        print(f"Image has fewer bands than expected: {band_count} vs {len(BANDS)}")
        return None, rect

    return image, rect

# ---- Extraction to NumPy ----
def extract_numpy(image, region, img_size=512, scale=SCALE):
    """Extracts image data as a NumPy array for the specified region."""
    if image is None:
        return None
    try:
        # Limit the region size to comply with GEE's size restrictions
        # Calculate a target size that's within GEE limits (approx. 50MB)
        target_dim = int(np.sqrt(45 * 1024 * 512 / 10))  # 10 bands, safely under 50MB limit

        # Create a smaller region or sample at a coarser scale if needed
        # Use specified scale as a minimum
        actual_scale = max(scale, scale * (512 / target_dim))

        # Get the centroid of the region for sampling
        if isinstance(region, ee.Geometry):
            centroid = region.centroid().getInfo()['coordinates']
            sample_point = ee.Geometry.Point(centroid)
            # Create a square region around the centroid with the target size
            buffer_size = (512 * actual_scale) / 2
            sample_region = sample_point.buffer(buffer_size).bounds()
        else:
            # If region is already a simple rectangle, just use it
            sample_region = region

        # Extract data using the adjusted region and scale
        arr = geemap.ee_to_numpy(image, region=sample_region, scale=actual_scale)

        # Expected shape from geemap: (H, W, C) -> Change to (C, H, W)
        arr = np.moveaxis(arr, -1, 0)

        # Ensure correct size by resampling with PIL
        c, h, w = arr.shape
        if h != 512 or w != 512:
            arr_resized = np.stack([
                np.array(Image.fromarray(band).resize((512, 512), Image.BILINEAR))
                for band in arr
            ])
            arr = arr_resized

        return arr.astype(np.float32)
    except Exception as e:
        print(f"Error extracting NumPy array: {e}")
        return None

def split_image_into_tiles_and_save_with_offset(arr, rect_coords, num_tiles=NUM_TILES, img_size=IMG_SIZE,
                                   output_dir="tiles", scale=SCALE, tile_index_offset=0):
    """
    Splits a large image into multiple tiles and saves them as GeoTIFF files.
    Input shape: (C, H, W)
    Saves tiles to disk and returns a list of file paths
    Uses tile_index_offset to ensure unique naming across multiple calls
    """
    if arr is None:
        return []

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    c, h, w = arr.shape
    saved_paths = []

    # Calculate geographic bounds of the original image
    min_lon, min_lat, max_lon, max_lat = rect_coords['min_lon'], rect_coords['min_lat'], \
                                         rect_coords['max_lon'], rect_coords['max_lat']
    full_width = max_lon - min_lon
    full_height = max_lat - min_lat

    # For larger images, proceed with actual splitting
    # Calculate tile dimensions - try to create square tiles
    tiles_per_side = int(np.ceil(np.sqrt(num_tiles)))
    tile_h = h // tiles_per_side
    tile_w = w // tiles_per_side

    # Ensure minimum tile size
    tile_h = max(tile_h, img_size)
    tile_w = max(tile_w, img_size)

    # Calculate pixel size in geographic units
    pixel_width = full_width / w
    pixel_height = full_height / h

    tile_counter = 0
    for i in range(tiles_per_side):
        if tile_counter >= num_tiles:
            break

        for j in range(tiles_per_side):
            if tile_counter >= num_tiles:
                break

            # Calculate tile boundaries in pixel space
            start_h = min(i * tile_h, h - tile_h)
            end_h = min(start_h + tile_h, h)
            start_w = min(j * tile_w, w - tile_w)
            end_w = min(start_w + tile_w, w)

            # Extract tile
            tile = arr[:, start_h:end_h, start_w:end_w]

            # Calculate geographic bounds for this tile
            tile_min_lon = min_lon + start_w * pixel_width
            tile_min_lat = min_lat + start_h * pixel_height
            tile_max_lon = min_lon + end_w * pixel_width
            tile_max_lat = min_lat + end_h * pixel_height

            # Ensure tile is the correct size
            if tile.shape[1] != img_size or tile.shape[2] != img_size:
                # Resize using PIL
                tile_resized = np.stack([
                    np.array(Image.fromarray(band).resize((img_size, img_size), Image.BILINEAR))
                    for band in tile
                ])
                tile = tile_resized

            # Use offset in file naming to ensure uniqueness
            tile_path = os.path.join(output_dir, f"tile_{tile_counter + tile_index_offset:06d}.tif")

            # Create geotransform
            transform = from_bounds(
                tile_min_lon, tile_min_lat,
                tile_max_lon, tile_max_lat,
                img_size, img_size
            )

            # Move channels to last dimension for rasterio (C, H, W) -> (H, W, C)
            tile_hwc = np.moveaxis(tile, 0, 2)

            with rasterio.open(
                tile_path,
                'w',
                driver='GTiff',
                height=img_size,
                width=img_size,
                count=c,
                dtype=tile.dtype,
                crs='+proj=longlat +ellps=WGS84 datum=WGS84 +no_defs',
                transform=transform,
            ) as dst:
                # Write each band
                for band_idx in range(c):
                    dst.write(tile[band_idx], band_idx + 1)  # 1-based indexing for rasterio

                # Add band names as metadata
                for band_idx, band_name in enumerate(BANDS):
                    dst.set_band_description(band_idx + 1, band_name)

            saved_paths.append(tile_path)
            tile_counter += 1

    # If we don't have enough tiles, duplicate some with variations
    while len(saved_paths) < num_tiles:
        # Choose a random existing tile
        src_path = saved_paths[np.random.randint(0, len(saved_paths))]

        # Read the tile
        with rasterio.open(src_path) as src:
            profile = src.profile.copy()
            tile = np.stack([src.read(i + 1) for i in range(c)])

        # Add a tiny bit of noise for variation
        noise_factor = 0.02  # 2% variation
        noise = np.random.normal(1.0, noise_factor, tile.shape)
        new_tile = tile * noise
        # Clip to valid range
        new_tile = np.clip(new_tile, 0, np.max(tile) * 1.1)

        # Use offset in file naming for duplicated tiles too
        tile_path = os.path.join(output_dir, f"tile_dup_{len(saved_paths) + tile_index_offset:06d}.tif")

        with rasterio.open(tile_path, 'w', **profile) as dst:
            for band_idx in range(c):
                dst.write(new_tile[band_idx], band_idx + 1)

        saved_paths.append(tile_path)

    return saved_paths[:num_tiles]  # Ensure we return exactly num_tiles file paths

def process_coordinates(csv_path, output_dir="all_tiles"):
    """Process all coordinates from CSV and save all tiles to one folder."""
    coordinates = read_coordinates_from_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Keep a global counter for unique tile naming
    global_tile_counter = 2780

    for i, coord in enumerate(tqdm(coordinates)):
        # Get image
        image, rect = get_s2_image(coord, coord['timestamp_ms'])
        if image is None:
            print(f"Skipping coordinate {i} - no image found")
            continue

        # Extract numpy array
        arr = extract_numpy(image, rect)
        if arr is None:
            print(f"Skipping coordinate {i} - failed to extract array")
            continue

        # Pass the global counter to the split function to maintain unique names
        tile_paths = split_image_into_tiles_and_save_with_offset(
            arr,
            coord,
            num_tiles=NUM_TILES,
            img_size=128,
            output_dir=output_dir,
            tile_index_offset=global_tile_counter
        )

        # Update the global counter
        global_tile_counter += len(tile_paths)
        print(f"Processed coordinate {i}: {len(tile_paths)} tiles saved, total: {global_tile_counter}")

    print(f"Total: {global_tile_counter} tiles saved to {output_dir}")

def save_batch_npz(output_dir, X_batch, Y_batch, batch_idx):
    """Saves a single batch to disk as a compressed .npz file."""
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"batch_{batch_idx:03d}.npz")
    np.savez_compressed(filename, X=np.stack(X_batch), Y=np.stack(Y_batch))

def build_dataset(folder_path1, folder_path2, output_dir, batch_size=500, attenuation=0.05):
    """
    Builds and saves dataset in batches to prevent RAM overflow.

    Args:
        folder_path1 (str): Folder with plume .tif files (e.g., "1_label.tif").
        folder_path2 (str): Folder with Sentinel .tif files (e.g., "tile_*.tif").
        output_dir (str): Directory to save .npz batch files.
        batch_size (int): Number of samples per saved file.
        attenuation (float): Methane plume absorption factor.
    """
    plume_files = sorted(glob(os.path.join(folder_path1, "*_label.tif")))
    sentinel_files = sorted(glob(os.path.join(folder_path2, "tile_*.tif")))

    X_batch, Y_batch = [], []
    corrupted_files = []
    num_plumes = len(plume_files)
    target_size = (128, 128)
    batch_idx = 0

    for i, sentinel_path in tqdm(enumerate(sentinel_files), total=len(sentinel_files)):
        plume_path = plume_files[i % num_plumes]  # cycle through plume masks

        try:
            with rasterio.open(sentinel_path) as src:
                sentinel = src.read().astype(np.float32) / 10000.0
            with rasterio.open(plume_path) as src:
                plume = src.read(1).astype(np.float32)
        except Exception as e:
            corrupted_files.append((os.path.basename(sentinel_path), os.path.basename(plume_path), str(e)))
            continue

        rgb_tensor = torch.tensor(sentinel)
        plume_tensor = torch.tensor(plume).unsqueeze(0)  # (1, H, W)

        rgb_resized = resize(rgb_tensor, target_size, interpolation=2)
        plume_resized = resize(plume_tensor, target_size, interpolation=2)
        plume_mask = (plume_resized > 0.1).float()

        t_minus_1 = rgb_resized.clone()
        t = rgb_resized.clone()
        t[-1] -= plume_mask[0] * attenuation
        t[-1] = torch.clamp(t[-1], 0, 1)

        input_pair = torch.cat([t_minus_1, t], dim=0)
        X_batch.append(input_pair.numpy())
        Y_batch.append(plume_mask.numpy())

        if len(X_batch) >= batch_size:
            save_batch_npz(output_dir, X_batch, Y_batch, batch_idx)
            batch_idx += 1
            X_batch, Y_batch = [], []  # clear memory

    # Save remaining data
    if X_batch:
        save_batch_npz(output_dir, X_batch, Y_batch, batch_idx)

    print(f"\n✅ Saved batches to: {output_dir}")
    print(f"Total batches: {batch_idx + (1 if X_batch else 0)}")
    print(f"Corrupted files: {len(corrupted_files)}")
    for item in corrupted_files:
        print("❌", item)
