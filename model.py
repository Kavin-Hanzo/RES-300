import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=20, embed_dim=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, embed_dim, H/patch_size, W/patch_size]
        x = self.proj(x)
        # [B, embed_dim, H/patch_size, W/patch_size] -> [B, (H/patch_size)*(W/patch_size), embed_dim]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (window_size, window_size)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        # Get pair-wise relative position index for each token in the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, window_size, window_size]
        coords_flatten = torch.flatten(coords, 1)  # [2, window_size*window_size]

        # [2, window_size*window_size, 1] - [2, 1, window_size*window_size]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, window_size*window_size, window_size*window_size]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [window_size*window_size, window_size*window_size, 2]
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # [window_size*window_size, window_size*window_size]

        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        """
        x: [num_windows*B, window_size*window_size, C]
        mask: [num_windows, window_size*window_size, window_size*window_size] or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B_, num_heads, N, C//num_heads]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # [B_, num_heads, N, N]

        # Add relative positional bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )  # [window_size*window_size, window_size*window_size, num_heads]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [num_heads, window_size*window_size, window_size*window_size]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # Apply mask
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size: window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, H, W, mask_matrix=None):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, window_size, window_size, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, window_size*window_size, C]

        # Window attention
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, window_size*window_size, C]

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, Hp, Wp, C]

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer.
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # Padding if needed
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float): Dropout rate.
        downsample (nn.Module | None): Downsample layer at the end of the layer.
    """
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0., downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = window_size // 2

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else self.shift_size,
                                 mlp_ratio=mlp_ratio, drop=drop)
            for i in range(depth)
        ])

        # Patch merging layer
        self.downsample = downsample

    def forward(self, x, H, W):
        # Calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]

        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, window_size, window_size, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, window_size*window_size]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, window_size*window_size, window_size*window_size]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        # Forward blocks
        for blk in self.blocks:
            x = blk(x, H, W, attn_mask)

        # Save the output before downsampling for skip connections
        skip = x

        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        return x, H, W, skip


class TransformerDecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=drop)

        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=drop)

        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, skip):
        # Self attention
        residual = x
        x = self.norm1(x)
        x = x.transpose(0, 1)  # [B, L, C] -> [L, B, C]
        skip = skip.transpose(0, 1)  # [B, L, C] -> [L, B, C]

        x, _ = self.self_attn(x, x, x)
        x = x.transpose(0, 1)  # [L, B, C] -> [B, L, C]
        x = residual + x

        # Cross attention
        residual = x
        x = self.norm2(x)
        x = x.transpose(0, 1)  # [B, L, C] -> [L, B, C]

        x, _ = self.cross_attn(x, skip, skip)
        x = x.transpose(0, 1)  # [L, B, C] -> [B, L, C]
        x = residual + x

        # MLP
        x = x + self.mlp(self.norm3(x))

        return x


class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, depth=1, mlp_ratio=4., drop=0.):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(out_dim, num_heads, mlp_ratio, drop)
            for _ in range(depth)
        ])

    def forward(self, x, skip, H, W):
        B = x.shape[0]

        # Rearrange to spatial format for upsampling
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.up(x)
        H, W = H * 2, W * 2

        # Rearrange back to sequence format
        x = rearrange(x, 'b c h w -> b (h w) c')

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, skip)

        return x, H, W


import numpy as np


class SwinUNETR(nn.Module):
    def __init__(
        self,
        in_channels=20,
        out_channels=1,
        img_size=128,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        drop_rate=0.1,
        decoder_depths=[1, 1, 1, 1]
    ):
        super().__init__()
        self.img_size = img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # Patch Embedding
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size=4)
        patches_resolution = img_size // 4
        self.patches_resolution = patches_resolution

        # Swin Transformer layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                downsample=PatchMerging(int(embed_dim * 2 ** i_layer)) if (i_layer < self.num_layers - 1) else None
            )
            self.layers.append(layer)

        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        for i_layer in range(self.num_layers-1, 0, -1):
            layer = UpBlock(
                in_dim=int(embed_dim * 2 ** i_layer),
                out_dim=int(embed_dim * 2 ** (i_layer-1)),
                num_heads=num_heads[i_layer-1],
                depth=decoder_depths[i_layer-1],
                mlp_ratio=mlp_ratio,
                drop=drop_rate
            )
            self.decoder_layers.append(layer)

        # Final layers
        self.norm = nn.LayerNorm(embed_dim)
        self.final_up = nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=4, stride=4)
        self.final_conv = nn.Conv2d(embed_dim // 2, out_channels, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # [B, patches_resolution*patches_resolution, embed_dim]
        H, W = self.patches_resolution, self.patches_resolution

        # Store skip connections
        skips = []

        # Encoder
        for i, layer in enumerate(self.layers):
            x, H, W, skip = layer(x, H, W)
            if i < len(self.layers) - 1:
                skips.append((skip, H, W))

        # Decoder
        for i, layer in enumerate(self.decoder_layers):
            skip, skip_H, skip_W = skips[-(i+1)]
            x, H, W = layer(x, skip, H, W)

        # Final output
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.final_up(x)  # [B, embed_dim//2, img_size, img_size]
        x = self.final_conv(x)  # [B, out_channels, img_size, img_size]

        return x


model = SwinUNETR()
trained_model, history, best_model_path = train_model(
    model,
    npz_path='/content/plume_dataset',
    epochs=10,
    save_dir='Swin',
    model_name='SwinUNETR',
    batch_size=8,
    lr=1e-3,
    device='cuda'
)

# Plot training history
plot_training_history(history, save_plot=True,plot_path="result.png")

