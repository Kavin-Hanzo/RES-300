# Deep Learning Framework for Methane Plume Segmentation using Sentinel-2 Multispectral Imagery

This repository implements a scalable deep learning framework for detecting methane emissions using dual-temporal Sentinel-2 imagery and synthetic training data. It is based on a transformer-driven architecture (SwinUNETR) trained entirely on physically simulated methane plumes.

📘 Overview
Methane (CH₄) is a potent greenhouse gas and a critical target for emission monitoring. This repository contains the full implementation of a framework that:

- Preprocesses Sentinel-2 imagery into paired temporal tiles

- Generates synthetic methane plumes using a Gaussian dispersion model

- Applies Beer–Lambert law to embed spectral absorption into Band 12

- Trains a SwinUNETR transformer-based segmentation model

- Supports evaluation on synthetic and real-world datasets (e.g., Carbon Mapper)

`Use quick_test.py to get the output`
 - saves the output plots in `segmentation_results`

📜 Citation
If you use this codebase, please cite our upcoming paper (pending DOI). Also cite:

Hatamizadeh et al. (2022) for SwinUNETR

Rouet-Leduc & Hulbert (2024) for transformer-based plume detection

Micallef & Micallef (2024), Kiteto & Mecha (2024) for plume simulation models
