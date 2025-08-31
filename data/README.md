# Data Directory

This directory is intended for storing the datasets required by the project.

**NOTE:** The actual data files are not tracked by Git.

## Required Datasets

### MVTec AD Dataset
- **Download**: Visit [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/)
- **Request Access**: Fill out the form to get download links
- **Required Classes**: `carpet` and `grid` classes
- **Structure**: Extract to `data/mvtec/` with the following layout:
  ```
  data/mvtec/
  ├── carpet/
  │   ├── train/          # Normal training images
  │   ├── test/           # Test images (normal + anomalous)
  │   └── ground_truth/   # Ground truth masks
  └── grid/
      ├── train/          # Normal training images
      ├── test/           # Test images (normal + anomalous)
      └── ground_truth/   # Ground truth masks
  ```

**Important**: This dataset cannot be downloaded automatically - manual download required.

### Note on GKD Dataset
The GKD dataset mentioned in the original research was a private industrial dataset and is not publicly available. This project focuses on the publicly available MVTec AD dataset for demonstration and research purposes.
