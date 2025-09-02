# Industrial Anomaly Detection Using Transformer-Based Super-Resolution

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An investigation into the novel application of Super-Resolution (SR) models, including a Transformer-based architecture, for unsupervised anomaly detection on industrial manufacturing data.

---

## 🎯 Abstract

In industrial manufacturing, reliably detecting visual anomalies is critical for quality control, but state-of-the-art methods often require extensive labeled data. This project introduces a novel, unsupervised approach by repurposing Super-Resolution (SR) models for anomaly detection. The core hypothesis is that models trained exclusively on anomaly-free images will fail to accurately reconstruct anomalous regions. By training two advanced SR models (DRN and the Transformer-based DRCT) on the MVTec AD dataset, this thesis measures the reconstruction error to distinguish between normal and defective samples. While the specialized EfficientAD model showed more robust overall performance, the SR-based approach achieved comparable, and in some cases near-equal, performance, validating its potential as a viable alternative for specific industrial use cases.

## ✨ Key Features & Skills Demonstrated

*   **Technologies:** Python, PyTorch, Pandas, Scikit-learn, Matplotlib, JupyterLab
*   **Infrastructure:** **NVIDIA H100 GPUs, Slurm Workload Manager, Bash Scripting**
*   **Concepts:** Deep Learning, Unsupervised Learning, Anomaly Detection, **Computer Vision**, **Generative Models**, Super-Resolution, Transformers, High-Performance Computing (HPC)

## 📊 Key Result

The chart below compares the anomaly detection performance (AUC Score) of the proposed Super-Resolution models (DRN-L, DRCT-L) against the state-of-the-art EfficientAD model across various industrial datasets and image configurations.

![Model Performance Comparison](assets/results_graph.png)

*Caption: The SR-based models demonstrate competitive performance against a specialized anomaly detection model, particularly on the MVTec Grid dataset, validating the core hypothesis that reconstruction error can serve as an effective proxy for anomaly detection.*

---

## 🚀 Quick Start

### Prerequisites

*   Python 3.9+
*   NVIDIA GPU with CUDA support (recommended for training)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Benedict3007/anomaly-detection-super-resolution.git
    cd anomaly-detection-super-resolution
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Setup the project:**
    ```bash
    python run.py setup
    ```

5.  **Download the MVTec AD dataset manually:**
    - Visit: https://www.mvtec.com/company/research/datasets/mvtec-ad/
    - Fill out the form to request access
    - Download the dataset from the email link
    - Extract to `data/mvtec/` folder
    - Ensure structure: `data/mvtec/carpet/`, `data/mvtec/grid/`

6.  **Test the installation:**
    ```bash
    python run.py test
    ```

### Basic Usage

The project provides a simple CLI for basic operations:

```bash
# Show help
python run.py help

# Setup project structure
python run.py setup

# Run basic tests
python run.py test
```

### Dataset Requirements

**MVTec AD Dataset:**
- **Download**: Visit [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/)
- **Access**: Fill out the form to request access (free for research)
- **Required Classes**: `carpet` and `grid` classes
- **Structure**: Extract to `data/mvtec/` with train/test splits
- **Note**: This dataset is not publicly downloadable via scripts - manual download required

### Training

You can train the SR models via the main CLI. Key arguments:

```bash
# DRCT (Transformer SR) on MVTec grid, 128px HR, scale x4
python src/main.py \
  --model-type drct \
  --dataset mvtec \
  --classe grid \
  --resolution 128 \
  --scale 4 \
  --epochs 100 \
  --batch-size 4

# DRN-L (Residual SR) on MVTec carpet, RGB
python src/main.py \
  --model-type drn-l \
  --dataset mvtec \
  --classe carpet \
  --resolution 128 \
  --scale 4 \
  --epochs 100 \
  --batch-size 4

# Optional: load defaults from YAML
python src/main.py --config configs/default.yaml --epochs 50

# Device override examples
python src/main.py --device cpu
python src/main.py --device mps   # Apple Silicon
```

### Evaluation

Evaluate a saved run directory or a specific checkpoint using the standalone evaluator:

```bash
# From a run directory (auto-detects settings)
python -m src.evaluate --run-dir workspace/experiment/drct/<your_run_dir>

# Or with an explicit checkpoint
python -m src.evaluate --checkpoint /path/to/model_best.pt --dataset mvtec --classe grid --resolution 128 --scale 4
```

Note:
- Training (`src/main.py`) validates super-resolution quality on a good-only validation set using PSNR/SSIM and does not compute anomaly AUC.
- Anomaly AUC is computed only via the evaluator (`src/evaluate.py`) on the test set (good + bad).

---

## 📁 Project Structure

```
.
├── src/                    # Source code
│   ├── main.py            # Main training script (CLI interface)
│   ├── model.py            # Model architectures
│   ├── data.py             # Data loading and preprocessing
│   ├── trainer.py          # Training loop
│   ├── loss.py             # Loss functions
│   ├── checkpoint.py       # Model checkpointing
│   └── helpers.py          # Utility functions
├── data/                   # Dataset directory
│   └── mvtec/             # MVTec AD dataset (carpet, grid classes)
├── results/                # Training outputs and results
├── notebooks/              # Jupyter notebooks for analysis
├── requirements.txt        # Python dependencies
├── setup.py               # Package configuration
├── run.py                 # Simple CLI runner
└── test_basic.py          # Basic functionality tests
```

### Architecture Overview (brief)

- `src/main.py`: CLI entry for training and config parsing; validates PSNR/SSIM on val.
- `src/data.py`: `SRData` dataset and `Data` loaders; MVTec-specific wrapper.
- `src/model.py`: Model wrapper, device handling, checkpoint load/save, `DownBlock`.
- `src/drn.py`: DRN-L super-resolution architecture.
- `src/drct.py`: Transformer-based DRCT architecture.
- `src/trainer.py`: Training/eval loop, optimizers, schedulers, AMP handling.
- `src/loss.py`: Loss factory (L1/MSE/PSNR/SSIM).
- `src/checkpoint.py`: Run directory, logging, image saving, metric plotting.
- `src/metrics.py`: Unified SSIM/PSNR utilities (numpy and torch).
- `src/evaluate.py`: Standalone evaluation entrypoint for saved checkpoints.

---

## 🔧 Development

- Run tests: `python run.py test`
- Useful Make targets:
  - `make setup` → create local dirs
  - `make test` → smoke tests
  - `make lint` → flake8 (optional)
  - `make format` → black+isort (optional)

---

## 📚 Documentation

- **Installation Guide**: See Quick Start section above
- **API Reference**: Coming soon as training pipeline is completed
- **Examples**: Check the `notebooks/` directory for analysis examples
- **Troubleshooting**: Run `python run.py test` to diagnose issues

---

## 🤝 Contributing

This project is currently in active development. Contributions are welcome once the basic training pipeline is complete.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This work was conducted as part of the Master's Thesis for the Information Systems Engineering program at University of Applied Sciences Aachen.

Special thanks to the authors of the DRN, DRCT, and EfficientAD papers.

---

## 📞 Support

If you encounter issues:

1. Run `python run.py test` to check your setup
2. **Ensure the MVTec AD dataset is downloaded and placed in `data/mvtec/`**
3. Check the error messages for specific issues
4. Ensure all dependencies are installed correctly
5. Verify your Python version (3.9+ required)

For development questions, please open an issue on GitHub.
