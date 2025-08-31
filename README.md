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

### Training (Coming Soon)

The training pipeline is currently being restructured. Once complete, you'll be able to run:

```bash
# Train DRN-L model on MVTec carpet dataset
python src/main.py --dataset mvtec --class-name carpet --model drn-l --epochs 100

# Train DRCT model on MVTec grid dataset  
python src/main.py --dataset mvtec --class-name grid --model drct --epochs 500
```

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

---

## 🔧 Development Status

**Current Status:** 🟡 Basic Setup Complete - Training Pipeline in Development

- ✅ **Project Structure**: Clean, professional organization
- ✅ **Dependencies**: Fixed requirements and setup
- ✅ **CLI Interface**: Basic command-line tools working
- ✅ **Testing**: Basic functionality tests implemented
- 🔄 **Training Pipeline**: Being restructured for production use
- 🔄 **Model Implementation**: Core models need integration
- 🔄 **Data Loading**: Dataset handling needs implementation

**Next Steps:**
1. Implement data loading pipeline
2. Integrate DRN-L and DRCT models
3. Complete training loop
4. Add configuration management
5. Implement evaluation metrics

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
