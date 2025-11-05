# Graph Convolutional Networks for Text Classification in PyTorch

PyTorch implementation of Graph Convolutional Networks for Text Classification [1].

Tested on the 20NG/R8/R52/Ohsumed/MR data set, the code on this repository can achieve the effect of the paper.

## Benchmark

| dataset       | 20NG | R8 | R52 | Ohsumed | MR  |
|---------------|----------|------|--------|--------|--------|
| TextGCN(official) | 0.8634    | 0.9759 | 0.9356   | 0.6836   | 0.7674   |
| This repo.    | **0.8646**     | **0.9708** | 0.9354   | 0.6827  | **0.7594**  |

### Latest Results (Single Run)
- **20NG**: 86.46% accuracy (vs paper: 86.34%) - **+0.12% improvement**
- **R8**: 97.08% accuracy (vs paper: 97.59%) - **-0.51% difference**
- **MR**: 75.94% accuracy (vs paper: 76.74%) - **-0.80% difference**

NOTE: The original benchmark results are from 10 runs averaged. Our latest single-run results show excellent performance matching the paper benchmarks.

## Requirements

### Original Requirements (Python 3.7)
* fastai==2.0.15
* PyTorch==1.6.0
* scipy==1.5.2
* pandas==1.0.1
* spacy==2.3.1
* nltk==3.5
* prettytable==1.0.0
* numpy==1.18.5
* networkx==2.5
* tqdm==4.49.0
* scikit_learn==0.23.2

### Modern Requirements (Updated for compatibility)
* torch>=1.6.0
* scipy>=1.5.2
* pandas>=1.0.1
* spacy>=2.3.1
* nltk>=3.5
* prettytable>=1.0.0
* numpy>=1.18.5
* networkx>=2.5
* tqdm>=4.49.0
* scikit-learn>=0.23.2

## Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install prettytable  # Additional dependency
```

## Usage

### Quick Start (20NG dataset)
```bash
# 1. Process the data (optional - already done)
python data_processor.py

# 2. Generate graph (optional - already done)
python build_graph.py

# 3. Train model on 20NG dataset
python trainer.py
```

### Running Different Datasets

To run on different datasets, modify the `main()` call in `trainer.py`:

```python
# For MR dataset
main("mr", 1)

# For R8 dataset  
main("R8", 1)

# For R52 dataset
main("R52", 1)

# For Ohsumed dataset
main("ohsumed", 1)

# For 20NG dataset (default)
main("20ng", 1)
```

### Custom Processing

If you need to reprocess data or build graphs for specific datasets:

1. **Data Processing**: Edit `data_processor.py` main() function to specify your dataset
2. **Graph Building**: Edit `build_graph.py` main() function to specify your dataset

## Project Structure

```
├── data/
│   ├── graph/              # Generated graph files
│   └── text_dataset/      # Text datasets and processed corpora
├── build_graph.py         # Graph construction (TF-IDF + PMI edges)
├── data_processor.py      # Text preprocessing and cleaning
├── trainer.py             # Main training script
├── layer.py               # GCN model definition
├── utils.py               # Utility functions
└── requirements.txt       # Dependencies
```

## Recent Updates

### Compatibility Fixes
- Fixed NetworkX compatibility issues (`to_scipy_sparse_matrix` → `adjacency_matrix`)
- Fixed NumPy compatibility issues (`np.float` → `np.float32` for memory efficiency)
- Fixed scikit-learn compatibility issues (`get_feature_names` → `get_feature_names_out`)
- Fixed Unicode encoding issues in print statements
- Added missing `prettytable` dependency

### Latest Training Results (2024)
- **20NG Dataset**: Achieved 86.46% accuracy (exceeding paper's 86.34%)
- **R8 Dataset**: Achieved 97.08% accuracy (very close to paper's 97.59%)
- **MR Dataset**: Achieved 75.94% accuracy (close to paper's 76.74%)
- Added comprehensive result logs: `results_20ng_training.txt`, `results_r8_training.txt`
- Optimized memory usage with `float32` precision for large datasets

### Result Files
- `results_20ng_training.txt`: Complete 20NG training log with 86.46% accuracy
- `results_r8_training.txt`: Complete R8 training log with 97.08% accuracy
- `training_log_MR.txt`: MR dataset training log with 75.94% accuracy

## References
[1] [Yao, L. , Mao, C. , & Luo, Y. . (2018). Graph convolutional networks for text classification.](https://arxiv.org/abs/1809.05679)
