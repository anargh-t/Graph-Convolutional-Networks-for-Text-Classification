# Topic Graph Convolutional Networks (TopicGCN)

TopicGCN replaces the classic TextGCN document-word graph with a **document-topic-topic** heterogeneous graph.  
Key ideas:
- Build topics with LDA and connect documents to topics via θ<sub>d,k</sub>
- Embed topics via the average Word2Vec representation of their top words
- Connect topics via cosine similarity to capture semantic proximity
- Train a two-layer GCN on this compact, interpretable graph

## Current Benchmark

| Dataset | Accuracy | Notes |
|---------|----------|-------|
| R8      | **94.11%** | TopicGCN (single run, 50 topics) |

> To reproduce the result, run `python build_graph.py --dataset R8 --num_topics 50` followed by  
> `python trainer.py --dataset R8 --times 1`.

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

### 1. (Optional) Clean raw corpora
```bash
python data_processor.py --dataset R8   # or 20ng / mr / ...
```
Pre-cleaned files already exist under `data/text_dataset/clean_corpus/`.

### 2. Build Document-Topic-Topic graph
```bash
# Build a graph for a single dataset
python build_graph.py --dataset R8 --num_topics 50

# Build graphs for all supported datasets with custom thresholds
python build_graph.py --num_topics 70 \
    --doc_topic_threshold 0.015 \
    --topic_topic_threshold 0.35
```
Artifacts are saved in `data/graph/`:
- `R8_topic.txt` – weighted edge list
- `R8_topic_model.pkl` – serialized LDA + embeddings
- `R8_topic_nodes.csv` – Protégé-compatible node definitions
- `R8_topic_edges.csv` – Protégé-compatible edge definitions

See [TOPIC_GCN_GUIDE.md](TOPIC_GCN_GUIDE.md#6-visualizing-graphs-in-protégé) for instructions on visualizing graphs in Protégé.

### 3. Train TopicGCN
```bash
# Train a single run
python trainer.py --dataset R8 --times 1

# Average across multiple random seeds
python trainer.py --dataset 20ng --times 5
```
Training automatically logs detailed metrics to  
`results/<dataset>_topic_training_results.txt`.

### 4. Reproducible experiments (recommended for publications)
```bash
# Edit the YAML to tweak hyperparameters
python run_experiment.py --config experiments/r8.yaml
```
This orchestrator builds the graph, trains TopicGCN, runs topic inspection, and stores all artifacts under `experiments/<dataset>/`.

### 5. Inspect topics (optional but recommended)
```bash
python inspect_topics.py --dataset R8 \
    --top_n_words 20 \
    --top_n_docs 3
```
This prints topic summaries and saves them to `results/<dataset>_topic_inspection.txt`.

## Project Structure

```
├── data/
│   ├── graph/              # Generated graph files
│   └── text_dataset/      # Text datasets and processed corpora
├── experiments/           # YAML configs + generated logs/results
├── build_graph.py         # Document-topic graph construction
├── trainer.py             # TopicGCN training script
├── run_experiment.py      # Pipeline orchestrator
├── topic_model.py         # LDA + topic embedding utilities
├── inspect_topics.py      # Topic visualization helpers
├── data_processor.py      # Text preprocessing and cleaning
├── layer.py               # GCN model definition
├── utils.py               # Utility functions
└── requirements.txt       # Dependencies
```

## Useful References
- [Graph Convolutional Networks for Text Classification](https://arxiv.org/abs/1809.05679)
- [Latent Dirichlet Allocation](https://jmlr.org/papers/v3/blei03a.html)

See `TOPIC_GCN_GUIDE.md` for a deeper dive into hyperparameters, thresholds, and troubleshooting tips.
