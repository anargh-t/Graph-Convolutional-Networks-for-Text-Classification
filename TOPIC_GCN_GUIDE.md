# TopicGCN: Document-Topic-Topic Graph Guide

This guide covers the full TopicGCN workflow—graph construction, model training, topic inspection, and reproducible experiments.

---

## 1. Quick Pipeline

```bash
# Optional: clean a dataset (already available for default corpora)
python data_processor.py --dataset R8

# Build the topic graph (creates data/graph/R8_topic.txt + topic model)
python build_graph.py --dataset R8 --num_topics 50

# Train TopicGCN (averages metrics over --times random seeds)
python trainer.py --dataset R8 --times 1

# Inspect topics (prints + saves to results/R8_topic_inspection.txt)
python inspect_topics.py --dataset R8 --top_n_words 20 --top_n_docs 5

# Recommended: orchestrate everything via YAML
python run_experiment.py --config experiments/r8.yaml
```

---

## 2. Graph Construction (`build_graph.py`)

```bash
# Single dataset with custom thresholds
python build_graph.py \
    --dataset 20ng \
    --num_topics 70 \
    --doc_topic_threshold 0.015 \
    --topic_topic_threshold 0.35

# All supported datasets (omit --dataset)
python build_graph.py --num_topics 60
```

| Flag | Description | Default |
|------|-------------|---------|
| `--dataset` | Dataset name (omit to process all) | all supported |
| `--num_topics` | Number of LDA topics | 50 |
| `--doc_topic_threshold` | Minimum θ<sub>d,k</sub> for doc-topic edge | 0.02 |
| `--topic_topic_threshold` | Minimum cosine similarity for topic-topic edge | 0.30 |
| `--no_word2vec` | Skip Word2Vec training | Disabled |
| `--min_df`, `--max_df` | LDA vocabulary filtering | 2 / 0.95 |

Outputs per dataset:
- `data/graph/<dataset>_topic.txt`
- `data/graph/<dataset>_topic_model.pkl`

---

## 3. Training (`trainer.py`)

```bash
# Average over multiple seeds and save to a custom dir
python trainer.py --dataset 20ng --times 5 --output_dir experiments/20ng/results
```

Arguments:
- `--dataset`: Dataset to train (default `R8`)
- `--times`: Number of random seeds (default `1`)
- `--output_dir`: Destination for logs + JSON summaries (default `results/`)

Outputs:
- `<output_dir>/<dataset>_topic_training_results.txt`
- `<output_dir>/<dataset>_topic_training_results.json`

Each file contains hyperparameters, aggregated metrics, per-epoch histories, and test results.

---

## 4. Topic Inspection (`inspect_topics.py`)

```bash
python inspect_topics.py --dataset R8 \
    --top_n_words 20 \
    --top_n_docs 5 \
    --output_dir experiments/R8/results \
    --heatmap_path experiments/R8/results/R8_heatmap.png
```

Key flags:
- `--top_n_words`, `--top_n_docs`: Display controls
- `--no_heatmap`: Skip similarity heatmap creation
- `--heatmap_path`: Custom output path (defaults to `output_dir/dataset_topic_heatmap.png`)
- `--output_dir`: Directory for inspection logs (default `results/`)
- `--no_save`: Print only (skip writing report)

---

## 5. Reproducible Experiments (`run_experiment.py`)

1. Create or edit a YAML config (see `experiments/r8.yaml`).
2. Run `python run_experiment.py --config experiments/r8.yaml`.
3. Artifacts are stored under `experiments/<dataset>/`:
   - `logs/`: build/train/inspect logs
   - `results/`: training summaries, JSON files, topic inspection outputs, heatmaps

This layout is publication-friendly—attach the folder as supplementary material for a paper.

---

## 6. Visualizing Graphs in Protégé

Every graph build now produces Protégé-compatible CSV files:
- `data/graph/<dataset>_topic_nodes.csv`
- `data/graph/<dataset>_topic_edges.csv`

### 6.1. Installing Protégé and CSV Import Plugin

1. **Download Protégé**: Get the latest version from [https://protege.stanford.edu/](https://protege.stanford.edu/)
2. **Install CSV Import Plugin (Cellfie)**:
   - Open Protégé
   - Go to `File → Check for plugins…`
   - Search for "Cellfie" or "CSV Import"
   - Install and restart Protégé
   - Enable the tab: `Window → Tabs → CSV Import`

### 6.2. Creating the Ontology Structure

Before importing, create the basic ontology structure:

1. **Create Classes**:
   - In the "Classes" tab, create two classes:
     - `DocumentNode` (subclass of `owl:Thing`)
     - `TopicNode` (subclass of `owl:Thing`)

2. **Create Object Properties**:
   - In the "Object Properties" tab, create:
     - `hasTopic` (domain: `DocumentNode`, range: `TopicNode`)
     - `relatedToTopic` (domain: `TopicNode`, range: `TopicNode`)

3. **Create Data Properties** (optional, for edge weights):
   - In the "Data Properties" tab, create:
     - `edgeWeight` (domain: `owl:Thing`, range: `xsd:float`)

### 6.3. Importing Nodes

1. **Open CSV Import Tab**:
   - `Window → Tabs → CSV Import` (if not visible)

2. **Load Nodes CSV**:
   - Click "Load CSV" and select `<dataset>_topic_nodes.csv`
   - Verify columns: `node_id`, `node_type`, `label`

3. **Map Columns to Ontology**:
   - **Individual Creation**: Map `node_id` to create individuals
     - Pattern: `http://example.org/topicgcn#Node_{node_id}`
   - **Class Assignment**: Map `node_type` to assign classes
     - When `node_type = "document"` → assign class `DocumentNode`
     - When `node_type = "topic"` → assign class `TopicNode`
   - **Label Assignment**: Map `label` to `rdfs:label` annotation property
     - This will display the label in the visualization

4. **Execute Import**:
   - Review the mapping preview
   - Click "Import" to create all node individuals

### 6.4. Importing Edges

1. **Load Edges CSV**:
   - Click "Load CSV" and select `<dataset>_topic_edges.csv`
   - Verify columns: `source_id`, `target_id`, `edge_type`, `weight`

2. **Map Columns to Properties**:
   - **Source/Target**: Map `source_id` and `target_id` to individuals
     - Pattern: `http://example.org/topicgcn#Node_{source_id}` and `Node_{target_id}`
   - **Property Selection**: Map `edge_type` to object properties
     - When `edge_type = "doc-topic"` → use property `hasTopic`
     - When `edge_type = "topic-topic"` → use property `relatedToTopic`
   - **Weight (Optional)**: Map `weight` to data property `edgeWeight`
     - This stores the edge weight as a numeric value

3. **Execute Import**:
   - Review the mapping preview
   - Click "Import" to create all edge relationships

### 6.5. Visualizing the Graph

1. **Using OntoGraf Plugin**:
   - Install OntoGraf: `File → Check for plugins…` → search "OntoGraf"
   - Enable: `Window → Tabs → OntoGraf`
   - Select a node in the "Individuals" tab
   - View its connections in the OntoGraf visualization

2. **Using OWLViz Plugin** (alternative):
   - Install OWLViz: `File → Check for plugins…` → search "OWLViz"
   - Enable: `Window → Tabs → OWLViz`
   - Select classes or individuals to visualize their relationships

3. **Custom Visualization Tips**:
   - **Filter by Class**: Show only `DocumentNode` or `TopicNode` to reduce clutter
   - **Filter by Property**: Focus on `hasTopic` or `relatedToTopic` relationships
   - **Color Coding**: Use different colors for document vs topic nodes
   - **Layout**: Try different layout algorithms (hierarchical, force-directed, etc.)

### 6.6. Exporting Visualizations

- **Screenshot**: Use Protégé's built-in export or take screenshots
- **GraphML Export**: Some plugins support exporting to GraphML format
- **OWL Export**: Save the ontology as OWL file for use in other tools

### 6.7. Troubleshooting

| Issue | Solution |
|-------|----------|
| CSV Import tab not visible | Install Cellfie plugin and restart Protégé |
| Individuals not created | Check IRI pattern matches existing individuals |
| Edges not showing | Verify object properties are correctly mapped |
| Graph too cluttered | Use filters to show subsets of nodes/edges |
| Performance issues | Import smaller subsets or use graph sampling |

---

## 7. Architecture Summary

| Component | Description |
|-----------|-------------|
| Nodes | Documents `[0, D)` and topics `[D, D+K)` |
| Document features | θ<sub>d</sub> (topic distribution, normalized/padded) |
| Topic features | Topic embedding (top-N Word2Vec average or topic-word distribution) |
| Edges | Doc-topic (θ<sub>d,k</sub> ≥ threshold) + topic-topic (cosine similarity ≥ threshold) |
| Model | Two-layer GCN (`layer.py`) with ReLU + dropout |

Topic modeling:
- LDA via `sklearn.decomposition.LatentDirichletAllocation`
- Word2Vec via `gensim.models.Word2Vec` (vector size 100, window 5, CBOW, 10 epochs)

---

## 7. Recommended Settings

- Small corpora (<10k docs): `num_topics = 20–50`
- Medium corpora (10k–100k docs): `num_topics = 50–100`
- Large corpora (>100k docs): `num_topics = 100–200`
- Raise thresholds for sparser graphs; lower them for denser graphs

---

## 8. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Topic model not found` | Run `build_graph.py` for that dataset first |
| `prettytable` / `gensim` missing | `pip install prettytable gensim` (or use `--no_word2vec`) |
| GPU OOM during training | Reduce `num_topics`, increase thresholds, or lower `nhid` |
| Heatmap fails to render | Install `matplotlib` + `seaborn` or pass `--no_heatmap` |

---

## 9. File Structure

```
GCN-Topic/
├── build_graph.py        # Document-topic graph builder
├── trainer.py            # TopicGCN trainer
├── inspect_topics.py     # Topic inspection helpers
├── run_experiment.py     # YAML-driven orchestrator
├── topic_model.py        # LDA + embedding utils
├── experiments/          # Configs + generated artifacts
├── data/
│   ├── graph/            # Generated graphs + topic models
│   └── text_dataset/     # Raw / cleaned corpora
└── results/              # Default logs (when --output_dir not supplied)
```

---

