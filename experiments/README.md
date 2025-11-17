# Experiments

This directory hosts configuration files and artifacts for structured TopicGCN experiments.

## Layout

```
experiments/
├── README.md                 # This file
├── r8.yaml                   # Example configuration
└── <dataset>/                # Auto-generated when running an experiment
    ├── config_used.yaml      # Copy of the YAML used for the run
    ├── logs/                 # build/train/inspect logs
    └── results/              # Training summaries, JSON reports, topic inspection outputs
```

## Running an Experiment

1. Create or edit a YAML file (see `r8.yaml` for reference).
2. Execute:
   ```bash
   python run_experiment.py --config experiments/r8.yaml
   ```
3. Review artifacts under `experiments/<dataset>/`.

The orchestrator handles graph building, training, and topic inspection with consistent logging, making it easier to cite or publish the results.

