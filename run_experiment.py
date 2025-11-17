"""
Experiment Orchestrator for TopicGCN

This script reads a YAML configuration file and automates the full pipeline:
1. Build the document-topic-topic graph
2. Train the TopicGCN model
3. Inspect topics and save qualitative summaries

Each step is logged to experiment-specific directories so the project can be
shared or published with reproducible artifacts.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent


def run_command(cmd, log_file):
    """Run a shell command and stream output to a log file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n\n")
        log.flush()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT,
            text=True,
        )
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="")
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(
                f"Command {' '.join(cmd)} failed with exit code {process.returncode}. "
                f"See log: {log_file}"
            )


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_graph(cfg: dict, logs_dir: Path):
    dataset = cfg["dataset"]
    build_cfg = cfg.get("build", {})

    cmd = [
        sys.executable,
        "build_graph.py",
        "--dataset",
        dataset,
        "--num_topics",
        str(build_cfg.get("num_topics", cfg.get("num_topics", 50))),
        "--doc_topic_threshold",
        str(build_cfg.get("doc_topic_threshold", 0.02)),
        "--topic_topic_threshold",
        str(build_cfg.get("topic_topic_threshold", 0.3)),
        "--min_df",
        str(build_cfg.get("min_df", 2)),
        "--max_df",
        str(build_cfg.get("max_df", 0.95)),
    ]

    if not build_cfg.get("use_word2vec", True):
        cmd.append("--no_word2vec")

    run_command(cmd, logs_dir / "build.log")


def train_model(cfg: dict, results_dir: Path, logs_dir: Path):
    dataset = cfg["dataset"]
    train_cfg = cfg.get("train", {})
    times = train_cfg.get("times", 1)

    cmd = [
        sys.executable,
        "trainer.py",
        "--dataset",
        dataset,
        "--times",
        str(times),
        "--output_dir",
        str(results_dir),
    ]

    run_command(cmd, logs_dir / "train.log")


def inspect_topics_cli(cfg: dict, results_dir: Path, logs_dir: Path):
    dataset = cfg["dataset"]
    inspect_cfg = cfg.get("inspect", {})

    cmd = [
        sys.executable,
        "inspect_topics.py",
        "--dataset",
        dataset,
        "--top_n_words",
        str(inspect_cfg.get("top_n_words", 20)),
        "--top_n_docs",
        str(inspect_cfg.get("top_n_docs", 5)),
        "--output_dir",
        str(results_dir),
    ]

    if not inspect_cfg.get("heatmap", True):
        cmd.append("--no_heatmap")
    if inspect_cfg.get("no_save", False):
        cmd.append("--no_save")
    else:
        # If we plan to save, optionally set custom heatmap path (within results dir)
        heatmap_path = inspect_cfg.get("heatmap_path")
        if heatmap_path:
            cmd.extend(["--heatmap_path", heatmap_path])

    run_command(cmd, logs_dir / "inspect.log")


def main():
    parser = argparse.ArgumentParser(description="Run a structured TopicGCN experiment.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g., experiments/r8.yaml)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_config(config_path)
    dataset = cfg["dataset"]

    exp_dir = config_path.parent / dataset.lower()
    logs_dir = exp_dir / "logs"
    results_dir = exp_dir / "results"

    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config_used.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    print(f"\n==> Running experiment for dataset: {dataset}")
    print(f"Config: {config_path}")
    print(f"Artifacts: {exp_dir}\n")

    build_graph(cfg, logs_dir)
    train_model(cfg, results_dir, logs_dir)
    inspect_topics_cli(cfg, results_dir, logs_dir)

    print(f"\nExperiment complete! Results saved under {exp_dir}")


if __name__ == "__main__":
    main()

