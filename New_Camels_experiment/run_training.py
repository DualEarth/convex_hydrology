"""
run_training.py
────────────────
Local training launcher for this folder's runoff.yml.

Bypasses neuralhydrology.training.train.start_training(), which in this
machine's installed package is hardcoded to import a trainer class from
`my_custom_trainer` (a leftover from a different experiment) instead of
reading the trainer module/class from the run config. This script does
what nh_run.py's start_run() does, but loads the trainer dynamically from
the `trainer: module / class` keys in the config file, as neuralhydrology
normally does.

Usage:
    python run_training.py [config_file]   # default: runoff.yml
"""
import importlib
import sys
from pathlib import Path

from neuralhydrology.utils.config import Config


def start_run(config_file: Path):
    cfg = Config(config_file)

    trainer_cfg = cfg.as_dict().get("trainer", {})
    module_name = trainer_cfg.get("module")
    class_name = trainer_cfg.get("class")
    if not module_name or not class_name:
        raise ValueError("Config must specify trainer.module and trainer.class")

    module = importlib.import_module(module_name)
    trainer_cls = getattr(module, class_name)

    trainer = trainer_cls(cfg=cfg)
    trainer.initialize_training()
    trainer.train_and_validate()

    return trainer.cfg.run_dir


if __name__ == "__main__":
    config_path = Path(sys.argv[1] if len(sys.argv) > 1 else "runoff.yml")
    run_dir = start_run(config_path)
    print(f"RUN_DIR={run_dir}")
