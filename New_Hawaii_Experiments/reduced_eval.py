import sys
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from neuralhydrology.utils.config import Config
from neuralhydrology.modelzoo import get_model
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datautils.utils import load_scaler
from neuralhydrology.evaluation.evaluate import start_evaluation

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class StateReducer(nn.Module):
    def __init__(self, original_dim: int = 256, reduced_dim: int = 3):
        super().__init__()
        if reduced_dim >= original_dim:
            raise ValueError(f"Reduced dimension ({reduced_dim}) must be less than original ({original_dim}).")
        self.reducer = nn.Sequential(
            nn.Linear(original_dim, 32), nn.ReLU(),
            nn.Linear(32, 16),           nn.ReLU(),
            nn.Linear(16, reduced_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(reduced_dim, 16),  nn.ReLU(),
            nn.Linear(16, 32),           nn.ReLU(),
            nn.Linear(32, original_dim),
        )

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x[-1, :, :] if (x.shape[0] <= 4 and x.shape[1] > 1) else x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        return x

    def forward(self, x):
        return self.reducer(self._normalize_input(x))

    def reconstruct(self, x):
        x_norm = self._normalize_input(x)
        return self.decoder(self.reducer(x_norm))

    def reconstruction_loss(self, x):
        x_norm = self._normalize_input(x)
        return F.mse_loss(self.decoder(self.reducer(x_norm)), x_norm)


def find_latest_epoch(run_dir: Path) -> int:
    epochs = [
        int(f.stem.replace("model_epoch", ""))
        for f in run_dir.glob("model_epoch*.pt")
    ]
    if not epochs:
        raise FileNotFoundError(f"No model epoch files found in {run_dir}")
    return max(epochs)


def main():
    if len(sys.argv) < 2:
        LOGGER.error("Usage: python eval_plot.py <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    cfg = Config(run_dir / "config.yml")
    cfg.run_dir = run_dir

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.device = str(device)

    epoch = find_latest_epoch(run_dir)
    LOGGER.info(f"Using epoch {epoch}")

    # Load model 
    model = get_model(cfg).to(device)
    model.load_state_dict(torch.load(run_dir / f"model_epoch{epoch:03d}.pt", map_location=device))
    model.eval()

    # Load reducer
    reducer_dim = cfg.as_dict().get("trainer", {}).get("reduced_state_dimension", 3)
    reducer = StateReducer(cfg.hidden_size, reducer_dim).to(device)
    reducer_path = run_dir / f"state_reducer_epoch{epoch:03d}.pth"
    if reducer_path.exists():
        reducer.load_state_dict(torch.load(reducer_path, map_location=device))
        LOGGER.info(f"Loaded reducer from {reducer_path}")
    else:
        LOGGER.warning("Reducer weights not found. Using untrained reducer.")
    reducer.eval()

    # Run NH metrics
    start_evaluation(cfg=cfg, run_dir=run_dir, epoch=epoch, period="test")

    # Collect states per basin
    scaler = load_scaler(run_dir)
    save_dir = run_dir / "internal_states" / "test"
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(cfg.test_basin_file), "r") as f:
        basin_ids = [l.strip() for l in f if l.strip()]

    for basin_id in tqdm(basin_ids, desc="Basins"):
        dataset = get_dataset(cfg=cfg, is_train=False, period="test", basin=basin_id, scaler=scaler)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, collate_fn=dataset.collate_fn)

        all_cell, all_reduced = [], []

        with torch.no_grad():
            for batch in loader:
                for key in batch:
                    if key.startswith("x_d"):
                        batch[key] = {k: v.to(device) for k, v in batch[key].items()}
                    elif not key.startswith("date"):
                        batch[key] = batch[key].to(device)

                batch = model.pre_model_hook(batch, is_train=False)
                preds = model(batch)

                c_n = preds.get("c_n")
                if c_n is None:
                    continue
                c_n = c_n.detach()

                all_cell.append(c_n.cpu())
                all_reduced.append(reducer(c_n).cpu())

        if not all_cell:
            LOGGER.warning(f"No states for basin {basin_id}, skipping.")
            continue

        safe_id = basin_id.replace(".", "_")
        torch.save(torch.cat(all_cell,          dim=0), save_dir / f"c_n_epoch{epoch:03d}_basin_{safe_id}.pt")
        torch.save(torch.cat(all_reduced,        dim=0), save_dir / f"c_n_reduced_epoch{epoch:03d}_basin_{safe_id}.pt")

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
