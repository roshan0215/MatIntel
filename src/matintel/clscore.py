from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from tqdm import tqdm

try:
    import torch
except Exception:  # pragma: no cover - runtime dependency guard
    torch = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLSCORE_REPO = PROJECT_ROOT / "external" / "Synthesizability-PU-CGCNN"
TRAINED_MODELS_DIR = CLSCORE_REPO / "trained_models"
DEFAULT_RADIUS = 8.0
DEFAULT_MAX_NEIGHBORS = 12


@dataclass
class CLscoreResult:
    material_id: str
    clscore: float
    status: str


class CLscorePredictor:
    """CPU-only CLscore predictor using KAIST Synthesizability-PU-CGCNN checkpoints."""

    def __init__(
        self,
        repo_dir: Path = CLSCORE_REPO,
        model_dir: Path = TRAINED_MODELS_DIR,
        radius: float = DEFAULT_RADIUS,
        max_neighbors: int = DEFAULT_MAX_NEIGHBORS,
    ) -> None:
        self.repo_dir = Path(repo_dir)
        self.model_dir = Path(model_dir)
        self.radius = radius
        self.max_neighbors = max_neighbors

        self.device = "cpu"
        self.loaded = False
        self.available = False
        self.last_error = ""
        self.models: list[Any] = []
        self.nbr_fea_len = 41
        self.atom_fea_len = 92

    def setup(self) -> bool:
        if self.loaded:
            return self.available

        self.loaded = True
        if torch is None:
            self.last_error = "PyTorch is not installed in the active environment."
            self.available = False
            return False

        if not self.repo_dir.exists() or not self.model_dir.exists():
            self.last_error = (
                "Synthesizability-PU-CGCNN repository or trained_models folder not found. "
                "Run scripts/setup_clscore.ps1 first."
            )
            self.available = False
            return False

        import sys

        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))

        try:
            import importlib

            CrystalGraphConvNet = importlib.import_module("cgcnn.model_PU_learning").CrystalGraphConvNet
        except Exception as exc:
            self.last_error = f"Failed to import KAIST CGCNN model: {exc}"
            self.available = False
            return False

        checkpoints = sorted(self.model_dir.glob("checkpoint_bag_*.pth.tar"), key=_bag_sort_key)
        max_models = int(os.getenv("MATINTEL_CLSCORE_MAX_MODELS", "0") or "0")
        if max_models > 0:
            checkpoints = checkpoints[:max_models]

        if not checkpoints:
            self.last_error = f"No model checkpoints found in {self.model_dir}"
            self.available = False
            return False

        try:
            # Load all bagging models once to avoid repeated disk I/O.
            for ckpt_path in checkpoints:
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                state_dict = checkpoint["state_dict"]

                # Derive architecture from checkpoint tensor shapes for robust compatibility.
                embedding_weight = state_dict["embedding.weight"]
                self.atom_fea_len = int(embedding_weight.shape[1])
                atom_hidden = int(embedding_weight.shape[0])

                fc_in = int(state_dict["convs.0.fc_full.weight"].shape[1])
                self.nbr_fea_len = int(fc_in - 2 * atom_hidden)

                conv_ids = {
                    int(k.split(".")[1])
                    for k in state_dict.keys()
                    if k.startswith("convs.") and k.endswith(".fc_full.weight")
                }
                n_conv = max(conv_ids) + 1 if conv_ids else 3

                h_fea_len = int(state_dict["fc_out.weight"].shape[1])
                fc_hidden_ids = {
                    int(k.split(".")[1])
                    for k in state_dict.keys()
                    if k.startswith("fcs.") and k.endswith(".weight")
                }
                n_h = (max(fc_hidden_ids) + 2) if fc_hidden_ids else 1

                model = CrystalGraphConvNet(
                    orig_atom_fea_len=self.atom_fea_len,
                    nbr_fea_len=self.nbr_fea_len,
                    atom_fea_len=atom_hidden,
                    n_conv=n_conv,
                    h_fea_len=h_fea_len,
                    n_h=n_h,
                    classification=True,
                )
                model.load_state_dict(state_dict, strict=True)
                model.eval()
                self.models.append(model)
        except Exception as exc:
            self.last_error = f"Failed to load model checkpoints: {exc}"
            self.available = False
            return False

        self.available = True
        return True

    def predict(self, material_id: str, cif_path: str) -> CLscoreResult:
        if not self.setup():
            return CLscoreResult(material_id=material_id, clscore=-1.0, status=self.last_error or "Model setup failed")

        cif = Path(cif_path)
        if not cif.exists():
            return CLscoreResult(material_id=material_id, clscore=-1.0, status="Missing CIF file")

        try:
            structure = Structure.from_file(str(cif))
            graph = self._build_graph(structure)
            if graph is None:
                return CLscoreResult(material_id=material_id, clscore=-1.0, status="Graph construction failed")

            atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = graph

            probs = []
            with torch.no_grad():
                for model in self.models:
                    out = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
                    prob = float(torch.exp(out)[0, 1].item())
                    probs.append(prob)

            if not probs:
                return CLscoreResult(material_id=material_id, clscore=-1.0, status="No model predictions")

            return CLscoreResult(material_id=material_id, clscore=float(np.mean(probs)), status="ok")
        except Exception as exc:
            return CLscoreResult(material_id=material_id, clscore=-1.0, status=f"Predict error: {exc}")

    def _build_graph(self, structure: Structure):
        if torch is None:
            return None

        n_atoms = len(structure)
        if n_atoms == 0:
            return None

        atom_fea = []
        for site in structure:
            z = int(site.specie.Z)
            atom_fea.append(_atomic_feature(z, self.atom_fea_len))

        atom_fea_t = torch.tensor(np.vstack(atom_fea), dtype=torch.float32)

        all_nbrs = structure.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

        nbr_fea_idx, nbr_dist = [], []
        for nbrs in all_nbrs:
            if len(nbrs) == 0:
                nbr_fea_idx.append([0] * self.max_neighbors)
                nbr_dist.append([self.radius + 1.0] * self.max_neighbors)
                continue

            if len(nbrs) < self.max_neighbors:
                idx = [int(n[2]) for n in nbrs] + [int(nbrs[0][2])] * (self.max_neighbors - len(nbrs))
                dist = [float(n[1]) for n in nbrs] + [self.radius + 1.0] * (self.max_neighbors - len(nbrs))
            else:
                nbrs = nbrs[: self.max_neighbors]
                idx = [int(n[2]) for n in nbrs]
                dist = [float(n[1]) for n in nbrs]

            nbr_fea_idx.append(idx)
            nbr_dist.append(dist)

        nbr_fea_idx_t = torch.tensor(np.array(nbr_fea_idx), dtype=torch.long)
        nbr_dist_arr = np.array(nbr_dist, dtype=float)

        nbr_fea_t = torch.tensor(_gaussian_expand(nbr_dist_arr, self.radius, self.nbr_fea_len), dtype=torch.float32)
        crystal_atom_idx = [torch.arange(n_atoms, dtype=torch.long)]

        return atom_fea_t, nbr_fea_t, nbr_fea_idx_t, crystal_atom_idx


def _atomic_feature(z: int, feature_len: int) -> np.ndarray:
    # Use robust one-hot encoding keyed by atomic number.
    fea = np.zeros(feature_len, dtype=float)
    idx = min(max(z - 1, 0), feature_len - 1)
    fea[idx] = 1.0
    return fea


def _gaussian_expand(distances: np.ndarray, radius: float, n_bins: int) -> np.ndarray:
    if n_bins <= 1:
        return distances[..., np.newaxis]
    centers = np.linspace(0.0, radius, num=n_bins)
    step = max(centers[1] - centers[0], 1e-6)
    return np.exp(-((distances[..., np.newaxis] - centers) ** 2) / (step ** 2))


def _bag_sort_key(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except Exception:
        return 10_000


_PREDICTOR: CLscorePredictor | None = None


def _predictor() -> CLscorePredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = CLscorePredictor()
    return _PREDICTOR


def get_clscore(material_id: str, cif_path: str) -> float:
    """
    Given a material ID and path to its CIF file, return CLscore in [0, 1].
    Returns -1 when the structure or model cannot be processed.
    """
    result = _predictor().predict(material_id=material_id, cif_path=cif_path)
    return float(result.clscore)


def batch_clscore(
    material_ids: list,
    cif_dir: str,
    output_csv: str,
    batch_size: int = 100,
    recompute_unknown: bool = False,
) -> pd.DataFrame:
    """
    Score material IDs in resumable batches.

    - Saves checkpoint every batch_size rows to output_csv
    - Resumes from existing output_csv
    - Writes failures to failed_clscore.csv
    """
    cif_root = Path(cif_dir)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path = out_path.parent / "failed_clscore.csv"

    existing: pd.DataFrame
    if out_path.exists():
        existing = pd.read_csv(out_path)
        if "MaterialId" not in existing.columns or "clscore" not in existing.columns:
            existing = pd.DataFrame(columns=["MaterialId", "clscore"])
    else:
        existing = pd.DataFrame(columns=["MaterialId", "clscore"])

    if recompute_unknown:
        done_ids = set(existing[existing["clscore"] != -1]["MaterialId"].astype(str).tolist())
    else:
        done_ids = set(existing["MaterialId"].astype(str).tolist())
    pending_ids = [str(mid) for mid in material_ids if str(mid) not in done_ids]

    failed_done_ids: set[str] = set()
    if failed_path.exists():
        try:
            failed_done_ids = set(pd.read_csv(failed_path)["MaterialId"].astype(str).tolist())
        except Exception:
            failed_done_ids = set()

    buffer: list[dict[str, Any]] = []
    failed_buffer: list[dict[str, Any]] = []

    # Ensure output files have headers before append-only writes.
    if not out_path.exists():
        pd.DataFrame(columns=["MaterialId", "clscore"]).to_csv(out_path, index=False)
    if not failed_path.exists():
        pd.DataFrame(columns=["MaterialId", "cif_path", "status"]).to_csv(failed_path, index=False)

    def flush() -> None:
        nonlocal buffer, failed_buffer
        if buffer:
            pd.DataFrame(buffer).to_csv(out_path, mode="a", header=False, index=False)
            buffer = []
        if failed_buffer:
            pd.DataFrame(failed_buffer).to_csv(failed_path, mode="a", header=False, index=False)
            failed_buffer = []
    pbar = tqdm(total=len(pending_ids), desc="CLscore")

    for idx, material_id in enumerate(pending_ids, start=1):
        cif_path = cif_root / f"{material_id}.cif"
        result = _predictor().predict(material_id=material_id, cif_path=str(cif_path))
        buffer.append({"MaterialId": material_id, "clscore": result.clscore})

        if result.clscore == -1 and material_id not in failed_done_ids:
            failed_buffer.append({"MaterialId": material_id, "cif_path": str(cif_path), "status": result.status})
            failed_done_ids.add(material_id)

        if idx % batch_size == 0:
            flush()

        pbar.update(1)

    pbar.close()

    flush()

    existing = pd.read_csv(out_path)
    existing = existing.drop_duplicates(subset=["MaterialId"], keep="last")
    existing.to_csv(out_path, index=False)
    return existing
