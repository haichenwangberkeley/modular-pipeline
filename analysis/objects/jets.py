from __future__ import annotations

import awkward as ak
import numpy as np


def build_jets(batch: ak.Array, cfg: dict, event_mask: np.ndarray) -> dict:
    jets = ak.zip(
        {
            "pt": batch["jet_pt"],
            "eta": batch["jet_eta"],
            "phi": batch["jet_phi"],
            "e": batch["jet_e"],
        }
    )
    jets = jets[event_mask]
    mask = (jets.pt > cfg["pt_min_gev"]) & (abs(jets.eta) < cfg["abs_eta_max"])
    selected = jets[mask]
    ordered = selected[ak.argsort(selected.pt, axis=1, ascending=False)]
    n_jets = ak.to_numpy(ak.num(ordered))
    has_two = n_jets >= 2
    mjj = np.full(n_jets.shape, np.nan)
    deta = np.full(n_jets.shape, np.nan)
    if np.any(has_two):
        leading = ordered[has_two][:, 0]
        subleading = ordered[has_two][:, 1]
        l_px = leading.pt * np.cos(leading.phi)
        l_py = leading.pt * np.sin(leading.phi)
        l_pz = leading.pt * np.sinh(leading.eta)
        s_px = subleading.pt * np.cos(subleading.phi)
        s_py = subleading.pt * np.sin(subleading.phi)
        s_pz = subleading.pt * np.sinh(subleading.eta)
        energy = leading.e + subleading.e
        px = l_px + s_px
        py = l_py + s_py
        pz = l_pz + s_pz
        mjj[has_two] = np.sqrt(np.clip(ak.to_numpy(energy**2 - px**2 - py**2 - pz**2), 0.0, None))
        deta[has_two] = np.abs(ak.to_numpy(leading.eta - subleading.eta))
    return {"n_jets": n_jets, "mjj": mjj, "delta_eta_jj": deta}
