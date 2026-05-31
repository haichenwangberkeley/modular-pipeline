from __future__ import annotations

import awkward as ak
import numpy as np


def build_photons(batch: ak.Array, cfg: dict) -> dict:
    abs_eta = abs(batch["photon_eta"])
    crack_lo, crack_hi = cfg["eta_crack"]
    mask = (batch["photon_pt"] > cfg["pt_min_gev"]) & (abs_eta < cfg["abs_eta_max"]) & ~((abs_eta > crack_lo) & (abs_eta < crack_hi))
    if cfg.get("require_tight_id", False):
        mask = mask & batch["photon_isTightID"]
    if cfg.get("require_tight_iso", False):
        mask = mask & batch["photon_isTightIso"]

    photons = ak.zip(
        {
            "pt": batch["photon_pt"][mask],
            "eta": batch["photon_eta"][mask],
            "phi": batch["photon_phi"][mask],
            "e": batch["photon_e"][mask],
        }
    )
    ordered = photons[ak.argsort(photons.pt, axis=1, ascending=False)]
    multiplicity = ak.to_numpy(ak.num(ordered))
    has_two = multiplicity >= 2
    selected = ordered[has_two]
    lead = selected[:, 0]
    sublead = selected[:, 1]

    lead_px = lead.pt * np.cos(lead.phi)
    lead_py = lead.pt * np.sin(lead.phi)
    lead_pz = lead.pt * np.sinh(lead.eta)
    sub_px = sublead.pt * np.cos(sublead.phi)
    sub_py = sublead.pt * np.sin(sublead.phi)
    sub_pz = sublead.pt * np.sinh(sublead.eta)

    diphoton_e = lead.e + sublead.e
    diphoton_px = lead_px + sub_px
    diphoton_py = lead_py + sub_py
    diphoton_pz = lead_pz + sub_pz
    mass2 = diphoton_e**2 - diphoton_px**2 - diphoton_py**2 - diphoton_pz**2
    diphoton_mass = np.sqrt(np.clip(ak.to_numpy(mass2), 0.0, None))

    thrust_x = ak.to_numpy(lead_px - sub_px)
    thrust_y = ak.to_numpy(lead_py - sub_py)
    dip_px = ak.to_numpy(diphoton_px)
    dip_py = ak.to_numpy(diphoton_py)
    thrust_norm = np.hypot(thrust_x, thrust_y)
    ptt = np.divide(np.abs(dip_px * thrust_y - dip_py * thrust_x), thrust_norm, out=np.zeros_like(diphoton_mass), where=thrust_norm > 0.0)

    lead_pt = ak.to_numpy(lead.pt)
    sublead_pt = ak.to_numpy(sublead.pt)
    lead_eta = ak.to_numpy(lead.eta)
    sublead_eta = ak.to_numpy(sublead.eta)
    lead_phi = ak.to_numpy(lead.phi)
    sublead_phi = ak.to_numpy(sublead.phi)
    delta_phi = np.arctan2(np.sin(lead_phi - sublead_phi), np.cos(lead_phi - sublead_phi))
    delta_r = np.hypot(lead_eta - sublead_eta, delta_phi)
    valid_mass = diphoton_mass > 0.0
    lead_fraction = np.divide(lead_pt, diphoton_mass, out=np.zeros_like(diphoton_mass), where=valid_mass)
    sublead_fraction = np.divide(sublead_pt, diphoton_mass, out=np.zeros_like(diphoton_mass), where=valid_mass)
    pt_fraction = (lead_fraction >= cfg["leading_pt_over_mgg_min"]) & (sublead_fraction >= cfg["subleading_pt_over_mgg_min"])

    return {
        "mask_has_two": ak.to_numpy(has_two),
        "selected_event_mask": ak.to_numpy(has_two),
        "diphoton_mass": diphoton_mass,
        "ptt": ptt,
        "delta_r": delta_r,
        "lead_pt": lead_pt,
        "sublead_pt": sublead_pt,
        "lead_eta": lead_eta,
        "sublead_eta": sublead_eta,
        "photon_multiplicity": multiplicity[has_two],
        "pt_fraction_mask": pt_fraction,
        "selected": selected,
    }
