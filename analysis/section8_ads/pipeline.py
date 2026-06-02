from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot

from analysis.common import ensure_dir, read_json, stable_hash, utcnow_iso, write_json, write_text
from analysis.samples.registry import build_registry
from analysis.section8_ads.ads_loader import load_ads
from analysis.section8_ads.branch_map import build_branch_mapping
from analysis.section8_ads.categories import ORDERED_CATEGORIES, assign_categories
from analysis.section8_ads.classifiers import (
    classifier_input_status,
    optimize_boundaries,
    score_samples,
    train_classifiers,
    write_training_sample_audit,
)


FIT_RANGE = (105.0, 160.0)
SIGNAL_WINDOW = (120.0, 130.0)
SIDEBAND_SCALE = 10.0 / 45.0


def _delta_phi(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))


def _rapidities(energy: np.ndarray, pz: np.ndarray) -> np.ndarray:
    denom = np.clip(energy - pz, 1e-6, None)
    return 0.5 * np.log(np.clip((energy + pz) / denom, 1e-6, None))


def _invariant_mass(px: np.ndarray, py: np.ndarray, pz: np.ndarray, energy: np.ndarray) -> np.ndarray:
    mass2 = energy**2 - px**2 - py**2 - pz**2
    return np.sqrt(np.clip(mass2, 0.0, None))


def _vector_components(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray, energy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return px, py, pz, energy


def _select_samples(registry: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for sample in registry:
        if sample["kind"] == "data":
            selected.append(sample)
            continue
        if sample["analysis_role"] == "signal_nominal":
            selected.append(sample)
            continue
        if sample["process_key"] == "prompt_diphoton" and sample["is_nominal"]:
            selected.append(sample)
    return selected


def _iterate_batches(
    files: list[str],
    branches: list[str],
    max_events: int | None = None,
    event_selector: Callable[[ak.Array], np.ndarray] | None = None,
):
    seen = 0
    for file_path in files:
        with uproot.open(file_path) as handle:
            tree = handle["analysis"]
            for batch in tree.iterate(branches, step_size="100 MB", library="ak"):
                if event_selector is not None:
                    selector = np.asarray(event_selector(batch), dtype=bool)
                    if not np.any(selector):
                        continue
                    batch = batch[selector]
                if max_events is None:
                    yield batch
                    continue
                remaining = max_events - seen
                if remaining <= 0:
                    return
                if len(batch[branches[0]]) > remaining:
                    yield batch[:remaining]
                    return
                yield batch
                seen += len(batch[branches[0]])


def _event_weights(batch: ak.Array, sample: dict[str, Any]) -> np.ndarray:
    size = len(batch["eventNumber"])
    if sample["kind"] == "data":
        return np.ones(size, dtype=float)
    denom = sample["sumw"]
    norm = 1.0 if not denom else sample["xsec_pb"] * sample["k_factor"] * sample["filter_eff"] * sample["lumi_fb"] * 1000.0 / denom
    weights = norm * np.asarray(batch["mcWeight"], dtype=float)
    for branch in ("ScaleFactor_PILEUP", "ScaleFactor_PHOTON", "ScaleFactor_JVT", "ScaleFactor_FTAG"):
        if branch in batch.fields:
            weights *= np.asarray(batch[branch], dtype=float)
    return weights


def _trigger_mask(batch: ak.Array, n_events: int, trigger_policy: str = "input_preselected") -> np.ndarray:
    if trigger_policy == "trigP":
        return np.asarray(batch["trigP"], dtype=bool) if "trigP" in batch.fields else np.ones(n_events, dtype=bool)
    if trigger_policy in {"input_preselected", "none"}:
        return np.ones(n_events, dtype=bool)
    raise ValueError(f"Unknown Section 8 trigger policy: {trigger_policy}")


def _empty_arrays() -> dict[str, list[np.ndarray]]:
    return {
        "event_number": [],
        "run_number": [],
        "weight": [],
        "trigger_passed": [],
        "baseline_selected": [],
        "is_sideband": [],
        "is_signal_window": [],
        "m_gammagamma": [],
        "pT_gammagamma": [],
        "eta_gammagamma": [],
        "pTt_gammagamma": [],
        "lead_pt": [],
        "sublead_pt": [],
        "lead_pt_over_mgg": [],
        "sublead_pt_over_mgg": [],
        "lead_eta": [],
        "sublead_eta": [],
        "max_abs_photon_eta": [],
        "N_jets_25": [],
        "N_jets_30": [],
        "N_jets_25_jvt_diagnostic": [],
        "N_jets_30_jvt_diagnostic": [],
        "N_central_jets_25": [],
        "N_forward_jets_25": [],
        "N_btag_25": [],
        "N_lep": [],
        "m_ll": [],
        "Z_ll_veto": [],
        "m_e_gamma_veto": [],
        "pT_lepton_plus_MET": [],
        "MET": [],
        "MET_significance": [],
        "leading_jet_pT_30": [],
        "m_jj_30": [],
        "abs_delta_eta_jj_30": [],
        "pT_Hjj_30": [],
        "deltaR_min_gamma_j": [],
        "VBF_centrality": [],
        "H_T": [],
        "m_all_jets": [],
        "delta_y_gammagamma_jj": [],
        "cos_theta_star_gammagamma_jj": [],
        "abs_delta_phi_gammagamma_jj_capped": [],
        "training_mask_tth": [],
        "training_mask_vh": [],
        "training_mask_vbf": [],
    }


def _concat_arrays(payload: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, chunks in payload.items():
        out[key] = np.concatenate(chunks) if chunks else np.array([])
    return out


def _append_chunk(store: dict[str, list[np.ndarray]], key: str, value: np.ndarray) -> None:
    store[key].append(np.asarray(value))


def _photon_iso_proxy(pt: ak.Array, topoetcone40: ak.Array, ptcone20: ak.Array) -> ak.Array:
    # Approximate the published calorimeter/track isolation using the closest visible ntuple branches.
    return (topoetcone40 <= 0.065 * pt) & (ptcone20 <= 0.05 * pt)


def _process_sample(
    sample: dict[str, Any],
    max_events: int | None = None,
    event_selector: Callable[[ak.Array], np.ndarray] | None = None,
    trigger_policy: str = "input_preselected",
) -> dict[str, Any]:
    branches = [
        "eventNumber",
        "runNumber",
        "mcWeight",
        "ScaleFactor_PILEUP",
        "ScaleFactor_PHOTON",
        "ScaleFactor_JVT",
        "ScaleFactor_FTAG",
        "trigP",
        "photon_pt",
        "photon_eta",
        "photon_phi",
        "photon_e",
        "photon_isLooseID",
        "photon_isTightID",
        "photon_ptcone20",
        "photon_topoetcone40",
        "photon_isLooseIso",
        "photon_isTightIso",
        "jet_pt",
        "jet_eta",
        "jet_phi",
        "jet_e",
        "jet_jvt",
        "jet_btag_quantile",
        "lep_type",
        "lep_pt",
        "lep_eta",
        "lep_phi",
        "lep_e",
        "lep_charge",
        "lep_isMediumID",
        "lep_isLooseIso",
        "lep_isTightIso",
        "lep_z0",
        "lep_d0sig",
        "met",
        "met_phi",
    ]
    cutflow = {
        "all_input_events": {"before": 0, "after": 0, "status": "implemented_faithfully", "notes": ""},
        "trigger_requirement": {
            "before": 0,
            "after": 0,
            "status": "implicit_in_input_format" if trigger_policy == "input_preselected" else "approximated",
            "notes": (
                "Treating GamGam ntuples as trigger-preselected because trigP does not match the ADS diphoton-trigger semantics."
                if trigger_policy == "input_preselected"
                else "Using trigP as an explicit photon-trigger proxy."
            ),
        },
        "at_least_two_photon_candidates": {"before": 0, "after": 0, "status": "implemented_faithfully", "notes": ""},
        "photon_kinematic_acceptance": {"before": 0, "after": 0, "status": "implemented_faithfully", "notes": ""},
        "loose_photon_id_preselection": {"before": 0, "after": 0, "status": "implemented_faithfully", "notes": ""},
        "select_two_highest_et_photons": {"before": 0, "after": 0, "status": "implemented_faithfully", "notes": ""},
        "diphoton_primary_vertex_handling": {"before": 0, "after": 0, "status": "unavailable", "notes": "The primary-vertex NN inputs are not present in the ntuples."},
        "leading_photon_et_over_mgg": {"before": 0, "after": 0, "status": "implemented_faithfully", "notes": ""},
        "subleading_photon_et_over_mgg": {"before": 0, "after": 0, "status": "implemented_faithfully", "notes": ""},
        "tight_photon_identification": {"before": 0, "after": 0, "status": "implemented_faithfully", "notes": ""},
        "photon_isolation": {
            "before": 0,
            "after": 0,
            "status": "approximated",
            "notes": "Using photon_topoetcone40 and photon_ptcone20 with the Section 5.1 thresholds because the ntuples do not expose the exact calorimeter-isolation cone-0.2 branch.",
        },
        "diphoton_mass_window": {"before": 0, "after": 0, "status": "implemented_faithfully", "notes": ""},
    }
    for entry in cutflow.values():
        entry["weighted_before"] = 0.0
        entry["weighted_after"] = 0.0
    arrays = _empty_arrays()

    for batch in _iterate_batches(sample["files"], branches, max_events=max_events, event_selector=event_selector):
        event_number = np.asarray(batch["eventNumber"], dtype=np.int64)
        run_number = np.asarray(batch["runNumber"], dtype=np.int64)
        weights = _event_weights(batch, sample)
        n_events = len(event_number)

        trigger_mask = _trigger_mask(batch, n_events, trigger_policy)

        abs_eta = abs(batch["photon_eta"])
        kin_mask = (batch["photon_pt"] > 25.0) & (abs_eta < 2.37) & ~((abs_eta > 1.37) & (abs_eta < 1.52))
        loose_mask = kin_mask & batch["photon_isLooseID"]
        photons = ak.zip(
            {
                "pt": batch["photon_pt"][loose_mask],
                "eta": batch["photon_eta"][loose_mask],
                "phi": batch["photon_phi"][loose_mask],
                "e": batch["photon_e"][loose_mask],
                "tight_id": batch["photon_isTightID"][loose_mask],
                "tight_iso": _photon_iso_proxy(
                    batch["photon_pt"][loose_mask],
                    batch["photon_topoetcone40"][loose_mask],
                    batch["photon_ptcone20"][loose_mask],
                ),
            }
        )
        ordered = photons[ak.argsort(photons.pt, axis=1, ascending=False)]
        loose_counts = ak.to_numpy(ak.num(ordered))
        has_two = loose_counts >= 2
        selected = ordered[has_two][:, :2]

        full_tight = np.zeros(n_events, dtype=bool)
        full_iso = np.zeros(n_events, dtype=bool)
        full_ptfrac = np.zeros(n_events, dtype=bool)
        full_mass = np.full(n_events, np.nan)
        full_ptgg = np.full(n_events, np.nan)
        full_etagg = np.full(n_events, np.nan)
        full_ptt = np.full(n_events, np.nan)
        full_lead_pt = np.full(n_events, np.nan)
        full_sub_pt = np.full(n_events, np.nan)
        full_lead_eta = np.full(n_events, np.nan)
        full_sub_eta = np.full(n_events, np.nan)
        full_abs_eta_max = np.full(n_events, np.nan)
        sel_indices = np.where(has_two)[0]

        if len(sel_indices):
            lead = selected[:, 0]
            sub = selected[:, 1]
            lead_px, lead_py, lead_pz, lead_e = _vector_components(ak.to_numpy(lead.pt), ak.to_numpy(lead.eta), ak.to_numpy(lead.phi), ak.to_numpy(lead.e))
            sub_px, sub_py, sub_pz, sub_e = _vector_components(ak.to_numpy(sub.pt), ak.to_numpy(sub.eta), ak.to_numpy(sub.phi), ak.to_numpy(sub.e))
            dip_px = lead_px + sub_px
            dip_py = lead_py + sub_py
            dip_pz = lead_pz + sub_pz
            dip_e = lead_e + sub_e
            mass = _invariant_mass(dip_px, dip_py, dip_pz, dip_e)
            ptt_x = lead_px - sub_px
            ptt_y = lead_py - sub_py
            ptt_norm = np.hypot(ptt_x, ptt_y)
            ptt = np.divide(np.abs(dip_px * ptt_y - dip_py * ptt_x), ptt_norm, out=np.zeros_like(mass), where=ptt_norm > 0.0)
            ptgg = np.hypot(dip_px, dip_py)
            etagg = np.arcsinh(np.divide(dip_pz, np.clip(ptgg, 1e-6, None)))
            lead_pt = ak.to_numpy(lead.pt)
            sub_pt = ak.to_numpy(sub.pt)
            lead_eta = ak.to_numpy(lead.eta)
            sub_eta = ak.to_numpy(sub.eta)
            tight_id = ak.to_numpy(lead.tight_id) & ak.to_numpy(sub.tight_id)
            tight_iso = ak.to_numpy(lead.tight_iso) & ak.to_numpy(sub.tight_iso)
            ptfrac = (lead_pt / np.clip(mass, 1e-6, None) > 0.35) & (sub_pt / np.clip(mass, 1e-6, None) > 0.25)
            full_tight[sel_indices] = tight_id
            full_iso[sel_indices] = tight_iso
            full_ptfrac[sel_indices] = ptfrac
            full_mass[sel_indices] = mass
            full_ptgg[sel_indices] = ptgg
            full_etagg[sel_indices] = etagg
            full_ptt[sel_indices] = ptt
            full_lead_pt[sel_indices] = lead_pt
            full_sub_pt[sel_indices] = sub_pt
            full_lead_eta[sel_indices] = lead_eta
            full_sub_eta[sel_indices] = sub_eta
            full_abs_eta_max[sel_indices] = np.maximum(np.abs(lead_eta), np.abs(sub_eta))

        baseline_mask = trigger_mask & has_two & full_tight & full_iso & full_ptfrac & np.isfinite(full_mass) & (full_mass > FIT_RANGE[0]) & (full_mass < FIT_RANGE[1])

        def update_cutflow(name: str, before_mask: np.ndarray, after_mask: np.ndarray) -> None:
            cutflow[name]["before"] += int(np.sum(before_mask))
            cutflow[name]["after"] += int(np.sum(after_mask))
            cutflow[name]["weighted_before"] += float(np.sum(weights[before_mask]))
            cutflow[name]["weighted_after"] += float(np.sum(weights[after_mask]))

        kin_counts = ak.to_numpy(ak.num(batch["photon_pt"][kin_mask]))
        has_two_kinematic_photons = kin_counts >= 2
        trigger_has_two = trigger_mask & has_two
        et_fraction_mask = trigger_has_two & full_ptfrac
        tight_id_mask = et_fraction_mask & full_tight
        tight_iso_mask = tight_id_mask & full_iso

        all_events = np.ones(n_events, dtype=bool)
        update_cutflow("all_input_events", all_events, all_events)
        update_cutflow("trigger_requirement", all_events, trigger_mask)
        update_cutflow("at_least_two_photon_candidates", trigger_mask, trigger_has_two)
        update_cutflow("photon_kinematic_acceptance", trigger_mask, trigger_mask & has_two_kinematic_photons)
        update_cutflow("loose_photon_id_preselection", trigger_mask & has_two_kinematic_photons, trigger_has_two)
        update_cutflow("select_two_highest_et_photons", trigger_has_two, trigger_has_two)
        update_cutflow("diphoton_primary_vertex_handling", trigger_has_two, trigger_has_two)
        update_cutflow("leading_photon_et_over_mgg", trigger_has_two, et_fraction_mask)
        update_cutflow("subleading_photon_et_over_mgg", trigger_has_two, et_fraction_mask)
        update_cutflow("tight_photon_identification", et_fraction_mask, tight_id_mask)
        update_cutflow("photon_isolation", tight_id_mask, tight_iso_mask)
        update_cutflow("diphoton_mass_window", tight_iso_mask, baseline_mask)

        jets = ak.zip(
            {
                "pt": batch["jet_pt"],
                "eta": batch["jet_eta"],
                "phi": batch["jet_phi"],
                "e": batch["jet_e"],
                "jvt": batch["jet_jvt"],
                "btag": batch["jet_btag_quantile"],
            }
        )
        jets25 = jets[(jets.pt > 25.0) & (abs(jets.eta) < 4.4)]
        jets30 = jets[(jets.pt > 30.0) & (abs(jets.eta) < 4.4)]
        jvt_pass = (abs(jets.eta) >= 2.4) | (jets.pt >= 60.0) | (jets.jvt > 0.59)
        jets25_jvt = jets[(jets.pt > 25.0) & (abs(jets.eta) < 4.4) & jvt_pass]
        jets30_jvt = jets[(jets.pt > 30.0) & (abs(jets.eta) < 4.4) & jvt_pass]
        leptons = ak.zip(
            {
                "type": batch["lep_type"],
                "pt": batch["lep_pt"],
                "eta": batch["lep_eta"],
                "phi": batch["lep_phi"],
                "e": batch["lep_e"],
                "charge": batch["lep_charge"],
                "medium_id": batch["lep_isMediumID"],
                "loose_iso": batch["lep_isLooseIso"],
                "tight_iso": batch["lep_isTightIso"],
                "z0": batch["lep_z0"],
                "d0sig": batch["lep_d0sig"],
            }
        )

        baseline_indices = np.where(baseline_mask)[0]
        if not len(baseline_indices):
            continue
        baseline_selector = ak.Array(baseline_mask)
        baseline_jets25 = jets25[baseline_selector]
        baseline_jets30 = jets30[baseline_selector]
        baseline_jets25_jvt = jets25_jvt[baseline_selector]
        baseline_jets30_jvt = jets30_jvt[baseline_selector]
        baseline_leptons = leptons[baseline_selector]

        n_base = len(baseline_indices)
        n_jets25 = ak.to_numpy(ak.num(baseline_jets25))
        n_jets30 = ak.to_numpy(ak.num(baseline_jets30))
        n_jets25_jvt = ak.to_numpy(ak.num(baseline_jets25_jvt))
        n_jets30_jvt = ak.to_numpy(ak.num(baseline_jets30_jvt))
        n_central25 = ak.to_numpy(ak.sum(abs(baseline_jets25.eta) < 2.5, axis=1))
        n_forward25 = ak.to_numpy(ak.sum(abs(baseline_jets25.eta) >= 2.5, axis=1))
        n_b25 = ak.to_numpy(ak.sum(baseline_jets25.btag >= 4, axis=1))

        electron_mask = (
            (baseline_leptons.type == 11)
            & (baseline_leptons.pt > 10.0)
            & (abs(baseline_leptons.eta) < 2.47)
            & ~((abs(baseline_leptons.eta) > 1.37) & (abs(baseline_leptons.eta) < 1.52))
            & baseline_leptons.medium_id
            & baseline_leptons.loose_iso
            & (abs(baseline_leptons.z0) < 0.5)
            & (abs(baseline_leptons.d0sig) < 5.0)
        )
        muon_mask = (
            (baseline_leptons.type == 13)
            & (baseline_leptons.pt > 10.0)
            & (abs(baseline_leptons.eta) < 2.7)
            & baseline_leptons.medium_id
            & baseline_leptons.loose_iso
            & (abs(baseline_leptons.z0) < 0.5)
            & (abs(baseline_leptons.d0sig) < 3.0)
        )
        selected_leptons = baseline_leptons[electron_mask | muon_mask]
        n_lep = ak.to_numpy(ak.num(selected_leptons))

        met = np.asarray(batch["met"], dtype=float)[baseline_mask]
        met_phi = np.asarray(batch["met_phi"], dtype=float)[baseline_mask]
        lead_eta = full_lead_eta[baseline_mask]
        sub_eta = full_sub_eta[baseline_mask]
        lead_pt = full_lead_pt[baseline_mask]
        sub_pt = full_sub_pt[baseline_mask]
        lead_phi = np.zeros(n_base)
        sub_phi = np.zeros(n_base)
        # We only need phi for derived angular variables used downstream.
        selected_baseline = selected[baseline_mask[sel_indices]]
        if len(selected_baseline):
            lead_phi = ak.to_numpy(selected_baseline[:, 0].phi)
            sub_phi = ak.to_numpy(selected_baseline[:, 1].phi)
        lead_e = lead_pt * np.cosh(lead_eta)
        sub_e = sub_pt * np.cosh(sub_eta)
        lead_px, lead_py, lead_pz, _ = _vector_components(lead_pt, lead_eta, lead_phi, lead_e)
        sub_px, sub_py, sub_pz, _ = _vector_components(sub_pt, sub_eta, sub_phi, sub_e)
        dip_px = lead_px + sub_px
        dip_py = lead_py + sub_py
        dip_pz = lead_pz + sub_pz
        dip_e = lead_e + sub_e
        dip_y = _rapidities(dip_e, dip_pz)

        leading_jet_pt30 = np.full(n_base, np.nan)
        mjj30 = np.full(n_base, np.nan)
        abs_deta30 = np.full(n_base, np.nan)
        pthjj30 = np.full(n_base, np.nan)
        dr_min_gamma_j = np.full(n_base, np.nan)
        vbf_centrality = np.full(n_base, np.nan)
        ht = ak.to_numpy(ak.sum(baseline_jets30.pt, axis=1))
        m_all_jets = np.full(n_base, np.nan)
        delta_y = np.full(n_base, np.nan)
        cos_theta_star = np.full(n_base, np.nan)
        abs_dphi_capped = np.full(n_base, np.nan)
        training_mask_tth = (n_lep == 0) & (n_jets30 >= 3) & (n_b25 >= 1)
        training_mask_vh = np.zeros(n_base, dtype=bool)
        training_mask_vbf = np.zeros(n_base, dtype=bool)

        if np.any(n_jets30 > 0):
            leading_jet_pt30[n_jets30 > 0] = ak.to_numpy(baseline_jets30[n_jets30 > 0][:, 0].pt)

        for idx in range(n_base):
            jets30_evt = baseline_jets30[idx]
            if len(jets30_evt):
                px = ak.to_numpy(jets30_evt.pt * np.cos(jets30_evt.phi))
                py = ak.to_numpy(jets30_evt.pt * np.sin(jets30_evt.phi))
                pz = ak.to_numpy(jets30_evt.pt * np.sinh(jets30_evt.eta))
                energy = ak.to_numpy(jets30_evt.e)
                m_all_jets[idx] = float(_invariant_mass(np.sum(px), np.sum(py), np.sum(pz), np.sum(energy)))
            if len(jets30_evt) >= 2:
                j1 = jets30_evt[0]
                j2 = jets30_evt[1]
                j1_px, j1_py, j1_pz, _ = _vector_components(np.asarray([j1.pt]), np.asarray([j1.eta]), np.asarray([j1.phi]), np.asarray([j1.e]))
                j2_px, j2_py, j2_pz, _ = _vector_components(np.asarray([j2.pt]), np.asarray([j2.eta]), np.asarray([j2.phi]), np.asarray([j2.e]))
                dijet_px = j1_px[0] + j2_px[0]
                dijet_py = j1_py[0] + j2_py[0]
                dijet_pz = j1_pz[0] + j2_pz[0]
                dijet_e = float(j1.e + j2.e)
                mjj30[idx] = float(_invariant_mass(dijet_px, dijet_py, dijet_pz, dijet_e))
                abs_deta30[idx] = abs(float(j1.eta - j2.eta))
                pthjj30[idx] = float(np.hypot(dip_px[idx] + dijet_px, dip_py[idx] + dijet_py))
                dijet_y = float(_rapidities(np.asarray([dijet_e]), np.asarray([dijet_pz]))[0])
                delta_y[idx] = dijet_y - dip_y[idx]
                vbf_centrality[idx] = abs(full_etagg[baseline_mask][idx] - 0.5 * (float(j1.eta) + float(j2.eta)))
                dphi = abs(_delta_phi(np.asarray([math.atan2(dip_py[idx], dip_px[idx])]), np.asarray([math.atan2(dijet_py, dijet_px)]))[0])
                abs_dphi_capped[idx] = min(dphi, 2.94)
                drs = []
                for photon_eta, photon_phi in ((lead_eta[idx], lead_phi[idx]), (sub_eta[idx], sub_phi[idx])):
                    for jet_eta, jet_phi in ((float(j1.eta), float(j1.phi)), (float(j2.eta), float(j2.phi))):
                        drs.append(float(np.hypot(photon_eta - jet_eta, _delta_phi(np.asarray([photon_phi]), np.asarray([jet_phi]))[0])))
                dr_min_gamma_j[idx] = min(drs) if drs else np.nan
                training_mask_vh[idx] = bool(60.0 < mjj30[idx] < 120.0)
                training_mask_vbf[idx] = bool(abs_deta30[idx] > 2.0 and vbf_centrality[idx] < 5.0)
                system_px = dip_px[idx] + dijet_px
                system_py = dip_py[idx] + dijet_py
                system_pz = dip_pz[idx] + dijet_pz
                system_p = np.sqrt(system_px**2 + system_py**2 + system_pz**2)
                dip_p = np.sqrt(dip_px[idx] ** 2 + dip_py[idx] ** 2 + dip_pz[idx] ** 2)
                if system_p > 0.0 and dip_p > 0.0:
                    cos_theta_star[idx] = (dip_px[idx] * system_px + dip_py[idx] * system_py + dip_pz[idx] * system_pz) / (dip_p * system_p)

        met_sig = met / np.sqrt(np.clip(ht, 1.0, None))
        mll = np.full(n_base, np.nan)
        z_veto = np.ones(n_base, dtype=bool)
        megamma_veto = np.ones(n_base, dtype=bool)
        ptlepmet = np.full(n_base, np.nan)

        for idx in range(n_base):
            leptons_evt = selected_leptons[idx]
            if len(leptons_evt):
                lead_lep = leptons_evt[0]
                lep_px = float(lead_lep.pt * np.cos(lead_lep.phi))
                lep_py = float(lead_lep.pt * np.sin(lead_lep.phi))
                ptlepmet[idx] = float(np.hypot(lep_px + met[idx] * np.cos(met_phi[idx]), lep_py + met[idx] * np.sin(met_phi[idx])))
            sfos_masses = []
            electron_pairings = []
            for i in range(len(leptons_evt)):
                li = leptons_evt[i]
                if int(li.type) == 11:
                    for photon_pt, photon_eta, photon_phi in ((lead_pt[idx], lead_eta[idx], lead_phi[idx]), (sub_pt[idx], sub_eta[idx], sub_phi[idx])):
                        lep_px = float(li.pt * np.cos(li.phi))
                        lep_py = float(li.pt * np.sin(li.phi))
                        lep_pz = float(li.pt * np.sinh(li.eta))
                        pho_px = float(photon_pt * np.cos(photon_phi))
                        pho_py = float(photon_pt * np.sin(photon_phi))
                        pho_pz = float(photon_pt * np.sinh(photon_eta))
                        electron_pairings.append(float(_invariant_mass(lep_px + pho_px, lep_py + pho_py, lep_pz + pho_pz, float(li.e) + float(photon_pt * np.cosh(photon_eta)))))
                for j in range(i + 1, len(leptons_evt)):
                    lj = leptons_evt[j]
                    if int(li.type) != int(lj.type):
                        continue
                    if int(li.charge + lj.charge) != 0:
                        continue
                    li_px = float(li.pt * np.cos(li.phi))
                    li_py = float(li.pt * np.sin(li.phi))
                    li_pz = float(li.pt * np.sinh(li.eta))
                    lj_px = float(lj.pt * np.cos(lj.phi))
                    lj_py = float(lj.pt * np.sin(lj.phi))
                    lj_pz = float(lj.pt * np.sinh(lj.eta))
                    sfos_masses.append(float(_invariant_mass(li_px + lj_px, li_py + lj_py, li_pz + lj_pz, float(li.e + lj.e))))
            if sfos_masses:
                closest = min(sfos_masses, key=lambda item: abs(item - 91.2))
                mll[idx] = closest
                z_veto[idx] = not (70.0 <= closest <= 110.0)
            if electron_pairings:
                megamma_veto[idx] = not any(84.0 <= mass <= 94.0 for mass in electron_pairings)

        sideband = (full_mass[baseline_mask] >= FIT_RANGE[0]) & (full_mass[baseline_mask] < SIGNAL_WINDOW[0]) | (full_mass[baseline_mask] > SIGNAL_WINDOW[1]) & (full_mass[baseline_mask] <= FIT_RANGE[1])
        signal_window = (full_mass[baseline_mask] >= SIGNAL_WINDOW[0]) & (full_mass[baseline_mask] <= SIGNAL_WINDOW[1])

        fields = {
            "event_number": event_number[baseline_mask],
            "run_number": run_number[baseline_mask],
            "weight": weights[baseline_mask],
            "trigger_passed": trigger_mask[baseline_mask].astype(int),
            "baseline_selected": np.ones(np.sum(baseline_mask), dtype=int),
            "is_sideband": sideband.astype(int),
            "is_signal_window": signal_window.astype(int),
            "m_gammagamma": full_mass[baseline_mask],
            "pT_gammagamma": full_ptgg[baseline_mask],
            "eta_gammagamma": full_etagg[baseline_mask],
            "pTt_gammagamma": full_ptt[baseline_mask],
            "lead_pt": lead_pt,
            "sublead_pt": sub_pt,
            "lead_pt_over_mgg": lead_pt / np.clip(full_mass[baseline_mask], 1e-6, None),
            "sublead_pt_over_mgg": sub_pt / np.clip(full_mass[baseline_mask], 1e-6, None),
            "lead_eta": lead_eta,
            "sublead_eta": sub_eta,
            "max_abs_photon_eta": full_abs_eta_max[baseline_mask],
            "N_jets_25": n_jets25,
            "N_jets_30": n_jets30,
            "N_jets_25_jvt_diagnostic": n_jets25_jvt,
            "N_jets_30_jvt_diagnostic": n_jets30_jvt,
            "N_central_jets_25": n_central25,
            "N_forward_jets_25": n_forward25,
            "N_btag_25": n_b25,
            "N_lep": n_lep,
            "m_ll": mll,
            "Z_ll_veto": z_veto.astype(int),
            "m_e_gamma_veto": megamma_veto.astype(int),
            "pT_lepton_plus_MET": ptlepmet,
            "MET": met,
            "MET_significance": met_sig,
            "leading_jet_pT_30": leading_jet_pt30,
            "m_jj_30": mjj30,
            "abs_delta_eta_jj_30": abs_deta30,
            "pT_Hjj_30": pthjj30,
            "deltaR_min_gamma_j": dr_min_gamma_j,
            "VBF_centrality": vbf_centrality,
            "H_T": ht,
            "m_all_jets": m_all_jets,
            "delta_y_gammagamma_jj": delta_y,
            "cos_theta_star_gammagamma_jj": cos_theta_star,
            "abs_delta_phi_gammagamma_jj_capped": abs_dphi_capped,
            "training_mask_tth": training_mask_tth.astype(int),
            "training_mask_vh": training_mask_vh.astype(int),
            "training_mask_vbf": training_mask_vbf.astype(int),
        }
        for key, value in fields.items():
            _append_chunk(arrays, key, np.asarray(value))

    return {
        "sample_id": sample["sample_id"],
        "process_key": sample["process_key"],
        "kind": sample["kind"],
        "analysis_role": sample["analysis_role"],
        "cutflow": cutflow,
        "arrays": _concat_arrays(arrays),
    }


def _empty_bdt_arrays() -> dict[str, list[np.ndarray]]:
    payload = _empty_arrays()
    payload.update(
        {
            "photon_region": [],
            "nominal_photon_region": [],
            "anti_id_or_iso_control_region": [],
            "bdt_subregion": [],
        }
    )
    return payload


def _process_bdt_candidates(
    sample: dict[str, Any],
    max_events: int | None = None,
    trigger_policy: str = "input_preselected",
) -> dict[str, Any]:
    branches = [
        "eventNumber",
        "runNumber",
        "mcWeight",
        "ScaleFactor_PILEUP",
        "ScaleFactor_PHOTON",
        "ScaleFactor_JVT",
        "ScaleFactor_FTAG",
        "trigP",
        "photon_pt",
        "photon_eta",
        "photon_phi",
        "photon_e",
        "photon_isLooseID",
        "photon_isTightID",
        "photon_ptcone20",
        "photon_topoetcone40",
        "photon_isLooseIso",
        "photon_isTightIso",
        "jet_pt",
        "jet_eta",
        "jet_phi",
        "jet_e",
        "jet_jvt",
        "jet_btag_quantile",
        "lep_type",
        "lep_pt",
        "lep_eta",
        "lep_phi",
        "lep_e",
        "lep_charge",
        "lep_isMediumID",
        "lep_isLooseIso",
        "lep_isTightIso",
        "lep_z0",
        "lep_d0sig",
        "met",
        "met_phi",
    ]
    arrays = _empty_bdt_arrays()
    for batch in _iterate_batches(sample["files"], branches, max_events=max_events):
        event_number = np.asarray(batch["eventNumber"], dtype=np.int64)
        run_number = np.asarray(batch["runNumber"], dtype=np.int64)
        weights = _event_weights(batch, sample)
        n_events = len(event_number)
        trigger_mask = _trigger_mask(batch, n_events, trigger_policy)

        abs_eta = abs(batch["photon_eta"])
        kin_mask = (batch["photon_pt"] > 25.0) & (abs_eta < 2.37) & ~((abs_eta > 1.37) & (abs_eta < 1.52))
        loose_mask = kin_mask & batch["photon_isLooseID"]
        photons = ak.zip(
            {
                "pt": batch["photon_pt"][loose_mask],
                "eta": batch["photon_eta"][loose_mask],
                "phi": batch["photon_phi"][loose_mask],
                "e": batch["photon_e"][loose_mask],
                "tight_id": batch["photon_isTightID"][loose_mask],
                "tight_iso": _photon_iso_proxy(
                    batch["photon_pt"][loose_mask],
                    batch["photon_topoetcone40"][loose_mask],
                    batch["photon_ptcone20"][loose_mask],
                ),
            }
        )
        ordered = photons[ak.argsort(photons.pt, axis=1, ascending=False)]
        has_two = ak.to_numpy(ak.num(ordered)) >= 2
        selected = ordered[has_two][:, :2]

        full_tight = np.zeros(n_events, dtype=bool)
        full_iso = np.zeros(n_events, dtype=bool)
        full_ptfrac = np.zeros(n_events, dtype=bool)
        full_mass = np.full(n_events, np.nan)
        full_ptgg = np.full(n_events, np.nan)
        full_etagg = np.full(n_events, np.nan)
        full_ptt = np.full(n_events, np.nan)
        full_lead_pt = np.full(n_events, np.nan)
        full_sub_pt = np.full(n_events, np.nan)
        full_lead_eta = np.full(n_events, np.nan)
        full_sub_eta = np.full(n_events, np.nan)
        full_abs_eta_max = np.full(n_events, np.nan)
        sel_indices = np.where(has_two)[0]

        if len(sel_indices):
            lead = selected[:, 0]
            sub = selected[:, 1]
            lead_px, lead_py, lead_pz, lead_e = _vector_components(ak.to_numpy(lead.pt), ak.to_numpy(lead.eta), ak.to_numpy(lead.phi), ak.to_numpy(lead.e))
            sub_px, sub_py, sub_pz, sub_e = _vector_components(ak.to_numpy(sub.pt), ak.to_numpy(sub.eta), ak.to_numpy(sub.phi), ak.to_numpy(sub.e))
            dip_px = lead_px + sub_px
            dip_py = lead_py + sub_py
            dip_pz = lead_pz + sub_pz
            dip_e = lead_e + sub_e
            mass = _invariant_mass(dip_px, dip_py, dip_pz, dip_e)
            ptt_x = lead_px - sub_px
            ptt_y = lead_py - sub_py
            ptt_norm = np.hypot(ptt_x, ptt_y)
            ptt = np.divide(np.abs(dip_px * ptt_y - dip_py * ptt_x), ptt_norm, out=np.zeros_like(mass), where=ptt_norm > 0.0)
            ptgg = np.hypot(dip_px, dip_py)
            etagg = np.arcsinh(np.divide(dip_pz, np.clip(ptgg, 1e-6, None)))
            lead_pt = ak.to_numpy(lead.pt)
            sub_pt = ak.to_numpy(sub.pt)
            lead_eta = ak.to_numpy(lead.eta)
            sub_eta = ak.to_numpy(sub.eta)
            tight_id = ak.to_numpy(lead.tight_id) & ak.to_numpy(sub.tight_id)
            tight_iso = ak.to_numpy(lead.tight_iso) & ak.to_numpy(sub.tight_iso)
            ptfrac = (lead_pt / np.clip(mass, 1e-6, None) > 0.35) & (sub_pt / np.clip(mass, 1e-6, None) > 0.25)
            full_tight[sel_indices] = tight_id
            full_iso[sel_indices] = tight_iso
            full_ptfrac[sel_indices] = ptfrac
            full_mass[sel_indices] = mass
            full_ptgg[sel_indices] = ptgg
            full_etagg[sel_indices] = etagg
            full_ptt[sel_indices] = ptt
            full_lead_pt[sel_indices] = lead_pt
            full_sub_pt[sel_indices] = sub_pt
            full_lead_eta[sel_indices] = lead_eta
            full_sub_eta[sel_indices] = sub_eta
            full_abs_eta_max[sel_indices] = np.maximum(np.abs(lead_eta), np.abs(sub_eta))

        nominal_region = full_tight & full_iso
        control_region = ~nominal_region
        candidate_mask = (
            trigger_mask
            & has_two
            & full_ptfrac
            & np.isfinite(full_mass)
            & (full_mass > FIT_RANGE[0])
            & (full_mass < FIT_RANGE[1])
            & (nominal_region | control_region)
        )
        if not np.any(candidate_mask):
            continue

        jets = ak.zip(
            {
                "pt": batch["jet_pt"],
                "eta": batch["jet_eta"],
                "phi": batch["jet_phi"],
                "e": batch["jet_e"],
                "jvt": batch["jet_jvt"],
                "btag": batch["jet_btag_quantile"],
            }
        )
        jets25 = jets[(jets.pt > 25.0) & (abs(jets.eta) < 4.4)]
        jets30 = jets[(jets.pt > 30.0) & (abs(jets.eta) < 4.4)]
        leptons = ak.zip(
            {
                "type": batch["lep_type"],
                "pt": batch["lep_pt"],
                "eta": batch["lep_eta"],
                "phi": batch["lep_phi"],
                "e": batch["lep_e"],
                "charge": batch["lep_charge"],
                "medium_id": batch["lep_isMediumID"],
                "loose_iso": batch["lep_isLooseIso"],
                "tight_iso": batch["lep_isTightIso"],
                "z0": batch["lep_z0"],
                "d0sig": batch["lep_d0sig"],
            }
        )
        selector = ak.Array(candidate_mask)
        cand_jets25 = jets25[selector]
        cand_jets30 = jets30[selector]
        cand_leptons = leptons[selector]
        n_cand = int(np.sum(candidate_mask))
        n_jets25 = ak.to_numpy(ak.num(cand_jets25))
        n_jets30 = ak.to_numpy(ak.num(cand_jets30))
        n_central25 = ak.to_numpy(ak.sum(abs(cand_jets25.eta) < 2.5, axis=1))
        n_forward25 = ak.to_numpy(ak.sum(abs(cand_jets25.eta) >= 2.5, axis=1))
        n_b25 = ak.to_numpy(ak.sum(cand_jets25.btag >= 4, axis=1))

        electron_mask = (
            (cand_leptons.type == 11)
            & (cand_leptons.pt > 10.0)
            & (abs(cand_leptons.eta) < 2.47)
            & ~((abs(cand_leptons.eta) > 1.37) & (abs(cand_leptons.eta) < 1.52))
            & cand_leptons.medium_id
            & cand_leptons.loose_iso
            & (abs(cand_leptons.z0) < 0.5)
            & (abs(cand_leptons.d0sig) < 5.0)
        )
        muon_mask = (
            (cand_leptons.type == 13)
            & (cand_leptons.pt > 10.0)
            & (abs(cand_leptons.eta) < 2.7)
            & cand_leptons.medium_id
            & cand_leptons.loose_iso
            & (abs(cand_leptons.z0) < 0.5)
            & (abs(cand_leptons.d0sig) < 3.0)
        )
        selected_leptons = cand_leptons[electron_mask | muon_mask]
        n_lep = ak.to_numpy(ak.num(selected_leptons))

        met = np.asarray(batch["met"], dtype=float)[candidate_mask]
        lead_eta = full_lead_eta[candidate_mask]
        sub_eta = full_sub_eta[candidate_mask]
        lead_pt = full_lead_pt[candidate_mask]
        sub_pt = full_sub_pt[candidate_mask]
        selected_candidate = selected[candidate_mask[sel_indices]]
        lead_phi = ak.to_numpy(selected_candidate[:, 0].phi) if len(selected_candidate) else np.zeros(n_cand)
        sub_phi = ak.to_numpy(selected_candidate[:, 1].phi) if len(selected_candidate) else np.zeros(n_cand)
        lead_e = lead_pt * np.cosh(lead_eta)
        sub_e = sub_pt * np.cosh(sub_eta)
        lead_px, lead_py, lead_pz, _ = _vector_components(lead_pt, lead_eta, lead_phi, lead_e)
        sub_px, sub_py, sub_pz, _ = _vector_components(sub_pt, sub_eta, sub_phi, sub_e)
        dip_px = lead_px + sub_px
        dip_py = lead_py + sub_py
        dip_pz = lead_pz + sub_pz
        dip_e = lead_e + sub_e
        dip_y = _rapidities(dip_e, dip_pz)

        leading_jet_pt30 = np.full(n_cand, np.nan)
        mjj30 = np.full(n_cand, np.nan)
        abs_deta30 = np.full(n_cand, np.nan)
        pthjj30 = np.full(n_cand, np.nan)
        dr_min_gamma_j = np.full(n_cand, np.nan)
        vbf_centrality = np.full(n_cand, np.nan)
        ht = ak.to_numpy(ak.sum(cand_jets30.pt, axis=1))
        m_all_jets = np.full(n_cand, np.nan)
        delta_y = np.full(n_cand, np.nan)
        cos_theta_star = np.full(n_cand, np.nan)
        abs_dphi_capped = np.full(n_cand, np.nan)
        training_mask_tth = (n_lep == 0) & (n_jets30 >= 3) & (n_b25 >= 1)
        training_mask_vh = np.zeros(n_cand, dtype=bool)
        training_mask_vbf = np.zeros(n_cand, dtype=bool)
        bdt_subregion = np.full(n_cand, "inclusive", dtype=object)

        if np.any(n_jets30 > 0):
            leading_jet_pt30[n_jets30 > 0] = ak.to_numpy(cand_jets30[n_jets30 > 0][:, 0].pt)
        for idx in range(n_cand):
            jets30_evt = cand_jets30[idx]
            if len(jets30_evt):
                px = ak.to_numpy(jets30_evt.pt * np.cos(jets30_evt.phi))
                py = ak.to_numpy(jets30_evt.pt * np.sin(jets30_evt.phi))
                pz = ak.to_numpy(jets30_evt.pt * np.sinh(jets30_evt.eta))
                energy = ak.to_numpy(jets30_evt.e)
                m_all_jets[idx] = float(_invariant_mass(np.sum(px), np.sum(py), np.sum(pz), np.sum(energy)))
            if len(jets30_evt) >= 2:
                j1 = jets30_evt[0]
                j2 = jets30_evt[1]
                j1_px, j1_py, j1_pz, _ = _vector_components(np.asarray([j1.pt]), np.asarray([j1.eta]), np.asarray([j1.phi]), np.asarray([j1.e]))
                j2_px, j2_py, j2_pz, _ = _vector_components(np.asarray([j2.pt]), np.asarray([j2.eta]), np.asarray([j2.phi]), np.asarray([j2.e]))
                dijet_px = j1_px[0] + j2_px[0]
                dijet_py = j1_py[0] + j2_py[0]
                dijet_pz = j1_pz[0] + j2_pz[0]
                dijet_e = float(j1.e + j2.e)
                mjj30[idx] = float(_invariant_mass(dijet_px, dijet_py, dijet_pz, dijet_e))
                abs_deta30[idx] = abs(float(j1.eta - j2.eta))
                pthjj30[idx] = float(np.hypot(dip_px[idx] + dijet_px, dip_py[idx] + dijet_py))
                dijet_y = float(_rapidities(np.asarray([dijet_e]), np.asarray([dijet_pz]))[0])
                delta_y[idx] = dijet_y - dip_y[idx]
                vbf_centrality[idx] = abs(full_etagg[candidate_mask][idx] - 0.5 * (float(j1.eta) + float(j2.eta)))
                dphi = abs(_delta_phi(np.asarray([math.atan2(dip_py[idx], dip_px[idx])]), np.asarray([math.atan2(dijet_py, dijet_px)]))[0])
                abs_dphi_capped[idx] = min(dphi, 2.94)
                drs = []
                for photon_eta, photon_phi in ((lead_eta[idx], lead_phi[idx]), (sub_eta[idx], sub_phi[idx])):
                    for jet_eta, jet_phi in ((float(j1.eta), float(j1.phi)), (float(j2.eta), float(j2.phi))):
                        drs.append(float(np.hypot(photon_eta - jet_eta, _delta_phi(np.asarray([photon_phi]), np.asarray([jet_phi]))[0])))
                dr_min_gamma_j[idx] = min(drs) if drs else np.nan
                training_mask_vh[idx] = bool(60.0 < mjj30[idx] < 120.0)
                training_mask_vbf[idx] = bool(abs_deta30[idx] > 2.0 and vbf_centrality[idx] < 5.0)
                if training_mask_vbf[idx]:
                    bdt_subregion[idx] = "vbf_low_pT_Hjj" if pthjj30[idx] < 25.0 else "vbf_high_pT_Hjj"
                system_px = dip_px[idx] + dijet_px
                system_py = dip_py[idx] + dijet_py
                system_pz = dip_pz[idx] + dijet_pz
                system_p = np.sqrt(system_px**2 + system_py**2 + system_pz**2)
                dip_p = np.sqrt(dip_px[idx] ** 2 + dip_py[idx] ** 2 + dip_pz[idx] ** 2)
                if system_p > 0.0 and dip_p > 0.0:
                    cos_theta_star[idx] = (dip_px[idx] * system_px + dip_py[idx] * system_py + dip_pz[idx] * system_pz) / (dip_p * system_p)

        sideband = (full_mass[candidate_mask] >= FIT_RANGE[0]) & (full_mass[candidate_mask] < SIGNAL_WINDOW[0]) | (full_mass[candidate_mask] > SIGNAL_WINDOW[1]) & (full_mass[candidate_mask] <= FIT_RANGE[1])
        signal_window = (full_mass[candidate_mask] >= SIGNAL_WINDOW[0]) & (full_mass[candidate_mask] <= SIGNAL_WINDOW[1])
        nominal = nominal_region[candidate_mask]
        control = control_region[candidate_mask]
        photon_region = np.where(nominal, "nominal_photon_region", "anti_id_or_iso_control_region")
        fields = {
            "event_number": event_number[candidate_mask],
            "run_number": run_number[candidate_mask],
            "weight": weights[candidate_mask],
            "trigger_passed": trigger_mask[candidate_mask].astype(int),
            "baseline_selected": nominal.astype(int),
            "is_sideband": sideband.astype(int),
            "is_signal_window": signal_window.astype(int),
            "m_gammagamma": full_mass[candidate_mask],
            "pT_gammagamma": full_ptgg[candidate_mask],
            "eta_gammagamma": full_etagg[candidate_mask],
            "pTt_gammagamma": full_ptt[candidate_mask],
            "lead_pt": lead_pt,
            "sublead_pt": sub_pt,
            "lead_eta": lead_eta,
            "sublead_eta": sub_eta,
            "max_abs_photon_eta": full_abs_eta_max[candidate_mask],
            "N_jets_25": n_jets25,
            "N_jets_30": n_jets30,
            "N_central_jets_25": n_central25,
            "N_forward_jets_25": n_forward25,
            "N_btag_25": n_b25,
            "N_lep": n_lep,
            "m_ll": np.full(n_cand, np.nan),
            "Z_ll_veto": np.ones(n_cand, dtype=int),
            "m_e_gamma_veto": np.ones(n_cand, dtype=int),
            "pT_lepton_plus_MET": np.full(n_cand, np.nan),
            "MET": met,
            "MET_significance": met / np.sqrt(np.clip(ht, 1.0, None)),
            "leading_jet_pT_30": leading_jet_pt30,
            "m_jj_30": mjj30,
            "abs_delta_eta_jj_30": abs_deta30,
            "pT_Hjj_30": pthjj30,
            "deltaR_min_gamma_j": dr_min_gamma_j,
            "VBF_centrality": vbf_centrality,
            "H_T": ht,
            "m_all_jets": m_all_jets,
            "delta_y_gammagamma_jj": delta_y,
            "cos_theta_star_gammagamma_jj": cos_theta_star,
            "abs_delta_phi_gammagamma_jj_capped": abs_dphi_capped,
            "training_mask_tth": training_mask_tth.astype(int),
            "training_mask_vh": training_mask_vh.astype(int),
            "training_mask_vbf": training_mask_vbf.astype(int),
            "photon_region": photon_region,
            "nominal_photon_region": nominal.astype(int),
            "anti_id_or_iso_control_region": control.astype(int),
            "bdt_subregion": bdt_subregion,
        }
        for key, value in fields.items():
            _append_chunk(arrays, key, np.asarray(value))

    return {
        "sample_id": sample["sample_id"],
        "process_key": sample["process_key"],
        "kind": sample["kind"],
        "analysis_role": sample["analysis_role"],
        "bdt_arrays": _concat_arrays(arrays),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _variable_status(outputs: Path) -> dict[str, Any]:
    payload = {
        "status": "ok",
        "variables": {
            "m_gammagamma": {"status": "implemented_faithfully"},
            "pT_gammagamma": {"status": "implemented_faithfully"},
            "eta_gammagamma": {"status": "implemented_faithfully"},
            "N_jets": {"status": "implemented_faithfully"},
            "N_btag": {"status": "approximated", "notes": "Using jet_btag_quantile >= 4 as the 70% b-tag proxy."},
            "N_lep": {"status": "implemented_faithfully"},
            "m_ll": {"status": "implemented_faithfully"},
            "m_e_gamma": {"status": "approximated", "notes": "Using reconstructed electrons from the open-data ntuples."},
            "pT_lepton_plus_MET": {"status": "implemented_faithfully"},
            "MET_significance": {"status": "approximated", "notes": "Using MET/sqrt(HT) as the user-approved approximation."},
            "cos_theta_star_gammagamma_jj": {"status": "approximated", "notes": "Using a reconstructed system-angle proxy for the supplemental BDT feature."},
        },
    }
    write_json(payload, outputs / "variable_implementation_status.json")
    write_text(
        "# Variable Implementation Status\n\n" + "\n".join(
            f"- `{name}`: {item['status']}" for name, item in payload["variables"].items()
        )
        + "\n",
        outputs / "variable_implementation_status.md",
    )
    return payload


def _summarize_categories(samples: list[dict[str, Any]], outputs: Path) -> None:
    rows = []
    markdown_lines = ["# Category Assignment Summary", ""]
    baseline_total = 0
    assigned_total = 0
    blocked_total = 0
    for sample in samples:
        categories = sample["arrays"]["assigned_category"].astype(str)
        reasons = sample["arrays"]["assignment_reason"].astype(str)
        baseline_total += len(categories)
        assigned_total += int(np.sum((categories != "unassigned") & (categories != "blocked_missing_input")))
        blocked_total += int(np.sum(categories == "blocked_missing_input"))
        for category in ORDERED_CATEGORIES + ["blocked_missing_input", "unassigned"]:
            count = int(np.sum(categories == category))
            if count == 0:
                continue
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "process_key": sample["process_key"],
                    "category": category,
                    "count": count,
                }
            )
        if np.any(categories == "unassigned"):
            unique_reasons, counts = np.unique(reasons[categories == "unassigned"], return_counts=True)
            markdown_lines.append(f"## Unassigned Reasons For `{sample['sample_id']}`")
            markdown_lines.append("")
            for reason, count in zip(unique_reasons.tolist(), counts.tolist()):
                markdown_lines.append(f"- {reason}: `{count}`")
            markdown_lines.append("")
    write_json({"status": "ok", "rows": rows}, outputs / "category_yields.json")
    _write_csv(rows, outputs / "category_yields.csv")
    markdown_lines[1:1] = [
        f"- Baseline-selected events: `{baseline_total}`",
        f"- Assigned events: `{assigned_total}`",
        f"- Blocked events: `{blocked_total}`",
        f"- Unassigned events: `{baseline_total - assigned_total - blocked_total}`",
        "",
    ]
    write_text("\n".join(markdown_lines) + "\n", outputs / "category_assignment_summary.md")


def _implementation_decisions(ads: dict[str, Any], outputs: Path) -> None:
    payload = {
        "status": "ok",
        "decisions": [
            {
                "affected_component": "b_tag_working_point",
                "ads_requirement": "70% b-tag efficiency working point",
                "operational_choice": "jet_btag_quantile >= 4",
                "confidence": "medium",
                "risk_level": "medium",
                "affected_categories": ["tH lep 0fwd", "tH lep 1fwd", "ttH lep", "ttH had BDT1-4", "tH had 4j1b", "tH had 4j2b"],
            },
            {
                "affected_component": "MET_significance",
                "ads_requirement": "MET/sqrt(sum_ET)",
                "operational_choice": "MET/sqrt(HT)",
                "confidence": "medium",
                "risk_level": "medium",
                "affected_categories": ["VH lep Low", "VH MET High", "VH MET Low"],
            },
            {
                "affected_component": "classifier_scores",
                "ads_requirement": "Official BDT outputs",
                "operational_choice": "Train supplemental local BDTs from ADS feature lists",
                "confidence": "high",
                "risk_level": "high",
                "affected_categories": [
                    "ttH had BDT1",
                    "ttH had BDT2",
                    "ttH had BDT3",
                    "ttH had BDT4",
                    "VH had tight",
                    "VH had loose",
                    "VBF tight, high pT_Hjj",
                    "VBF loose, high pT_Hjj",
                    "VBF tight, low pT_Hjj",
                    "VBF loose, low pT_Hjj",
                ],
            },
        ],
        "ads_ambiguities": ads["ambiguity_registry"],
    }
    write_json(payload, outputs / "implementation_decisions.json")
    lines = ["# Implementation Decisions", ""]
    for item in payload["decisions"]:
        lines.append(f"- `{item['affected_component']}`: {item['operational_choice']}")
    write_text("\n".join(lines) + "\n", outputs / "implementation_decisions.md")


def _write_cutflow(samples: list[dict[str, Any]], outputs: Path) -> None:
    ordered_steps = [
        "all_input_events",
        "trigger_requirement",
        "at_least_two_photon_candidates",
        "photon_kinematic_acceptance",
        "loose_photon_id_preselection",
        "select_two_highest_et_photons",
        "diphoton_primary_vertex_handling",
        "leading_photon_et_over_mgg",
        "subleading_photon_et_over_mgg",
        "tight_photon_identification",
        "photon_isolation",
        "diphoton_mass_window",
    ]
    aggregated = {}
    for step in ordered_steps:
        before = sum(sample["cutflow"][step]["before"] for sample in samples if sample["kind"] == "data")
        after = sum(sample["cutflow"][step]["after"] for sample in samples if sample["kind"] == "data")
        aggregated[step] = {
            "selection_name": step,
            "events_before": before,
            "events_after": after,
            "incremental_efficiency": 0.0 if before == 0 else after / before,
            "cumulative_efficiency": 0.0 if aggregated == {} else after / max(aggregated["all_input_events"]["events_after"], 1),
            "implementation_status": samples[0]["cutflow"][step]["status"],
            "notes": samples[0]["cutflow"][step]["notes"],
        }
    rows = list(aggregated.values())
    write_json({"status": "ok", "aggregated": aggregated}, outputs / "cutflow_baseline.json")
    _write_csv(rows, outputs / "cutflow_baseline.csv")
    lines = ["# Cut Flow", ""]
    for row in rows:
        lines.append(
            f"- `{row['selection_name']}`: before `{row['events_before']}`, after `{row['events_after']}`, incremental `{row['incremental_efficiency']:.4f}`, cumulative `{row['cumulative_efficiency']:.4f}`"
        )
    write_text("\n".join(lines) + "\n", outputs / "cutflow_baseline.md")


def _validation_plots(samples: list[dict[str, Any]], outputs: Path) -> None:
    plot_dir = ensure_dir(outputs / "plots")
    manifest = []
    for key, title, bins in [
        ("m_gammagamma", "Diphoton Mass", np.linspace(105.0, 160.0, 56)),
        ("lead_pt", "Leading Photon pT", np.linspace(0.0, 300.0, 31)),
        ("sublead_pt", "Subleading Photon pT", np.linspace(0.0, 250.0, 26)),
        ("N_jets_30", "Jet Multiplicity", np.arange(-0.5, 8.5, 1.0)),
        ("N_btag_25", "B-Tagged Jet Multiplicity", np.arange(-0.5, 6.5, 1.0)),
        ("N_lep", "Lepton Multiplicity", np.arange(-0.5, 5.5, 1.0)),
        ("MET", "Missing Transverse Momentum", np.linspace(0.0, 300.0, 31)),
    ]:
        plt.figure(figsize=(8, 5))
        for sample in samples:
            if len(sample["arrays"][key]) == 0:
                continue
            weights = sample["arrays"]["weight"] if sample["kind"] != "data" else None
            plt.hist(sample["arrays"][key], bins=bins, histtype="step", label=sample["sample_id"], weights=weights)
        plt.title(title)
        plt.xlabel(key)
        plt.ylabel("Events")
        plt.legend(fontsize=7)
        out_path = plot_dir / f"{key}.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        manifest.append({"plot": key, "path": str(out_path)})
    lines = ["# Validation Report", ""]
    for item in manifest:
        lines.append(f"- `{item['plot']}`: `{item['path']}`")
    write_text("\n".join(lines) + "\n", outputs / "validation_report.md")


def _implementation_status(outputs: Path) -> None:
    lines = [
        "# IMPLEMENTATION STATUS",
        "",
        "## Implemented faithfully",
        "- ADS loading and ordered category parsing",
        "- Diphoton kinematics, baseline ET/mgg cuts, and main multiplicity variables",
        "",
        "## Implemented with approximations",
        "- Supplemental BDT training in place of official ATLAS classifier artifacts",
        "- MET_significance approximated as MET/sqrt(HT)",
        "- B-tag working-point proxy from jet_btag_quantile >= 4",
        "",
        "## Blocked by unavailable inputs",
        "- Official diphoton primary-vertex NN reconstruction inputs",
        "",
        "## Recommended next steps",
        "- Validate supplemental BDTs against external references if score branches become available",
        "- Replace the fast significance proxy with a fit-based validation pass for winning boundaries",
        "",
    ]
    write_text("\n".join(lines) + "\n", outputs / "IMPLEMENTATION_STATUS.md")


def _write_bdt_optimization_metadata(
    *,
    outputs: Path,
    ads: dict[str, Any],
    inputs: Path,
    selected_samples: list[dict[str, Any]],
    training_report: dict[str, Any] | None,
    audit_report: dict[str, Any] | None,
    prepare_bdt_training: bool,
    train_bdts: bool,
    score_bdts: bool,
) -> None:
    if not (prepare_bdt_training or train_bdts or score_bdts):
        return
    run_id = f"section8_bdt_{stable_hash({'outputs': str(outputs), 'time': utcnow_iso()})[:12]}"
    sample_hash = stable_hash(
        [
            {
                "sample_id": sample["sample_id"],
                "process_key": sample["process_key"],
                "kind": sample["kind"],
                "files": sample["files"],
            }
            for sample in selected_samples
        ]
    )
    metric_values = {
        "trained_classifiers": 0,
        "blocked_classifiers": 0,
    }
    if training_report:
        statuses = [item["status"] for item in training_report.get("classifiers", {}).values()]
        metric_values["trained_classifiers"] = statuses.count("trained")
        metric_values["blocked_classifiers"] = len([status for status in statuses if status != "trained"])
    observation = {
        "configuration_changes": {
            "changed_parameters": ["section8_ads.bdt_training"],
            "rationale": "Enable Section 8 supplemental BDT training with anti-ID/anti-isolation data controls.",
            "change_mode": "qualitative_strategy",
        },
        "primary_metric_response": {
            "metric": "trained_classifiers",
            "baseline_value": 0,
            "candidate_value": metric_values["trained_classifiers"],
            "absolute_change": metric_values["trained_classifiers"],
            "relative_change": None,
            "uncertainty": None,
            "meaningfulness": "infrastructure validation",
        },
        "yield_cutflow_response": {
            "yield_changes": None,
            "cutflow_changes": None,
            "signal_background_pattern": "See bdt_training_sample_audit.json for per-process training rows.",
            "unexpected_empty_or_unstable_categories": None,
        },
        "shape_response": {
            "changed_distributions": ["BDT_ttH", "BDT_VH", "BDT_VBF"],
            "localized_or_global": "localized to BDT-dependent Section 8 categories",
            "suspicious_structures": None,
            "binning_assessment": None,
        },
        "fit_inference_response": {
            "parameter_changes": None,
            "nuisance_changes": None,
            "goodness_of_fit_changes": None,
            "uncertainty_changes": None,
            "dominant_categories": None,
        },
        "validation_response": {
            "verifier_results": "smoke-test dependent",
            "warnings": [] if metric_values["blocked_classifiers"] == 0 else ["One or more BDTs did not train; inspect classifier_training_report.json."],
            "failed_checks": [],
            "artifacts_reused": [],
            "stages_rerun": ["section8_bdt_training"],
            "cache_concerns": None,
        },
        "interpretation": {
            "status": "new qualitative opportunity" if metric_values["trained_classifiers"] else "requires human review",
            "direct_observations": metric_values,
            "plausible_interpretation": "Prepared BDT artifacts are suitable for future optimization only when all needed classifiers train.",
            "unresolved_uncertainty": "Supplemental BDTs are not official ATLAS artifacts.",
            "possible_implementation_issue": None,
        },
    }
    run_dir = ensure_dir(Path("/Users/haichenwang/Work/newpipeline/modular-pipeline/runs") / run_id)
    write_json(observation, run_dir / "observations.yaml")
    registry_record = {
        "run_id": run_id,
        "timestamp": utcnow_iso(),
        "parent_run_id": None,
        "branch_id": "section8_ads_bdt",
        "strategy_id": "xgboost_anti_id_iso_controls",
        "run_type": "service_extension_validation",
        "objective": "Prepare and train supplemental Section 8 BDTs for continuous optimization.",
        "configuration_snapshot_path": str(outputs / "run_manifest.json"),
        "changed_parameters": ["prepare_bdt_training", "train_bdts", "score_bdts"],
        "reused_artifacts": [],
        "invalidated_artifacts": ["section8_classifier_scores", "section8_category_assignment"],
        "stages_executed": ["bdt_training_sample_audit", "classifier_training", "classifier_scoring"],
        "stages_skipped": [],
        "verifier_status": "ok",
        "cut_flow_path": str(outputs / "cutflow_baseline.json"),
        "yield_table_path": str(outputs / "category_yields.json"),
        "fit_output_path": None,
        "metric_values": metric_values,
        "expected_significance": None,
        "observed_significance": None,
        "runtime": None,
        "warnings": observation["validation_response"]["warnings"],
        "failure_reason": None,
        "human_or_agent_note": "Section 8 BDT training artifacts are supplemental and local.",
        "git_state": {"dirty": None, "commit": None},
        "service_versions": {"section8_ads_bdt": stable_hash({"ads": ads["config_hash"], "samples": sample_hash})},
        "version_name": "section8-bdt-xgboost-controls",
        "version_ref": None,
        "artifacts": {
            "audit": None if audit_report is None else str(outputs / "bdt_training_sample_audit.json"),
            "training_report": None if training_report is None else str(outputs / "classifier_training_report.json"),
        },
        "inputs": str(inputs),
    }
    registry_path = Path("/Users/haichenwang/Work/newpipeline/modular-pipeline/optimization_infra/runs.jsonl")
    with registry_path.open("a") as handle:
        handle.write(json.dumps(registry_record, sort_keys=True) + "\n")


def run_section8_ads(
    *,
    ads_path: Path,
    inputs: Path,
    outputs: Path,
    max_events: int | None = None,
    prepare_bdt_training: bool = False,
    train_bdts: bool = False,
    score_bdts: bool = False,
    optimize_boundaries_flag: bool = False,
    reuse_bdt_artifacts: Path | None = None,
    trigger_policy: str = "input_preselected",
) -> dict[str, Any]:
    outputs = ensure_dir(outputs)
    ads = load_ads(ads_path)
    summary_path = Path(__file__).resolve().parents[1] / "analysis.summary.json"
    registry, _ = build_registry(inputs, read_json(summary_path), 36.1)
    selected_samples = _select_samples(registry)
    branch_map = build_branch_mapping([sample["files"][0] for sample in selected_samples], ads, outputs)
    classifier_input_status(branch_map["available_fields"], outputs)
    processed = [_process_sample(sample, max_events=max_events, trigger_policy=trigger_policy) for sample in selected_samples]
    _write_cutflow(processed, outputs)
    variable_status = _variable_status(outputs)
    training_report = None
    audit_report = None
    bdt_training_samples = None
    bdt_metadata = {
        "ads_hash": ads["config_hash"],
        "branch_map_hash": stable_hash(branch_map),
        "sample_list_hash": stable_hash([sample["sample_id"] for sample in selected_samples]),
        "selection_policy_hash": stable_hash(
            {
                "control_region": "anti_id_or_iso_control_region",
                "mass_range": FIT_RANGE,
                "tth": "N_lep == 0 and N_jets_30 >= 3 and N_btag_25 >= 1",
                "vh": "N_jets_30 >= 2 and 60 < m_jj_30 < 120",
                "vbf": "N_jets_30 >= 2 and abs_delta_eta_jj_30 > 2 and VBF_centrality < 5",
            }
        ),
    }
    if prepare_bdt_training or train_bdts:
        bdt_training_samples = [
            _process_bdt_candidates(sample, max_events=max_events, trigger_policy=trigger_policy)
            for sample in selected_samples
        ]
        audit_report = write_training_sample_audit(bdt_training_samples, outputs, metadata=bdt_metadata)
    if train_bdts:
        training_report = train_classifiers(processed, outputs, training_samples=bdt_training_samples, metadata=bdt_metadata)
    elif reuse_bdt_artifacts is not None:
        candidate = reuse_bdt_artifacts / "classifier_training_report.json"
        if candidate.exists():
            training_report = read_json(candidate)
    if training_report and (score_bdts or train_bdts or reuse_bdt_artifacts is not None):
        score_samples(processed, training_report)
    for sample in processed:
        for name in ("BDT_ttH", "BDT_VH", "BDT_VBF"):
            if name not in sample["arrays"]:
                sample["arrays"][name] = np.full(len(sample["arrays"]["event_number"]), np.nan)

    boundary_report = None
    if optimize_boundaries_flag and train_bdts:
        boundary_report = optimize_boundaries(processed, outputs)
    boundaries = None if boundary_report is None else boundary_report["selected_boundaries"]
    for sample in processed:
        assigned, reasons, blocked = assign_categories(sample["arrays"], boundaries)
        sample["arrays"]["assigned_category"] = assigned
        sample["arrays"]["assignment_reason"] = reasons
        sample["arrays"]["assignment_blocked"] = blocked.astype(int)
    _summarize_categories(processed, outputs)
    _implementation_decisions(ads, outputs)
    _validation_plots(processed, outputs)
    _implementation_status(outputs)
    _write_bdt_optimization_metadata(
        outputs=outputs,
        ads=ads,
        inputs=inputs,
        selected_samples=selected_samples,
        training_report=training_report,
        audit_report=audit_report,
        prepare_bdt_training=prepare_bdt_training,
        train_bdts=train_bdts,
        score_bdts=score_bdts,
    )
    write_json(
        {
            "status": "ok",
            "timestamp_utc": utcnow_iso(),
            "ads_path": str(ads_path),
            "inputs": str(inputs),
            "outputs": str(outputs),
            "max_events": max_events,
            "prepare_bdt_training": prepare_bdt_training,
            "train_bdts": train_bdts,
            "score_bdts": score_bdts,
            "optimize_boundaries": optimize_boundaries_flag,
            "reuse_bdt_artifacts": None if reuse_bdt_artifacts is None else str(reuse_bdt_artifacts),
            "trigger_policy": trigger_policy,
            "config_hash": stable_hash(
                {
                    "ads": ads["config_hash"],
                    "max_events": max_events,
                    "prepare_bdt_training": prepare_bdt_training,
                    "train_bdts": train_bdts,
                    "score_bdts": score_bdts,
                    "optimize_boundaries": optimize_boundaries_flag,
                    "trigger_policy": trigger_policy,
                    "bdt_metadata": bdt_metadata,
                }
            ),
        },
        outputs / "run_manifest.json",
    )
    return {
        "ads": ads,
        "branch_map": branch_map,
        "variable_status": variable_status,
        "audit_report": audit_report,
        "training_report": training_report,
        "boundary_report": boundary_report,
        "processed": processed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Section 8 ADS reconstruction pipeline.")
    parser.add_argument("--ads", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--prepare-bdt-training", action="store_true")
    parser.add_argument("--train-bdts", action="store_true")
    parser.add_argument("--score-bdts", action="store_true")
    parser.add_argument("--optimize-boundaries", action="store_true")
    parser.add_argument("--reuse-bdt-artifacts", type=Path)
    parser.add_argument("--trigger-policy", choices=["input_preselected", "trigP", "none"], default="input_preselected")
    args = parser.parse_args()
    run_section8_ads(
        ads_path=Path(args.ads),
        inputs=Path(args.inputs),
        outputs=Path(args.outputs),
        max_events=args.max_events,
        prepare_bdt_training=args.prepare_bdt_training,
        train_bdts=args.train_bdts,
        score_bdts=args.score_bdts,
        optimize_boundaries_flag=args.optimize_boundaries,
        reuse_bdt_artifacts=args.reuse_bdt_artifacts,
        trigger_policy=args.trigger_policy,
    )


if __name__ == "__main__":
    main()
