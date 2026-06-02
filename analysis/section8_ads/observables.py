from __future__ import annotations

from dataclasses import dataclass
import math

import awkward as ak
import numpy as np


FIT_RANGE = (105.0, 160.0)
SIGNAL_WINDOW = (120.0, 130.0)
SIDEBAND_SCALE = 10.0 / 45.0

SECTION8_REQUIRED_BRANCHES = [
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


@dataclass(frozen=True)
class DiphotonEventView:
    """Per-input-event diphoton preselection and kinematic arrays."""

    event_number: np.ndarray
    run_number: np.ndarray
    weight: np.ndarray
    trigger_mask: np.ndarray
    has_two_kinematic_photons: np.ndarray
    has_two: np.ndarray
    selected_photons: ak.Array
    selected_input_indices: np.ndarray
    full_tight: np.ndarray
    full_iso: np.ndarray
    full_ptfrac: np.ndarray
    full_mass: np.ndarray
    full_ptgg: np.ndarray
    full_etagg: np.ndarray
    full_ptt: np.ndarray
    full_lead_pt: np.ndarray
    full_sub_pt: np.ndarray
    full_lead_eta: np.ndarray
    full_sub_eta: np.ndarray
    full_abs_eta_max: np.ndarray

    @property
    def nominal_photon_region(self) -> np.ndarray:
        return self.full_tight & self.full_iso

    @property
    def anti_id_or_iso_control_region(self) -> np.ndarray:
        return ~self.nominal_photon_region


@dataclass(frozen=True)
class Section8ObservableResult:
    """Materialized Section 8 observables for a selected event view."""

    fields: dict[str, np.ndarray]
    selection_mask: np.ndarray


def delta_phi(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))


def rapidities(energy: np.ndarray, pz: np.ndarray) -> np.ndarray:
    denom = np.clip(energy - pz, 1e-6, None)
    return 0.5 * np.log(np.clip((energy + pz) / denom, 1e-6, None))


def invariant_mass(px: np.ndarray, py: np.ndarray, pz: np.ndarray, energy: np.ndarray) -> np.ndarray:
    mass2 = energy**2 - px**2 - py**2 - pz**2
    return np.sqrt(np.clip(mass2, 0.0, None))


def vector_components(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray, energy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return px, py, pz, energy


def photon_iso_proxy(pt: ak.Array, topoetcone40: ak.Array, ptcone20: ak.Array) -> ak.Array:
    # Approximate the published calorimeter/track isolation using the closest visible ntuple branches.
    return (topoetcone40 <= 0.065 * pt) & (ptcone20 <= 0.05 * pt)


def build_diphoton_event_view(
    batch: ak.Array,
    *,
    weights: np.ndarray,
    trigger_mask: np.ndarray,
) -> DiphotonEventView:
    event_number = np.asarray(batch["eventNumber"], dtype=np.int64)
    run_number = np.asarray(batch["runNumber"], dtype=np.int64)
    n_events = len(event_number)

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
            "tight_iso": photon_iso_proxy(
                batch["photon_pt"][loose_mask],
                batch["photon_topoetcone40"][loose_mask],
                batch["photon_ptcone20"][loose_mask],
            ),
        }
    )
    ordered = photons[ak.argsort(photons.pt, axis=1, ascending=False)]
    has_two_kinematic_photons = ak.to_numpy(ak.num(batch["photon_pt"][kin_mask])) >= 2
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
    selected_input_indices = np.where(has_two)[0]

    if len(selected_input_indices):
        lead = selected[:, 0]
        sub = selected[:, 1]
        lead_px, lead_py, lead_pz, lead_e = vector_components(ak.to_numpy(lead.pt), ak.to_numpy(lead.eta), ak.to_numpy(lead.phi), ak.to_numpy(lead.e))
        sub_px, sub_py, sub_pz, sub_e = vector_components(ak.to_numpy(sub.pt), ak.to_numpy(sub.eta), ak.to_numpy(sub.phi), ak.to_numpy(sub.e))
        dip_px = lead_px + sub_px
        dip_py = lead_py + sub_py
        dip_pz = lead_pz + sub_pz
        dip_e = lead_e + sub_e
        mass = invariant_mass(dip_px, dip_py, dip_pz, dip_e)
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
        full_tight[selected_input_indices] = tight_id
        full_iso[selected_input_indices] = tight_iso
        full_ptfrac[selected_input_indices] = ptfrac
        full_mass[selected_input_indices] = mass
        full_ptgg[selected_input_indices] = ptgg
        full_etagg[selected_input_indices] = etagg
        full_ptt[selected_input_indices] = ptt
        full_lead_pt[selected_input_indices] = lead_pt
        full_sub_pt[selected_input_indices] = sub_pt
        full_lead_eta[selected_input_indices] = lead_eta
        full_sub_eta[selected_input_indices] = sub_eta
        full_abs_eta_max[selected_input_indices] = np.maximum(np.abs(lead_eta), np.abs(sub_eta))

    return DiphotonEventView(
        event_number=event_number,
        run_number=run_number,
        weight=np.asarray(weights, dtype=float),
        trigger_mask=np.asarray(trigger_mask, dtype=bool),
        has_two_kinematic_photons=has_two_kinematic_photons,
        has_two=has_two,
        selected_photons=selected,
        selected_input_indices=selected_input_indices,
        full_tight=full_tight,
        full_iso=full_iso,
        full_ptfrac=full_ptfrac,
        full_mass=full_mass,
        full_ptgg=full_ptgg,
        full_etagg=full_etagg,
        full_ptt=full_ptt,
        full_lead_pt=full_lead_pt,
        full_sub_pt=full_sub_pt,
        full_lead_eta=full_lead_eta,
        full_sub_eta=full_sub_eta,
        full_abs_eta_max=full_abs_eta_max,
    )


def build_section8_observables(
    batch: ak.Array,
    view: DiphotonEventView,
    selection_mask: np.ndarray,
    *,
    baseline_selected: np.ndarray | None = None,
    compute_lepton_vetoes: bool = True,
    include_bdt_subregion: bool = False,
) -> Section8ObservableResult:
    selection_mask = np.asarray(selection_mask, dtype=bool)
    n_selected = int(np.sum(selection_mask))

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

    selector = ak.Array(selection_mask)
    selected_jets25 = jets25[selector]
    selected_jets30 = jets30[selector]
    selected_jets25_jvt = jets25_jvt[selector]
    selected_jets30_jvt = jets30_jvt[selector]
    selected_leptons_all = leptons[selector]

    n_jets25 = ak.to_numpy(ak.num(selected_jets25))
    n_jets30 = ak.to_numpy(ak.num(selected_jets30))
    n_jets25_jvt = ak.to_numpy(ak.num(selected_jets25_jvt))
    n_jets30_jvt = ak.to_numpy(ak.num(selected_jets30_jvt))
    n_central25 = ak.to_numpy(ak.sum(abs(selected_jets25.eta) < 2.5, axis=1))
    n_forward25 = ak.to_numpy(ak.sum(abs(selected_jets25.eta) >= 2.5, axis=1))
    n_b25 = ak.to_numpy(ak.sum(selected_jets25.btag >= 4, axis=1))

    electron_mask = (
        (selected_leptons_all.type == 11)
        & (selected_leptons_all.pt > 10.0)
        & (abs(selected_leptons_all.eta) < 2.47)
        & ~((abs(selected_leptons_all.eta) > 1.37) & (abs(selected_leptons_all.eta) < 1.52))
        & selected_leptons_all.medium_id
        & selected_leptons_all.loose_iso
        & (abs(selected_leptons_all.z0) < 0.5)
        & (abs(selected_leptons_all.d0sig) < 5.0)
    )
    muon_mask = (
        (selected_leptons_all.type == 13)
        & (selected_leptons_all.pt > 10.0)
        & (abs(selected_leptons_all.eta) < 2.7)
        & selected_leptons_all.medium_id
        & selected_leptons_all.loose_iso
        & (abs(selected_leptons_all.z0) < 0.5)
        & (abs(selected_leptons_all.d0sig) < 3.0)
    )
    selected_leptons = selected_leptons_all[electron_mask | muon_mask]
    n_lep = ak.to_numpy(ak.num(selected_leptons))

    met = np.asarray(batch["met"], dtype=float)[selection_mask]
    met_phi = np.asarray(batch["met_phi"], dtype=float)[selection_mask]
    lead_eta = view.full_lead_eta[selection_mask]
    sub_eta = view.full_sub_eta[selection_mask]
    lead_pt = view.full_lead_pt[selection_mask]
    sub_pt = view.full_sub_pt[selection_mask]
    selected_photons = view.selected_photons[selection_mask[view.selected_input_indices]]
    lead_phi = ak.to_numpy(selected_photons[:, 0].phi) if len(selected_photons) else np.zeros(n_selected)
    sub_phi = ak.to_numpy(selected_photons[:, 1].phi) if len(selected_photons) else np.zeros(n_selected)
    lead_e = lead_pt * np.cosh(lead_eta)
    sub_e = sub_pt * np.cosh(sub_eta)
    lead_px, lead_py, lead_pz, _ = vector_components(lead_pt, lead_eta, lead_phi, lead_e)
    sub_px, sub_py, sub_pz, _ = vector_components(sub_pt, sub_eta, sub_phi, sub_e)
    dip_px = lead_px + sub_px
    dip_py = lead_py + sub_py
    dip_pz = lead_pz + sub_pz
    dip_e = lead_e + sub_e
    dip_y = rapidities(dip_e, dip_pz)
    selected_etagg = view.full_etagg[selection_mask]

    leading_jet_pt30 = np.full(n_selected, np.nan)
    mjj30 = np.full(n_selected, np.nan)
    abs_deta30 = np.full(n_selected, np.nan)
    pthjj30 = np.full(n_selected, np.nan)
    dr_min_gamma_j = np.full(n_selected, np.nan)
    vbf_centrality = np.full(n_selected, np.nan)
    ht = ak.to_numpy(ak.sum(selected_jets30.pt, axis=1))
    m_all_jets = np.full(n_selected, np.nan)
    delta_y = np.full(n_selected, np.nan)
    cos_theta_star = np.full(n_selected, np.nan)
    abs_dphi_capped = np.full(n_selected, np.nan)
    training_mask_tth = (n_lep == 0) & (n_jets30 >= 3) & (n_b25 >= 1)
    training_mask_vh = np.zeros(n_selected, dtype=bool)
    training_mask_vbf = np.zeros(n_selected, dtype=bool)
    bdt_subregion = np.full(n_selected, "inclusive", dtype=object)

    if np.any(n_jets30 > 0):
        leading_jet_pt30[n_jets30 > 0] = ak.to_numpy(selected_jets30[n_jets30 > 0][:, 0].pt)
    for idx in range(n_selected):
        jets30_evt = selected_jets30[idx]
        if len(jets30_evt):
            px = ak.to_numpy(jets30_evt.pt * np.cos(jets30_evt.phi))
            py = ak.to_numpy(jets30_evt.pt * np.sin(jets30_evt.phi))
            pz = ak.to_numpy(jets30_evt.pt * np.sinh(jets30_evt.eta))
            energy = ak.to_numpy(jets30_evt.e)
            m_all_jets[idx] = float(invariant_mass(np.sum(px), np.sum(py), np.sum(pz), np.sum(energy)))
        if len(jets30_evt) >= 2:
            j1 = jets30_evt[0]
            j2 = jets30_evt[1]
            j1_px, j1_py, j1_pz, _ = vector_components(np.asarray([j1.pt]), np.asarray([j1.eta]), np.asarray([j1.phi]), np.asarray([j1.e]))
            j2_px, j2_py, j2_pz, _ = vector_components(np.asarray([j2.pt]), np.asarray([j2.eta]), np.asarray([j2.phi]), np.asarray([j2.e]))
            dijet_px = j1_px[0] + j2_px[0]
            dijet_py = j1_py[0] + j2_py[0]
            dijet_pz = j1_pz[0] + j2_pz[0]
            dijet_e = float(j1.e + j2.e)
            mjj30[idx] = float(invariant_mass(dijet_px, dijet_py, dijet_pz, dijet_e))
            abs_deta30[idx] = abs(float(j1.eta - j2.eta))
            pthjj30[idx] = float(np.hypot(dip_px[idx] + dijet_px, dip_py[idx] + dijet_py))
            dijet_y = float(rapidities(np.asarray([dijet_e]), np.asarray([dijet_pz]))[0])
            delta_y[idx] = dijet_y - dip_y[idx]
            vbf_centrality[idx] = abs(selected_etagg[idx] - 0.5 * (float(j1.eta) + float(j2.eta)))
            dphi = abs(delta_phi(np.asarray([math.atan2(dip_py[idx], dip_px[idx])]), np.asarray([math.atan2(dijet_py, dijet_px)]))[0])
            abs_dphi_capped[idx] = min(dphi, 2.94)
            drs = []
            for photon_eta, photon_phi in ((lead_eta[idx], lead_phi[idx]), (sub_eta[idx], sub_phi[idx])):
                for jet_eta, jet_phi in ((float(j1.eta), float(j1.phi)), (float(j2.eta), float(j2.phi))):
                    drs.append(float(np.hypot(photon_eta - jet_eta, delta_phi(np.asarray([photon_phi]), np.asarray([jet_phi]))[0])))
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

    met_sig = met / np.sqrt(np.clip(ht, 1.0, None))
    mll = np.full(n_selected, np.nan)
    z_veto = np.ones(n_selected, dtype=bool)
    megamma_veto = np.ones(n_selected, dtype=bool)
    ptlepmet = np.full(n_selected, np.nan)

    if compute_lepton_vetoes:
        for idx in range(n_selected):
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
                        electron_pairings.append(float(invariant_mass(lep_px + pho_px, lep_py + pho_py, lep_pz + pho_pz, float(li.e) + float(photon_pt * np.cosh(photon_eta)))))
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
                    sfos_masses.append(float(invariant_mass(li_px + lj_px, li_py + lj_py, li_pz + lj_pz, float(li.e + lj.e))))
            if sfos_masses:
                closest = min(sfos_masses, key=lambda item: abs(item - 91.2))
                mll[idx] = closest
                z_veto[idx] = not (70.0 <= closest <= 110.0)
            if electron_pairings:
                megamma_veto[idx] = not any(84.0 <= mass <= 94.0 for mass in electron_pairings)

    selected_mass = view.full_mass[selection_mask]
    sideband = (selected_mass >= FIT_RANGE[0]) & (selected_mass < SIGNAL_WINDOW[0]) | (selected_mass > SIGNAL_WINDOW[1]) & (selected_mass <= FIT_RANGE[1])
    signal_window = (selected_mass >= SIGNAL_WINDOW[0]) & (selected_mass <= SIGNAL_WINDOW[1])
    baseline_selected_array = np.ones(n_selected, dtype=int) if baseline_selected is None else np.asarray(baseline_selected, dtype=int)

    fields = {
        "event_number": view.event_number[selection_mask],
        "run_number": view.run_number[selection_mask],
        "weight": view.weight[selection_mask],
        "trigger_passed": view.trigger_mask[selection_mask].astype(int),
        "baseline_selected": baseline_selected_array,
        "is_sideband": sideband.astype(int),
        "is_signal_window": signal_window.astype(int),
        "m_gammagamma": selected_mass,
        "pT_gammagamma": view.full_ptgg[selection_mask],
        "eta_gammagamma": view.full_etagg[selection_mask],
        "pTt_gammagamma": view.full_ptt[selection_mask],
        "lead_pt": lead_pt,
        "sublead_pt": sub_pt,
        "lead_pt_over_mgg": lead_pt / np.clip(selected_mass, 1e-6, None),
        "sublead_pt_over_mgg": sub_pt / np.clip(selected_mass, 1e-6, None),
        "lead_eta": lead_eta,
        "sublead_eta": sub_eta,
        "max_abs_photon_eta": view.full_abs_eta_max[selection_mask],
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
    if include_bdt_subregion:
        fields["bdt_subregion"] = bdt_subregion
    return Section8ObservableResult(fields=fields, selection_mask=selection_mask)
