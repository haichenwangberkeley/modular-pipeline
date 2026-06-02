from __future__ import annotations

import csv
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from analysis.common import ensure_dir, stable_hash, utcnow_iso, write_json, write_text

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:  # pragma: no cover - exercised when dependency is missing
    XGBClassifier = None

try:
    from sklearn.metrics import roc_auc_score
except ModuleNotFoundError:  # pragma: no cover - xgboost is the required backend
    roc_auc_score = None


RANDOM_SEED = 20260601
BACKEND_NAME = "xgboost"
PEAK_WINDOW = (123.0, 127.0)
CONTINUUM_FALLBACK_TRANSFER_FACTOR = 1.0 / 11.0
XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.7,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
}


@dataclass(frozen=True)
class ClassifierSpec:
    name: str
    features: tuple[str, ...]
    signal_processes: tuple[str, ...]
    mc_background_processes: tuple[str, ...]
    training_mask_name: str


CLASSIFIER_SPECS = {
    "BDT_ttH": ClassifierSpec(
        name="BDT_ttH",
        features=("H_T", "m_all_jets", "N_jets_30", "N_central_jets_25", "N_btag_25"),
        signal_processes=("tth",),
        mc_background_processes=("ggh",),
        training_mask_name="training_mask_tth",
    ),
    "BDT_VH": ClassifierSpec(
        name="BDT_VH",
        features=("m_jj_30", "pTt_gammagamma", "delta_y_gammagamma_jj", "cos_theta_star_gammagamma_jj"),
        signal_processes=("wmh", "wph", "zh", "ggzh"),
        mc_background_processes=("ggh", "vbf", "tth", "prompt_diphoton"),
        training_mask_name="training_mask_vh",
    ),
    "BDT_VBF": ClassifierSpec(
        name="BDT_VBF",
        features=("m_jj_30", "abs_delta_eta_jj_30", "pTt_gammagamma", "abs_delta_phi_gammagamma_jj_capped", "deltaR_min_gamma_j", "VBF_centrality"),
        signal_processes=("vbf",),
        mc_background_processes=("ggh", "prompt_diphoton"),
        training_mask_name="training_mask_vbf",
    ),
}


def classifier_input_status(available_fields: list[str], outputs: Path) -> dict[str, Any]:
    payload = {"status": "ok", "classifiers": {}}
    available = set(available_fields)
    for name in CLASSIFIER_SPECS:
        if name in available:
            status = "official_score_branch_found"
        else:
            status = "unavailable"
        payload["classifiers"][name] = {
            "status": status,
            "branch": name if name in available else None,
            "supplemental_backend": BACKEND_NAME,
        }
    write_json(payload, outputs / "classifier_input_status.json")
    write_text(render_classifier_status_markdown(payload), outputs / "classifier_input_status.md")
    return payload


def render_classifier_status_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Classifier Input Status", ""]
    for name, item in payload["classifiers"].items():
        lines.append(f"- `{name}`: {item['status']} supplemental backend `{item['supplemental_backend']}`")
    return "\n".join(lines) + "\n"


def deterministic_split(sample_ids: np.ndarray, event_numbers: np.ndarray) -> np.ndarray:
    values = []
    for sample_id, event_number in zip(sample_ids.astype(str), event_numbers.astype(np.int64)):
        digest = stable_hash(f"{sample_id}:{int(event_number)}")
        values.append(int(digest[:16], 16) % 10)
    return np.asarray(values, dtype=np.int32)


def _split_name(split_value: int) -> str:
    if split_value <= 5:
        return "train"
    if split_value <= 7:
        return "validation"
    return "test"


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys()) if rows else ["status"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _training_role(spec: ClassifierSpec, process_key: str, kind: str, photon_region: str) -> tuple[int | None, str | None, str | None]:
    if kind == "data":
        if photon_region == "anti_id_or_iso_control_region":
            return 0, "background", "data_control"
        return None, None, None
    if photon_region != "nominal_photon_region":
        return None, None, None
    if process_key in spec.signal_processes:
        return 1, "signal", "mc_signal"
    if process_key in spec.mc_background_processes:
        return 0, "background", "mc_background"
    return None, None, None


def _component_type(label: int, source_kind: str) -> str:
    return "signal" if label == 1 else source_kind


def _component_type_for_role(label: int, process_key: str, source_kind: str) -> str:
    if label == 1:
        return "signal"
    if source_kind == "data_control":
        return "continuum_background"
    return "continuum_background" if process_key == "prompt_diphoton" else "resonant_background"


def _candidate_rows_for_spec(
    spec: ClassifierSpec,
    training_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    chunks = []
    labels = []
    raw_weights = []
    masses = []
    process_keys = []
    source_kinds = []
    component_types = []
    event_numbers = []
    sample_ids = []
    provenance = []
    audit_rows = []

    for sample in training_samples:
        arrays = sample["bdt_arrays"]
        if len(arrays.get("event_number", [])) == 0:
            continue
        mask = arrays[spec.training_mask_name].astype(bool)
        if not np.any(mask):
            continue
        features = np.column_stack([arrays[feature] for feature in spec.features])
        finite = np.all(np.isfinite(features), axis=1)
        split_values = deterministic_split(
            np.full(len(arrays["event_number"]), sample["sample_id"], dtype=object),
            arrays["event_number"],
        )
        for label, class_label, source_kind in {_training_role(spec, sample["process_key"], sample["kind"], str(region)) for region in np.unique(arrays["photon_region"].astype(str))}:
            if label is None:
                continue
            region_mask = arrays["photon_region"].astype(str) == ("anti_id_or_iso_control_region" if source_kind == "data_control" else "nominal_photon_region")
            role_mask = mask & region_mask
            if not np.any(role_mask):
                continue
            role_finite = role_mask & finite
            subregions = arrays.get("bdt_subregion", np.full(len(mask), "inclusive", dtype=object)).astype(str)
            for subregion in np.unique(subregions[role_mask]):
                for split in ("train", "validation", "test"):
                    split_mask = role_mask & (subregions == subregion) & np.asarray([_split_name(value) == split for value in split_values])
                    audit_rows.append(
                        {
                            "classifier": spec.name,
                            "sample_id": sample["sample_id"],
                            "process_key": sample["process_key"],
                            "class_label": class_label,
                            "source_kind": source_kind,
                            "photon_region": "anti_id_or_iso_control_region" if source_kind == "data_control" else "nominal_photon_region",
                            "bdt_subregion": subregion,
                            "split": split,
                            "finite_features": bool(np.any(split_mask & finite)),
                            "event_count": int(np.sum(split_mask)),
                            "finite_event_count": int(np.sum(split_mask & finite)),
                            "weighted_yield": float(np.sum(arrays["weight"][split_mask])),
                            "finite_weighted_yield": float(np.sum(arrays["weight"][split_mask & finite])),
                        }
                    )
            if not np.any(role_finite):
                continue
            chunks.append(features[role_finite])
            labels.append(np.full(int(np.sum(role_finite)), label, dtype=np.int32))
            raw_weights.append(np.asarray(arrays["weight"][role_finite], dtype=float))
            masses.append(np.asarray(arrays["m_gammagamma"][role_finite], dtype=float))
            process_keys.append(np.full(int(np.sum(role_finite)), sample["process_key"], dtype=object))
            source_kinds.append(np.full(int(np.sum(role_finite)), source_kind, dtype=object))
            component_types.append(
                np.full(
                    int(np.sum(role_finite)),
                    _component_type_for_role(label, sample["process_key"], source_kind),
                    dtype=object,
                )
            )
            event_numbers.append(np.asarray(arrays["event_number"][role_finite], dtype=np.int64))
            sample_ids.append(np.full(int(np.sum(role_finite)), sample["sample_id"], dtype=object))
            provenance.append(
                {
                    "sample_id": sample["sample_id"],
                    "process_key": sample["process_key"],
                    "kind": class_label,
                    "source_kind": source_kind,
                    "row_count": int(np.sum(role_finite)),
                    "weighted_yield": float(np.sum(arrays["weight"][role_finite])),
                }
            )

    if not chunks:
        return {"audit_rows": audit_rows, "provenance": provenance}

    rows = np.concatenate(chunks, axis=0)
    label_array = np.concatenate(labels)
    weight_array = np.concatenate(raw_weights)
    mass_array = np.concatenate(masses)
    process_array = np.concatenate(process_keys)
    source_array = np.concatenate(source_kinds)
    component_array = np.concatenate(component_types)
    event_array = np.concatenate(event_numbers)
    sample_array = np.concatenate(sample_ids)
    split = deterministic_split(sample_array, event_array)
    return {
        "features": rows,
        "labels": label_array,
        "raw_weights": weight_array,
        "masses": mass_array,
        "process_keys": process_array,
        "source_kinds": source_array,
        "component_types": component_array,
        "event_numbers": event_array,
        "sample_ids": sample_array,
        "split": split,
        "audit_rows": audit_rows,
        "provenance": provenance,
    }


def write_training_sample_audit(
    training_samples: list[dict[str, Any]],
    outputs: Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    audit_rows = []
    summaries = {}
    for name, spec in CLASSIFIER_SPECS.items():
        payload = _candidate_rows_for_spec(spec, training_samples)
        audit_rows.extend(payload["audit_rows"])
        summaries[name] = {
            "features": list(spec.features),
            "total_finite_rows": int(0 if "features" not in payload else payload["features"].shape[0]),
            "signal_rows": int(0 if "labels" not in payload else np.sum(payload["labels"] == 1)),
            "background_rows": int(0 if "labels" not in payload else np.sum(payload["labels"] == 0)),
            "training_samples": payload["provenance"],
        }

    artifact_hash = stable_hash({"rows": audit_rows, "metadata": metadata or {}})
    report = {
        "status": "ok",
        "timestamp_utc": utcnow_iso(),
        "backend": BACKEND_NAME,
        "random_seed": RANDOM_SEED,
        "artifact_hash": artifact_hash,
        "metadata": metadata or {},
        "classifiers": summaries,
        "rows": audit_rows,
    }
    write_json(report, outputs / "bdt_training_sample_audit.json")
    _write_csv(audit_rows, outputs / "bdt_training_sample_audit.csv")
    write_text(render_training_audit_markdown(report), outputs / "bdt_training_sample_audit.md")
    return report


def render_training_audit_markdown(report: dict[str, Any]) -> str:
    lines = ["# BDT Training Sample Audit", "", f"- Backend: `{report['backend']}`", f"- Artifact hash: `{report['artifact_hash']}`", ""]
    for name, item in report["classifiers"].items():
        lines.append(
            f"- `{name}`: finite rows `{item['total_finite_rows']}`, signal `{item['signal_rows']}`, background `{item['background_rows']}`"
        )
    return "\n".join(lines) + "\n"


def _normalized_training_weights(labels: np.ndarray, raw_weights: np.ndarray) -> np.ndarray:
    weights = np.abs(raw_weights.astype(float))
    normalized = np.zeros_like(weights, dtype=float)
    for label in (0, 1):
        mask = labels == label
        total = float(np.sum(weights[mask]))
        if total > 0.0:
            normalized[mask] = weights[mask] * (float(np.sum(mask)) / total)
        else:
            normalized[mask] = 1.0
    return normalized


def _peak_mask(masses: np.ndarray) -> np.ndarray:
    return (masses >= PEAK_WINDOW[0]) & (masses <= PEAK_WINDOW[1])


def _continuum_transfer_factor(masses: np.ndarray, weights: np.ndarray) -> tuple[float, str]:
    finite = np.isfinite(masses) & np.isfinite(weights) & (weights > 0.0)
    if np.sum(finite) < 20:
        return CONTINUUM_FALLBACK_TRANSFER_FACTOR, "fallback_shape_factor_low_statistics"
    hist, edges = np.histogram(masses[finite], bins=np.linspace(105.0, 160.0, 12), weights=weights[finite])
    centers = 0.5 * (edges[:-1] + edges[1:])
    positive = hist > 0.0
    if np.sum(positive) < 3:
        return CONTINUUM_FALLBACK_TRANSFER_FACTOR, "fallback_shape_factor_sparse_histogram"
    try:
        slope, intercept = np.polyfit(centers[positive], np.log(hist[positive]), deg=1)
        if abs(slope) < 1e-9:
            numerator = PEAK_WINDOW[1] - PEAK_WINDOW[0]
            denominator = 160.0 - 105.0
        else:
            numerator = math.exp(intercept) * (math.exp(slope * PEAK_WINDOW[1]) - math.exp(slope * PEAK_WINDOW[0])) / slope
            denominator = math.exp(intercept) * (math.exp(slope * 160.0) - math.exp(slope * 105.0)) / slope
        factor = float(numerator / denominator)
    except (FloatingPointError, ValueError, ZeroDivisionError, OverflowError):
        return CONTINUUM_FALLBACK_TRANSFER_FACTOR, "fallback_shape_factor_fit_failed"
    if not math.isfinite(factor) or factor <= 0.0 or factor > 1.0:
        return CONTINUUM_FALLBACK_TRANSFER_FACTOR, "fallback_shape_factor_unphysical_fit"
    return factor, "exponential_fit"


def _peak_normalized_weights(name: str, payload: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels = payload["labels"]
    raw_weights = np.abs(payload["raw_weights"].astype(float))
    masses = payload["masses"]
    process_keys = payload["process_keys"].astype(str)
    source_kinds = payload["source_kinds"].astype(str)
    component_types = payload["component_types"].astype(str)
    component_weights = np.zeros_like(raw_weights, dtype=float)
    rows = []
    component_keys = sorted(set(zip(labels.tolist(), process_keys.tolist(), source_kinds.tolist(), component_types.tolist())))
    for label, process_key, source_kind, component_type in component_keys:
        mask = (labels == int(label)) & (process_keys == process_key) & (source_kinds == source_kind) & (component_types == component_type)
        full_yield = float(np.sum(raw_weights[mask]))
        peak_yield = float(np.sum(raw_weights[mask & _peak_mask(masses)]))
        transfer_factor = None
        normalization_status = "direct_peak_yield"
        if component_type == "continuum_background":
            transfer_factor, normalization_status = _continuum_transfer_factor(masses[mask], raw_weights[mask])
            target_yield = full_yield * transfer_factor
        else:
            target_yield = peak_yield
            if target_yield <= 0.0 and full_yield > 0.0:
                target_yield = full_yield
                normalization_status = "fallback_full_range_no_peak_rows"
        if full_yield > 0.0 and target_yield > 0.0:
            component_weights[mask] = raw_weights[mask] * (target_yield / full_yield)
        rows.append(
            {
                "classifier": name,
                "process_key": process_key,
                "source_kind": source_kind,
                "component_type": component_type,
                "class_label": "signal" if int(label) == 1 else "background",
                "raw_row_count": int(np.sum(mask)),
                "full_range_weighted_yield": full_yield,
                "peak_window_weighted_yield": peak_yield,
                "transfer_factor": transfer_factor,
                "final_effective_training_yield": float(np.sum(component_weights[mask])),
                "normalization_status": normalization_status,
            }
        )
    return component_weights, rows


def _balance_class_weights(labels: np.ndarray, component_weights: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    weights = component_weights.copy()
    train_rows = max(float(np.sum(train_mask)), 1.0)
    target_per_class = 0.5 * train_rows
    for label in (0, 1):
        mask = train_mask & (labels == label)
        total = float(np.sum(weights[mask]))
        if total > 0.0:
            weights[labels == label] *= target_per_class / total
    return weights


def _mass_correlation_rows(name: str, spec: ClassifierSpec, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    continuum = (payload["labels"] == 0) & (payload["component_types"].astype(str) == "continuum_background")
    masses = payload["masses"]
    for idx, feature in enumerate(spec.features):
        mask = continuum & np.isfinite(masses) & np.isfinite(payload["features"][:, idx])
        correlation = None
        status = "insufficient_rows"
        if np.sum(mask) >= 3 and np.std(masses[mask]) > 0.0 and np.std(payload["features"][mask, idx]) > 0.0:
            correlation = float(np.corrcoef(masses[mask], payload["features"][mask, idx])[0, 1])
            status = "ok"
        rows.append(
            {
                "classifier": name,
                "feature": feature,
                "background_component": "continuum_background",
                "row_count": int(np.sum(mask)),
                "pearson_correlation_with_myy": correlation,
                "status": status,
                "warning": bool(correlation is not None and abs(correlation) > 0.3),
            }
        )
    return rows


def _write_background_normalization_report(rows: list[dict[str, Any]], outputs: Path) -> None:
    payload = {
        "status": "ok",
        "peak_window": list(PEAK_WINDOW),
        "continuum_fallback_transfer_factor": CONTINUUM_FALLBACK_TRANSFER_FACTOR,
        "rows": rows,
    }
    write_json(payload, outputs / "bdt_background_normalization.json")
    _write_csv(rows, outputs / "bdt_background_normalization.csv")
    lines = ["# BDT Background Normalization", "", f"- Peak window: `{PEAK_WINDOW[0]}-{PEAK_WINDOW[1]} GeV`", f"- Fallback continuum transfer factor: `{CONTINUUM_FALLBACK_TRANSFER_FACTOR:.6f}`", ""]
    for row in rows:
        lines.append(
            f"- `{row['classifier']}` `{row['process_key']}` `{row['component_type']}`: effective yield `{row['final_effective_training_yield']:.6g}`, status `{row['normalization_status']}`"
        )
    write_text("\n".join(lines) + "\n", outputs / "bdt_background_normalization.md")


def _write_mass_correlation_report(rows: list[dict[str, Any]], outputs: Path) -> None:
    payload = {"status": "ok", "rows": rows}
    write_json(payload, outputs / "bdt_mass_correlation_report.json")
    write_text(
        "# BDT Mass Correlation Report\n\n"
        + "\n".join(
            f"- `{row['classifier']}` `{row['feature']}`: correlation `{row['pearson_correlation_with_myy']}`, rows `{row['row_count']}`, warning `{row['warning']}`"
            for row in rows
        )
        + "\n",
        outputs / "bdt_mass_correlation_report.md",
    )


def _auc(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray | None = None) -> float | None:
    if roc_auc_score is None or len(labels) == 0 or len(np.unique(labels)) < 2:
        return None
    kwargs = {}
    if weights is not None:
        kwargs["sample_weight"] = weights
    return float(roc_auc_score(labels, scores, **kwargs))


def train_classifiers(
    samples: list[dict[str, Any]],
    outputs: Path,
    *,
    training_samples: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if XGBClassifier is None:
        raise RuntimeError("xgboost is required for --train-bdts but is not installed.")

    classifiers_dir = ensure_dir(outputs / "classifiers")
    training_samples = samples if training_samples is None else training_samples
    audit = write_training_sample_audit(training_samples, outputs, metadata=metadata)
    report = {
        "status": "ok",
        "backend": BACKEND_NAME,
        "backend_parameters": XGBOOST_PARAMS,
        "random_seed": RANDOM_SEED,
        "metadata": metadata or {},
        "compatibility_hash": stable_hash({"backend": BACKEND_NAME, "params": XGBOOST_PARAMS, "metadata": metadata or {}}),
        "training_audit_path": str(outputs / "bdt_training_sample_audit.json"),
        "classifiers": {},
        "approximations": [
            "Supplemental classifiers are locally trained approximations and are not official ATLAS model artifacts.",
            "Gamma-plus-jet and dijet data controls are combined as one anti-ID/anti-isolation reducible-background control class.",
            "Continuum backgrounds use full 105-160 GeV statistics with a peak-window normalization factor; fallback is 1/11.",
        ],
    }

    normalization_rows = []
    correlation_rows = []
    for name, spec in CLASSIFIER_SPECS.items():
        payload = _candidate_rows_for_spec(spec, training_samples)
        if "features" not in payload:
            report["classifiers"][name] = {
                "status": "blocked_no_training_rows",
                "features": list(spec.features),
                "training_samples": payload["provenance"],
            }
            continue
        features = payload["features"]
        labels = payload["labels"]
        raw_weights = payload["raw_weights"]
        component_weights, component_rows = _peak_normalized_weights(name, payload)
        normalization_rows.extend(component_rows)
        correlation_rows.extend(_mass_correlation_rows(name, spec, payload))
        split = payload["split"]
        train_mask = split <= 5
        valid_mask = (split >= 6) & (split <= 7)
        test_mask = split >= 8

        if np.sum(labels[train_mask] == 1) == 0:
            status = "blocked_no_signal_rows"
        elif np.sum(labels[train_mask] == 0) == 0:
            status = "blocked_no_background_rows"
        elif features.shape[0] == 0:
            status = "blocked_no_finite_features"
        else:
            status = "ready"
        if status != "ready":
            report["classifiers"][name] = {
                "status": status,
                "features": list(spec.features),
                "candidate_rows": int(features.shape[0]),
                "signal_rows": int(np.sum(labels == 1)),
                "background_rows": int(np.sum(labels == 0)),
                "normalization_rows": component_rows,
                "training_samples": payload["provenance"],
            }
            continue

        final_weights = _balance_class_weights(labels, component_weights, train_mask)
        train_weights = final_weights[train_mask]
        model = XGBClassifier(**XGBOOST_PARAMS)
        eval_set = None
        sample_weight_eval_set = None
        if np.any(valid_mask) and len(np.unique(labels[valid_mask])) == 2:
            eval_set = [(features[valid_mask], labels[valid_mask])]
            sample_weight_eval_set = [final_weights[valid_mask]]
        fit_kwargs: dict[str, Any] = {"sample_weight": train_weights, "verbose": False}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["sample_weight_eval_set"] = sample_weight_eval_set
        model.fit(features[train_mask], labels[train_mask], **fit_kwargs)

        metrics = {}
        for split_name, split_mask in (("train", train_mask), ("validation", valid_mask), ("test", test_mask)):
            if not np.any(split_mask):
                metrics[f"{split_name}_auc"] = None
                continue
            scores = model.predict_proba(features[split_mask])[:, 1]
            metrics[f"{split_name}_auc"] = _auc(labels[split_mask], scores, final_weights[split_mask])

        model_path = classifiers_dir / f"{name}.pkl"
        json_path = classifiers_dir / f"{name}.json"
        manifest_path = classifiers_dir / f"{name}_manifest.json"
        with model_path.open("wb") as handle:
            pickle.dump(model, handle)
        model.save_model(json_path)
        manifest = {
            "classifier": name,
            "backend": BACKEND_NAME,
            "backend_parameters": XGBOOST_PARAMS,
            "features": list(spec.features),
            "feature_manifest_hash": stable_hash({"name": name, "features": spec.features}),
            "training_audit_hash": audit["artifact_hash"],
            "model_path": str(model_path),
            "model_json_path": str(json_path),
            "created_utc": utcnow_iso(),
        }
        write_json(manifest, manifest_path)
        report["classifiers"][name] = {
            "status": "trained",
            "features": list(spec.features),
            "training_rows": int(np.sum(train_mask)),
            "validation_rows": int(np.sum(valid_mask)),
            "test_rows": int(np.sum(test_mask)),
            "signal_rows": int(np.sum(labels == 1)),
            "background_rows": int(np.sum(labels == 0)),
            "metrics": metrics,
            "raw_signal_weight": float(np.sum(np.abs(raw_weights[labels == 1]))),
            "raw_background_weight": float(np.sum(np.abs(raw_weights[labels == 0]))),
            "peak_normalized_signal_weight": float(np.sum(component_weights[labels == 1])),
            "peak_normalized_background_weight": float(np.sum(component_weights[labels == 0])),
            "final_train_signal_weight": float(np.sum(final_weights[train_mask & (labels == 1)])),
            "final_train_background_weight": float(np.sum(final_weights[train_mask & (labels == 0)])),
            "model_path": str(model_path),
            "model_json_path": str(json_path),
            "manifest_path": str(manifest_path),
            "training_samples": payload["provenance"],
            "normalization_rows": component_rows,
            "feature_manifest_hash": manifest["feature_manifest_hash"],
        }

    _write_background_normalization_report(normalization_rows, outputs)
    _write_mass_correlation_report(correlation_rows, outputs)
    write_json(report, outputs / "classifier_training_report.json")
    write_text(render_classifier_training_markdown(report), outputs / "classifier_training_report.md")
    return report


def render_classifier_training_markdown(report: dict[str, Any]) -> str:
    lines = ["# Classifier Training Report", "", f"- Backend: `{report['backend']}`", ""]
    for name, item in report["classifiers"].items():
        if item["status"] != "trained":
            lines.append(f"- `{name}`: {item['status']}")
            continue
        test_auc = item["metrics"].get("test_auc")
        auc_text = "n/a" if test_auc is None else f"{test_auc:.4f}"
        lines.append(
            f"- `{name}`: trained on {item['training_rows']} rows, test AUC `{auc_text}`, features `{', '.join(item['features'])}`"
        )
    return "\n".join(lines) + "\n"


def score_samples(samples: list[dict[str, Any]], report: dict[str, Any]) -> None:
    for name, item in report["classifiers"].items():
        if item.get("status") != "trained":
            continue
        with Path(item["model_path"]).open("rb") as handle:
            model = pickle.load(handle)
        features = item["features"]
        for sample in samples:
            rows = np.column_stack([sample["arrays"][feature] for feature in features])
            finite_mask = np.all(np.isfinite(rows), axis=1)
            scores = np.full(rows.shape[0], np.nan)
            if np.any(finite_mask):
                scores[finite_mask] = model.predict_proba(rows[finite_mask])[:, 1]
            sample["arrays"][name] = scores


def optimize_boundaries(samples: list[dict[str, Any]], outputs: Path) -> dict[str, Any]:
    data_events = [sample for sample in samples if sample["kind"] == "data"]
    signal_events = [sample for sample in samples if sample["analysis_role"] == "signal_nominal"]
    seed = {
        "BDT_ttH": [0.52, 0.79, 0.83, 0.92],
        "BDT_VH": [0.35, 0.78],
        "BDT_VBF_high": [-0.32, 0.47],
        "BDT_VBF_low": [0.26, 0.87],
    }
    candidates = {
        "BDT_ttH": [0.45, 0.52, 0.60, 0.70, 0.79, 0.83, 0.90, 0.92, 0.96],
        "BDT_VH": [0.20, 0.35, 0.50, 0.65, 0.78, 0.88],
        "BDT_VBF_high": [-0.50, -0.32, 0.00, 0.20, 0.47, 0.70],
        "BDT_VBF_low": [0.10, 0.26, 0.45, 0.65, 0.87, 0.95],
    }
    results = []
    width_scale = 10.0 / 45.0

    score_availability = {}
    for score_name in ("BDT_ttH", "BDT_VH", "BDT_VBF"):
        finite_count = 0
        for sample in signal_events + data_events:
            values = sample["arrays"].get(score_name)
            if values is None:
                continue
            finite_count += int(np.sum(np.isfinite(values)))
        score_availability[score_name] = finite_count

    if not data_events or not signal_events:
        return _blocked_boundary_report(outputs, seed, "blocked_missing_samples", "Boundary optimization requires both data sidebands and nominal signal samples.", score_availability)

    if not any(score_availability.values()):
        return _blocked_boundary_report(outputs, seed, "blocked_missing_classifier_scores", "No finite supplemental or official BDT scores were available, so boundary optimization was skipped.", score_availability)

    def evaluate(boundaries: dict[str, list[float]]) -> dict[str, Any]:
        from analysis.section8_ads.categories import assign_categories

        assigned_signal = 0.0
        assigned_background = 0.0
        per_category = {}
        for sample in signal_events:
            categories, _, _ = assign_categories(sample["arrays"], boundaries)
            weights = sample["arrays"]["weight"]
            for category in np.unique(categories):
                if category in {"unassigned", "blocked_missing_input"}:
                    continue
                mask = categories == category
                per_category.setdefault(category, {"signal": 0.0, "background": 0.0})
                per_category[category]["signal"] += float(np.sum(weights[mask]))
        for sample in data_events:
            categories, _, _ = assign_categories(sample["arrays"], boundaries)
            sideband = sample["arrays"]["is_sideband"]
            for category in np.unique(categories):
                if category in {"unassigned", "blocked_missing_input"}:
                    continue
                mask = (categories == category) & sideband
                per_category.setdefault(category, {"signal": 0.0, "background": 0.0})
                per_category[category]["background"] += float(np.sum(mask)) * width_scale
        q0_sum = 0.0
        for payload in per_category.values():
            z = _asimov_significance(payload["signal"], payload["background"])
            q0_sum += z * z
            assigned_signal += payload["signal"]
            assigned_background += payload["background"]
        return {
            "boundaries": boundaries,
            "objective": math.sqrt(max(q0_sum, 0.0)),
            "signal_yield": assigned_signal,
            "background_estimate": assigned_background,
            "per_category": per_category,
        }

    best = evaluate(seed)
    results.append(best)

    def scan_pair(name: str, values: list[float], current: dict[str, list[float]]) -> dict[str, list[float]]:
        local_best = current
        local_score = evaluate(current)["objective"]
        for low in values:
            for high in values:
                if low >= high:
                    continue
                candidate = dict(local_best)
                candidate[name] = [low, high]
                result = evaluate(candidate)
                results.append(result)
                if result["objective"] > local_score:
                    local_best = candidate
                    local_score = result["objective"]
        return local_best

    current = dict(seed)
    if score_availability["BDT_ttH"]:
        local_score = evaluate(current)["objective"]
        for low1 in candidates["BDT_ttH"]:
            for low2 in candidates["BDT_ttH"]:
                for low3 in candidates["BDT_ttH"]:
                    for low4 in candidates["BDT_ttH"]:
                        ordered = sorted({low1, low2, low3, low4})
                        if len(ordered) != 4:
                            continue
                        candidate = dict(current)
                        candidate["BDT_ttH"] = ordered
                        result = evaluate(candidate)
                        results.append(result)
                        if result["objective"] > local_score:
                            current = candidate
                            local_score = result["objective"]
    if score_availability["BDT_VH"]:
        current = scan_pair("BDT_VH", candidates["BDT_VH"], current)
    if score_availability["BDT_VBF"]:
        current = scan_pair("BDT_VBF_high", candidates["BDT_VBF_high"], current)
        current = scan_pair("BDT_VBF_low", candidates["BDT_VBF_low"], current)
    best = evaluate(current)
    results.append(best)

    payload = {
        "status": "ok",
        "seed_boundaries": seed,
        "selected_boundaries": best["boundaries"],
        "best_expected_significance_proxy": best["objective"],
        "scan_result_count": len(results),
        "score_availability": score_availability,
        "objective_definition": "sqrt(sum_c q0_c) with q0_c from a blinded Asimov counting proxy using sideband-data background estimates scaled by signal-window width.",
    }
    write_json(payload, outputs / "bdt_boundary_optimization.json")
    write_text(render_boundary_markdown(payload), outputs / "bdt_boundary_optimization.md")
    return payload


def _blocked_boundary_report(outputs: Path, seed: dict[str, list[float]], status: str, reason: str, score_availability: dict[str, int]) -> dict[str, Any]:
    payload = {
        "status": status,
        "seed_boundaries": seed,
        "selected_boundaries": seed,
        "best_expected_significance_proxy": None,
        "scan_result_count": 0,
        "objective_definition": "sqrt(sum_c q0_c) with q0_c from a blinded Asimov counting proxy using sideband-data background estimates scaled by signal-window width.",
        "blocking_reason": reason,
        "score_availability": score_availability,
    }
    write_json(payload, outputs / "bdt_boundary_optimization.json")
    write_text(render_boundary_markdown(payload), outputs / "bdt_boundary_optimization.md")
    return payload


def _asimov_significance(signal_yield: float, background_yield: float) -> float:
    if signal_yield <= 0.0 or background_yield <= 0.0:
        return 0.0
    q0 = 2.0 * ((signal_yield + background_yield) * math.log(1.0 + signal_yield / background_yield) - signal_yield)
    return math.sqrt(max(q0, 0.0))


def render_boundary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# BDT Boundary Optimization",
        "",
        f"- Status: `{payload['status']}`",
        f"- Scan points evaluated: `{payload['scan_result_count']}`",
        f"- Best expected-significance proxy: `{payload['best_expected_significance_proxy']}`",
        "",
    ]
    if payload.get("blocking_reason"):
        lines.extend([f"- Blocking reason: `{payload['blocking_reason']}`", ""])
    if payload.get("score_availability"):
        lines.extend(["## Score Availability", ""])
        for name, count in payload["score_availability"].items():
            lines.append(f"- `{name}` finite rows: `{count}`")
        lines.append("")
    lines.extend(["## Selected Boundaries", ""])
    for name, values in (payload.get("selected_boundaries") or {}).items():
        lines.append(f"- `{name}`: {values}")
    return "\n".join(lines) + "\n"
