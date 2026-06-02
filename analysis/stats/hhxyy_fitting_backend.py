from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sysconfig
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from analysis.selections.engine import CATEGORY_ORDER as DEFAULT_CATEGORY_ORDER

FIT_ID = "FIT1"
MASS_LOW = 105.0
MASS_HIGH = 160.0
N_BINS = 55
CATEGORY_ORDER = DEFAULT_CATEGORY_ORDER
FIXED_SIGNAL_SHAPE_NP = "sigmaCB_*,alphaCBLo_*,alphaCBHi_*,nCBLo_*,nCBHi_*"

SIGNAL_TEMPLATE = """\
<!DOCTYPE Model SYSTEM 'AnaWSBuilder.dtd'>
<Model Type="UserDef">
  <Item Name="meanCB_:category:[{mean}]"/>
  <Item Name="sigmaCB_:category:[{sigma},0.001,20]"/>
  <Item Name="alphaCBLo_:category:[{alpha_low},0.001,20]"/>
  <Item Name="nCBLo_:category:[{n_low},0.001,200]"/>
  <Item Name="alphaCBHi_:category:[{alpha_high},0.001,20]"/>
  <Item Name="nCBHi_:category:[{n_high},0.001,200]"/>
  <ModelItem Name="RooCrystalBall::sigPdf(:observable:, meanCB_:category:, sigmaCB_:category:, alphaCBLo_:category:, nCBLo_:category:, alphaCBHi_:category:, nCBHi_:category:)"/>
</Model>
"""

BKG_TEMPLATES = {
    "Exponential": """\
<!DOCTYPE Model SYSTEM 'AnaWSBuilder.dtd'>
<Model Type="UserDef">
  <ModelItem Name="EXPR::bkgPdf('exp((@0-125)*@1)',:observable:,p1_:category:[0,-100,100])"/>
</Model>
""",
    "ExpPoly2": """\
<!DOCTYPE Model SYSTEM 'AnaWSBuilder.dtd'>
<Model Type="UserDef">
  <ModelItem Name="EXPR::bkgPdf('exp((@0-125)*@1+(@0-125)*(@0-125)*@2)',:observable:,p1_:category:[0,-100,100],p2_:category:[0,-100,100])"/>
</Model>
""",
    "ExpPoly3": """\
<!DOCTYPE Model SYSTEM 'AnaWSBuilder.dtd'>
<Model Type="UserDef">
  <ModelItem Name="EXPR::bkgPdf('exp((@0-125)*@1+(@0-125)*(@0-125)*@2+(@0-125)*(@0-125)*(@0-125)*@3)',:observable:,p1_:category:[0,-100,100],p2_:category:[0,-100,100],p3_:category:[0,-100,100])"/>
</Model>
""",
    "Pow": """\
<!DOCTYPE Model SYSTEM 'AnaWSBuilder.dtd'>
<Model Type="UserDef">
  <ModelItem Name="EXPR::bkgPdf('pow(@0/125.0, @1)',:observable:,p1_:category:[0,-100,100])"/>
</Model>
""",
    "Bern2": """\
<!DOCTYPE Model SYSTEM 'AnaWSBuilder.dtd'>
<Model Type="UserDef">
  <ModelItem Name="RooBernstein::bkgPdf(:observable:,{p1_:category:[0,-100,100],p2_:category:[0,-100,100],1})"/>
</Model>
""",
    "Bern3": """\
<!DOCTYPE Model SYSTEM 'AnaWSBuilder.dtd'>
<Model Type="UserDef">
  <ModelItem Name="RooBernstein::bkgPdf(:observable:,{p1_:category:[0,-100,100],p2_:category:[0,-100,100],p3_:category:[0,-100,100],1})"/>
</Model>
""",
    "Bern4": """\
<!DOCTYPE Model SYSTEM 'AnaWSBuilder.dtd'>
<Model Type="UserDef">
  <ModelItem Name="RooBernstein::bkgPdf(:observable:,{p1_:category:[0,-100,100],p2_:category:[0,-100,100],p3_:category:[0,-100,100],p4_:category:[0,-100,100],1})"/>
</Model>
""",
    "Bern5": """\
<!DOCTYPE Model SYSTEM 'AnaWSBuilder.dtd'>
<Model Type="UserDef">
  <ModelItem Name="RooBernstein::bkgPdf(:observable:,{p1_:category:[0,-100,100],p2_:category:[0,-100,100],p3_:category:[0,-100,100],p4_:category:[0,-100,100],p5_:category:[0,-100,100],1})"/>
</Model>
""",
}


def _which(name: str) -> str | None:
    return shutil.which(name)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _path_from_env(name: str) -> Path | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.exists() else None


def _hhxyy_reference_root() -> Path | None:
    candidates = [
        _path_from_env("HHXYY_REFERENCE_ROOT"),
        Path.home() / "disk" / "hhxyy",
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def _hhxyy_fitting_root() -> Path | None:
    try:
        import hhxyy_fitting

        return Path(hhxyy_fitting.__file__).resolve().parents[1]
    except Exception:
        candidates = [
            _path_from_env("HHXYY_FITTING_ROOT"),
            Path.home() / "disk" / "hhxyy-codex" / "fitting",
        ]
        for candidate in candidates:
            if candidate and candidate.exists():
                return candidate
    return None


def _xmlreader_path() -> Path | None:
    env_path = _path_from_env("HHXYY_XMLREADER")
    if env_path is not None and os.access(env_path, os.X_OK):
        return env_path
    from_path = _which("XMLReader")
    if from_path:
        return Path(from_path)
    reference_root = _hhxyy_reference_root()
    if reference_root is not None:
        candidate = reference_root / "statistics-xcheck" / "tools" / "xmlAnaWSBuilder-build" / "bin" / "XMLReader"
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _xmlreader_env(xmlreader: Path) -> dict[str, str]:
    env = os.environ.copy()
    xml_build = xmlreader.resolve().parents[1]
    site_prefix = Path(sysconfig.get_paths().get("purelib", "")).resolve()
    root_module = None
    try:
        import ROOT  # noqa: F401

        import ROOT as root_module_import

        root_module = Path(root_module_import.__file__).resolve().parent
    except Exception:
        pass

    library_paths = [xml_build / "lib"]
    if root_module is not None:
        library_paths.append(root_module / "lib")
    if site_prefix.exists():
        library_paths.append(site_prefix / "root.libs")
        local_site = Path.home() / ".local" / "lib" / f"python{site_prefix.parent.name.removeprefix('python')}" / "site-packages"
        if local_site.exists() and local_site != site_prefix:
            library_paths.append(local_site / "ROOT" / "lib")
            library_paths.append(local_site / "root.libs")
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(str(path) for path in library_paths if path.exists()) + (
        f":{existing}" if existing else ""
    )
    return env


def _quickfit_setup_candidates() -> list[Path]:
    candidates = [
        _path_from_env("HHXYY_QUICKFIT_SETUP"),
        _path_from_env("QUICKFIT_XCHECK_SETUP"),
        _project_root() / ".cache" / "hhxyy_tools" / "quickfit-xcheck" / "setup.sh",
    ]
    reference_root = _hhxyy_reference_root()
    if reference_root is not None:
        candidates.append(reference_root / "statistics-xcheck" / "tools" / "quickfit-xcheck" / "setup.sh")
    return [path for path in candidates if path is not None and path.exists()]


def _shell_command_with_setup(setup: Path, command: list[str]) -> list[str]:
    quoted_setup = shlex.quote(str(setup.resolve()))
    quoted_command = shlex.join([str(part) for part in command])
    return ["bash", "-lc", f"source {quoted_setup} >/dev/null 2>&1 && exec {quoted_command}"]


def _command_succeeds(command: list[str]) -> bool:
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0


def _quickfit_command(command: list[str]) -> list[str]:
    for setup in _quickfit_setup_candidates():
        probe = _shell_command_with_setup(setup, ["quickFit", "--help"])
        if _command_succeeds(probe):
            return _shell_command_with_setup(setup, command)
    if _which("quickFit"):
        return command
    return command


def _quickfit_is_runnable() -> bool:
    return _command_succeeds(_quickfit_command(["quickFit", "--help"]))


def is_atlas_env_available(*, require_quicklimit: bool = False) -> bool:
    if _xmlreader_path() is None:
        return False
    if not _quickfit_is_runnable():
        return False
    if require_quicklimit:
        return bool(_which("quickLimit"))
    return True


def _active_categories(category_context: dict[str, Any], category_order: list[str] | None = None) -> list[str]:
    order = category_order or CATEGORY_ORDER
    active = [category for category in order if category in category_context]
    extras = [category for category in category_context if category not in active]
    return active + sorted(extras)


def _as_counts(value: Any, *, default: float = 0.0) -> np.ndarray:
    if value is None:
        return np.full(N_BINS, default, dtype=float)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if len(arr) == N_BINS:
        return np.clip(arr, 0.0, None)
    out = np.full(N_BINS, default, dtype=float)
    out[: min(len(arr), N_BINS)] = np.clip(arr[:N_BINS], 0.0, None)
    return out


def export_histograms_to_root(
    category_context: dict[str, Any],
    output_dir: Path,
    *,
    category_order: list[str] | None = None,
) -> dict[str, Path]:
    import ROOT

    output_dir.mkdir(parents=True, exist_ok=True)
    active = _active_categories(category_context, category_order)

    signal_path = output_dir / "signal.root"
    yyjets_path = output_dir / "yyjets_myy_categories.root"
    data_path = output_dir / "data_myy_categories.root"

    def _write_hist(root_file, name: str, counts: np.ndarray, *, quantize_for_xmlreader: bool = False) -> None:
        hist = ROOT.TH1D(name, name, N_BINS, MASS_LOW, MASS_HIGH)
        hist.Sumw2(False)
        values = np.asarray(counts, dtype=float)
        if quantize_for_xmlreader:
            # XMLReader compares TH1::Integral() and RooDataSet::sumEntries()
            # with DBL_EPSILON tolerance, so use binary-exact weights for Data.
            values = np.rint(values * 1024.0) / 1024.0
        for bin_index, value in enumerate(values, start=1):
            value = float(max(value, 0.0))
            hist.SetBinContent(bin_index, value)
            hist.SetBinError(bin_index, math.sqrt(value))
        hist.SetEntries(float(hist.Integral()))
        root_file.cd()
        hist.Write(name)

    inclusive = np.zeros(N_BINS, dtype=float)
    background_hists: dict[str, np.ndarray] = {}
    for idx, category in enumerate(active):
        counts = _as_counts(category_context[category].get("selection_counts"))
        background_hists[f"category{idx}"] = counts
        inclusive += counts
    background_hists["inclusive"] = inclusive

    signal_file = ROOT.TFile(str(signal_path), "RECREATE")
    try:
        for idx, category in enumerate(active):
            ctx = category_context[category]
            counts = _as_counts(ctx.get("signal_counts"))
            total = float(np.sum(counts))
            target = float(ctx.get("expected_signal_yield", total))
            if total > 0.0 and target > 0.0:
                counts = counts * target / total
            _write_hist(signal_file, f"signal_{idx}", counts)
    finally:
        signal_file.Close()

    for path in (yyjets_path, data_path):
        root_file = ROOT.TFile(str(path), "RECREATE")
        try:
            _write_hist(root_file, "inclusive", background_hists["inclusive"], quantize_for_xmlreader=True)
            for idx, _category in enumerate(active):
                _write_hist(root_file, f"category{idx}", background_hists[f"category{idx}"], quantize_for_xmlreader=True)
        finally:
            root_file.Close()

    return {"signal": signal_path, "yyjets": yyjets_path, "data": data_path}


def _signal_param(params: dict[str, Any], category: str, stem: str, fallback: float) -> float:
    prefixes = {
        "mean": ["mean"],
        "sigma_l": ["sigmaL", "sigma"],
        "sigma_r": ["sigmaR", "sigma"],
        "alpha_l": ["alphaL", "alpha_low"],
        "alpha_r": ["alphaR", "alpha_high"],
        "n_l": ["nL", "n_low"],
        "n_r": ["nR", "n_high"],
    }
    for prefix in prefixes[stem]:
        for key, payload in params.items():
            if key.startswith(prefix) and key.endswith(category):
                if isinstance(payload, dict):
                    return float(payload.get("value", fallback))
                return float(payload)
    return float(fallback)


def _write_signal_parameter_files(category_context: dict[str, Any], parameters_dir: Path, active: list[str]) -> None:
    parameters_dir.mkdir(parents=True, exist_ok=True)
    for idx, category in enumerate(active):
        params = category_context[category]["signal_artifact"]["parameters"]
        sigma_l = _signal_param(params, category, "sigma_l", 1.6)
        sigma_r = _signal_param(params, category, "sigma_r", sigma_l)
        values = {
            "mean": _signal_param(params, category, "mean", 125.0),
            "sigma": 0.5 * (sigma_l + sigma_r),
            "alpha_low": _signal_param(params, category, "alpha_l", 1.5),
            "alpha_high": _signal_param(params, category, "alpha_r", 1.5),
            "n_low": _signal_param(params, category, "n_l", 5.0),
            "n_high": _signal_param(params, category, "n_r", 5.0),
        }
        lines = [f"{key}: {value:.12g}" for key, value in values.items()]
        (parameters_dir / f"signal_cat{idx}.txt").write_text("\n".join(lines) + "\n")


def _external_background_model(local_model: str) -> str:
    mapping = {
        "exponential": "Exponential",
        "exp_poly2": "ExpPoly2",
        "exp_poly3": "ExpPoly3",
        "power": "Pow",
        "pow": "Pow",
        "bernstein2": "Bern2",
        "bernstein3": "Bern3",
        "bernstein4": "Bern4",
    }
    return mapping.get(local_model, local_model)


def _write_background_result_files(category_context: dict[str, Any], bkg_dir: Path, active: list[str]) -> None:
    bkg_dir.mkdir(parents=True, exist_ok=True)
    for idx, category in enumerate(active):
        ctx = category_context[category]
        choice = ctx.get("background_choice", {})
        model = _external_background_model(str(choice.get("selected_model", "exponential")))
        n_spur = abs(float(choice.get("selected_n_spur", 0.0)))
        if not n_spur:
            n_spur = abs(float((ctx.get("spurious_signal") or {}).get("N_spur", 0.0)))
        cat_dir = bkg_dir / f"cat{idx}"
        cat_dir.mkdir(parents=True, exist_ok=True)
        # The reference parser reads token 0 as the selected model and token 5 as |max(S)|.
        (cat_dir / "results.txt").write_text(
            "Name max(S/dS) max(1sigma/dS) max(2sigma/dS) max(S/Sref) max(S) S[125] Sref[125] dS[125]\n"
            f"{model} 0 0 0 0 {n_spur:.12g} {n_spur:.12g} {float(ctx.get('expected_signal_yield', 0.0)):.12g} 1 <== Selected\n"
        )


def _write_category_yaml(path: Path, active: list[str]) -> None:
    payload = [{"name": category, "selection": "1", "fit": True} for category in active]
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_category_yield_yaml(path: Path, category_context: dict[str, Any], active: list[str]) -> None:
    signal = {
        f"category_{idx}": float(category_context[category].get("expected_signal_yield", 0.0))
        for idx, category in enumerate(active)
    }
    path.write_text(yaml.safe_dump({"signal": signal, "ggF": signal}, sort_keys=False))


def generate_fit_config(
    workspace_root: Path,
    category_yaml: Path,
    hist_dir: Path,
    summary: dict[str, Any],
    active: list[str],
) -> Path:
    cfg = summary["runtime_defaults"]
    blinding = cfg["blinding"]
    signal_window = cfg["signal_window_gev"]
    fit_range = cfg["fit_mass_range_gev"]
    config = {
        "histogramming": {
            "category_yaml": str(category_yaml.resolve()),
            "category_names": active,
            "myy_min": float(fit_range[0]),
            "myy_max": float(fit_range[1]),
            "myy_bins": N_BINS,
            "blinding_low": float(signal_window[0]),
            "blinding_high": float(signal_window[1]),
        },
        "fit": {
            "input_dir": str(hist_dir.resolve()),
            "category_yaml": str(category_yaml.resolve()),
            "n_categories": len(active),
            "sideband_only": not bool(blinding.get("observed_significance_allowed", False)),
            "blinding_low": float(signal_window[0]),
            "blinding_high": float(signal_window[1]),
            "hist_xmin": float(fit_range[0]),
            "hist_xmax": float(fit_range[1]),
            "myy_min": float(fit_range[0]),
            "myy_max": float(fit_range[1]),
            "ws_output_file": "ws_combined.root",
            "run_quicklimit": False,
            "spurious_signal": True,
            "bkg_fallback_model": "Exponential",
            "bkg_models": ["Exponential", "ExpPoly2", "ExpPoly3", "Pow", "Bern2", "Bern3", "Bern4", "Bern5"],
            "poi_name": "mu",
            "mu_range": [0.0, 5.0],
        },
    }
    path = workspace_root / "fit_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _toolkit_data_path(name: str) -> Path | None:
    root = _hhxyy_fitting_root()
    if root is None:
        return None
    path = root / "data" / name
    return path if path.exists() else None


def _write_model_xmls(model_dir: Path, parameters_dir: Path, active: list[str]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    for idx, _category in enumerate(active):
        raw = {}
        param_path = parameters_dir / f"signal_cat{idx}.txt"
        for line in param_path.read_text().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                raw[key.strip()] = float(value)
        (model_dir / f"signal_cat{idx}.xml").write_text(SIGNAL_TEMPLATE.format(**raw))
    for model_name, xml in BKG_TEMPLATES.items():
        (model_dir / f"background_{model_name}.xml").write_text(xml)


def _category_xml(
    *,
    idx: int,
    category: str,
    ctx: dict[str, Any],
    data_path: Path,
    signal_window: list[float],
    fit_range: list[float],
) -> str:
    choice = ctx.get("background_choice", {})
    model = _external_background_model(str(choice.get("selected_model", "exponential")))
    bkg_seed = max(float(ctx.get("template_total_yield", choice.get("observed_data_sideband_count", ctx.get("observed_count", 1.0)))), 1.0)
    signal_yield = float(ctx.get("expected_signal_yield", 0.0))
    return f"""<!DOCTYPE Channel SYSTEM 'AnaWSBuilder.dtd'>
<Channel Name="category_{idx}" Type="shape" Lumi="1">
  <Data InputFile="{data_path.resolve()}" FileType="histogram" HistName="category{idx}" Observable="atlas_invMass_:category:[{fit_range[0]}, {fit_range[1]}]" Binning="{N_BINS}" />
  <Sample Name="signal" InputFile="model/signal_cat{idx}.xml">
    <NormFactor Name="yield_signal_:category:[{signal_yield:.12g}]" />
    <NormFactor Name="mu[1,0,5]" />
  </Sample>
  <Sample Name="background" InputFile="model/background_{model}.xml" MultiplyLumi="false">
    <NormFactor Name="nbkg_:category:[{bkg_seed:.12g},0,{10.0 * bkg_seed:.12g}]" />
  </Sample>
</Channel>
"""


def _write_workspace_xmls(
    workspace_dir: Path,
    category_context: dict[str, Any],
    hist_paths: dict[str, Path],
    summary: dict[str, Any],
    active: list[str],
) -> None:
    model_dir = workspace_dir / "model"
    parameters_dir = workspace_dir.parent / "parameters"
    _write_model_xmls(model_dir, parameters_dir, active)

    cfg = summary["runtime_defaults"]
    for idx, category in enumerate(active):
        xml = _category_xml(
            idx=idx,
            category=category,
            ctx=category_context[category],
            data_path=hist_paths["data"],
            signal_window=cfg["signal_window_gev"],
            fit_range=cfg["fit_mass_range_gev"],
        )
        (workspace_dir / f"category_{idx}.xml").write_text(xml)

    inputs = "\n".join(f"  <Input>category_{idx}.xml</Input>" for idx in range(len(active)))
    combination = f"""<!DOCTYPE Combination SYSTEM 'AnaWSBuilder.dtd'>
<Combination Blind="false" DataName="obsData" ModelConfigName="ModelConfig" OutputFile="ws_combined.root" WorkspaceName="combined" Integrator="RooAdaptiveGaussKronrodIntegrator1D">
{inputs}
  <POI>mu</POI>
  <Asimov Setup="mu=0" Action="fixsyst:fit:genasimov:float:savesnapshot" Name="asimovData_0" SnapshotGlob="nominalGlobs" SnapshotNuis="nominalNuis"/>
  <Asimov Setup="mu=1" Action="genasimov" Name="asimovData_1"/>
</Combination>
"""
    (workspace_dir / "combination.xml").write_text(combination)

    dtd = _toolkit_data_path("AnaWSBuilder.dtd")
    if dtd is not None:
        shutil.copyfile(dtd, workspace_dir / "AnaWSBuilder.dtd")
        shutil.copyfile(dtd, model_dir / "AnaWSBuilder.dtd")


def prepare_hhxyy_workspace(
    category_context: dict[str, Any],
    fit_dir: Path,
    summary: dict[str, Any],
    *,
    category_order: list[str] | None = None,
) -> dict[str, Any]:
    active = _active_categories(category_context, category_order)
    workspace_root = Path(fit_dir) / "hhxyy_workspace"
    hist_dir = workspace_root / "hists"
    fitting_dir = workspace_root / "fitting"
    parameters_dir = fitting_dir / "parameters"
    bkg_dir = workspace_root / "bkg_model"
    workspace_dir = fitting_dir / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    hist_paths = export_histograms_to_root(category_context, hist_dir, category_order=active)
    _write_signal_parameter_files(category_context, parameters_dir, active)
    _write_background_result_files(category_context, bkg_dir, active)
    category_yaml = workspace_root / "category_config.yaml"
    _write_category_yaml(category_yaml, active)
    _write_category_yield_yaml(hist_dir / "category_yield.yaml", category_context, active)
    config_path = generate_fit_config(workspace_root, category_yaml, hist_dir, summary, active)
    _write_workspace_xmls(workspace_dir, category_context, hist_paths, summary, active)

    manifest = {
        "status": "ok",
        "implementation": "hhxyy_fitting_equivalent",
        "reference": str(_hhxyy_fitting_root()) if _hhxyy_fitting_root() else None,
        "hhxyy_reference_root": str(_hhxyy_reference_root()) if _hhxyy_reference_root() else None,
        "categories": active,
        "workspace_root": str(workspace_root),
        "histograms": {key: str(value) for key, value in hist_paths.items()},
        "category_yaml": str(category_yaml),
        "config": str(config_path),
        "workspace_dir": str(workspace_dir),
        "combination_xml": str(workspace_dir / "combination.xml"),
        "signal_parameter_dir": str(parameters_dir),
        "background_model_dir": str(bkg_dir),
    }
    (workspace_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _run_xmlreader(workspace_dir: Path, log_path: Path) -> Path:
    xmlreader = _xmlreader_path()
    if xmlreader is None:
        raise RuntimeError("XMLReader is not available; set HHXYY_XMLREADER or install the HHXYY xmlAnaWSBuilder tool")
    cmd = [str(xmlreader), "-x", "combination.xml"]
    env = _xmlreader_env(xmlreader)
    with log_path.open("w") as log:
        log.write(f"$ {shlex.join(cmd)}\n")
        result = subprocess.run(cmd, cwd=workspace_dir, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"XMLReader failed with return code {result.returncode}; see {log_path}")
    workspace = workspace_dir / "ws_combined.root"
    if not workspace.exists():
        raise RuntimeError(f"XMLReader did not produce {workspace}")
    return workspace


def _run_quickfit(
    workspace: Path,
    fit_result_dir: Path,
    *,
    dataset: str,
    poi_setup: str,
    output_name: str,
) -> Path:
    fit_result_dir.mkdir(parents=True, exist_ok=True)
    output = fit_result_dir / output_name
    log_path = fit_result_dir / f"{Path(output_name).stem}.log"
    cmd = [
        "quickFit",
        "-f",
        str(workspace.resolve()),
        "-d",
        dataset,
        "-w",
        "combined",
        "-m",
        "ModelConfig",
        "-n",
        FIXED_SIGNAL_SHAPE_NP,
        "-p",
        poi_setup,
        "-o",
        str(output.resolve()),
        "--savefitresult",
        "1",
        "--saveWS",
        "0",
        "--saveNP",
        "1",
        "--printLevel",
        "-1",
    ]
    runnable_cmd = _quickfit_command(cmd)
    with log_path.open("w") as log:
        log.write(f"$ {shlex.join(runnable_cmd)}\n")
        result = subprocess.run(runnable_cmd, cwd=fit_result_dir, stdout=log, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0 or not output.exists():
        raise RuntimeError(f"quickFit failed for {dataset} ({poi_setup}); see {log_path}")
    return output


def _format_number(value: float) -> str:
    return f"{value:.15g}"


def _workspace_poi_setup(workspace_path: Path, poi_name: str = "mu") -> str:
    import ROOT

    root_file = ROOT.TFile.Open(str(workspace_path))
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open workspace {workspace_path}")
    try:
        workspace = root_file.Get("combined")
        if not workspace:
            raise RuntimeError(f"Workspace 'combined' not found in {workspace_path}")
        poi = workspace.var(poi_name)
        if not poi:
            raise RuntimeError(f"POI '{poi_name}' not found in workspace {workspace_path}")
        return (
            f"{poi_name}="
            f"{_format_number(float(poi.getVal()))}_"
            f"{_format_number(float(poi.getMin()))}_"
            f"{_format_number(float(poi.getMax()))}"
        )
    finally:
        root_file.Close()


def _read_quickfit_tree(path: Path, poi_name: str = "mu") -> dict[str, Any]:
    import ROOT

    root_file = ROOT.TFile.Open(str(path))
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open quickFit output {path}")
    try:
        tree = root_file.Get("nllscan")
        if not tree or tree.GetEntries() < 1:
            raise RuntimeError(f"quickFit output {path} has no nllscan entry")
        tree.GetEntry(0)
        result = {
            "status": int(getattr(tree, "status")),
            "nll": float(getattr(tree, "nll")),
            poi_name: float(getattr(tree, poi_name)),
        }
        err_hi_name = f"{poi_name}__up"
        err_lo_name = f"{poi_name}__down"
        if hasattr(tree, err_hi_name) and hasattr(tree, err_lo_name):
            result[f"{poi_name}_err_hi"] = float(getattr(tree, err_hi_name))
            result[f"{poi_name}_err_lo"] = float(getattr(tree, err_lo_name))
        fit_result = root_file.Get("fitResult")
        if fit_result:
            result["fit_result_status"] = int(fit_result.status())
            result["fit_result_covqual"] = int(fit_result.covQual())
            poi_var = fit_result.floatParsFinal().find(poi_name)
            if poi_var:
                result[f"{poi_name}_err"] = float(poi_var.getError())
                result[f"{poi_name}_err_hi"] = float(poi_var.getErrorHi())
                result[f"{poi_name}_err_lo"] = float(poi_var.getErrorLo())
        return result
    finally:
        root_file.Close()


def _format_significance_artifact(
    *,
    free_fit: dict[str, Any],
    mu0_fit: dict[str, Any],
    active: list[str],
    fit_range: list[float],
    manifest: dict[str, Any],
    workspace: Path,
    free_output: Path,
    mu0_output: Path,
) -> dict[str, Any]:
    q0 = max(2.0 * (float(mu0_fit["nll"]) - float(free_fit["nll"])), 0.0)
    mu_hat = float(free_fit["mu"])
    err_hi = free_fit.get("mu_err_hi")
    err_lo = free_fit.get("mu_err_lo")
    if err_hi is not None or err_lo is not None:
        mu_uncertainty = max(abs(float(err_hi or 0.0)), abs(float(err_lo or 0.0)))
    else:
        mu_uncertainty = float("nan")
    fisher = 1.0 / (mu_uncertainty * mu_uncertainty) if math.isfinite(mu_uncertainty) and mu_uncertainty > 0.0 else None
    diagnostics = []
    if int(free_fit["status"]) != 0:
        diagnostics.append("quickFit free-mu fit returned non-zero status.")
    if int(mu0_fit["status"]) != 0:
        diagnostics.append("quickFit fixed-mu=0 fit returned non-zero status.")
    if not math.isfinite(mu_uncertainty) or mu_uncertainty <= 0.0:
        diagnostics.append("quickFit output did not provide a finite positive mu uncertainty.")
    return {
        "fit_id": FIT_ID,
        "status": "ok" if not diagnostics else "warning",
        "dataset_type": "asimov",
        "generation_hypothesis": "signal_plus_background",
        "mu_gen": 1.0,
        "backend": "pyroot_roofit",
        "fit_driver": "hhxyy_fitting_quickfit",
        "poi_name": "signal_strength_mu",
        "mu_hat": mu_hat,
        "mu_uncertainty": mu_uncertainty,
        "twice_nll_mu0": 2.0 * float(mu0_fit["nll"]),
        "twice_nll_free": 2.0 * float(free_fit["nll"]),
        "q0": q0,
        "z_discovery": math.sqrt(q0),
        "fit_range": fit_range,
        "fisher_information_mu": fisher,
        "categories": active,
        "shared_mu": True,
        "background_parameter_policy": "floating_shape_and_normalization",
        "asimov_profile_method": "hhxyy_fitting_XMLReader_quickFit",
        "fixed_np_patterns": FIXED_SIGNAL_SHAPE_NP,
        "quickfit_outputs": {
            "workspace": str(workspace),
            "free_mu": str(free_output),
            "mu0": str(mu0_output),
        },
        "hhxyy_workspace_manifest": manifest,
        "diagnostics": diagnostics,
        "_construction": {
            "fit_id": FIT_ID,
            "status": "ok",
            "dataset_type": "asimov",
            "generation_range": fit_range,
            "construction_mode": "hhxyy_fitting_XMLReader_quickFit",
            "binning": {"observable": "m_gg", "n_bins": N_BINS, "range": fit_range},
            "workspace": str(workspace),
            "manifest": str(Path(manifest["workspace_root"]) / "manifest.json"),
            "datasets": ["asimovData_0", "asimovData_1"],
            "free_fit_parameters": [
                "mu",
                "nbkg_*",
                "background shape parameters",
            ],
            "fixed_hypothesis_parameters": {"mu0_fit": {"mu": 0.0}},
        },
    }


def run_hhxyy_significance(
    category_context: dict[str, Any],
    fit_dir: Path,
    summary: dict[str, Any],
    *,
    category_order: list[str] | None = None,
) -> dict[str, Any]:
    if not is_atlas_env_available():
        raise RuntimeError(
            "HHXYY fitting executables are not available: expected XMLReader plus a runnable quickFit sandbox "
            "(run scripts/bootstrap_hhxyy_quickfit.sh or set HHXYY_QUICKFIT_SETUP)"
        )

    manifest = prepare_hhxyy_workspace(category_context, fit_dir, summary, category_order=category_order)
    workspace_dir = Path(manifest["workspace_dir"])
    log_dir = Path(manifest["workspace_root"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    workspace = _run_xmlreader(workspace_dir, log_dir / "XMLReader.log")
    free_poi_setup = _workspace_poi_setup(workspace, "mu")

    fit_result_dir = Path(manifest["workspace_root"]) / "fitting" / "fit"
    free_output = _run_quickfit(
        workspace,
        fit_result_dir,
        dataset="asimovData_1",
        poi_setup=free_poi_setup,
        output_name="bestfit_mu_asimovData_1.root",
    )
    mu0_output = _run_quickfit(
        workspace,
        fit_result_dir,
        dataset="asimovData_1",
        poi_setup="mu=0",
        output_name="bestfit_mu0_asimovData_1.root",
    )
    free_fit = _read_quickfit_tree(free_output, "mu")
    mu0_fit = _read_quickfit_tree(mu0_output, "mu")
    return _format_significance_artifact(
        free_fit=free_fit,
        mu0_fit=mu0_fit,
        active=manifest["categories"],
        fit_range=summary["runtime_defaults"]["fit_mass_range_gev"],
        manifest=manifest,
        workspace=workspace,
        free_output=free_output,
        mu0_output=mu0_output,
    )
