"""Frugal Phase 2 inference entrypoint.

Loads the artifact bundle produced by the training entrypoint and writes the forecast
submission CSV. No training, notebooks, PowerShell, or previous submissions
are used during inference.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
):
    os.environ[_thread_env] = "1"

import joblib
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_KIT_NAME = "Hackathon-Sea-Winds-Predictions-phase2"
FOOTPRINT_ROWS = 43_715
HOURS = (0, 6, 12, 18)
LEADS = (1, 7, 14)
SOURCE_LEAD = {1: 1, 7: 7, 14: 10}
QUANTILES = (0.05, 0.5, 0.95)
AUXILIARY_OUTPUT_FILES = (
    "siting_submission.json",
    "competition_evidence.json",
    "methodology_economics_compute.md",
)
_HRES_CACHE = None
_ANALYSIS_CACHE = {}
_CONTEXT_PARQUET_CACHE = {}
ANALYSIS_BLEND = 0.30
CONTEXT_BLEND = 0.25
CONTEXT_LAGS = (0, 1, 2, 3, 7, 13)
CONTEXT_SELECTED_SLOTS = ((5, 20, 18), (9, 23, 12))
D7_SPEED_CONTEXT_LAGS = (0, 1, 3, 7, 13)
D1_DENSE_DAILY_LAGS = (0, 1, 3, 7, 13)
D1_DENSE_DAILY_MEANS = (3, 7, 14)
FINE_D7_NEIGHBOR_COUNTS = (8, 32)
FINE_D7_CONTEXT_LAGS = (0, 1, 3, 7, 13)
FINE_D7_CONTEXT_MEAN_DAYS = (3, 7, 14)
FINE_D7_SPATIAL_FEATURES = ["d7_u", "d7_v", "d1_u", "d1_v"]
for _fine_count in FINE_D7_NEIGHBOR_COUNTS:
    _fine_suffix = f"k{_fine_count}"
    FINE_D7_SPATIAL_FEATURES.extend(
        [
            f"d7_u_mean_{_fine_suffix}",
            f"d7_v_mean_{_fine_suffix}",
            f"d7_speed_std_{_fine_suffix}",
            f"d7_u_highpass_{_fine_suffix}",
            f"d7_v_highpass_{_fine_suffix}",
            f"d7_local_cross_{_fine_suffix}",
            f"d7_local_dot_{_fine_suffix}",
            f"delta_u_mean_{_fine_suffix}",
            f"delta_v_mean_{_fine_suffix}",
            f"delta_u_highpass_{_fine_suffix}",
            f"delta_v_highpass_{_fine_suffix}",
        ]
    )
FINE_D7_CONTEXT_FEATURES = tuple(
    [
        name
        for lag in FINE_D7_CONTEXT_LAGS
        for name in (f"ctx_lag{lag}_u", f"ctx_lag{lag}_v")
    ]
    + [
        name
        for days in FINE_D7_CONTEXT_MEAN_DAYS
        for name in (
            f"ctx_mean{days}_u",
            f"ctx_mean{days}_v",
            f"ctx_concentration{days}",
        )
    ]
)
D7_CONDITIONAL_ENDPOINT_ISSUE_SLOTS = (
    (1, 14),
    (2, 25),
    (4, 8),
    (5, 20),
    (7, 1),
    (8, 12),
    (9, 23),
    (11, 4),
)
PROTECTED_SPEED_INFLATION = {1: 1.25}
D14_DIRECTION_POLICY = {
    (4, 8, 0): ("direct", 0.50),
    (4, 8, 6): ("vector", 1.00),
    (4, 8, 12): ("direct", 0.50),
    (7, 1, 18): ("direct", 0.30),
}
D14_SPEED_ENDPOINT_GUARDS = {}
FINAL_EVALUATION_YEAR = 2022


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frugal Phase 2 inference.")
    parser.add_argument(
        "--kit-dir",
        type=Path,
        default=None,
        help="Path to the cloned official Phase 2 kit. Auto-detected by default.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Unpacked Phase 2 data root. Sets PHASE2_DATA_ROOT before importing the kit.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=SCRIPT_DIR / "artifacts",
        help="Directory containing phase2_forecast_artifacts.joblib.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "predictions.csv",
        help="Output path for the required forecast predictions.csv.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Submission ZIP path. Defaults to the output path with .zip suffix.",
    )
    parser.add_argument(
        "--eval-year",
        type=int,
        default=FINAL_EVALUATION_YEAR,
        help="Evaluation year; defaults to the definitive competition year.",
    )
    parser.add_argument(
        "--window-base",
        type=int,
        default=0,
        choices=(0, 1),
        help="Window ids in the output. The official notebook uses 0; use 1 only if required.",
    )
    parser.add_argument(
        "--speed-width-scale",
        default="1:0.75",
        help=(
            "Optional comma-separated lead:scale map applied around q50 after "
            "assembly. The default sharpens only d1, whose public interval score "
            "regressed while the new d7/d14 centers improved materially."
        ),
    )
    parser.add_argument(
        "--dir-halfwidth-scale",
        default="",
        help=(
            "Optional comma-separated lead:scale map for circular direction "
            "half-widths around dir_50, e.g. '7:0.97,14:0.955'."
        ),
    )
    parser.add_argument(
        "--dir-halfwidth-deg",
        default="",
        help=(
            "Optional comma-separated lead:degrees map overriding direction "
            "half-widths around dir_50. Takes precedence over "
            "--dir-halfwidth-scale and organizer-trained artifacts for listed "
            "leads. By default d1, d7, and d14 use strict-gated artifact "
            "policies; d7 falls back to its protected 138-degree constant."
        ),
    )
    parser.add_argument(
        "--d7-center-policy-max-weight",
        type=float,
        default=0.4,
        help=(
            "Maximum permitted calendar-specific d7 center-transfer weight. "
            "The default retains the two conservative blends that survived "
            "both chronological validation and definitive hidden scoring."
        ),
    )
    parser.add_argument(
        "--d1-context-blend-scale",
        type=float,
        default=1.25,
        help="Multiplier for the strictly gated February d1 upper-endpoint blend.",
    )
    parser.add_argument(
        "--dir-halfwidth-cap-deg",
        default="14:145",
        help=(
            "Optional lead:degrees caps applied after all trained direction "
            "interval policies; already-narrow intervals remain unchanged."
        ),
    )
    parser.add_argument(
        "--single-process",
        action="store_true",
        help="Process all windows in one Python process instead of spawning one worker per window.",
    )
    parser.add_argument(
        "--worker-retries",
        type=int,
        default=3,
        help="Retries per window when using default per-window subprocess inference.",
    )
    parser.add_argument("--worker-window-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--append-output", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def resolve_kit_root(kit_dir: Path | None) -> Path:
    candidates: list[Path] = []
    if kit_dir is not None:
        candidates.append(kit_dir)
    candidates.extend(
        [
            Path.cwd(),
            Path.cwd() / DEFAULT_KIT_NAME,
            Path.cwd() / "phase2_workspace" / DEFAULT_KIT_NAME,
            SCRIPT_DIR,
            SCRIPT_DIR / DEFAULT_KIT_NAME,
            SCRIPT_DIR.parent,
            SCRIPT_DIR.parent / DEFAULT_KIT_NAME,
            SCRIPT_DIR.parent / "phase2_workspace" / DEFAULT_KIT_NAME,
            SCRIPT_DIR.parent / DEFAULT_KIT_NAME / "phase_2",
        ]
    )

    for candidate in candidates:
        p = candidate.expanduser().resolve()
        if (p / "phase_2" / "part1_forecast" / "forecast_pipeline.py").exists():
            return p
        if (p / "part1_forecast" / "forecast_pipeline.py").exists():
            return p.parent
    raise FileNotFoundError(
        "Could not find the official Phase 2 kit. Pass --kit-dir pointing at "
        "Hackathon-Sea-Winds-Predictions-phase2."
    )


def add_kit_paths(kit_root: Path) -> Path:
    phase2 = kit_root / "phase_2"
    for sub in (
        phase2,
        phase2 / "part0_dataset_setup",
        phase2 / "part1_forecast",
        phase2 / "part2_siting",
    ):
        sys.path.insert(0, str(sub))
    return phase2


def configure_data_root(data_root: Path | None) -> None:
    if data_root is not None:
        os.environ["PHASE2_DATA_ROOT"] = str(data_root.expanduser().resolve())
        return
    if os.environ.get("PHASE2_DATA_ROOT"):
        return

    candidates = [
        SCRIPT_DIR / "data",
        SCRIPT_DIR / "data" / "phase2_dataset_ship",
        SCRIPT_DIR / "data" / "unpacked" / "phase2_dataset_ship",
        SCRIPT_DIR.parent / "data",
        SCRIPT_DIR.parent / "data" / "phase2_dataset_ship",
        SCRIPT_DIR.parent / "data" / "unpacked" / "phase2_dataset_ship",
        SCRIPT_DIR.parent / "phase2_workspace" / "data" / "unpacked" / "phase2_dataset_ship",
        Path.cwd() / "data",
        Path.cwd() / "data" / "phase2_dataset_ship",
        Path.cwd() / "data" / "unpacked" / "phase2_dataset_ship",
        Path.cwd() / "phase2_workspace" / "data" / "unpacked" / "phase2_dataset_ship",
    ]
    for candidate in candidates:
        if candidate.exists():
            os.environ["PHASE2_DATA_ROOT"] = str(candidate.resolve())
            return


def inspect_final_inference_metadata(
    inference_root: Path, requested_year: int | None
) -> dict:
    """Validate and fingerprint the eight definitive organiser windows."""
    inference_root = Path(inference_root).expanduser().resolve()
    metadata_paths = sorted(inference_root.glob("window_*/metadata.json"))
    if len(metadata_paths) != 8:
        raise RuntimeError(
            f"Final evaluation requires exactly 8 metadata files, found "
            f"{len(metadata_paths)} under {inference_root}"
        )
    years = set()
    digest = hashlib.sha256()
    files = []
    for path in metadata_paths:
        raw = path.read_bytes()
        metadata = json.loads(raw.decode("utf-8"))
        relative = path.relative_to(inference_root).as_posix()
        file_sha256 = hashlib.sha256(raw).hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(raw)
        files.append({"path": relative, "sha256": file_sha256})
        for key in ("context_start", "context_end", "predict_start", "predict_end"):
            years.add(pd.Timestamp(metadata[key]).year)
        years.update(pd.Timestamp(value).year for value in metadata["score_days"].values())
    if years != {FINAL_EVALUATION_YEAR}:
        raise RuntimeError(
            "Refusing non-final inference metadata: expected only 2022 dates, "
            f"found years={sorted(years)} under {inference_root}"
        )
    if requested_year is not None and requested_year != FINAL_EVALUATION_YEAR:
        raise ValueError(
            f"--eval-year={requested_year} conflicts with final metadata year "
            f"{FINAL_EVALUATION_YEAR}"
        )
    return {
        "root": str(inference_root),
        "eval_year": FINAL_EVALUATION_YEAR,
        "metadata_sha256": digest.hexdigest(),
        "files": files,
    }


def resolve_final_eval_year(config, requested_year: int | None) -> int:
    """Derive the evaluation year from all organiser inference metadata."""
    inference_root = config.inference_root()
    if inference_root is None:
        raise FileNotFoundError("No organiser inference directory was found")
    return int(
        inspect_final_inference_metadata(inference_root, requested_year)["eval_year"]
    )


def locate_final_inference_metadata(
    data_root: Path | None, requested_year: int | None
) -> dict:
    """Find definitive metadata without importing the modelling stack."""
    configure_data_root(data_root)
    configured = os.environ.get("PHASE2_DATA_ROOT")
    if not configured:
        raise FileNotFoundError("No organiser data root was found")
    root = Path(configured).expanduser().resolve()
    candidates = (
        root,
        root / "inference",
        root / "phase2_dataset_ship" / "inference",
        root / "unpacked" / "phase2_dataset_ship" / "inference",
    )
    errors = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            return inspect_final_inference_metadata(candidate, requested_year)
        except (RuntimeError, ValueError) as exc:
            errors.append(str(exc))
    detail = "; ".join(errors) if errors else f"searched under {root}"
    raise FileNotFoundError(f"No valid final 2022 inference metadata found: {detail}")


def resolve_phase1_pressure_path() -> Path:
    roots = []
    for name in ("PHASE1_DATA_ROOT", "PHASE2_DATA_ROOT"):
        value = os.environ.get(name)
        if value:
            root = Path(value).expanduser()
            roots.extend([root, root.parent, root.parent.parent])
    roots.extend(
        [
            SCRIPT_DIR / "data",
            SCRIPT_DIR.parent / "data",
            SCRIPT_DIR.parent / "phase2_workspace" / "data" / "unpacked",
            Path.cwd() / "data",
            Path.cwd() / "phase2_workspace" / "data" / "unpacked",
        ]
    )
    relative = (
        Path("hres_pressure_north_sea.parquet"),
        Path("train") / "hres_pressure_north_sea.parquet",
        Path("phase1_dataset") / "train" / "hres_pressure_north_sea.parquet",
        Path("unpacked")
        / "phase1_dataset"
        / "train"
        / "hres_pressure_north_sea.parquet",
    )
    seen = set()
    for root in roots:
        for suffix in relative:
            candidate = (root / suffix).resolve()
            key = str(candidate).lower()
            if key in seen:
                continue
            seen.add(key)
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        "Could not find hres_pressure_north_sea.parquet. Put the Phase 1 "
        "dataset beside the Phase 2 dataset or set PHASE1_DATA_ROOT."
    )


def validate_submission(df, expected_windows: int) -> dict:
    required = [
        "type",
        "window",
        "region",
        "latitude",
        "longitude",
        "horizon",
        "hour",
        "level",
        "q05",
        "q50",
        "q95",
        "dir_05",
        "dir_50",
        "dir_95",
    ]
    if list(df.columns) != required:
        raise ValueError(f"Unexpected submission columns: {list(df.columns)}")
    if df[required].isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Submission contains NaNs in columns: {bad}")
    if not ((df["q05"] <= df["q50"]) & (df["q50"] <= df["q95"])).all():
        raise ValueError("Speed quantiles are not monotone")
    if set(df["horizon"].unique()) != {1, 7, 14}:
        raise ValueError(f"Unexpected horizons: {sorted(df['horizon'].unique())}")
    if set(df["hour"].unique()) != {0, 6, 12, 18}:
        raise ValueError(f"Unexpected hours: {sorted(df['hour'].unique())}")
    if not ((df[["dir_05", "dir_50", "dir_95"]] >= 0.0) & (df[["dir_05", "dir_50", "dir_95"]] < 360.0)).all().all():
        raise ValueError("Direction columns must be in [0, 360)")
    if df["window"].nunique() != expected_windows:
        raise ValueError(
            f"Expected {expected_windows} windows, got {df['window'].nunique()}"
        )
    return {
        "rows": int(len(df)),
        "windows": int(df["window"].nunique()),
        "footprint_rows": int(len(df) // (expected_windows * 3 * 4)),
        "q50_mean": float(df["q50"].mean()),
    }


def validate_submission_rows(df) -> None:
    required = [
        "type",
        "window",
        "region",
        "latitude",
        "longitude",
        "horizon",
        "hour",
        "level",
        "q05",
        "q50",
        "q95",
        "dir_05",
        "dir_50",
        "dir_95",
    ]
    if list(df.columns) != required:
        raise ValueError(f"Unexpected submission columns: {list(df.columns)}")
    if df[required].isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Submission contains NaNs in columns: {bad}")
    if not ((df["q05"] <= df["q50"]) & (df["q50"] <= df["q95"])).all():
        raise ValueError("Speed quantiles are not monotone")
    if not ((df[["dir_05", "dir_50", "dir_95"]] >= 0.0) & (df[["dir_05", "dir_50", "dir_95"]] < 360.0)).all().all():
        raise ValueError("Direction columns must be in [0, 360)")


def normalize_directions(df):
    for col in ("dir_05", "dir_50", "dir_95"):
        df[col] = (df[col].astype("float64") % 360.0).astype("float32")
        df.loc[df[col] >= 360.0, col] = np.float32(0.0)
    return df


def apply_external_trajectory_policy(df, issue_date, artifact: dict):
    """Apply the strictly gated causal GraphCast d1 and d7 corrections."""
    policy = artifact.get("external_trajectory_policy")
    if policy is None:
        return df
    if policy.get("input_only_training") is not True:
        raise ValueError("External trajectory policy lacks input-only provenance")
    if policy.get("previous_submission_inputs") != []:
        raise ValueError("External trajectory policy used a previous submission")
    if policy.get("final_evaluation_labels_used") is not False:
        raise ValueError(
            "External trajectory policy lacks a no-final-labels declaration"
        )
    gates = policy.get("gates", {})
    expected_signals = {"speed_d1", "direction_d1", "speed_d7"}
    if not expected_signals.issubset(gates):
        raise ValueError(f"Incomplete external trajectory gates: {sorted(gates)}")
    if not all(gates[name].get("passed", False) for name in expected_signals):
        raise ValueError("An external trajectory signal failed its historical gate")

    issue = pd.Timestamp(issue_date).normalize()
    issue_key = str(issue.date())
    issue_slots = tuple(tuple(int(v) for v in slot) for slot in policy["issue_slots"])
    issue_slot = (int(issue.month), int(issue.day))
    if issue_slot not in issue_slots:
        raise KeyError(f"No external trajectory slot for issue {issue_key}")
    slot_index = issue_slots.index(issue_slot)
    hours = tuple(int(value) for value in policy.get("hours", ()))
    if hours != HOURS:
        raise ValueError(f"Unexpected external trajectory hour ordering: {hours}")
    graphcast_by_issue = policy.get("graphcast_by_issue", {})
    if issue_key not in graphcast_by_issue:
        raise KeyError(f"Missing external trajectory fields for issue {issue_key}")
    required_graphcast = {
        "d1_1000_u",
        "d1_1000_v",
        "d7_10m_u",
        "d7_10m_v",
        "d7_1000_u",
        "d7_1000_v",
    }
    available_graphcast = graphcast_by_issue[issue_key]
    if not required_graphcast.issubset(available_graphcast):
        raise ValueError(
            f"Unexpected GraphCast fields for {issue_key}: "
            f"{sorted(available_graphcast)}"
        )
    graphcast = {
        name: np.asarray(values, dtype="float64")
        for name, values in available_graphcast.items()
        if name in required_graphcast
    }
    for name, values in graphcast.items():
        if values.shape != (len(HOURS), FOOTPRINT_ROWS):
            raise ValueError(f"Unexpected {name} shape: {values.shape}")
    expected_lat = np.asarray(policy["latitude"], dtype="float64")
    expected_lon = np.asarray(policy["longitude"], dtype="float64")
    if expected_lat.shape != (FOOTPRINT_ROWS,) or expected_lon.shape != (FOOTPRINT_ROWS,):
        raise ValueError("External target-coordinate artifact is incomplete")
    def positions_for(lead, hour):
        selected = (df["horizon"] == lead) & (df["hour"] == hour)
        positions = np.flatnonzero(selected.to_numpy())
        if len(positions) != FOOTPRINT_ROWS:
            raise RuntimeError(
                f"External trajectory footprint mismatch d{lead} h{hour}: "
                f"{len(positions)}"
            )
        rows = df.iloc[positions]
        if not (
            np.allclose(
                rows["latitude"].to_numpy(dtype="float64"),
                expected_lat,
                rtol=0.0,
                atol=0.0051,
            )
            and np.allclose(
                rows["longitude"].to_numpy(dtype="float64"),
                expected_lon,
                rtol=0.0,
                atol=0.0051,
            )
        ):
            raise RuntimeError(
                f"External target-coordinate order changed d{lead} h{hour}"
            )
        return positions, rows

    def circular_blend(current, challenger, weight):
        current_rad = np.radians(current)
        challenger_rad = np.radians(challenger)
        return np.mod(
            np.degrees(
                np.arctan2(
                    (1.0 - weight) * np.sin(current_rad)
                    + weight * np.sin(challenger_rad),
                    (1.0 - weight) * np.cos(current_rad)
                    + weight * np.cos(challenger_rad),
                )
            ),
            360.0,
        )

    def translate_speed(positions, rows, shift):
        center = np.maximum(0.0, rows["q50"].to_numpy(dtype="float64") + shift)
        lower = np.maximum(0.0, rows["q05"].to_numpy(dtype="float64") + shift)
        upper = np.maximum(center, rows["q95"].to_numpy(dtype="float64") + shift)
        lower = np.minimum(lower, center)
        for column, values in (("q05", lower), ("q50", center), ("q95", upper)):
            df.iloc[positions, df.columns.get_loc(column)] = values.astype("float32")

    def translate_direction(positions, rows, candidate):
        current = rows["dir_50"].to_numpy(dtype="float64")
        shift = np.mod(candidate - current + 180.0, 360.0) - 180.0
        for column in ("dir_05", "dir_50", "dir_95"):
            values = rows[column].to_numpy(dtype="float64")
            df.iloc[positions, df.columns.get_loc(column)] = np.mod(
                values + shift, 360.0
            ).astype("float32")
        return shift

    changes = {name: [] for name in expected_signals}
    d1_speed_gate = gates["speed_d1"]
    d1_direction_gate = gates["direction_d1"]
    d7_speed_gate = gates["speed_d7"]
    d7_residual_quantiles = np.asarray(
        policy.get("d7_speed_residual_quantiles"), dtype="float64"
    )
    if d7_residual_quantiles.shape != (len(issue_slots), len(HOURS), 3):
        raise ValueError(
            "Unexpected GraphCast d7 residual-quantile shape: "
            f"{d7_residual_quantiles.shape}"
        )
    active_d7_cells = {
        int(value) for value in d7_speed_gate.get("active_cells", ())
    }
    if active_d7_cells != {
        4, 5, 6, 7, 8, 9, 10, 11, 14, 15,
        16, 18, 20, 21, 22, 24, 25, 27, 28, 30,
    }:
        raise ValueError("Unexpected external d7 speed activation cells")
    d7_level_blend = float(d7_speed_gate["level_blend_1000_hpa"])
    d7_center_weight = float(d7_speed_gate["center_weight"])
    d7_lower_width_weight = float(d7_speed_gate["lower_width_weight"])
    d7_upper_width_weight = float(d7_speed_gate["upper_width_weight"])
    if not (
        d7_level_blend == 0.50
        and d7_center_weight == 0.50
        and d7_lower_width_weight == 0.20
        and d7_upper_width_weight == 0.60
    ):
        raise ValueError("Unexpected external d7 speed production weights")
    speed_label_blends = {
        int(label): float(weight)
        for label, weight in d1_speed_gate.get("label_blends", {}).items()
    }
    selected_speed_labels = {
        int(label) for label in d1_speed_gate["selected_labels"]
    }
    if speed_label_blends and set(speed_label_blends) != selected_speed_labels:
        raise ValueError("External d1 speed labels and weights do not align")
    for hour_index, hour in enumerate(HOURS):
        slot_hour = slot_index * len(HOURS) + hour_index

        d1_positions, d1_rows = positions_for(1, hour)
        if slot_hour in selected_speed_labels:
            source = np.hypot(
                graphcast["d1_1000_u"][hour_index],
                graphcast["d1_1000_v"][hour_index],
            )
            current = d1_rows["q50"].to_numpy(dtype="float64")
            weight = speed_label_blends.get(
                slot_hour, float(d1_speed_gate.get("production_blend", 0.0))
            )
            if not 0.0 < weight <= 1.0:
                raise ValueError(
                    f"Invalid external d1 speed weight for label {slot_hour}: "
                    f"{weight}"
                )
            shift = weight * (source - current)
            translate_speed(d1_positions, d1_rows, shift)
            changes["speed_d1"].append(np.abs(shift))

        current_direction = d1_rows["dir_50"].to_numpy(dtype="float64")
        source_direction = np.mod(
            np.degrees(
                np.arctan2(
                    -graphcast["d1_1000_u"][hour_index],
                    -graphcast["d1_1000_v"][hour_index],
                )
            ),
            360.0,
        )
        direction_weight = float(
            d1_direction_gate.get("production_blend", 0.30)
        )
        guarded = d1_direction_gate.get("guarded_incremental_blend")
        if guarded is not None:
            if guarded.get("passed") is not True:
                raise ValueError("Guarded GraphCast d1 direction gate failed")
            if guarded.get("final_2022_input_support_passed") is not True:
                raise ValueError(
                    "Guarded GraphCast d1 direction support gate failed"
                )
            if not (
                float(guarded.get("base_weight", np.nan)) == 0.30
                and float(guarded.get("strong_weight", np.nan)) == 0.40
                and guarded.get("guarded_slot_indices") == [0, 1]
                and guarded.get("guarded_direction_sectors_degrees")
                == [[45.0, 90.0], [180.0, 225.0]]
                and guarded.get("guarded_predicted_speed_interval")
                == [15.0, 20.0]
            ):
                raise ValueError("Unexpected guarded GraphCast d1 policy")
            base_speed = d1_rows["q50"].to_numpy(dtype="float64")
            protected = (
                (slot_index <= 1)
                | (
                    (current_direction >= 45.0)
                    & (current_direction < 90.0)
                )
                | (
                    (current_direction >= 180.0)
                    & (current_direction < 225.0)
                )
                | ((base_speed >= 15.0) & (base_speed < 20.0))
            )
            direction_weight = np.where(protected, 0.30, 0.40)
        candidate_direction = circular_blend(
            current_direction,
            source_direction,
            direction_weight,
        )
        changes["direction_d1"].append(
            np.abs(
                translate_direction(
                    d1_positions, d1_rows, candidate_direction
                )
            )
        )
        if slot_hour in active_d7_cells:
            d7_positions, d7_rows = positions_for(7, hour)
            source_u = (
                (1.0 - d7_level_blend) * graphcast["d7_10m_u"][hour_index]
                + d7_level_blend * graphcast["d7_1000_u"][hour_index]
            )
            source_v = (
                (1.0 - d7_level_blend) * graphcast["d7_10m_v"][hour_index]
                + d7_level_blend * graphcast["d7_1000_v"][hour_index]
            )
            source_speed = np.hypot(source_u, source_v)
            residual = d7_residual_quantiles[slot_index, hour_index]
            candidate_lower = np.maximum(0.0, source_speed + residual[0])
            candidate_center = np.maximum(0.0, source_speed + residual[1])
            candidate_upper = np.maximum(0.0, source_speed + residual[2])
            candidate_lower = np.minimum(candidate_lower, candidate_center)
            candidate_upper = np.maximum(candidate_center, candidate_upper)

            base_lower = d7_rows["q05"].to_numpy(dtype="float64")
            base_center = d7_rows["q50"].to_numpy(dtype="float64")
            base_upper = d7_rows["q95"].to_numpy(dtype="float64")
            center = base_center + d7_center_weight * (
                candidate_center - base_center
            )
            base_left = base_center - base_lower
            base_right = base_upper - base_center
            candidate_left = candidate_center - candidate_lower
            candidate_right = candidate_upper - candidate_center
            left = base_left + d7_lower_width_weight * (
                candidate_left - base_left
            )
            right = base_right + d7_upper_width_weight * (
                candidate_right - base_right
            )
            lower = np.maximum(0.0, center - np.maximum(left, 0.0))
            upper = np.maximum(center, center + np.maximum(right, 0.0))
            lower = np.minimum(lower, center)
            for column, values in (
                ("q05", lower),
                ("q50", center),
                ("q95", upper),
            ):
                df.iloc[
                    d7_positions, df.columns.get_loc(column)
                ] = values.astype("float32")
            changes["speed_d7"].append(np.abs(center - base_center))
    summary = {
        name: {
            "rows": int(sum(len(values) for values in parts)),
            "mean_abs_center_change": (
                float(np.mean(np.concatenate(parts))) if parts else 0.0
            ),
        }
        for name, parts in sorted(changes.items())
    }
    print(
        f"[infer] gated external d1/d7 trajectory {issue_key}: "
        f"{json.dumps(summary, sort_keys=True)}",
        flush=True,
    )
    return df


def apply_hres_analog_policy(df, issue_date, artifact: dict):
    """Apply the frozen two-view d1 speed endpoint consensus."""
    policy = artifact.get("hres_analog_policy")
    if policy is None:
        return df
    if policy.get("input_only_training") is not True:
        raise ValueError("HRES analogue policy lacks input-only provenance")
    if policy.get("previous_submission_inputs") != []:
        raise ValueError("HRES analogue policy used a previous submission")
    if policy.get("final_evaluation_labels_used") is not False:
        raise ValueError("HRES analogue policy used final labels")
    if not policy.get("gate", {}).get("passed", False):
        raise ValueError("HRES analogue historical gate did not pass")
    if policy.get("support_passed") is not True:
        raise ValueError("HRES analogue final-input support gate did not pass")

    issue_key = str(pd.Timestamp(issue_date).normalize().date())
    issue_values = policy.get("values_by_issue", {}).get(issue_key)
    if issue_values is None:
        raise KeyError(f"Missing HRES analogue fields for {issue_key}")
    mixed = np.asarray(issue_values["mixed"], dtype="float64")
    lead = np.asarray(issue_values["lead"], dtype="float64")
    expected_shape = (3, len(HOURS), FOOTPRINT_ROWS)
    if mixed.shape != expected_shape or lead.shape != expected_shape:
        raise ValueError(
            f"Unexpected HRES analogue shapes: {mixed.shape}, {lead.shape}"
        )

    expected_lat = np.asarray(policy["latitude"], dtype="float64")
    expected_lon = np.asarray(policy["longitude"], dtype="float64")
    if expected_lat.shape != (FOOTPRINT_ROWS,) or expected_lon.shape != (
        FOOTPRINT_ROWS,
    ):
        raise ValueError("HRES analogue coordinate artifact is incomplete")
    lower_weight = float(policy["lower_endpoint_weight"])
    upper_weight = float(policy["upper_endpoint_weight"])
    if not (0.0 < lower_weight <= 1.0 and 0.0 < upper_weight <= 1.0):
        raise ValueError("Invalid HRES analogue endpoint weights")

    active_count = 0
    lower_changes = []
    upper_changes = []
    for hour_index, hour in enumerate(HOURS):
        selected = (df["horizon"] == 1) & (df["hour"] == hour)
        positions = np.flatnonzero(selected.to_numpy())
        if len(positions) != FOOTPRINT_ROWS:
            raise RuntimeError(
                f"HRES analogue footprint mismatch d1 h{hour}: {len(positions)}"
            )
        rows = df.iloc[positions]
        if not (
            np.allclose(
                rows["latitude"].to_numpy(dtype="float64"),
                expected_lat,
                rtol=0.0,
                atol=0.0051,
            )
            and np.allclose(
                rows["longitude"].to_numpy(dtype="float64"),
                expected_lon,
                rtol=0.0,
                atol=0.0051,
            )
        ):
            raise RuntimeError(
                f"HRES analogue target-coordinate order changed d1 h{hour}"
            )

        center = rows["q50"].to_numpy(dtype="float64")
        lower = rows["q05"].to_numpy(dtype="float64")
        upper = rows["q95"].to_numpy(dtype="float64")
        mixed_shift = mixed[1, hour_index] - center
        lead_shift = lead[1, hour_index] - center
        active = np.sign(mixed_shift) == np.sign(lead_shift)
        target_lower = 0.5 * (
            mixed[0, hour_index] + lead[0, hour_index]
        )
        target_upper = 0.5 * (
            mixed[2, hour_index] + lead[2, hour_index]
        )
        revised_lower = lower.copy()
        revised_upper = upper.copy()
        revised_lower[active] = np.clip(
            (1.0 - lower_weight) * lower[active]
            + lower_weight * target_lower[active],
            0.0,
            center[active],
        )
        revised_upper[active] = np.maximum(
            center[active],
            (1.0 - upper_weight) * upper[active]
            + upper_weight * target_upper[active],
        )
        # Re-project all rows after the selective update. Earlier policies are
        # stored as float32, so endpoint/centre equality can differ by one ULP
        # even on inactive rows after conversion through float64.
        revised_lower = np.minimum(np.maximum(0.0, revised_lower), center)
        revised_upper = np.maximum(revised_upper, center)
        if np.any(revised_lower > center) or np.any(revised_upper < center):
            raise RuntimeError("HRES analogue correction broke speed monotonicity")
        df.iloc[positions, df.columns.get_loc("q05")] = revised_lower.astype(
            "float32"
        )
        df.iloc[positions, df.columns.get_loc("q95")] = revised_upper.astype(
            "float32"
        )
        active_count += int(np.count_nonzero(active))
        lower_changes.append(np.abs(revised_lower[active] - lower[active]))
        upper_changes.append(np.abs(revised_upper[active] - upper[active]))

    total = len(HOURS) * FOOTPRINT_ROWS
    summary = {
        "active_rows": active_count,
        "active_fraction": active_count / total,
        "mean_abs_lower_change": float(
            np.mean(np.concatenate(lower_changes))
        ),
        "mean_abs_upper_change": float(
            np.mean(np.concatenate(upper_changes))
        ),
    }
    print(
        f"[infer] two-view HRES d1 analogue {issue_key}: "
        f"{json.dumps(summary, sort_keys=True)}",
        flush=True,
    )
    return df


def fine_d7_direction_features(
    latitude,
    longitude,
    issue_date,
    hour,
    d1_speed,
    d1_direction,
    d7_speed,
    d7_direction,
) -> pd.DataFrame:
    """Build the training-identical 22-feature fine-grid d7 matrix."""
    latitude = np.asarray(latitude, dtype="float32")
    longitude = np.asarray(longitude, dtype="float32")
    d1_speed = np.asarray(d1_speed, dtype="float32")
    d7_speed = np.asarray(d7_speed, dtype="float32")
    d1_direction = np.asarray(d1_direction, dtype="float32") % 360.0
    d7_direction = np.asarray(d7_direction, dtype="float32") % 360.0
    n = len(latitude)
    if not all(
        len(values) == n
        for values in (
            longitude,
            d1_speed,
            d7_speed,
            d1_direction,
            d7_direction,
        )
    ):
        raise ValueError("Fine d7 feature arrays have inconsistent lengths")
    valid_date = pd.Timestamp(issue_date).normalize() + pd.Timedelta(days=7)
    week = int(valid_date.isocalendar().week)
    d1_rad = np.radians(d1_direction)
    d7_rad = np.radians(d7_direction)
    signed_tendency = (
        d7_direction.astype("float64")
        - d1_direction.astype("float64")
        + 180.0
    ) % 360.0 - 180.0
    tendency_rad = np.radians(signed_tendency)
    vector_change = np.hypot(
        -d7_speed * np.sin(d7_rad) + d1_speed * np.sin(d1_rad),
        -d7_speed * np.cos(d7_rad) + d1_speed * np.cos(d1_rad),
    )
    return pd.DataFrame(
        {
            "lat": latitude,
            "lon": longitude,
            "lat2": latitude ** 2,
            "lon2": longitude ** 2,
            "lat_lon": latitude * longitude,
            "hour_sin": np.full(
                n, np.sin(2.0 * np.pi * hour / 24.0), dtype="float32"
            ),
            "hour_cos": np.full(
                n, np.cos(2.0 * np.pi * hour / 24.0), dtype="float32"
            ),
            "woy_sin": np.full(
                n, np.sin(2.0 * np.pi * week / 52.0), dtype="float32"
            ),
            "woy_cos": np.full(
                n, np.cos(2.0 * np.pi * week / 52.0), dtype="float32"
            ),
            "speed": d7_speed,
            "speed_sq": d7_speed ** 2,
            "d1_speed": d1_speed,
            "speed_delta": d7_speed - d1_speed,
            "speed_ratio": d7_speed / np.maximum(d1_speed, 0.25),
            "raw_dir_sin": np.sin(d7_rad),
            "raw_dir_cos": np.cos(d7_rad),
            "d1_dir_sin": np.sin(d1_rad),
            "d1_dir_cos": np.cos(d1_rad),
            "tendency_sin": np.sin(tendency_rad),
            "tendency_cos": np.cos(tendency_rad),
            "lead_disagreement": np.abs(signed_tendency),
            "vector_change": vector_change,
        }
    ).astype("float32")


def fine_d7_neighbor_indices(latitude, longitude) -> np.ndarray:
    from scipy.spatial import cKDTree

    latitude = np.asarray(latitude, dtype="float64")
    longitude = np.asarray(longitude, dtype="float64")
    longitude_scale = np.cos(np.radians(float(np.mean(latitude))))
    coordinates = np.column_stack((latitude, longitude * longitude_scale))
    tree = cKDTree(coordinates)
    _, indices = tree.query(
        coordinates,
        k=max(FINE_D7_NEIGHBOR_COUNTS) + 1,
        workers=1,
    )
    if not np.array_equal(indices[:, 0], np.arange(len(latitude))):
        raise RuntimeError("Target point was not its first nearest neighbor")
    return indices[:, 1:].astype("int32")


def fine_d7_spatial_features(
    d1_speed,
    d1_direction,
    d7_speed,
    d7_direction,
    neighbors,
) -> np.ndarray:
    d1_speed = np.asarray(d1_speed, dtype="float32")
    d7_speed = np.asarray(d7_speed, dtype="float32")
    d1_radians = np.radians(np.asarray(d1_direction, dtype="float32"))
    d7_radians = np.radians(np.asarray(d7_direction, dtype="float32"))
    d7_unit_u = -np.sin(d7_radians)
    d7_unit_v = -np.cos(d7_radians)
    d1_u = -d1_speed * np.sin(d1_radians)
    d1_v = -d1_speed * np.cos(d1_radians)
    d7_u = d7_speed * d7_unit_u
    d7_v = d7_speed * d7_unit_v
    delta_u = d7_u - d1_u
    delta_v = d7_v - d1_v
    columns = [d7_u, d7_v, d1_u, d1_v]
    for count in FINE_D7_NEIGHBOR_COUNTS:
        selected = neighbors[:, :count]
        local_u = np.mean(d7_u[selected], axis=1)
        local_v = np.mean(d7_v[selected], axis=1)
        speed_std = np.std(d7_speed[selected], axis=1)
        local_norm = np.maximum(np.hypot(local_u, local_v), 1e-6)
        local_unit_u = local_u / local_norm
        local_unit_v = local_v / local_norm
        local_delta_u = np.mean(delta_u[selected], axis=1)
        local_delta_v = np.mean(delta_v[selected], axis=1)
        columns.extend(
            [
                local_u,
                local_v,
                speed_std,
                d7_u - local_u,
                d7_v - local_v,
                d7_unit_u * local_unit_v - d7_unit_v * local_unit_u,
                d7_unit_u * local_unit_u + d7_unit_v * local_unit_v,
                local_delta_u,
                local_delta_v,
                delta_u - local_delta_u,
                delta_v - local_delta_v,
            ]
        )
    result = np.column_stack(columns).astype("float32")
    if (
        result.shape[1] != len(FINE_D7_SPATIAL_FEATURES)
        or not np.isfinite(result).all()
    ):
        raise RuntimeError("Invalid fine d7 spatial feature block")
    return result


def _interpolate_fine_reanalysis(snapshot, latitude, longitude):
    from scipy.interpolate import RegularGridInterpolator

    points = np.column_stack(
        (
            np.clip(latitude, snapshot.lats[0], snapshot.lats[-1]),
            np.clip(longitude, snapshot.lons[0], snapshot.lons[-1]),
        )
    )
    u = RegularGridInterpolator(
        (snapshot.lats, snapshot.lons),
        snapshot.u100,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )(points)
    v = RegularGridInterpolator(
        (snapshot.lats, snapshot.lons),
        snapshot.v100,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )(points)
    return u.astype("float32"), v.astype("float32")


def fine_d7_lagged_context_features(
    config, issue_date, latitude, longitude
) -> dict[int, pd.DataFrame]:
    import reanalysis_loader
    from types import SimpleNamespace

    issue_date = pd.Timestamp(issue_date).normalize()
    latitude = np.asarray(latitude, dtype="float32")
    longitude = np.asarray(longitude, dtype="float32")
    u_hist = np.empty(
        (14, len(HOURS), len(latitude)), dtype="float32"
    )
    v_hist = np.empty_like(u_hist)
    inference_frame = None
    try:
        reanalysis_loader.load_reanalysis(
            issue_date.date(),
            HOURS[0],
            root=config.reanalysis_root(),
        )
    except FileNotFoundError:
        inference_root = config.inference_root()
        for window_dir in sorted(inference_root.glob("window_*")):
            metadata_path = window_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                pd.Timestamp(metadata["context_end"]).normalize()
                != issue_date
            ):
                continue
            context_path = (
                window_dir / "context_reanalysis_north_sea.parquet"
            )
            inference_frame = pd.read_parquet(context_path)
            inference_frame["time"] = pd.to_datetime(
                inference_frame["time"]
            )
            break
        if inference_frame is None:
            raise FileNotFoundError(
                f"No inference reanalysis window ending {issue_date.date()}"
            )
    for lag in range(14):
        date = (issue_date - pd.Timedelta(days=lag)).date()
        for hour_index, hour in enumerate(HOURS):
            if inference_frame is None:
                snapshot = reanalysis_loader.load_reanalysis(
                    date, hour, root=config.reanalysis_root()
                )
            else:
                timestamp = pd.Timestamp(date) + pd.Timedelta(hours=hour)
                subset = inference_frame[
                    inference_frame["time"] == timestamp
                ]
                lats = np.sort(subset["latitude"].unique())
                lons = np.sort(subset["longitude"].unique())
                u100 = subset.pivot(
                    index="latitude", columns="longitude", values="u100"
                ).reindex(index=lats, columns=lons).to_numpy()
                v100 = subset.pivot(
                    index="latitude", columns="longitude", values="v100"
                ).reindex(index=lats, columns=lons).to_numpy()
                if (
                    u100.shape != (len(lats), len(lons))
                    or not np.isfinite(u100).all()
                    or not np.isfinite(v100).all()
                ):
                    raise RuntimeError(
                        f"Invalid inference reanalysis at {timestamp}"
                    )
                snapshot = SimpleNamespace(
                    lats=lats,
                    lons=lons,
                    u100=u100,
                    v100=v100,
                )
            u, v = _interpolate_fine_reanalysis(
                snapshot, latitude, longitude
            )
            u_hist[lag, hour_index] = u
            v_hist[lag, hour_index] = v
    output = {}
    for hour_index, hour in enumerate(HOURS):
        columns = []
        for lag in FINE_D7_CONTEXT_LAGS:
            columns.extend(
                (u_hist[lag, hour_index], v_hist[lag, hour_index])
            )
        for days in FINE_D7_CONTEXT_MEAN_DAYS:
            u = np.mean(u_hist[:days, hour_index], axis=0)
            v = np.mean(v_hist[:days, hour_index], axis=0)
            mean_speed = np.mean(
                np.hypot(
                    u_hist[:days, hour_index],
                    v_hist[:days, hour_index],
                ),
                axis=0,
            )
            concentration = np.hypot(u, v) / np.maximum(mean_speed, 0.1)
            columns.extend((u, v, concentration))
        values = np.column_stack(columns).astype("float32")
        if (
            values.shape[1] != len(FINE_D7_CONTEXT_FEATURES)
            or not np.isfinite(values).all()
        ):
            raise RuntimeError(
                f"Invalid fine d7 context for {issue_date.date()} h{hour:02d}"
            )
        output[hour] = pd.DataFrame(
            values, columns=list(FINE_D7_CONTEXT_FEATURES)
        )
    return output


def fine_raw_d10_directions(pipeline, artifact, fields) -> dict[int, np.ndarray]:
    import footprint

    mask = footprint.footprint_mask()
    output = {}
    for hour in HOURS:
        raw_u, raw_v = fields[(10, hour, "raw")]
        fine_u, fine_v = pipeline.dn.downscale(
            artifact["downscaler"], raw_u, raw_v
        )
        output[hour] = (
            np.degrees(np.arctan2(-fine_u, -fine_v)) % 360.0
        )[mask].astype("float32")
    return output


def predict_fine_d7_direction_correction(
    df: pd.DataFrame,
    issue_date,
    artifact: dict,
    config=None,
    raw_d10_directions=None,
) -> np.ndarray:
    """Predict v18, context-residual, and zero-model d10 d7 shifts."""
    models = artifact.get("fine_d7_direction_models")
    policy = artifact.get("fine_d7_direction_policy", {})
    context_models = artifact.get("fine_d7_context_models")
    context_policy = artifact.get("fine_d7_context_policy", {})
    d10_policy = artifact.get("d7_d10_tendency_policy", {})
    correction_out = np.zeros(len(df), dtype="float32")
    if models is None:
        return correction_out
    issue_date = pd.Timestamp(issue_date).normalize()
    relevant_hours = sorted(
        {
            hour
            for source in (policy, context_policy, d10_policy)
            for month, day, hour in source
            if month == issue_date.month and day == issue_date.day
        }
    )
    if not relevant_hours:
        return correction_out

    first_d7 = df.loc[
        (df["horizon"] == 7) & (df["hour"] == relevant_hours[0]),
        ["latitude", "longitude"],
    ]
    latitude = first_d7["latitude"].to_numpy(dtype="float32")
    longitude = first_d7["longitude"].to_numpy(dtype="float32")
    neighbors = (
        fine_d7_neighbor_indices(latitude, longitude)
        if context_models is not None and context_policy
        else None
    )
    lagged_context = (
        fine_d7_lagged_context_features(
            config, issue_date, latitude, longitude
        )
        if context_models is not None and context_policy and config is not None
        else {}
    )

    for hour in relevant_hours:
        d1 = df.loc[
            (df["horizon"] == 1) & (df["hour"] == hour),
            ["latitude", "longitude", "q50", "dir_50"],
        ]
        d7 = df.loc[
            (df["horizon"] == 7) & (df["hour"] == hour),
            ["latitude", "longitude", "q50", "dir_50"],
        ]
        if len(d1) != len(d7) or len(d7) != FOOTPRINT_ROWS:
            raise ValueError(
                f"Fine d7 correction alignment failed at hour={hour}: "
                f"d1={len(d1)} d7={len(d7)}"
            )
        d1_coordinates = d1[["latitude", "longitude"]].to_numpy(dtype="float64")
        d7_coordinates = d7[["latitude", "longitude"]].to_numpy(dtype="float64")
        if not np.allclose(d1_coordinates, d7_coordinates, atol=1e-6, rtol=0.0):
            raise ValueError(f"Fine d1/d7 coordinates differ at hour={hour}")
        features = fine_d7_direction_features(
            d7["latitude"].to_numpy(),
            d7["longitude"].to_numpy(),
            issue_date,
            hour,
            d1["q50"].to_numpy(),
            d1["dir_50"].to_numpy(),
            d7["q50"].to_numpy(),
            d7["dir_50"].to_numpy(),
        )
        total = np.zeros(len(d7), dtype="float32")
        rule = policy.get((issue_date.month, issue_date.day, hour))
        if rule is not None:
            sign, weight, confidence_min, correction_cap = map(float, rule)
            base_features = features
            expected = models.get("features")
            if expected is not None:
                base_features = base_features[expected]
            pred_sin = models["sin"].predict(base_features)
            pred_cos = models["cos"].predict(base_features)
            prediction = np.degrees(np.arctan2(pred_sin, pred_cos))
            confidence = np.hypot(pred_sin, pred_cos)
            correction = sign * weight * prediction
            active = (confidence >= confidence_min) & (
                np.abs(correction) <= correction_cap
            )
            total[active] += correction[active].astype("float32")

        context_rule = context_policy.get(
            (issue_date.month, issue_date.day, hour)
        )
        if context_models is not None and context_rule is not None:
            if hour not in lagged_context or neighbors is None:
                raise RuntimeError("Fine d7 context inputs were not prepared")
            spatial = fine_d7_spatial_features(
                d1["q50"].to_numpy(),
                d1["dir_50"].to_numpy(),
                d7["q50"].to_numpy(),
                d7["dir_50"].to_numpy(),
                neighbors,
            )
            context_features = pd.concat(
                (
                    features.reset_index(drop=True),
                    pd.DataFrame(
                        spatial, columns=FINE_D7_SPATIAL_FEATURES
                    ),
                    lagged_context[hour].reset_index(drop=True),
                ),
                axis=1,
            )
            expected = context_models.get("features")
            if expected is not None:
                context_features = context_features[expected]
            pred_sin = context_models["sin"].predict(context_features)
            pred_cos = context_models["cos"].predict(context_features)
            prediction = np.degrees(np.arctan2(pred_sin, pred_cos))
            confidence = np.hypot(pred_sin, pred_cos)
            sign, weight, confidence_min, correction_cap = map(
                float, context_rule
            )
            correction = sign * weight * prediction
            active = (confidence >= confidence_min) & (
                np.abs(correction) <= correction_cap
            )
            total[active] += correction[active].astype("float32")

        d10_rule = d10_policy.get(
            (issue_date.month, issue_date.day, hour)
        )
        if d10_rule is not None:
            if raw_d10_directions is None or hour not in raw_d10_directions:
                raise RuntimeError("Raw d10 direction input was not prepared")
            tendency = (
                np.asarray(raw_d10_directions[hour], dtype="float64")
                - d7["dir_50"].to_numpy(dtype="float64")
                + 180.0
            ) % 360.0 - 180.0
            weight, tendency_max = map(float, d10_rule)
            active = np.abs(tendency) <= tendency_max
            total[active] += (weight * tendency[active]).astype("float32")

        target_indices = d7.index.to_numpy(dtype="int64")
        correction_out[target_indices] = total
    return correction_out


def apply_fine_d7_direction_correction(
    df: pd.DataFrame, correction: np.ndarray
) -> pd.DataFrame:
    correction = np.asarray(correction, dtype="float64")
    if len(correction) != len(df):
        raise ValueError("Fine d7 correction length does not match submission")
    selected = correction != 0.0
    if selected.any():
        columns = ["dir_05", "dir_50", "dir_95"]
        shifted = (
            df.loc[selected, columns].to_numpy(dtype="float64")
            + correction[selected, None]
        ) % 360.0
        df.loc[selected, columns] = shifted.astype("float32")
    return df


def apply_d7_conditional_endpoint(
    df: pd.DataFrame,
    issue_date,
    artifact: dict,
    config,
) -> pd.DataFrame:
    """Apply the single four-fold-safe q05/q95 endpoint action."""
    payload = artifact.get("d7_conditional_endpoint")
    if payload is None:
        return df
    gate = payload.get("gate", {})
    if not gate.get("passed", False) or int(payload.get("new_models", -1)) != 2:
        raise RuntimeError("Invalid d7 conditional endpoint artifact")
    issue = pd.Timestamp(issue_date).normalize()
    slot_key = (issue.month, issue.day)
    if slot_key not in D7_CONDITIONAL_ENDPOINT_ISSUE_SLOTS:
        return df
    issue_index = D7_CONDITIONAL_ENDPOINT_ISSUE_SLOTS.index(slot_key)
    active_cells = np.asarray(payload.get("active_cells", []), dtype="int64")
    active_slots = np.unique(active_cells // 16)
    relevant_hours = [
        hour
        for hour_index, hour in enumerate(HOURS)
        if issue_index * len(HOURS) + hour_index in active_slots
    ]
    if not relevant_hours:
        return df

    first = df.loc[
        (df["horizon"] == 7) & (df["hour"] == relevant_hours[0]),
        ["latitude", "longitude"],
    ]
    if len(first) != FOOTPRINT_ROWS:
        raise RuntimeError("Conditional endpoint footprint is incomplete")
    latitude = first["latitude"].to_numpy(dtype="float32")
    longitude = first["longitude"].to_numpy(dtype="float32")
    neighbors = fine_d7_neighbor_indices(latitude, longitude)
    context = fine_d7_lagged_context_features(
        config, issue, latitude, longitude
    )

    amplitude = float(payload["amplitude"])
    for hour in relevant_hours:
        d1 = df.loc[
            (df["horizon"] == 1) & (df["hour"] == hour),
            ["latitude", "longitude", "q50", "dir_50"],
        ]
        d7 = df.loc[
            (df["horizon"] == 7) & (df["hour"] == hour),
            ["latitude", "longitude", "q50", "dir_05", "dir_50", "dir_95"],
        ]
        if len(d1) != FOOTPRINT_ROWS or len(d7) != FOOTPRINT_ROWS:
            raise RuntimeError(f"Conditional endpoint alignment failed at h{hour:02d}")
        if not np.allclose(
            d1[["latitude", "longitude"]].to_numpy(dtype="float64"),
            d7[["latitude", "longitude"]].to_numpy(dtype="float64"),
            atol=1e-6,
            rtol=0.0,
        ):
            raise RuntimeError("Conditional endpoint coordinates differ by lead")
        base = fine_d7_direction_features(
            d7["latitude"].to_numpy(),
            d7["longitude"].to_numpy(),
            issue,
            hour,
            d1["q50"].to_numpy(),
            d1["dir_50"].to_numpy(),
            d7["q50"].to_numpy(),
            d7["dir_50"].to_numpy(),
        )
        spatial = fine_d7_spatial_features(
            d1["q50"].to_numpy(),
            d1["dir_50"].to_numpy(),
            d7["q50"].to_numpy(),
            d7["dir_50"].to_numpy(),
            neighbors,
        )
        features = pd.concat(
            (
                base.reset_index(drop=True),
                pd.DataFrame(spatial, columns=FINE_D7_SPATIAL_FEATURES),
                context[hour].reset_index(drop=True),
            ),
            axis=1,
        )
        expected = list(payload.get("features", []))
        if not expected or any(name not in features for name in expected):
            raise RuntimeError("Conditional endpoint features do not match training")
        features = features[expected]
        lower_prediction = np.asarray(
            payload["lower"].predict(features), dtype="float64"
        )
        upper_prediction = np.asarray(
            payload["upper"].predict(features), dtype="float64"
        )
        crossed = upper_prediction <= lower_prediction + 5.0
        if np.any(crossed):
            midpoint = 0.5 * (
                lower_prediction[crossed] + upper_prediction[crossed]
            )
            lower_prediction[crossed] = midpoint - 2.5
            upper_prediction[crossed] = midpoint + 2.5
        lower_prediction = np.clip(lower_prediction, -179.0, 179.0)
        upper_prediction = np.clip(upper_prediction, -179.0, 179.0)
        lower_signal = np.clip(
            (lower_prediction - float(payload["lower_location"]))
            / float(payload["lower_scale"]),
            -2.5,
            2.5,
        )
        upper_signal = np.clip(
            (upper_prediction - float(payload["upper_location"]))
            / float(payload["upper_scale"]),
            -2.5,
            2.5,
        )
        log_width = np.log(
            np.maximum(upper_prediction - lower_prediction, 5.0)
        )
        width_signal = np.clip(
            (log_width - float(payload["log_width_location"]))
            / float(payload["log_width_scale"]),
            -2.5,
            2.5,
        )
        width_rank = np.digitize(width_signal, (-0.674, 0.0, 0.674))
        spatial_bin = (
            (latitude >= float(payload["latitude_median"])).astype("int64") * 2
            + (longitude >= float(payload["longitude_median"])).astype("int64")
        )
        slot = issue_index * len(HOURS) + HOURS.index(hour)
        cell = (slot * 4 + spatial_bin) * 4 + width_rank
        active = np.isin(cell, active_cells)
        if not np.any(active):
            continue

        center = d7["dir_50"].to_numpy(dtype="float64")
        lower_offset = (
            d7["dir_05"].to_numpy(dtype="float64") - center + 180.0
        ) % 360.0 - 180.0
        upper_offset = (
            d7["dir_95"].to_numpy(dtype="float64") - center + 180.0
        ) % 360.0 - 180.0
        lower_offset[active] += amplitude * lower_signal[active]
        upper_offset[active] += amplitude * upper_signal[active]
        invalid = active & (upper_offset < lower_offset + 5.0)
        if np.any(invalid):
            midpoint = 0.5 * (lower_offset[invalid] + upper_offset[invalid])
            lower_offset[invalid] = midpoint - 2.5
            upper_offset[invalid] = midpoint + 2.5
        target_index = d7.index.to_numpy(dtype="int64")
        df.loc[target_index[active], "dir_05"] = (
            (center[active] + lower_offset[active]) % 360.0
        ).astype("float32")
        df.loc[target_index[active], "dir_95"] = (
            (center[active] + upper_offset[active]) % 360.0
        ).astype("float32")
    return df


def load_hres_frame(config) -> pd.DataFrame:
    """Load official HRES inputs, including organiser inference windows."""
    global _HRES_CACHE
    if _HRES_CACHE is not None:
        return _HRES_CACHE
    paths = list(config.hres_parquets())
    inference_root = config.inference_root()
    if inference_root is not None:
        paths.extend(sorted(inference_root.glob("window_*/context_hres_*.parquet")))
    if not paths:
        raise FileNotFoundError("No HRES parquet inputs were found")
    frame = pd.concat(
        [pd.read_parquet(path) for path in paths],
        ignore_index=True,
        sort=False,
    )
    frame["time"] = pd.to_datetime(frame["time"])
    _HRES_CACHE = frame.drop_duplicates(
        subset=["time", "latitude", "longitude"]
    ).reset_index(drop=True)
    return _HRES_CACHE


def _analysis_grid_index(source_lat, source_lon, lat, lon) -> np.ndarray:
    source_key = {
        (round(la, 3), round(lo, 3)): i
        for i, (la, lo) in enumerate(zip(source_lat, source_lon))
    }
    return np.array(
        [source_key.get((round(la, 3), round(lo, 3)), -1) for la, lo in zip(lat, lon)],
        dtype="int32",
    )


def _aligned_analysis(source_lat, source_lon, source_u, source_v, lat, lon, index=None):
    if index is None:
        index = _analysis_grid_index(source_lat, source_lon, lat, lon)
    ok = index >= 0
    analysis_u = np.full(len(index), np.nan, dtype="float32")
    analysis_v = np.full(len(index), np.nan, dtype="float32")
    analysis_u[ok] = np.asarray(source_u, dtype="float32")[index[ok]]
    analysis_v[ok] = np.asarray(source_v, dtype="float32")[index[ok]]
    return analysis_u, analysis_v, index


def load_issue_analysis(config, issue, lat, lon) -> dict:
    """Load official issue-time u100/v100 for one inference window."""
    issue = pd.Timestamp(issue).normalize()
    cached = _ANALYSIS_CACHE.get(issue)
    if cached is not None:
        return cached

    training_path = (
        Path(config.reanalysis_root())
        / f"{issue.year}"
        / f"reanalysis_{issue:%Y%m%d}.nc"
    )
    fields = {}
    index = None
    if training_path.exists():
        import reanalysis_loader

        for hour in HOURS:
            snapshot = reanalysis_loader.load_reanalysis(
                issue.date(), hour, root=config.reanalysis_root()
            )
            source_lon, source_lat = np.meshgrid(snapshot.lons, snapshot.lats)
            analysis_u, analysis_v, index = _aligned_analysis(
                source_lat.ravel(),
                source_lon.ravel(),
                snapshot.u100.ravel(),
                snapshot.v100.ravel(),
                lat,
                lon,
                index=index,
            )
            fields[hour] = (analysis_u, analysis_v)
    else:
        inference_root = config.inference_root()
        if inference_root is None:
            raise FileNotFoundError(
                f"No reanalysis input found for issue date {issue.date()}"
            )
        context_path = None
        for path in sorted(inference_root.glob("window_*/context_reanalysis*.parquet")):
            times = pd.to_datetime(pd.read_parquet(path, columns=["time"])["time"])
            if ((times >= issue) & (times < issue + pd.Timedelta(days=1))).any():
                context_path = path
                break
        if context_path is None:
            raise FileNotFoundError(
                f"No organiser context reanalysis contains issue date {issue.date()}"
            )
        context = pd.read_parquet(
            context_path,
            columns=["time", "latitude", "longitude", "u100", "v100"],
        )
        context["time"] = pd.to_datetime(context["time"])
        for hour in HOURS:
            snapshot = context[context["time"] == issue + pd.Timedelta(hours=hour)]
            if snapshot.empty:
                raise ValueError(
                    f"Missing reanalysis hour={hour} for issue date {issue.date()}"
                )
            analysis_u, analysis_v, index = _aligned_analysis(
                snapshot["latitude"].to_numpy(),
                snapshot["longitude"].to_numpy(),
                snapshot["u100"].to_numpy(),
                snapshot["v100"].to_numpy(),
                lat,
                lon,
                index=index,
            )
            fields[hour] = (analysis_u, analysis_v)

    if any(np.isnan(values[0]).any() or np.isnan(values[1]).any() for values in fields.values()):
        raise ValueError(f"Reanalysis and HRES grids did not align for {issue.date()}")
    _ANALYSIS_CACHE[issue] = fields
    return fields


def build_hybrid_table(
    fh, config, issue_dates, with_analysis: bool = False
) -> pd.DataFrame:
    """Build d1/d7 features and d10-source features for the d14 target."""
    hres = load_hres_frame(config)
    blocks = []
    for issue in issue_dates:
        issue = pd.Timestamp(issue)
        hrow = hres[hres["time"] == issue]
        if hrow.empty:
            continue
        lat = hrow["latitude"].to_numpy()
        lon = hrow["longitude"].to_numpy()
        analysis = (
            load_issue_analysis(config, issue, lat, lon)
            if with_analysis
            else None
        )
        for lead in LEADS:
            source_lead = SOURCE_LEAD[lead]
            valid = issue + pd.Timedelta(days=lead)
            week = int(valid.isocalendar().week)
            for hour in HOURS:
                speed_col = f"fcst_speed_d{source_lead}_h{hour}"
                dir_col = f"fcst_dir_d{source_lead}_h{hour}"
                if speed_col not in hrow.columns or dir_col not in hrow.columns:
                    continue
                speed = hrow[speed_col].to_numpy(dtype="float64")
                direction = hrow[dir_col].to_numpy(dtype="float64")
                fcst_u, fcst_v = fh._uv_from_speed_dir(speed, direction)
                if lead == 1 and analysis is not None:
                    analysis_u, analysis_v = analysis[hour]
                else:
                    analysis_u = np.full(lat.shape, np.nan, dtype="float32")
                    analysis_v = np.full(lat.shape, np.nan, dtype="float32")
                blocks.append(
                    pd.DataFrame(
                        {
                            "issue_date": issue,
                            "lat": lat,
                            "lon": lon,
                            "lead": lead,
                            "hour": hour,
                            "fcst_u": fcst_u,
                            "fcst_v": fcst_v,
                            "fcst_speed": speed,
                            "woy_sin": np.sin(2 * np.pi * week / 52.0),
                            "woy_cos": np.cos(2 * np.pi * week / 52.0),
                            "u125c": np.nan,
                            "v125c": np.nan,
                            "analysis_u": analysis_u,
                            "analysis_v": analysis_v,
                        }
                    )
                )
    if not blocks:
        raise ValueError(f"No HRES rows matched issue dates: {list(issue_dates)}")
    table = pd.concat(blocks, ignore_index=True)
    table = table.dropna(subset=["fcst_u", "fcst_v", "fcst_speed"]).reset_index(drop=True)
    return table


def engineered_features(fh, table: pd.DataFrame) -> pd.DataFrame:
    features = table[fh.FEATURES].copy()
    hour = table["hour"].to_numpy(dtype="float32")
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    features["speed_sq"] = table["fcst_speed"].to_numpy(dtype="float32") ** 2
    features["uv_cross"] = (
        table["fcst_u"].to_numpy(dtype="float32")
        * table["fcst_v"].to_numpy(dtype="float32")
    )
    return features.astype("float32")


def augmented_direction_features(fh, table: pd.DataFrame) -> pd.DataFrame:
    features = engineered_features(fh, table)
    analysis_u = table["analysis_u"].to_numpy(dtype="float32")
    analysis_v = table["analysis_v"].to_numpy(dtype="float32")
    forecast_u = table["fcst_u"].to_numpy(dtype="float32")
    forecast_v = table["fcst_v"].to_numpy(dtype="float32")
    analysis_speed = np.hypot(analysis_u, analysis_v)
    denom = np.maximum(
        analysis_speed * table["fcst_speed"].to_numpy(dtype="float32"), 0.25
    )
    features["analysis_u"] = analysis_u
    features["analysis_v"] = analysis_v
    features["analysis_speed"] = analysis_speed
    features["analysis_du"] = analysis_u - forecast_u
    features["analysis_dv"] = analysis_v - forecast_v
    features["analysis_dot"] = (
        analysis_u * forecast_u + analysis_v * forecast_v
    ) / denom
    features["analysis_cross"] = (
        analysis_u * forecast_v - analysis_v * forecast_u
    ) / denom
    return features.astype("float32")


def add_normalized_flow_features(
    frame: pd.DataFrame,
    prefix: str,
    u: np.ndarray,
    v: np.ndarray,
    forecast_u: np.ndarray,
    forecast_v: np.ndarray,
    forecast_speed: np.ndarray,
) -> None:
    speed = np.hypot(u, v)
    denom = np.maximum(speed * forecast_speed, 0.25)
    frame[f"{prefix}_u"] = np.asarray(u, dtype="float32")
    frame[f"{prefix}_v"] = np.asarray(v, dtype="float32")
    frame[f"{prefix}_speed"] = speed.astype("float32")
    frame[f"{prefix}_dot"] = (
        (u * forecast_u + v * forecast_v) / denom
    ).astype("float32")
    frame[f"{prefix}_cross"] = (
        (u * forecast_v - v * forecast_u) / denom
    ).astype("float32")


def add_lagged_context_features(
    config,
    fh,
    issue_date,
    table: pd.DataFrame,
) -> pd.DataFrame:
    issue_date = pd.Timestamp(issue_date).normalize()
    table = table.reset_index(drop=True)
    if set(table["lead"].unique()) != {1}:
        raise ValueError("Lagged context features require a d1-only table")

    base = engineered_features(fh, table).reset_index(drop=True)
    forecast_u = table["fcst_u"].to_numpy(dtype="float32")
    forecast_v = table["fcst_v"].to_numpy(dtype="float32")
    forecast_speed = table["fcst_speed"].to_numpy(dtype="float32")
    hour_index = (table["hour"].to_numpy(dtype="int16") // 6).astype("int8")

    grid = table[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    grid_lookup = {
        (round(float(lat), 3), round(float(lon), 3)): index
        for index, (lat, lon) in enumerate(zip(grid["lat"], grid["lon"]))
    }
    grid_index = np.asarray(
        [
            grid_lookup[(round(float(lat), 3), round(float(lon), 3))]
            for lat, lon in zip(table["lat"], table["lon"])
        ],
        dtype="int32",
    )

    u_days = []
    v_days = []
    for lag in range(14):
        fields = load_issue_analysis(
            config,
            issue_date - pd.Timedelta(days=lag),
            grid["lat"].to_numpy(),
            grid["lon"].to_numpy(),
        )
        u_days.append(np.stack([fields[hour][0] for hour in HOURS]))
        v_days.append(np.stack([fields[hour][1] for hour in HOURS]))
    u_hist = np.stack(u_days)
    v_hist = np.stack(v_days)

    for lag in CONTEXT_LAGS:
        u = u_hist[lag, hour_index, grid_index]
        v = v_hist[lag, hour_index, grid_index]
        add_normalized_flow_features(
            base,
            f"same_h_lag{lag}",
            u,
            v,
            forecast_u,
            forecast_v,
            forecast_speed,
        )

    for source_hour_index, source_hour in enumerate(HOURS):
        u = u_hist[0, source_hour_index, grid_index]
        v = v_hist[0, source_hour_index, grid_index]
        base[f"issue_h{source_hour}_u"] = u.astype("float32")
        base[f"issue_h{source_hour}_v"] = v.astype("float32")
        base[f"issue_h{source_hour}_speed"] = np.hypot(u, v).astype("float32")

    for days in (3, 7, 14):
        u_same = np.mean(u_hist[:days, hour_index, grid_index], axis=0)
        v_same = np.mean(v_hist[:days, hour_index, grid_index], axis=0)
        add_normalized_flow_features(
            base,
            f"same_h_mean{days}",
            u_same,
            v_same,
            forecast_u,
            forecast_v,
            forecast_speed,
        )
        mean_speed = np.mean(
            np.hypot(
                u_hist[:days, hour_index, grid_index],
                v_hist[:days, hour_index, grid_index],
            ),
            axis=0,
        )
        base[f"same_h_concentration{days}"] = (
            np.hypot(u_same, v_same) / np.maximum(mean_speed, 0.1)
        ).astype("float32")

    latest_u = u_hist[0, 3, grid_index]
    latest_v = v_hist[0, 3, grid_index]
    add_normalized_flow_features(
        base,
        "latest_h18",
        latest_u,
        latest_v,
        forecast_u,
        forecast_v,
        forecast_speed,
    )
    base["latest_age_hours"] = (6 + table["hour"].to_numpy()).astype("float32")
    base["forecast_minus_latest_u"] = (forecast_u - latest_u).astype("float32")
    base["forecast_minus_latest_v"] = (forecast_v - latest_v).astype("float32")

    current_u = u_hist[0, hour_index, grid_index]
    current_v = v_hist[0, hour_index, grid_index]
    for lag in (1, 3, 7, 13):
        base[f"tendency{lag}_u"] = (
            current_u - u_hist[lag, hour_index, grid_index]
        ).astype("float32")
        base[f"tendency{lag}_v"] = (
            current_v - v_hist[lag, hour_index, grid_index]
        ).astype("float32")

    base["issue_h18_du6"] = (
        u_hist[0, 3, grid_index] - u_hist[0, 2, grid_index]
    ).astype("float32")
    base["issue_h18_dv6"] = (
        v_hist[0, 3, grid_index] - v_hist[0, 2, grid_index]
    ).astype("float32")
    base["issue_h12_du6"] = (
        u_hist[0, 2, grid_index] - u_hist[0, 1, grid_index]
    ).astype("float32")
    base["issue_h12_dv6"] = (
        v_hist[0, 2, grid_index] - v_hist[0, 1, grid_index]
    ).astype("float32")
    values = base.to_numpy(dtype="float32")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite lagged context features for {issue_date.date()}")
    return base.astype("float32")


def _load_d7_context_snapshot(config, day, hour, lat, lon, index=None):
    """Load aligned u/v at 10 m and 100 m from training or inference context."""
    day = pd.Timestamp(day).normalize()
    training_path = (
        Path(config.reanalysis_root())
        / f"{day.year}"
        / f"reanalysis_{day:%Y%m%d}.nc"
    )
    if training_path.exists():
        import reanalysis_loader

        snapshot = reanalysis_loader.load_reanalysis(
            day.date(), hour, root=config.reanalysis_root()
        )
        source_lon, source_lat = np.meshgrid(snapshot.lons, snapshot.lats)
        source_lat = source_lat.ravel()
        source_lon = source_lon.ravel()
        u100 = np.asarray(snapshot.u100, dtype="float32").ravel()
        v100 = np.asarray(snapshot.v100, dtype="float32").ravel()
        u10 = np.asarray(snapshot.u10, dtype="float32").ravel()
        v10 = np.asarray(snapshot.v10, dtype="float32").ravel()
    else:
        inference_root = config.inference_root()
        if inference_root is None:
            raise FileNotFoundError(f"No context reanalysis for {day.date()}")
        target_time = day + pd.Timedelta(hours=hour)
        frame = None
        for path in sorted(
            inference_root.glob("window_*/context_reanalysis*.parquet")
        ):
            cached = _CONTEXT_PARQUET_CACHE.get(path)
            if cached is None:
                cached = pd.read_parquet(
                    path,
                    columns=[
                        "time", "latitude", "longitude", "u10", "v10", "u100", "v100"
                    ],
                )
                cached["time"] = pd.to_datetime(cached["time"])
                _CONTEXT_PARQUET_CACHE[path] = cached
            selected = cached[cached["time"] == target_time]
            if not selected.empty:
                frame = selected
                break
        if frame is None:
            raise FileNotFoundError(
                f"No organiser context snapshot for {target_time}"
            )
        source_lat = frame["latitude"].to_numpy(dtype="float32")
        source_lon = frame["longitude"].to_numpy(dtype="float32")
        u100 = frame["u100"].to_numpy(dtype="float32")
        v100 = frame["v100"].to_numpy(dtype="float32")
        u10 = frame["u10"].to_numpy(dtype="float32")
        v10 = frame["v10"].to_numpy(dtype="float32")
    aligned_u100, aligned_v100, index = _aligned_analysis(
        source_lat, source_lon, u100, v100, lat, lon, index=index
    )
    aligned_u10, aligned_v10, _ = _aligned_analysis(
        source_lat, source_lon, u10, v10, lat, lon, index=index
    )
    return aligned_u100, aligned_v100, aligned_u10, aligned_v10, index


def d1_dense_daily_feature_names() -> list[str]:
    names = ["hres_u", "hres_v", "hres_speed"]
    names.extend(f"hres_speed_h{hour}" for hour in HOURS)
    names.extend(
        (
            "ctx_u10",
            "ctx_v10",
            "ctx_u100",
            "ctx_v100",
            "ctx_speed10",
            "ctx_speed100",
            "ctx_shear",
        )
    )
    for lag in D1_DENSE_DAILY_LAGS[1:]:
        names.extend((f"ctx_u100_lag{lag}", f"ctx_v100_lag{lag}"))
    for days in D1_DENSE_DAILY_MEANS:
        names.extend(
            (
                f"ctx_u100_mean{days}",
                f"ctx_v100_mean{days}",
                f"ctx_concentration{days}",
            )
        )
    names.extend(("forecast_du", "forecast_dv", "forecast_dot", "forecast_cross"))
    names.extend(
        ("latitude", "longitude", "season_sin", "season_cos", "hour_sin", "hour_cos")
    )
    return names


def add_d1_dense_daily_features(config, issue_date, table: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the exact organizer-only dense-daily d1 feature contract."""
    issue_date = pd.Timestamp(issue_date).normalize()
    table = table.reset_index(drop=True)
    if set(table["lead"].unique()) != {1}:
        raise ValueError("Dense-daily features require a d1-only table")

    grid = table[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    lookup = {
        (round(float(lat), 3), round(float(lon), 3)): index
        for index, (lat, lon) in enumerate(zip(grid["lat"], grid["lon"]))
    }
    grid_index = np.asarray(
        [
            lookup[(round(float(lat), 3), round(float(lon), 3))]
            for lat, lon in zip(table["lat"], table["lon"])
        ],
        dtype="int32",
    )
    hour_index = (table["hour"].to_numpy(dtype="int16") // 6).astype("int8")
    forecast_u = table["fcst_u"].to_numpy(dtype="float32")
    forecast_v = table["fcst_v"].to_numpy(dtype="float32")
    forecast_speed = table["fcst_speed"].to_numpy(dtype="float32")

    hres = np.full((len(HOURS), len(grid), 3), np.nan, dtype="float32")
    hres[hour_index, grid_index, 0] = forecast_u
    hres[hour_index, grid_index, 1] = forecast_v
    hres[hour_index, grid_index, 2] = forecast_speed

    u100 = np.empty((14, len(HOURS), len(grid)), dtype="float32")
    v100 = np.empty_like(u100)
    u10 = np.empty_like(u100)
    v10 = np.empty_like(u100)
    source_index = None
    lat = grid["lat"].to_numpy(dtype="float32")
    lon = grid["lon"].to_numpy(dtype="float32")
    for lag in range(14):
        day = issue_date - pd.Timedelta(days=lag)
        for source_hour_index, source_hour in enumerate(HOURS):
            (
                aligned_u100,
                aligned_v100,
                aligned_u10,
                aligned_v10,
                source_index,
            ) = _load_d7_context_snapshot(
                config,
                day,
                source_hour,
                lat,
                lon,
                index=source_index,
            )
            if (source_index < 0).any():
                raise ValueError("Could not align dense-daily context to HRES")
            u100[lag, source_hour_index] = aligned_u100
            v100[lag, source_hour_index] = aligned_v100
            u10[lag, source_hour_index] = aligned_u10
            v10[lag, source_hour_index] = aligned_v10

    current_u100 = u100[0, hour_index, grid_index]
    current_v100 = v100[0, hour_index, grid_index]
    current_u10 = u10[0, hour_index, grid_index]
    current_v10 = v10[0, hour_index, grid_index]
    speed100 = np.hypot(current_u100, current_v100)
    speed10 = np.hypot(current_u10, current_v10)
    denominator = np.maximum(forecast_speed * speed100, 0.25)

    columns = {
        "hres_u": forecast_u,
        "hres_v": forecast_v,
        "hres_speed": forecast_speed,
    }
    for source_hour_index, source_hour in enumerate(HOURS):
        columns[f"hres_speed_h{source_hour}"] = hres[
            source_hour_index, grid_index, 2
        ]
    columns.update(
        {
            "ctx_u10": current_u10,
            "ctx_v10": current_v10,
            "ctx_u100": current_u100,
            "ctx_v100": current_v100,
            "ctx_speed10": speed10,
            "ctx_speed100": speed100,
            "ctx_shear": speed100 - speed10,
        }
    )
    for lag in D1_DENSE_DAILY_LAGS[1:]:
        columns[f"ctx_u100_lag{lag}"] = u100[lag, hour_index, grid_index]
        columns[f"ctx_v100_lag{lag}"] = v100[lag, hour_index, grid_index]
    for days in D1_DENSE_DAILY_MEANS:
        mean_u = np.mean(u100[:days, hour_index, grid_index], axis=0)
        mean_v = np.mean(v100[:days, hour_index, grid_index], axis=0)
        mean_speed = np.mean(
            np.hypot(
                u100[:days, hour_index, grid_index],
                v100[:days, hour_index, grid_index],
            ),
            axis=0,
        )
        columns[f"ctx_u100_mean{days}"] = mean_u
        columns[f"ctx_v100_mean{days}"] = mean_v
        columns[f"ctx_concentration{days}"] = (
            np.hypot(mean_u, mean_v) / np.maximum(mean_speed, 0.1)
        )
    columns.update(
        {
            "forecast_du": forecast_u - current_u100,
            "forecast_dv": forecast_v - current_v100,
            "forecast_dot": (
                forecast_u * current_u100 + forecast_v * current_v100
            )
            / denominator,
            "forecast_cross": (
                forecast_u * current_v100 - forecast_v * current_u100
            )
            / denominator,
            "latitude": table["lat"].to_numpy(dtype="float32"),
            "longitude": table["lon"].to_numpy(dtype="float32"),
        }
    )
    season = 2.0 * np.pi * float(issue_date.dayofyear) / 365.2425
    hour_angle = 2.0 * np.pi * table["hour"].to_numpy(dtype="float32") / 24.0
    columns["season_sin"] = np.full(len(table), np.sin(season), dtype="float32")
    columns["season_cos"] = np.full(len(table), np.cos(season), dtype="float32")
    columns["hour_sin"] = np.sin(hour_angle)
    columns["hour_cos"] = np.cos(hour_angle)
    names = d1_dense_daily_feature_names()
    features = pd.DataFrame({name: columns[name] for name in names}).astype("float32")
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError(f"Non-finite dense-daily d1 features for {issue_date.date()}")
    return features


def add_d7_speed_context_features(
    config,
    fh,
    issue_date,
    table: pd.DataFrame,
) -> pd.DataFrame:
    """Rebuild the exact compact d7 endpoint features used during training."""
    issue_date = pd.Timestamp(issue_date).normalize()
    table = table.reset_index(drop=True)
    if set(table["lead"].unique()) != {7}:
        raise ValueError("d7 speed context features require a d7-only table")

    features = engineered_features(fh, table).reset_index(drop=True)
    grid = table[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    lookup = {
        (round(float(lat), 3), round(float(lon), 3)): index
        for index, (lat, lon) in enumerate(zip(grid["lat"], grid["lon"]))
    }
    grid_index = np.asarray(
        [
            lookup[(round(float(lat), 3), round(float(lon), 3))]
            for lat, lon in zip(table["lat"], table["lon"])
        ],
        dtype="int32",
    )
    hour_index = (table["hour"].to_numpy(dtype="int16") // 6).astype("int8")

    hres = np.full((len(HOURS), len(grid), 3), np.nan, dtype="float32")
    hres[hour_index, grid_index, 0] = table["fcst_u"].to_numpy(dtype="float32")
    hres[hour_index, grid_index, 1] = table["fcst_v"].to_numpy(dtype="float32")
    hres[hour_index, grid_index, 2] = table["fcst_speed"].to_numpy(dtype="float32")
    for source_index, source_hour in enumerate(HOURS):
        values = hres[source_index, grid_index]
        features[f"same_lead_h{source_hour}_u"] = values[:, 0]
        features[f"same_lead_h{source_hour}_v"] = values[:, 1]
        features[f"same_lead_h{source_hour}_speed"] = values[:, 2]
    speed_hours = hres[:, grid_index, 2].T
    u_hours = hres[:, grid_index, 0].T
    v_hours = hres[:, grid_index, 1].T
    features["hres_hour_speed_std"] = np.std(speed_hours, axis=1)
    features["hres_hour_vector_concentration"] = np.hypot(
        np.mean(u_hours, axis=1), np.mean(v_hours, axis=1)
    ) / np.maximum(np.mean(speed_hours, axis=1), 0.1)
    features["hres_du_18_0"] = u_hours[:, 3] - u_hours[:, 0]
    features["hres_dv_18_0"] = v_hours[:, 3] - v_hours[:, 0]

    u100 = np.empty(
        (len(D7_SPEED_CONTEXT_LAGS), len(HOURS), len(grid)), dtype="float32"
    )
    v100 = np.empty_like(u100)
    u10 = np.empty_like(u100)
    v10 = np.empty_like(u100)
    source_index = None
    lat = grid["lat"].to_numpy()
    lon = grid["lon"].to_numpy()
    for lag_index, lag in enumerate(D7_SPEED_CONTEXT_LAGS):
        day = issue_date - pd.Timedelta(days=lag)
        for source_hour_index, source_hour in enumerate(HOURS):
            (
                aligned_u100,
                aligned_v100,
                aligned_u10,
                aligned_v10,
                source_index,
            ) = _load_d7_context_snapshot(
                config, day, source_hour, lat, lon, index=source_index
            )
            u100[lag_index, source_hour_index] = aligned_u100
            v100[lag_index, source_hour_index] = aligned_v100
            u10[lag_index, source_hour_index] = aligned_u10
            v10[lag_index, source_hour_index] = aligned_v10

    current_u = u100[0, hour_index, grid_index]
    current_v = v100[0, hour_index, grid_index]
    current_speed = np.hypot(current_u, current_v)
    current_speed10 = np.hypot(
        u10[0, hour_index, grid_index], v10[0, hour_index, grid_index]
    )
    forecast_u = table["fcst_u"].to_numpy(dtype="float32")
    forecast_v = table["fcst_v"].to_numpy(dtype="float32")
    forecast_speed = table["fcst_speed"].to_numpy(dtype="float32")
    denominator = np.maximum(current_speed * forecast_speed, 0.25)
    features["ctx_shear_speed"] = current_speed - current_speed10
    features["ctx_shear_ratio"] = current_speed / np.maximum(current_speed10, 0.25)
    features["ctx_forecast_dot"] = (
        current_u * forecast_u + current_v * forecast_v
    ) / denominator
    features["ctx_forecast_cross"] = (
        current_u * forecast_v - current_v * forecast_u
    ) / denominator
    features["ctx_forecast_du"] = forecast_u - current_u
    features["ctx_forecast_dv"] = forecast_v - current_v
    for lag_index, lag in enumerate(D7_SPEED_CONTEXT_LAGS[1:], start=1):
        features[f"ctx_tendency{lag}_u"] = (
            current_u - u100[lag_index, hour_index, grid_index]
        )
        features[f"ctx_tendency{lag}_v"] = (
            current_v - v100[lag_index, hour_index, grid_index]
        )
    values = features.to_numpy(dtype="float32")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite d7 speed context for {issue_date.date()}")
    return features.astype("float32")


def circular_blend(a, b, weight: float) -> np.ndarray:
    a_rad = np.radians(np.asarray(a, dtype="float64"))
    b_rad = np.radians(np.asarray(b, dtype="float64"))
    y = (1.0 - weight) * np.sin(a_rad) + weight * np.sin(b_rad)
    x = (1.0 - weight) * np.cos(a_rad) + weight * np.cos(b_rad)
    return np.degrees(np.arctan2(y, x)) % 360.0


def _spatial_box_mean(values: np.ndarray, radius: int) -> np.ndarray:
    values = np.asarray(values, dtype="float32")
    padded = np.pad(values, radius, mode="edge")
    total = np.zeros_like(values, dtype="float64")
    count = np.zeros_like(values, dtype="float64")
    height, width = values.shape
    for row_shift in range(2 * radius + 1):
        for col_shift in range(2 * radius + 1):
            part = padded[
                row_shift : row_shift + height,
                col_shift : col_shift + width,
            ]
            valid = np.isfinite(part)
            total += np.where(valid, part, 0.0)
            count += valid
    return (total / np.maximum(count, 1.0)).astype("float32")


def shared_spatial_direction_features(
    table: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Rebuild the train-time multi-scale d7 vector features exactly."""
    subset = table.loc[table["lead"] == 7].copy()
    if subset.empty:
        return np.empty(0, dtype="int64"), pd.DataFrame(dtype="float32")
    subset["_source_index"] = subset.index.to_numpy(dtype="int64")
    subset = subset.reset_index(drop=True)
    latitude_values = np.sort(subset["lat"].unique())
    longitude_values = np.sort(subset["lon"].unique())
    latitude_index = np.searchsorted(
        latitude_values, subset["lat"].to_numpy()
    )
    longitude_index = np.searchsorted(
        longitude_values, subset["lon"].to_numpy()
    )
    shape = (len(latitude_values), len(longitude_values))
    n_rows = len(subset)
    spatial = {
        name: np.empty(n_rows, dtype="float32")
        for name in (
            "u_mean3", "v_mean3", "u_mean5", "v_mean5",
            "u_grad_lat", "u_grad_lon", "v_grad_lat", "v_grad_lon",
            "u_laplacian", "v_laplacian",
        )
    }
    grouped = subset.groupby(["issue_date", "hour"], sort=False).indices
    raw_u = subset["fcst_u"].to_numpy(dtype="float32")
    raw_v = subset["fcst_v"].to_numpy(dtype="float32")
    for positions in grouped.values():
        positions = np.asarray(positions, dtype="int64")
        rows = latitude_index[positions]
        cols = longitude_index[positions]
        u_grid = np.full(shape, np.nan, dtype="float32")
        v_grid = np.full(shape, np.nan, dtype="float32")
        u_grid[rows, cols] = raw_u[positions]
        v_grid[rows, cols] = raw_v[positions]
        u3 = _spatial_box_mean(u_grid, 1)
        v3 = _spatial_box_mean(v_grid, 1)
        u5 = _spatial_box_mean(u_grid, 2)
        v5 = _spatial_box_mean(v_grid, 2)
        u_grad_lat, u_grad_lon = np.gradient(u3)
        v_grad_lat, v_grad_lon = np.gradient(v3)
        spatial["u_mean3"][positions] = u3[rows, cols]
        spatial["v_mean3"][positions] = v3[rows, cols]
        spatial["u_mean5"][positions] = u5[rows, cols]
        spatial["v_mean5"][positions] = v5[rows, cols]
        spatial["u_grad_lat"][positions] = u_grad_lat[rows, cols]
        spatial["u_grad_lon"][positions] = u_grad_lon[rows, cols]
        spatial["v_grad_lat"][positions] = v_grad_lat[rows, cols]
        spatial["v_grad_lon"][positions] = v_grad_lon[rows, cols]
        spatial["u_laplacian"][positions] = (
            u3[rows, cols] - raw_u[positions]
        )
        spatial["v_laplacian"][positions] = (
            v3[rows, cols] - raw_v[positions]
        )
    hours = subset["hour"].to_numpy(dtype="float32")
    valid_dates = pd.to_datetime(subset["issue_date"]) + pd.Timedelta(days=7)
    weeks = valid_dates.dt.isocalendar().week.to_numpy(dtype="float32")
    speed = np.hypot(raw_u, raw_v).astype("float32")
    direction = np.arctan2(-raw_u, -raw_v)
    features = pd.DataFrame(
        {
            "lat": subset["lat"].to_numpy(dtype="float32"),
            "lon": subset["lon"].to_numpy(dtype="float32"),
            "fcst_u": raw_u,
            "fcst_v": raw_v,
            "fcst_speed": speed,
            "speed_sq": speed ** 2,
            "raw_dir_sin": np.sin(direction),
            "raw_dir_cos": np.cos(direction),
            "hour_sin": np.sin(2.0 * np.pi * hours / 24.0),
            "hour_cos": np.cos(2.0 * np.pi * hours / 24.0),
            "woy_sin": np.sin(2.0 * np.pi * weeks / 52.0),
            "woy_cos": np.cos(2.0 * np.pi * weeks / 52.0),
            **spatial,
        }
    ).astype("float32")
    if not np.isfinite(features.to_numpy()).all():
        raise RuntimeError("Non-finite shared spatial direction features")
    return subset["_source_index"].to_numpy(dtype="int64"), features


def _shared_spatial_prediction(model: dict, x: np.ndarray):
    normalized = (
        (x - model["mean"]) / np.maximum(model["scale"], 1e-5)
    ).astype("float32")
    vector = model["model"].predict(normalized)
    correction = np.degrees(np.arctan2(vector[:, 0], vector[:, 1]))
    confidence = np.hypot(vector[:, 0], vector[:, 1])
    return correction, confidence


def apply_shared_spatial_direction_model(
    models: dict,
    table: pd.DataFrame,
    centers,
) -> np.ndarray:
    payload = models.get("shared_spatial_direction")
    result = np.asarray(centers, dtype="float64").copy()
    if not payload or not payload.get("gate", {}).get("passed", False):
        return result
    source_indices, features = shared_spatial_direction_features(table)
    if not len(source_indices):
        return result
    expected = payload.get("features")
    if expected is not None:
        features = features[expected]
    correction, confidence = _shared_spatial_prediction(
        payload, features.to_numpy(dtype="float32")
    )
    subset = table.loc[source_indices]
    dates = pd.to_datetime(subset["issue_date"])
    hours = subset["hour"].to_numpy(dtype="int16")
    selected_slots = {
        tuple(map(int, slot)) for slot in payload.get("selected_slots", ())
    }
    active = np.asarray(
        [
            (date.month, date.day, int(hour)) in selected_slots
            for date, hour in zip(dates, hours)
        ],
        dtype=bool,
    )
    active &= confidence >= float(payload["minimum_confidence"])
    if active.any():
        target = source_indices[active]
        result[target] = (
            result[target] + float(payload["blend"]) * correction[active]
        ) % 360.0
    return result


def apply_d7_direction_center_policy(
    fh,
    models: dict,
    table: pd.DataFrame,
    raw_direction,
    centers,
) -> np.ndarray:
    """Apply the trained zero-model d7 center policy from the artifact."""
    policy = models.get("d7_center_policy", {})
    result = np.asarray(centers, dtype="float64").copy()
    if not policy or not (table["lead"].to_numpy() == 7).any():
        return result

    raw = np.asarray(raw_direction, dtype="float64")
    dates = pd.to_datetime(table["issue_date"]).dt.normalize()
    leads = table["lead"].to_numpy()
    hours = table["hour"].to_numpy()
    latitudes = table["lat"].to_numpy(dtype="float64")
    longitudes = table["lon"].to_numpy(dtype="float64")

    for issue_date in pd.DatetimeIndex(dates.unique()):
        rules = [
            (hour, family, float(weight))
            for (month, day, hour), (family, weight) in policy.items()
            if month == issue_date.month and day == issue_date.day
        ]
        if not rules:
            continue
        date_mask = (dates == issue_date).to_numpy()
        climatology_hours = tuple(
            sorted({hour for hour, family, _ in rules if family == "climatology"})
        )
        climatology = (
            fh.build_climatology_forecast(
                [issue_date],
                hours=climatology_hours,
                lead=7,
                with_truth=False,
            )
            if climatology_hours
            else None
        )
        for hour, family, weight in rules:
            target_mask = date_mask & (leads == 7) & (hours == hour)
            target_indices = np.flatnonzero(target_mask)
            if not len(target_indices):
                continue
            if family == "d1":
                source_mask = date_mask & (leads == 1) & (hours == hour)
                source_frame = table.loc[source_mask, ["lat", "lon"]]
                source_values = raw[source_mask]
            elif family == "climatology":
                source_frame = climatology.loc[
                    climatology["hour"] == hour, ["lat", "lon"]
                ]
                source_values = (
                    np.degrees(
                        np.arctan2(
                            -climatology.loc[climatology["hour"] == hour, "u_pred"],
                            -climatology.loc[climatology["hour"] == hour, "v_pred"],
                        )
                    )
                    % 360.0
                ).to_numpy(dtype="float64")
            else:
                raise ValueError(f"Unknown d7 direction center family: {family}")

            lookup = {
                (round(float(lat), 3), round(float(lon), 3)): float(value)
                for lat, lon, value in zip(
                    source_frame["lat"], source_frame["lon"], source_values
                )
            }
            signal = np.asarray(
                [
                    lookup.get(
                        (
                            round(float(latitudes[index]), 3),
                            round(float(longitudes[index]), 3),
                        ),
                        np.nan,
                    )
                    for index in target_indices
                ],
                dtype="float64",
            )
            if not np.isfinite(signal).all():
                raise ValueError(
                    f"Could not align d7 {family} signal for "
                    f"{issue_date.date()} hour={hour}"
                )
            result[target_indices] = circular_blend(
                result[target_indices], signal, weight
            )
    return result


def apply_d14_direction_signal(base, source_climatology, raw_d7, family, weight):
    """Transfer a gated d7 signal onto the d14 climatological center."""
    if family == "direct":
        return circular_blend(base, raw_d7, weight)
    if family != "vector":
        raise ValueError(f"Unknown d14 direction family: {family}")
    base_rad = np.radians(np.asarray(base, dtype="float64"))
    source_rad = np.radians(np.asarray(source_climatology, dtype="float64"))
    raw_rad = np.radians(np.asarray(raw_d7, dtype="float64"))
    u = -np.sin(base_rad) + weight * (-np.sin(raw_rad) + np.sin(source_rad))
    v = -np.cos(base_rad) + weight * (-np.cos(raw_rad) + np.cos(source_rad))
    return np.degrees(np.arctan2(-u, -v)) % 360.0


def predict_quantiles(
    fh,
    models: dict,
    table: pd.DataFrame,
    adjust=None,
    config=None,
    d7_speed_context=None,
) -> pd.DataFrame:
    out = table.copy()
    columns = [f"spd_q{int(round(q * 100)):02d}" for q in QUANTILES]
    for column in columns:
        out[column] = np.nan
    for lead in LEADS:
        mask = out["lead"] == lead
        if not mask.any():
            continue
        x = engineered_features(fh, out.loc[mask])
        for quantile, column in zip(QUANTILES, columns):
            out.loc[mask, column] = models[(lead, quantile)].predict(x)
        if lead == 7 and d7_speed_context is not None:
            if config is None:
                raise ValueError("d7 speed context prediction requires config")
            d7_table = out.loc[mask].reset_index(drop=True)
            issue_dates = pd.to_datetime(d7_table["issue_date"]).dt.normalize().unique()
            if len(issue_dates) != 1:
                raise ValueError("d7 speed context expects one issue date")
            selected_slots = {
                tuple(map(int, slot))
                for slot in d7_speed_context.get("selected_slots", ())
            }
            issue_date = pd.Timestamp(issue_dates[0])
            active = np.asarray(
                [
                    (issue_date.month, issue_date.day, int(hour)) in selected_slots
                    for hour in d7_table["hour"].to_numpy()
                ],
                dtype=bool,
            )
            if active.any():
                context_x = add_d7_speed_context_features(
                    config,
                    fh,
                    issue_date,
                    d7_table,
                )
                context_x = context_x[d7_speed_context["features"]]
                lower_candidate = d7_speed_context["models"][0.05].predict(context_x)
                upper_candidate = d7_speed_context["models"][0.95].predict(context_x)
                lower_base = out.loc[mask, columns[0]].to_numpy(dtype="float64")
                upper_base = out.loc[mask, columns[-1]].to_numpy(dtype="float64")
                lower = lower_base.copy()
                upper = upper_base.copy()
                lower[active] += float(d7_speed_context["lower_blend"]) * (
                    lower_candidate[active] - lower_base[active]
                )
                upper[active] += float(d7_speed_context["upper_blend"]) * (
                    upper_candidate[active] - upper_base[active]
                )
                out.loc[mask, columns[0]] = lower
                out.loc[mask, columns[-1]] = upper
        if adjust and lead in adjust:
            out.loc[mask, columns[0]] -= adjust[lead]
            out.loc[mask, columns[-1]] += adjust[lead]
    values = np.sort(out[columns].to_numpy(dtype="float64"), axis=1)
    out[columns] = np.clip(values, 0.0, None)
    return out


def predict_direction_centers(
    fh,
    models: dict,
    table: pd.DataFrame,
    config=None,
) -> np.ndarray:
    base = np.degrees(np.arctan2(-table["fcst_u"], -table["fcst_v"])) % 360
    result = base.to_numpy(dtype="float64")
    mask = (table["lead"] == 1).to_numpy()
    if mask.any():
        x = engineered_features(fh, table.loc[mask])
        base_models = models.get("base", models)
        residual = np.degrees(
            np.arctan2(
                base_models["sin"].predict(x), base_models["cos"].predict(x)
            )
        )
        baseline_center = (result[mask] + residual) % 360
        result[mask] = baseline_center
        analysis_models = models.get("analysis")
        if analysis_models is not None:
            analysis_x = augmented_direction_features(fh, table.loc[mask])
            analysis_residual = np.degrees(
                np.arctan2(
                    analysis_models["sin"].predict(analysis_x),
                    analysis_models["cos"].predict(analysis_x),
                )
            )
            analysis_center = (base.to_numpy(dtype="float64")[mask] + analysis_residual) % 360
            result[mask] = circular_blend(
                baseline_center,
                analysis_center,
                float(models.get("analysis_blend", ANALYSIS_BLEND)),
            )
        context_models = models.get("context")
        if context_models is not None:
            d1 = table.loc[mask].reset_index(drop=True)
            selected = np.zeros(len(d1), dtype=bool)
            dates = pd.to_datetime(d1["issue_date"])
            for month, day, hour in models.get(
                "context_slots", CONTEXT_SELECTED_SLOTS
            ):
                selected |= (
                    (dates.dt.month.to_numpy() == month)
                    & (dates.dt.day.to_numpy() == day)
                    & (d1["hour"].to_numpy() == hour)
                )
            if selected.any():
                if config is None:
                    raise ValueError("Lagged context direction model requires config")
                unique_dates = dates.dt.normalize().unique()
                if len(unique_dates) != 1:
                    raise ValueError("Context inference expects one issue date at a time")
                context_x = add_lagged_context_features(
                    config,
                    fh,
                    pd.Timestamp(unique_dates[0]),
                    d1,
                )
                expected = models.get("context_features")
                if expected is not None:
                    context_x = context_x[expected]
                context_residual = np.degrees(
                    np.arctan2(
                        context_models["sin"].predict(context_x.loc[selected]),
                        context_models["cos"].predict(context_x.loc[selected]),
                    )
                )
                raw = base.to_numpy(dtype="float64")[mask]
                context_center = (raw[selected] + context_residual) % 360.0
                result_indices = np.flatnonzero(mask)[selected]
                result[result_indices] = circular_blend(
                    baseline_center[selected],
                    context_center,
                    float(models.get("context_blend", CONTEXT_BLEND)),
                )
    result = apply_d7_direction_center_policy(
        fh,
        models,
        table,
        base.to_numpy(dtype="float64"),
        result,
    )
    return apply_shared_spatial_direction_model(models, table, result)


def qmos_refit_movement_summary(base: pd.DataFrame, candidate: pd.DataFrame) -> dict:
    columns = ("spd_q05", "spd_q50", "spd_q95")
    old = base.loc[:, columns].to_numpy(dtype="float64")
    new = candidate.loc[:, columns].to_numpy(dtype="float64")
    width = np.maximum(old[:, 2] - old[:, 0], 0.1)
    normalized = (new - old) / width[:, None]
    return {
        "q05_shift": float(np.median(normalized[:, 0])),
        "q50_shift": float(np.median(normalized[:, 1])),
        "q95_shift": float(np.median(normalized[:, 2])),
        "normalized_abs_movement": float(np.mean(np.abs(normalized))),
    }


def qmos_refit_is_supported(rule: dict, base, candidate) -> tuple[bool, dict]:
    summary = qmos_refit_movement_summary(base, candidate)
    bounds = rule.get("support", {}).get("bounds", {})
    supported = bool(bounds) and all(
        metric in bounds
        and float(bounds[metric][0]) <= value <= float(bounds[metric][1])
        for metric, value in summary.items()
    )
    return supported, summary


def coarse_fields_hybrid(fh, config, artifact: dict, issue_date) -> dict:
    direction_models = artifact["direction_models"]
    has_analysis_center = "analysis" in direction_models
    has_d7_center = bool(direction_models.get("d7_center_policy"))
    has_alternate_center = (
        has_analysis_center or "context" in direction_models or has_d7_center
    )
    table = build_hybrid_table(
        fh, config, [issue_date], with_analysis=has_analysis_center
    )
    quantiles = predict_quantiles(
        fh,
        artifact["qmos"],
        table,
        adjust=artifact["conformal_adjust"],
        config=config,
        d7_speed_context=artifact.get("d7_speed_context"),
    )
    issue_date = pd.Timestamp(issue_date)
    quantiles["dir_pred"] = predict_direction_centers(
        fh, artifact["direction_models"], table, config=config
    )
    if has_alternate_center:
        baseline_models = direction_models.get("base", direction_models)
        quantiles["dir_speed_baseline"] = predict_direction_centers(
            fh, baseline_models, table, config=config
        )
    climatology = fh.build_climatology_forecast(
        [issue_date], lead=14, with_truth=False
    )
    d14_policy = artifact.get("d14_direction_policy", {})
    uses_d14_signal = any(
        (issue_date.month, issue_date.day, hour) in d14_policy
        for hour in HOURS
    )
    source_climatology = (
        fh.build_climatology_forecast([issue_date], lead=7, with_truth=False)
        if uses_d14_signal
        else None
    )
    fields = {}
    for lead in LEADS:
        for hour in HOURS:
            subset = quantiles[
                (quantiles["lead"] == lead) & (quantiles["hour"] == hour)
            ].copy()
            if subset.empty:
                raise ValueError(f"No rows for lead={lead}, hour={hour}")
            speed_stack = np.stack(
                [
                    fh.predictions_to_grid(
                        subset.assign(u_pred=subset[column], v_pred=0.0),
                        lead,
                        hour,
                    )[0]
                    for column in ("spd_q05", "spd_q50", "spd_q95")
                ]
            ).astype("float32")
            speed_direction_grid = None
            if lead == 14:
                raw_d10_u, raw_d10_v = fh.predictions_to_grid(
                    subset.assign(
                        u_pred=subset["fcst_u"],
                        v_pred=subset["fcst_v"],
                    ),
                    lead,
                    hour,
                )
                fields[(10, hour, "raw")] = np.stack(
                    [raw_d10_u, raw_d10_v]
                ).astype("float32")
                clim_u, clim_v = fh.predictions_to_grid(climatology, lead, hour)
                direction_grid = np.degrees(np.arctan2(-clim_u, -clim_v)) % 360
                rule = d14_policy.get((issue_date.month, issue_date.day, hour))
                if rule is not None:
                    speed_direction_grid = direction_grid.copy()
                    source_u, source_v = fh.predictions_to_grid(
                        source_climatology, 7, hour
                    )
                    source_direction = (
                        np.degrees(np.arctan2(-source_u, -source_v)) % 360
                    )
                    d7 = quantiles[
                        (quantiles["lead"] == 7) & (quantiles["hour"] == hour)
                    ].copy()
                    raw_d7 = fh.predictions_to_grid(
                        d7.assign(u_pred=d7["dir_pred"], v_pred=0.0), 7, hour
                    )[0]
                    direction_grid = apply_d14_direction_signal(
                        direction_grid, source_direction, raw_d7, *rule
                    )
            else:
                direction_grid = fh.predictions_to_grid(
                    subset.assign(u_pred=subset["dir_pred"], v_pred=0.0),
                    lead,
                    hour,
                )[0]
            radians = np.radians(direction_grid)
            median = speed_stack[1]
            det_u = -median * np.sin(radians)
            det_v = -median * np.cos(radians)
            fields[(lead, hour, "det")] = np.stack([det_u, det_v]).astype("float32")
            if speed_direction_grid is not None:
                speed_radians = np.radians(speed_direction_grid)
                speed_u = -median * np.sin(speed_radians)
                speed_v = -median * np.cos(speed_radians)
                fields[(lead, hour, "speed_det")] = np.stack(
                    [speed_u, speed_v]
                ).astype("float32")
            protect_speed_center = (
                lead == 1 and has_alternate_center
            ) or (
                lead == 7 and has_d7_center
            )
            if protect_speed_center:
                baseline_grid = fh.predictions_to_grid(
                    subset.assign(u_pred=subset["dir_speed_baseline"], v_pred=0.0),
                    lead,
                    hour,
                )[0]
                baseline_rad = np.radians(baseline_grid)
                baseline_u = -median * np.sin(baseline_rad)
                baseline_v = -median * np.cos(baseline_rad)
                fields[(lead, hour, "speed_det")] = np.stack(
                    [baseline_u, baseline_v]
                ).astype("float32")
            fields[(lead, hour, "spd")] = speed_stack

    # Evaluate optional speed refits only after the protected forecast is fully
    # materialized. Some underlying estimators share numerical thread state;
    # this ordering guarantees that evaluating a speed challenger cannot alter
    # any protected direction center or interval decision.
    refit_policy = artifact.get("qmos_refit_policy")
    if refit_policy is not None:
        matching_rules = [
            rule
            for rule in refit_policy.get("rules", ())
            if (int(rule["month"]), int(rule["day"]))
            == (issue_date.month, issue_date.day)
        ]
        if matching_rules:
            refit_table = table[table["lead"].isin((1, 7))].copy()
            refit_quantiles = predict_quantiles(
                fh,
                refit_policy["models"],
                refit_table,
                adjust=artifact["conformal_adjust"],
            )
            for refit_rule in matching_rules:
                lead = int(refit_rule["lead"])
                hour = int(refit_rule["hour"])
                base_selected = (
                    (quantiles["lead"] == lead)
                    & (quantiles["hour"] == hour)
                )
                refit_selected = (
                    (refit_quantiles["lead"] == lead)
                    & (refit_quantiles["hour"] == hour)
                )
                supported, summary = qmos_refit_is_supported(
                    refit_rule,
                    quantiles.loc[base_selected],
                    refit_quantiles.loc[refit_selected],
                )
                print(
                    f"[infer] qMOS refit {issue_date.date()} d{lead} "
                    f"h{hour:02d} supported={supported} movement={summary}",
                    flush=True,
                )
                if not supported:
                    continue
                refit_subset = refit_quantiles.loc[refit_selected].copy()
                refit_stack = np.stack(
                    [
                        fh.predictions_to_grid(
                            refit_subset.assign(
                                u_pred=refit_subset[column], v_pred=0.0
                            ),
                            lead,
                            hour,
                        )[0]
                        for column in ("spd_q05", "spd_q50", "spd_q95")
                    ]
                ).astype("float32")
                fields[(lead, hour, "qmos_refit_spd")] = refit_stack
                fields[(lead, hour, "qmos_refit_rule")] = refit_rule

    d1_context = artifact.get("d1_speed_context")
    if d1_context is not None and (
        issue_date.month,
        issue_date.day,
    ) == (int(d1_context["month"]), int(d1_context["day"])):
        d1_table = table[table["lead"] == 1].reset_index(drop=True)
        context_x = add_lagged_context_features(
            config, fh, issue_date, d1_table
        )
        context_x = context_x[d1_context["features"]]
        candidate = np.sort(
            np.column_stack(
                [
                    d1_context["models"][quantile].predict(context_x)
                    for quantile in QUANTILES
                ]
            ),
            axis=1,
        )
        for hour in HOURS:
            selected = d1_table["hour"].to_numpy() == hour
            candidate_stack = np.stack(
                [
                    fh.predictions_to_grid(
                        d1_table.loc[selected].assign(
                            u_pred=candidate[selected, index], v_pred=0.0
                        ),
                        1,
                        hour,
                    )[0]
                    for index in range(3)
                ]
            ).astype("float32")
            fields[(1, hour, "d1_context_spd")] = candidate_stack
        print(
            f"[infer] d1 causal context candidate ready for "
            f"{issue_date.date()}",
            flush=True,
        )

    dense_daily = artifact.get("d1_dense_daily")
    dense_rules = [] if dense_daily is None else [
        rule
        for rule in dense_daily.get("rules", ())
        if (int(rule["month"]), int(rule["day"]))
        == (issue_date.month, issue_date.day)
    ]
    if dense_rules:
        d1_table = table[table["lead"] == 1].reset_index(drop=True)
        grid_latitude = np.sort(d1_table["lat"].unique())
        grid_longitude = np.sort(d1_table["lon"].unique())
        if not (
            np.allclose(
                grid_latitude,
                np.asarray(dense_daily["coarse_latitude"]),
                atol=1e-5,
            )
            and np.allclose(
                grid_longitude,
                np.asarray(dense_daily["coarse_longitude"]),
                atol=1e-5,
            )
        ):
            raise RuntimeError("Dense-daily artifact and inference grids differ")
        dense_x = add_d1_dense_daily_features(config, issue_date, d1_table)
        if list(dense_x.columns) != list(dense_daily["features"]):
            raise RuntimeError("Dense-daily d1 feature contract changed")
        candidate = np.column_stack(
            [
                dense_daily["models"][quantile].predict(dense_x)
                for quantile in (0.05, 0.95)
            ]
        )
        candidate += np.asarray(dense_daily["calibration_offsets"])[None, :]
        candidate.sort(axis=1)
        valid_flat = np.asarray(
            dense_daily["coarse_valid_flat_indices"], dtype="int32"
        )
        for hour in HOURS:
            selected = d1_table["hour"].to_numpy() == hour
            lower_grid = fh.predictions_to_grid(
                d1_table.loc[selected].assign(
                    u_pred=candidate[selected, 0], v_pred=0.0
                ),
                1,
                hour,
            )[0]
            upper_grid = fh.predictions_to_grid(
                d1_table.loc[selected].assign(
                    u_pred=candidate[selected, 1], v_pred=0.0
                ),
                1,
                hour,
            )[0]
            stack = np.asarray(fields[(1, hour, "spd")], dtype="float64").copy()
            flat = stack.reshape(3, -1)
            flat[0, valid_flat] = np.minimum(
                lower_grid.ravel()[valid_flat], flat[1, valid_flat]
            )
            flat[2, valid_flat] = np.maximum(
                upper_grid.ravel()[valid_flat], flat[1, valid_flat]
            )
            fields[(1, hour, "d1_dense_daily_spd")] = stack.astype("float32")
        print(
            f"[infer] dense-daily d1 candidate ready for {issue_date.date()} "
            f"rules={len(dense_rules)}",
            flush=True,
        )
    return fields


def parse_lead_float_map(text: str) -> dict[int, float]:
    """Parse '1:0.9,7:0.55' into {1: 0.9, 7: 0.55}."""
    out: dict[int, float] = {}
    if not text:
        return out
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Expected lead:value item, got {item!r}")
        lead_s, value_s = item.split(":", 1)
        lead = int(lead_s)
        value = float(value_s)
        if lead not in (1, 7, 14):
            raise ValueError(f"Unsupported lead in map: {lead}")
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid value for lead {lead}: {value}")
        out[lead] = value
    return out


def cap_direction_intervals(df, caps: dict[int, float]) -> pd.DataFrame:
    """Symmetrically cap only intervals wider than the configured limit."""
    for lead, half_width in sorted(caps.items()):
        if not 0.0 < half_width < 180.0:
            raise ValueError(
                f"Direction half-width cap for d{lead} must be in (0, 180)"
            )
        selected = df["horizon"] == lead
        if not selected.any():
            continue
        lower = df.loc[selected, "dir_05"].to_numpy(dtype="float64")
        upper = df.loc[selected, "dir_95"].to_numpy(dtype="float64")
        width = (upper - lower) % 360.0
        active = width > 2.0 * half_width
        if not active.any():
            continue
        center = df.loc[selected, "dir_50"].to_numpy(dtype="float64") % 360.0
        lower[active] = (center[active] - half_width) % 360.0
        upper[active] = (center[active] + half_width) % 360.0
        df.loc[selected, "dir_05"] = lower.astype("float32")
        df.loc[selected, "dir_95"] = upper.astype("float32")
        print(
            f"[infer] d{lead} direction half-width cap={half_width:.3f} "
            f"active_rows={int(active.sum()):,}",
            flush=True,
        )
    return df


def apply_fine_d14_climatology(df, issue_date, artifact):
    """Apply training-built native-grid d14 centers and strict widths."""
    payload = artifact.get("fine_d14_climatology")
    if payload is None:
        return df
    endpoint_policy = payload.get("endpoint_policy", {})
    if set(endpoint_policy) - set(payload["policy"]):
        raise ValueError("Fine d14 endpoint policy contains unknown cells")
    issue = pd.Timestamp(issue_date)
    source_lat = np.asarray(payload["latitude"], dtype="float32")
    source_lon = np.asarray(payload["longitude"], dtype="float32")
    expected_lat = source_lat.round(2)
    expected_lon = source_lon.round(2)
    for key, rule in payload["policy"].items():
        month, day, hour = (int(value) for value in key)
        if month != issue.month or day != issue.day:
            continue
        selected = (df["horizon"] == 14) & (df["hour"] == hour)
        if int(selected.sum()) != len(source_lat):
            raise RuntimeError(
                f"Fine d14 row mismatch for {issue.date()} h{hour:02d}: "
                f"{int(selected.sum())}/{len(source_lat)}"
            )
        target_lat = df.loc[selected, "latitude"].to_numpy(dtype="float32")
        target_lon = df.loc[selected, "longitude"].to_numpy(dtype="float32")
        center = np.asarray(payload["centers"][key], dtype="float64")
        # The official writer preserves canonical np.where(mask) order but
        # rounds native curvilinear coordinates to two float32 decimals.
        if not (
            np.array_equal(target_lat, expected_lat)
            and np.array_equal(target_lon, expected_lon)
        ):
            raise RuntimeError(
                f"Fine d14 canonical grid order mismatch for {issue.date()} "
                f"h{hour:02d}"
            )
        half_width = float(rule["half_width"])
        if not 0.0 < half_width < 180.0:
            raise ValueError(f"Invalid fine d14 half-width: {half_width}")
        endpoint_rule = endpoint_policy.get(key, {})
        lower_factor = float(endpoint_rule.get("lower_factor", 1.0))
        upper_factor = float(endpoint_rule.get("upper_factor", 1.0))
        lower_width = half_width * lower_factor
        upper_width = half_width * upper_factor
        if (
            not 0.0 < lower_factor <= 1.10
            or not 0.0 < upper_factor <= 1.10
            or lower_width + upper_width >= 359.0
        ):
            raise ValueError(
                f"Invalid fine d14 endpoint factors for {key}: "
                f"{lower_factor}/{upper_factor}"
            )
        df.loc[selected, "dir_50"] = center.astype("float32")
        df.loc[selected, "dir_05"] = (
            (center - lower_width) % 360.0
        ).astype("float32")
        df.loc[selected, "dir_95"] = (
            (center + upper_width) % 360.0
        ).astype("float32")
    return df


def apply_fine_d7_climatology(df, issue_date, artifact):
    # Shift exact d7 intervals toward strictly gated native-grid centers.
    payload = artifact.get("fine_d7_climatology")
    if payload is None:
        return df
    issue = pd.Timestamp(issue_date)
    source_lat = np.asarray(payload["latitude"], dtype="float32")
    source_lon = np.asarray(payload["longitude"], dtype="float32")
    expected_lat = source_lat.round(2)
    expected_lon = source_lon.round(2)
    lat_mid = float(np.median(source_lat))
    lon_mid = float(np.median(source_lon))
    spatial_bin = (
        (source_lat >= lat_mid).astype("int8") * 2
        + (source_lon >= lon_mid).astype("int8")
    )
    for key, rule in payload["policy"].items():
        month, day, hour = (int(value) for value in key)
        if month != issue.month or day != issue.day:
            continue
        selected = (df["horizon"] == 7) & (df["hour"] == hour)
        if int(selected.sum()) != len(source_lat):
            raise RuntimeError(
                f"Fine d7 row mismatch for {issue.date()} h{hour:02d}: "
                f"{int(selected.sum())}/{len(source_lat)}"
            )
        target_lat = df.loc[selected, "latitude"].to_numpy(dtype="float32")
        target_lon = df.loc[selected, "longitude"].to_numpy(dtype="float32")
        if not (
            np.array_equal(target_lat, expected_lat)
            and np.array_equal(target_lon, expected_lon)
        ):
            raise RuntimeError(
                f"Fine d7 canonical grid order mismatch for {issue.date()} "
                f"h{hour:02d}"
            )
        center = np.asarray(payload["centers"][key], dtype="float64")
        baseline = df.loc[selected, "dir_50"].to_numpy(dtype="float64")
        lower = df.loc[selected, "dir_05"].to_numpy(dtype="float64")
        upper = df.loc[selected, "dir_95"].to_numpy(dtype="float64")
        blend = float(rule["blend"])
        if not 0.0 <= blend <= 1.0:
            raise ValueError(f"Invalid fine d7 blend: {blend}")
        disagreement = np.abs((center - baseline + 180.0) % 360.0 - 180.0)
        activation = rule.get("activation", {})
        minimum = float(activation.get("minimum", 0.0))
        maximum = float(activation.get("maximum", 180.0))
        if not 0.0 <= minimum <= maximum <= 180.0:
            raise ValueError(f"Invalid fine d7 activation: {activation}")
        active = disagreement >= minimum
        if maximum < 180.0:
            active &= disagreement < maximum
        else:
            active &= disagreement <= maximum
        context = rule.get("context", {})
        if set(context) - {"exclude_spatial"}:
            raise ValueError(f"Unsupported fine d7 context: {context}")
        if "exclude_spatial" in context:
            excluded = int(context["exclude_spatial"])
            if excluded not in (0, 1, 2, 3):
                raise ValueError(f"Invalid fine d7 spatial exclusion: {excluded}")
            active &= spatial_bin != excluded
        target_shift = (center - baseline + 180.0) % 360.0 - 180.0
        shift = np.where(active, blend * target_shift, 0.0)
        df.loc[selected, "dir_50"] = ((baseline + shift) % 360.0).astype(
            "float32"
        )
        df.loc[selected, "dir_05"] = ((lower + shift) % 360.0).astype(
            "float32"
        )
        df.loc[selected, "dir_95"] = ((upper + shift) % 360.0).astype(
            "float32"
        )
    return df


def apply_d7_pressure_policy(
    df,
    issue_date,
    artifact,
    raw_d7_center: pd.Series,
):
    """Shift d7 intervals with the strictly gated Phase 1 pressure signal."""

    payload = artifact.get("d7_pressure_policy")
    if payload is None:
        return df
    issue = pd.Timestamp(issue_date).normalize()
    rules = [
        rule
        for rule in payload.get("rules", [])
        if int(rule["month"]) == issue.month and int(rule["day"]) == issue.day
    ]
    if not rules:
        return df
    if not isinstance(raw_d7_center, pd.Series):
        raise TypeError("raw_d7_center must be an index-aligned Series")

    def signal_levels(signal):
        if signal == "low2":
            return ("1000", "925")
        if signal == "low3":
            return ("1000", "925", "850")
        if signal == "low_30_70":
            return ("1000", "925")
        if signal not in {"1000", "925", "850", "700", "500"}:
            raise ValueError(f"Unsupported pressure signal: {signal}")
        return (signal,)

    required_columns = {"time", "latitude", "longitude"}
    for rule in rules:
        hour = int(rule["hour"])
        for level in signal_levels(str(rule["signal"])):
            required_columns.update(
                [f"fcst_u_{level}_d7_h{hour}", f"fcst_v_{level}_d7_h{hour}"]
            )
    pressure_path = resolve_phase1_pressure_path()
    pressure = pd.read_parquet(
        pressure_path,
        columns=sorted(required_columns),
        filters=[("time", "==", issue)],
    )
    pressure["time"] = pd.to_datetime(pressure["time"]).dt.normalize()
    pressure = pressure.loc[pressure["time"] == issue].sort_values(
        ["latitude", "longitude"]
    )
    source_latitude = np.asarray(payload["source_latitude"], dtype="float64")
    source_longitude = np.asarray(payload["source_longitude"], dtype="float64")
    expected_rows = len(source_latitude) * len(source_longitude)
    if len(pressure) != expected_rows:
        raise RuntimeError(
            f"Incomplete pressure grid for {issue.date()}: "
            f"{len(pressure)}/{expected_rows}"
        )
    observed_latitude = np.sort(pressure["latitude"].unique()).astype("float64")
    observed_longitude = np.sort(pressure["longitude"].unique()).astype("float64")
    if not (
        np.array_equal(observed_latitude, source_latitude)
        and np.array_equal(observed_longitude, source_longitude)
    ):
        raise RuntimeError("Pressure source grid differs from the training grid")

    nearest = np.asarray(payload["nearest_flat_index"], dtype="int64")
    target_latitude = np.asarray(payload["target_latitude"], dtype="float32")
    target_longitude = np.asarray(payload["target_longitude"], dtype="float32")
    if (
        nearest.shape != target_latitude.shape
        or target_longitude.shape != target_latitude.shape
        or np.any(nearest < 0)
        or np.any(nearest >= expected_rows)
    ):
        raise RuntimeError("Invalid pressure nearest-grid artifact")
    expected_target_latitude = target_latitude.round(2)
    expected_target_longitude = target_longitude.round(2)
    spatial = (
        (target_latitude >= np.median(target_latitude)).astype("int8") * 2
        + (target_longitude >= np.median(target_longitude)).astype("int8")
    ).astype("int64")
    disagreement_edges = np.asarray(
        payload["disagreement_edges"], dtype="float64"
    )

    def circular_delta(left, right):
        return (
            np.asarray(left, dtype="float64")
            - np.asarray(right, dtype="float64")
            + 180.0
        ) % 360.0 - 180.0

    def fixed_bins(values, edges):
        return np.clip(
            np.digitize(values, edges) - 1, 0, len(edges) - 2
        ).astype("int64")

    def pressure_direction(rule):
        hour = int(rule["hour"])
        signal = str(rule["signal"])
        levels = signal_levels(signal)
        if signal == "low_30_70":
            coefficients = (0.30, 0.70)
            u = sum(
                coefficient
                * pressure[f"fcst_u_{level}_d7_h{hour}"].to_numpy(
                    dtype="float64"
                )
                for level, coefficient in zip(levels, coefficients)
            )
            v = sum(
                coefficient
                * pressure[f"fcst_v_{level}_d7_h{hour}"].to_numpy(
                    dtype="float64"
                )
                for level, coefficient in zip(levels, coefficients)
            )
        else:
            u = np.mean(
                [
                    pressure[f"fcst_u_{level}_d7_h{hour}"].to_numpy(
                        dtype="float64"
                    )
                    for level in levels
                ],
                axis=0,
            )
            v = np.mean(
                [
                    pressure[f"fcst_v_{level}_d7_h{hour}"].to_numpy(
                        dtype="float64"
                    )
                    for level in levels
                ],
                axis=0,
            )
        values = (np.degrees(np.arctan2(-u, -v)) % 360.0)[nearest]
        if values.shape != target_latitude.shape or not np.all(np.isfinite(values)):
            raise RuntimeError(f"Invalid pressure direction for rule {rule}")
        return values

    def group_bins(family, pressure_center, raw_center):
        pressure4 = np.floor((pressure_center % 360.0) / 90.0).astype("int64")
        pressure8 = np.floor((pressure_center % 360.0) / 45.0).astype("int64")
        base4 = np.floor((raw_center % 360.0) / 90.0).astype("int64")
        disagreement4 = fixed_bins(
            np.abs(circular_delta(pressure_center, raw_center)),
            disagreement_edges,
        )
        if family == "scalar":
            return np.zeros(len(raw_center), dtype="int64")
        if family == "pdir4":
            return pressure4
        if family == "pdir8":
            return pressure8
        if family == "spatial4":
            return spatial
        if family == "base4_spatial4":
            return base4 * 4 + spatial
        if family == "pdir4_spatial4":
            return pressure4 * 4 + spatial
        if family == "dis4":
            return disagreement4
        if family == "dis4_pdir4":
            return disagreement4 * 4 + pressure4
        raise ValueError(f"Unsupported pressure calibration family: {family}")

    shifted_rows = 0
    for rule in rules:
        hour = int(rule["hour"])
        selected = (df["horizon"] == 7) & (df["hour"] == hour)
        if int(selected.sum()) != len(target_latitude):
            raise RuntimeError(
                f"Pressure policy row mismatch for {issue.date()} h{hour:02d}: "
                f"{int(selected.sum())}/{len(target_latitude)}"
            )
        selected_index = df.index[selected]
        observed_target_latitude = df.loc[selected, "latitude"].to_numpy(
            dtype="float32"
        )
        observed_target_longitude = df.loc[selected, "longitude"].to_numpy(
            dtype="float32"
        )
        if not (
            np.array_equal(observed_target_latitude, expected_target_latitude)
            and np.array_equal(observed_target_longitude, expected_target_longitude)
        ):
            raise RuntimeError(
                f"Pressure target grid order mismatch for {issue.date()} h{hour:02d}"
            )
        raw_center = raw_d7_center.loc[selected_index].to_numpy(dtype="float64")
        if raw_center.shape != target_latitude.shape or not np.all(
            np.isfinite(raw_center)
        ):
            raise RuntimeError("Invalid raw d7 center alignment")
        pressure_center = pressure_direction(rule)
        mode = str(rule["mode"])
        if mode == "raw":
            active = np.ones(len(raw_center), dtype=bool)
            target = pressure_center
        elif mode in {"calibrated", "calibrated_spatial"}:
            bins = group_bins(str(rule["family"]), pressure_center, raw_center)
            biases = np.asarray(rule["biases"], dtype="float64")
            supported = np.asarray(rule["supported"], dtype=bool)
            if (
                biases.ndim != 1
                or supported.shape != biases.shape
                or np.any(bins < 0)
                or np.any(bins >= len(biases))
                or not np.all(np.isfinite(biases))
            ):
                raise RuntimeError(f"Invalid pressure lookup rule: {rule}")
            active = supported[bins]
            if mode == "calibrated_spatial":
                spatial_supported = np.asarray(
                    rule.get("spatial_supported"), dtype=bool
                )
                if spatial_supported.shape != active.shape:
                    raise RuntimeError(
                        f"Invalid pressure spatial mask: {rule}"
                    )
                active &= spatial_supported
            target = (pressure_center + biases[bins]) % 360.0
        else:
            raise ValueError(f"Unsupported pressure policy mode: {mode}")
        weight = float(rule["weight"])
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Invalid pressure center weight: {weight}")
        shift = np.zeros(len(raw_center), dtype="float64")
        shift[active] = weight * circular_delta(
            target[active], raw_center[active]
        )
        for column in ("dir_05", "dir_50", "dir_95"):
            current = df.loc[selected, column].to_numpy(dtype="float64")
            df.loc[selected, column] = ((current + shift) % 360.0).astype(
                "float32"
            )
        shifted_rows += int(np.count_nonzero(active))
    print(
        f"[infer] d7 pressure policy {issue.date()} "
        f"rules={len(rules)} active_rows={shifted_rows:,}",
        flush=True,
    )
    return df


def apply_interval_postprocess(
    df,
    speed_width_scale: dict[int, float],
    dir_halfwidth_scale: dict[int, float],
    dir_halfwidth_deg: dict[int, float],
    d1_direction_speed_interval: dict | None = None,
    d7_direction_interval_policy: dict | None = None,
    d14_direction_speed_interval: dict | None = None,
    issue_date=None,
):
    """Deterministic interval-only adjustment.

    The public baseline was dominated by over-wide d7/d14 intervals. This keeps
    the learned centers unchanged and only changes interval half-widths, so it
    remains frugal and does not introduce any additional models.
    """
    if speed_width_scale:
        for lead, scale in speed_width_scale.items():
            m = df["horizon"] == lead
            q50 = df.loc[m, "q50"].astype("float64")
            lo = q50 - scale * (q50 - df.loc[m, "q05"].astype("float64"))
            hi = q50 + scale * (df.loc[m, "q95"].astype("float64") - q50)
            df.loc[m, "q05"] = np.maximum(0.0, lo).astype("float32")
            df.loc[m, "q95"] = hi.astype("float32")

    explicit_dir_leads = set(dir_halfwidth_scale) | set(dir_halfwidth_deg)
    speed_conditioned_intervals = (
        (1, d1_direction_speed_interval),
        (14, d14_direction_speed_interval),
    )
    for lead, interval_artifact in speed_conditioned_intervals:
        if interval_artifact is None or lead in explicit_dir_leads:
            continue
        edges = np.asarray(
            interval_artifact.get("edges", []), dtype="float64"
        )
        widths = np.asarray(
            interval_artifact.get("half_widths", []), dtype="float64"
        )
        if (
            edges.ndim != 1
            or widths.ndim != 1
            or len(edges) != len(widths) + 1
            or len(widths) == 0
            or not np.all(np.diff(edges) > 0)
            or not np.all(np.isfinite(widths))
            or np.any(widths <= 0)
            or np.any(widths >= 180)
        ):
            raise ValueError(
                f"Invalid d{lead} speed-conditioned direction interval artifact"
            )
        m = df["horizon"] == lead
        if lead == 14:
            # Preserve the public-gated v7 interval outside the sparse slots.
            # The learned table is evaluated only as a local challenger.
            public_half = 158.0
            d50_all = df.loc[m, "dir_50"].to_numpy(dtype="float64") % 360.0
            df.loc[m, "dir_05"] = (
                (d50_all - public_half) % 360.0
            ).astype("float32")
            df.loc[m, "dir_95"] = (
                (d50_all + public_half) % 360.0
            ).astype("float32")
            if issue_date is None:
                raise ValueError("d14 direction interval policy requires issue_date")
            issue = pd.Timestamp(issue_date)
            selected_hours = {
                int(hour)
                for month, day, hour in interval_artifact.get(
                    "selected_slots", []
                )
                if int(month) == issue.month and int(day) == issue.day
            }
            if not selected_hours:
                continue
            m &= df["hour"].isin(selected_hours)
        if not m.any():
            continue
        speed = df.loc[m, "q50"].to_numpy(dtype="float64")
        if not np.all(np.isfinite(speed)) or np.any(speed < 0):
            raise ValueError(
                f"Invalid d{lead} q50 values for direction interval lookup"
            )
        mapping = interval_artifact.get("mapping", "step")
        if mapping == "step":
            bins = np.clip(np.digitize(speed, edges) - 1, 0, len(widths) - 1)
            half = widths[bins]
        elif mapping == "linear":
            centers = np.empty(len(widths), dtype="float64")
            centers[:-1] = 0.5 * (edges[:-2] + edges[1:-1])
            centers[-1] = edges[-2] + 0.5 * (edges[-2] - edges[-3])
            half = np.interp(
                speed,
                centers,
                widths,
                left=widths[0],
                right=widths[-1],
            )
        else:
            raise ValueError(
                f"Unsupported d{lead} direction interval mapping: {mapping}"
            )
        d50 = df.loc[m, "dir_50"].to_numpy(dtype="float64") % 360.0
        df.loc[m, "dir_05"] = ((d50 - half) % 360.0).astype("float32")
        df.loc[m, "dir_95"] = ((d50 + half) % 360.0).astype("float32")

    if 7 not in explicit_dir_leads:
        base_half_width = 138.0
        if d7_direction_interval_policy is not None:
            base_half_width = float(
                d7_direction_interval_policy.get(
                    "base_half_width", base_half_width
                )
            )
        if not 0.0 < base_half_width < 180.0:
            raise ValueError(f"Invalid d7 base half-width: {base_half_width}")
        d7_mask = df["horizon"] == 7
        d7_center = df.loc[d7_mask, "dir_50"].to_numpy(dtype="float64") % 360.0
        d7_raw_center = pd.Series(
            d7_center,
            index=df.index[d7_mask],
            dtype="float64",
        )
        df.loc[d7_mask, "dir_05"] = (
            (d7_center - base_half_width) % 360.0
        ).astype("float32")
        df.loc[d7_mask, "dir_95"] = (
            (d7_center + base_half_width) % 360.0
        ).astype("float32")
        if d7_direction_interval_policy is not None:
            if issue_date is None:
                raise ValueError("d7 direction interval policy requires issue_date")
            issue = pd.Timestamp(issue_date)
            for rule in d7_direction_interval_policy.get("rules", []):
                if (
                    int(rule["month"]) != issue.month
                    or int(rule["day"]) != issue.day
                ):
                    continue
                hour = int(rule["hour"])
                selected = d7_mask & (df["hour"] == hour)
                family = rule.get("family", "scalar")
                original_center = d7_raw_center.loc[
                    df.index[selected]
                ].to_numpy(dtype="float64")
                if "half_widths" in rule:
                    widths = np.asarray(
                        rule["half_widths"], dtype="float64"
                    )
                else:
                    widths = np.asarray(
                        [rule["half_width"]], dtype="float64"
                    )
                if (
                    widths.ndim != 1
                    or len(widths) == 0
                    or not np.all(np.isfinite(widths))
                    or np.any(widths <= 0.0)
                    or np.any(widths >= 180.0)
                ):
                    raise ValueError(f"Invalid d7 direction widths: {rule}")
                if family == "scalar":
                    if len(widths) != 1:
                        raise ValueError(f"Invalid scalar d7 rule: {rule}")
                    half_width = float(widths[0])
                elif family == "speed_linear":
                    edges = np.asarray(
                        rule.get("speed_edges", []), dtype="float64"
                    )
                    if (
                        edges.ndim != 1
                        or len(edges) != len(widths) + 1
                        or not np.all(np.diff(edges) > 0.0)
                    ):
                        raise ValueError(
                            f"Invalid speed-conditioned d7 rule: {rule}"
                        )
                    speed = df.loc[selected, "q50"].to_numpy(
                        dtype="float64"
                    )
                    if not np.all(np.isfinite(speed)) or np.any(speed < 0.0):
                        raise ValueError(
                            f"Invalid d7 q50 values for direction lookup"
                        )
                    centers = np.empty(len(widths), dtype="float64")
                    centers[:-1] = 0.5 * (
                        edges[:-2] + edges[1:-1]
                    )
                    centers[-1] = edges[-2] + 0.5 * (
                        edges[-2] - edges[-3]
                    )
                    half_width = np.interp(
                        speed,
                        centers,
                        widths,
                        left=widths[0],
                        right=widths[-1],
                    )
                elif family == "direction_sector":
                    edges = np.asarray(
                        rule.get("direction_sector_edges", []),
                        dtype="float64",
                    )
                    if (
                        edges.ndim != 1
                        or len(edges) != len(widths) + 1
                        or not np.all(np.diff(edges) > 0.0)
                        or edges[0] != 0.0
                        or edges[-1] != 360.0
                    ):
                        raise ValueError(
                            f"Invalid direction-sector d7 rule: {rule}"
                        )
                    bins = np.clip(
                        np.digitize(original_center, edges) - 1,
                        0,
                        len(widths) - 1,
                    )
                    half_width = widths[bins]
                else:
                    raise ValueError(f"Unsupported d7 rule family: {family}")
                bias_family = rule.get("bias_family", "scalar")
                if bias_family == "scalar":
                    if "biases" in rule:
                        biases = np.asarray(rule["biases"], dtype="float64")
                        if biases.shape != (1,):
                            raise ValueError(f"Invalid scalar d7 bias: {rule}")
                        center_bias = float(biases[0])
                    else:
                        center_bias = float(rule["bias"])
                    if not np.isfinite(center_bias):
                        raise ValueError(f"Invalid d7 direction bias: {rule}")
                elif bias_family.startswith("direction_sector_"):
                    edges = np.asarray(
                        rule.get(
                            "bias_direction_sector_edges",
                            rule.get("direction_sector_edges", []),
                        ),
                        dtype="float64",
                    )
                    biases = np.asarray(rule.get("biases", []), dtype="float64")
                    if (
                        biases.ndim != 1
                        or edges.ndim != 1
                        or len(edges) != len(biases) + 1
                        or not np.all(np.isfinite(biases))
                        or np.any(np.abs(biases) > 30.0)
                        or not np.all(np.diff(edges) > 0.0)
                        or edges[0] != 0.0
                        or edges[-1] != 360.0
                    ):
                        raise ValueError(
                            f"Invalid direction-sector d7 bias: {rule}"
                        )
                    bias_bins = np.clip(
                        np.digitize(original_center, edges) - 1,
                        0,
                        len(biases) - 1,
                    )
                    center_bias = biases[bias_bins]
                else:
                    raise ValueError(
                        f"Unsupported d7 bias family: {bias_family}"
                    )
                center = (original_center + center_bias) % 360.0
                df.loc[selected, "dir_50"] = center.astype("float32")
                df.loc[selected, "dir_05"] = (
                    (center - half_width) % 360.0
                ).astype("float32")
                df.loc[selected, "dir_95"] = (
                    (center + half_width) % 360.0
                ).astype("float32")

            for rule in d7_direction_interval_policy.get(
                "asymmetric_rules", []
            ):
                if (
                    int(rule["month"]) != issue.month
                    or int(rule["day"]) != issue.day
                ):
                    continue
                hour = int(rule["hour"])
                selected = d7_mask & (df["hour"] == hour)
                if not selected.any():
                    continue
                raw_center = d7_raw_center.loc[
                    df.index[selected]
                ].to_numpy(dtype="float64")
                current_lower = df.loc[
                    selected, "dir_05"
                ].to_numpy(dtype="float64")
                current_upper = df.loc[
                    selected, "dir_95"
                ].to_numpy(dtype="float64")
                base_lower_offset = (
                    (current_lower - raw_center + 180.0) % 360.0 - 180.0
                )
                base_upper_offset = (
                    (current_upper - raw_center + 180.0) % 360.0 - 180.0
                )

                spec = rule.get("spec", {})
                kind = spec.get("kind")
                count = int(spec.get("count", 0))
                if count <= 0:
                    raise ValueError(
                        f"Invalid d7 asymmetric family count: {rule}"
                    )
                if kind == "scalar":
                    bins = np.zeros(len(raw_center), dtype="int64")
                elif kind == "direction":
                    shift = float(spec.get("shift", 0.0))
                    bins = np.floor(
                        ((raw_center - shift) % 360.0)
                        / (360.0 / count)
                    ).astype("int64")
                elif kind == "speed":
                    edges = np.asarray(
                        spec.get("edges", []), dtype="float64"
                    )
                    if (
                        edges.ndim != 1
                        or len(edges) != count + 1
                        or not np.all(np.diff(edges) > 0.0)
                    ):
                        raise ValueError(
                            f"Invalid d7 asymmetric speed family: {rule}"
                        )
                    speed = df.loc[selected, "q50"].to_numpy(
                        dtype="float64"
                    )
                    if not np.all(np.isfinite(speed)) or np.any(speed < 0.0):
                        raise ValueError(
                            "Invalid q50 in d7 asymmetric speed lookup"
                        )
                    bins = np.clip(
                        np.digitize(speed, edges) - 1, 0, count - 1
                    )
                elif kind == "spatial":
                    lat_count = int(spec.get("lat_count", 0))
                    lon_count = int(spec.get("lon_count", 0))
                    lat_edges = np.asarray(
                        spec.get("lat_edges", []), dtype="float64"
                    )
                    lon_edges = np.asarray(
                        spec.get("lon_edges", []), dtype="float64"
                    )
                    if (
                        lat_count <= 0
                        or lon_count <= 0
                        or lat_count * lon_count != count
                        or len(lat_edges) != lat_count + 1
                        or len(lon_edges) != lon_count + 1
                        or not np.all(np.diff(lat_edges) > 0.0)
                        or not np.all(np.diff(lon_edges) > 0.0)
                    ):
                        raise ValueError(
                            f"Invalid d7 asymmetric spatial family: {rule}"
                        )
                    latitude = df.loc[selected, "latitude"].to_numpy(
                        dtype="float64"
                    )
                    longitude = df.loc[selected, "longitude"].to_numpy(
                        dtype="float64"
                    )
                    if (
                        not np.all(np.isfinite(latitude))
                        or not np.all(np.isfinite(longitude))
                    ):
                        raise ValueError(
                            "Invalid coordinates in d7 asymmetric lookup"
                        )
                    lat_bins = np.clip(
                        np.digitize(latitude, lat_edges) - 1,
                        0,
                        lat_count - 1,
                    )
                    lon_bins = np.clip(
                        np.digitize(longitude, lon_edges) - 1,
                        0,
                        lon_count - 1,
                    )
                    bins = lat_bins * lon_count + lon_bins
                elif kind == "direction_speed":
                    direction_count = int(
                        spec.get("direction_count", 0)
                    )
                    speed_count = int(spec.get("speed_count", 0))
                    shift = float(spec.get("shift", 0.0))
                    speed_edges = np.asarray(
                        spec.get("speed_edges", []), dtype="float64"
                    )
                    if (
                        direction_count <= 0
                        or speed_count <= 0
                        or direction_count * speed_count != count
                        or len(speed_edges) != speed_count + 1
                        or not np.all(np.diff(speed_edges) > 0.0)
                    ):
                        raise ValueError(
                            "Invalid d7 asymmetric direction-speed family: "
                            f"{rule}"
                        )
                    direction_bins = np.floor(
                        ((raw_center - shift) % 360.0)
                        / (360.0 / direction_count)
                    ).astype("int64")
                    speed = df.loc[selected, "q50"].to_numpy(
                        dtype="float64"
                    )
                    if not np.all(np.isfinite(speed)) or np.any(speed < 0.0):
                        raise ValueError(
                            "Invalid q50 in d7 asymmetric interaction lookup"
                        )
                    speed_bins = np.clip(
                        np.digitize(speed, speed_edges) - 1,
                        0,
                        speed_count - 1,
                    )
                    bins = direction_bins * speed_count + speed_bins
                else:
                    raise ValueError(
                        f"Unsupported d7 asymmetric family: {kind}"
                    )

                raw_lower = np.asarray(
                    rule.get("raw_lower", []), dtype="float64"
                )
                raw_upper = np.asarray(
                    rule.get("raw_upper", []), dtype="float64"
                )
                shrinkage = float(rule.get("shrinkage", np.nan))
                if (
                    raw_lower.shape != (count,)
                    or raw_upper.shape != (count,)
                    or not np.all(np.isfinite(raw_lower))
                    or not np.all(np.isfinite(raw_upper))
                    or np.any(raw_lower < -180.0)
                    or np.any(raw_lower > 180.0)
                    or np.any(raw_upper < -180.0)
                    or np.any(raw_upper > 180.0)
                    or not 0.0 <= shrinkage <= 1.0
                ):
                    raise ValueError(
                        f"Invalid d7 asymmetric endpoint rule: {rule}"
                    )
                target_lower = raw_lower[bins]
                target_upper = raw_upper[bins]
                lower_offset = (
                    (1.0 - shrinkage) * base_lower_offset
                    + shrinkage * target_lower
                )
                upper_offset = (
                    (1.0 - shrinkage) * base_upper_offset
                    + shrinkage * target_upper
                )
                df.loc[selected, "dir_05"] = (
                    (raw_center + lower_offset) % 360.0
                ).astype("float32")
                df.loc[selected, "dir_95"] = (
                    (raw_center + upper_offset) % 360.0
                ).astype("float32")

            for rule in d7_direction_interval_policy.get(
                "conditional_width_rules", []
            ):
                if (
                    int(rule["month"]) != issue.month
                    or int(rule["day"]) != issue.day
                ):
                    continue
                hour = int(rule["hour"])
                selected = d7_mask & (df["hour"] == hour)
                if not selected.any():
                    continue

                spec = rule.get("spec", {})
                kind = spec.get("kind")
                count = int(spec.get("count", 0))
                supported = np.asarray(
                    rule.get("supported", []), dtype="bool"
                )
                scale = float(rule.get("scale", np.nan))
                if (
                    count <= 0
                    or supported.shape != (count,)
                    or not np.any(supported)
                    or not 0.0 < scale <= 1.0
                ):
                    raise ValueError(
                        f"Invalid d7 conditional-width rule: {rule}"
                    )

                raw_center = d7_raw_center.loc[
                    df.index[selected]
                ].to_numpy(dtype="float64")
                if kind == "direction":
                    shift = float(spec.get("shift", 0.0))
                    bins = np.floor(
                        ((raw_center - shift) % 360.0)
                        / (360.0 / count)
                    ).astype("int64")
                elif kind == "spatial":
                    lat_count = int(spec.get("lat_count", 0))
                    lon_count = int(spec.get("lon_count", 0))
                    lat_edges = np.asarray(
                        spec.get("lat_edges", []), dtype="float64"
                    )
                    lon_edges = np.asarray(
                        spec.get("lon_edges", []), dtype="float64"
                    )
                    if (
                        lat_count <= 0
                        or lon_count <= 0
                        or lat_count * lon_count != count
                        or len(lat_edges) != lat_count + 1
                        or len(lon_edges) != lon_count + 1
                        or not np.all(np.diff(lat_edges) > 0.0)
                        or not np.all(np.diff(lon_edges) > 0.0)
                    ):
                        raise ValueError(
                            "Invalid d7 conditional-width spatial family: "
                            f"{rule}"
                        )
                    latitude = df.loc[selected, "latitude"].to_numpy(
                        dtype="float64"
                    )
                    longitude = df.loc[selected, "longitude"].to_numpy(
                        dtype="float64"
                    )
                    if (
                        not np.all(np.isfinite(latitude))
                        or not np.all(np.isfinite(longitude))
                    ):
                        raise ValueError(
                            "Invalid coordinates in d7 conditional-width lookup"
                        )
                    lat_bins = np.clip(
                        np.digitize(latitude, lat_edges) - 1,
                        0,
                        lat_count - 1,
                    )
                    lon_bins = np.clip(
                        np.digitize(longitude, lon_edges) - 1,
                        0,
                        lon_count - 1,
                    )
                    bins = lat_bins * lon_count + lon_bins
                else:
                    raise ValueError(
                        f"Unsupported d7 conditional-width family: {kind}"
                    )

                active = supported[bins]
                if not np.any(active):
                    continue
                current_lower = df.loc[
                    selected, "dir_05"
                ].to_numpy(dtype="float64")
                current_upper = df.loc[
                    selected, "dir_95"
                ].to_numpy(dtype="float64")
                width = (current_upper - current_lower) % 360.0
                midpoint = (current_lower + width / 2.0) % 360.0
                target_width = width.copy()
                target_width[active] *= scale
                if (
                    not np.all(np.isfinite(target_width))
                    or np.any(target_width <= 0.0)
                    or np.any(target_width >= 360.0)
                ):
                    raise ValueError(
                        "Invalid d7 conditional-width interval geometry"
                    )
                df.loc[selected, "dir_05"] = (
                    (midpoint - target_width / 2.0) % 360.0
                ).astype("float32")
                df.loc[selected, "dir_95"] = (
                    (midpoint + target_width / 2.0) % 360.0
                ).astype("float32")

            for rule in d7_direction_interval_policy.get(
                "lead_ratio_rules", []
            ):
                if (
                    int(rule["month"]) != issue.month
                    or int(rule["day"]) != issue.day
                ):
                    continue
                hour = int(rule["hour"])
                selected = d7_mask & (df["hour"] == hour)
                if not selected.any():
                    continue
                d1_selected = (df["horizon"] == 1) & (df["hour"] == hour)
                d7_coordinates = df.loc[
                    selected, ["latitude", "longitude"]
                ].to_numpy(dtype="float64")
                d1_coordinates = df.loc[
                    d1_selected, ["latitude", "longitude"]
                ].to_numpy(dtype="float64")
                if (
                    len(d1_coordinates) == len(d7_coordinates)
                    and np.array_equal(d1_coordinates, d7_coordinates)
                ):
                    d1_speed = df.loc[d1_selected, "q50"].to_numpy(
                        dtype="float64"
                    )
                else:
                    source = df.loc[
                        d1_selected, ["latitude", "longitude", "q50"]
                    ].copy()
                    if source.duplicated(["latitude", "longitude"]).any():
                        raise RuntimeError(
                            "Duplicate d1 coordinates in the d7 lead-ratio rule"
                        )
                    source = source.set_index(["latitude", "longitude"])["q50"]
                    target_index = pd.MultiIndex.from_arrays(
                        [d7_coordinates[:, 0], d7_coordinates[:, 1]],
                        names=["latitude", "longitude"],
                    )
                    d1_speed = source.reindex(target_index).to_numpy(
                        dtype="float64"
                    )
                d7_speed = df.loc[selected, "q50"].to_numpy(dtype="float64")
                if (
                    d1_speed.shape != d7_speed.shape
                    or not np.all(np.isfinite(d1_speed))
                    or not np.all(np.isfinite(d7_speed))
                    or np.any(d1_speed < 0.0)
                    or np.any(d7_speed < 0.0)
                ):
                    raise RuntimeError(
                        "Invalid matched speeds in the d7 lead-ratio rule"
                    )
                spec = rule.get("spec", {})
                count = int(spec.get("count", 0))
                edges = np.asarray(spec.get("edges", []), dtype="float64")
                if (
                    spec.get("kind") != "fixed"
                    or spec.get("feature") != "speed_ratio"
                    or count <= 0
                    or edges.shape != (count + 1,)
                    or not np.all(np.diff(edges) > 0.0)
                ):
                    raise ValueError(f"Invalid d7 lead-ratio spec: {rule}")
                bins = np.clip(
                    np.digitize(d7_speed / np.maximum(d1_speed, 0.25), edges)
                    - 1,
                    0,
                    count - 1,
                )
                raw_lower = np.asarray(
                    rule.get("raw_lower", []), dtype="float64"
                )
                raw_upper = np.asarray(
                    rule.get("raw_upper", []), dtype="float64"
                )
                shrinkage = float(rule.get("shrinkage", np.nan))
                if (
                    raw_lower.shape != (count,)
                    or raw_upper.shape != (count,)
                    or not np.all(np.isfinite(raw_lower))
                    or not np.all(np.isfinite(raw_upper))
                    or np.any(raw_lower < -180.0)
                    or np.any(raw_lower > 180.0)
                    or np.any(raw_upper < -180.0)
                    or np.any(raw_upper > 180.0)
                    or not 0.0 <= shrinkage <= 1.0
                ):
                    raise ValueError(
                        f"Invalid d7 lead-ratio endpoint rule: {rule}"
                    )
                raw_center = d7_raw_center.loc[
                    df.index[selected]
                ].to_numpy(dtype="float64")
                current_lower = df.loc[
                    selected, "dir_05"
                ].to_numpy(dtype="float64")
                current_upper = df.loc[
                    selected, "dir_95"
                ].to_numpy(dtype="float64")
                base_lower_offset = (
                    (current_lower - raw_center + 180.0) % 360.0 - 180.0
                )
                base_upper_offset = (
                    (current_upper - raw_center + 180.0) % 360.0 - 180.0
                )
                lower_offset = (
                    (1.0 - shrinkage) * base_lower_offset
                    + shrinkage * raw_lower[bins]
                )
                upper_offset = (
                    (1.0 - shrinkage) * base_upper_offset
                    + shrinkage * raw_upper[bins]
                )
                df.loc[selected, "dir_05"] = (
                    (raw_center + lower_offset) % 360.0
                ).astype("float32")
                df.loc[selected, "dir_95"] = (
                    (raw_center + upper_offset) % 360.0
                ).astype("float32")

    affected_dir_leads = explicit_dir_leads
    for lead in sorted(affected_dir_leads):
        m = df["horizon"] == lead
        d50 = df.loc[m, "dir_50"].astype("float64") % 360.0
        if lead in dir_halfwidth_deg:
            half = np.full(int(m.sum()), dir_halfwidth_deg[lead], dtype="float64")
        else:
            width = (
                df.loc[m, "dir_95"].astype("float64")
                - df.loc[m, "dir_05"].astype("float64")
            ) % 360.0
            half = 0.5 * width * dir_halfwidth_scale[lead]
        df.loc[m, "dir_05"] = ((d50 - half) % 360.0).astype("float32")
        df.loc[m, "dir_95"] = ((d50 + half) % 360.0).astype("float32")

    return normalize_directions(df)


def apply_d7_speed_endpoint_policy(df, issue_date, artifact: dict):
    """Apply the input-trained d7 endpoint rules without moving the center."""
    payload = artifact.get("d7_speed_endpoint_policy")
    if payload is None or not payload.get("rules"):
        return df
    if payload.get("input_only_training") is not True:
        raise ValueError("d7 speed endpoint policy lacks input-only provenance")
    if payload.get("previous_submission_inputs") != []:
        raise ValueError("d7 speed endpoint policy used a previous submission")
    if payload.get("new_models") != 0:
        raise ValueError("d7 speed endpoint policy unexpectedly adds models")
    if payload.get("gate", {}).get("passed") is not True:
        raise ValueError("d7 speed endpoint policy did not pass its strict gate")

    issue = pd.Timestamp(issue_date)
    active = {}
    for rule in payload["rules"]:
        if (int(rule["month"]), int(rule["day"])) != (
            issue.month,
            issue.day,
        ):
            continue
        hour = int(rule["hour"])
        if hour in active:
            raise ValueError(f"Duplicate d7 speed endpoint rule: {hour}")
        lower_factor = float(rule.get("lower_factor", np.nan))
        upper_factor = float(rule.get("upper_factor", np.nan))
        threshold = float(rule.get("median_speed_threshold", np.inf))
        high_ratio = float(rule.get("high_ratio", 1.0))
        if not (
            0.0 < lower_factor <= 1.025
            and 0.0 < upper_factor <= 1.025
            and threshold >= 0.0
            and 1.0 <= high_ratio <= 3.0
        ):
            raise ValueError("Invalid d7 speed endpoint factors")
        active[hour] = (lower_factor, upper_factor, threshold, high_ratio)
    if not active:
        return df

    changed = 0
    for hour, (
        lower_factor,
        upper_factor,
        threshold,
        high_ratio,
    ) in sorted(active.items()):
        selected = (df["horizon"] == 7) & (df["hour"] == hour)
        q05 = df.loc[selected, "q05"].to_numpy(dtype="float64")
        q50 = df.loc[selected, "q50"].to_numpy(dtype="float64")
        q95 = df.loc[selected, "q95"].to_numpy(dtype="float64")
        if (
            not np.all(np.isfinite(q05))
            or not np.all(np.isfinite(q50))
            or not np.all(np.isfinite(q95))
            or np.any(q05 > q50)
            or np.any(q50 > q95)
        ):
            raise ValueError("Invalid speed interval before d7 endpoint policy")
        ratio = np.where(q50 < threshold, 1.0, high_ratio)
        local_lower_factor = np.where(
            lower_factor < 1.0,
            1.0 + ratio * (lower_factor - 1.0),
            lower_factor,
        )
        local_upper_factor = np.where(
            upper_factor < 1.0,
            1.0 + ratio * (upper_factor - 1.0),
            upper_factor,
        )
        lower = np.maximum(
            0.0, q50 - local_lower_factor * (q50 - q05)
        )
        upper = q50 + local_upper_factor * (q95 - q50)
        if np.any(lower > q50) or np.any(q50 > upper):
            raise RuntimeError("d7 speed endpoint policy broke quantile ordering")
        df.loc[selected, "q05"] = lower.astype("float32")
        df.loc[selected, "q95"] = upper.astype("float32")
        changed += int(selected.sum())
    print(
        f"[infer] d7 speed endpoint policy {issue.date()} "
        f"hours={sorted(active)} rows={changed:,}",
        flush=True,
    )
    return df


def apply_d14_speed_endpoint_policy(df, issue_date, artifact: dict):
    """Apply the input-trained d14 endpoint policy without moving its center."""
    payload = artifact.get("d14_speed_endpoint_policy")
    if payload is None or not payload.get("rules"):
        return df
    if payload.get("input_only_training") is not True:
        raise ValueError("d14 speed endpoint policy lacks input-only provenance")
    if payload.get("previous_submission_inputs") != []:
        raise ValueError("d14 speed endpoint policy used a previous submission")
    if payload.get("gate", {}).get("passed") is not True:
        raise ValueError("d14 speed endpoint policy did not pass its strict gate")

    default_lower = float(payload.get("lower_factor", np.nan))
    default_upper = float(payload.get("upper_factor", np.nan))
    slots = []
    active = {}
    issue = pd.Timestamp(issue_date)
    for rule in payload["rules"]:
        slot = (int(rule["month"]), int(rule["day"]), int(rule["hour"]))
        if slot in slots:
            raise ValueError(f"Duplicate d14 speed endpoint rule: {slot}")
        slots.append(slot)
        lower_factor = float(rule.get("lower_factor", default_lower))
        upper_factor = float(rule.get("upper_factor", default_upper))
        fixed_guard = D14_SPEED_ENDPOINT_GUARDS.get(slot)
        threshold = float(
            rule.get(
                "median_speed_threshold",
                np.inf if fixed_guard is None else fixed_guard[0],
            )
        )
        high_strength = float(
            rule.get(
                "high_strength",
                3.0 if fixed_guard is None else fixed_guard[1],
            )
        )
        if not (
            0.0 <= lower_factor <= 1.0 and 0.0 <= upper_factor <= 1.0
            and threshold >= 0.0
            and 3.0 <= high_strength <= 6.0
        ):
            raise ValueError(f"Invalid d14 speed endpoint factors for {slot}")
        if slot[:2] == (issue.month, issue.day):
            active[slot[2]] = (
                lower_factor,
                upper_factor,
                threshold,
                high_strength,
            )
    if not active:
        return df
    changed = 0
    for hour, (
        lower_factor,
        upper_factor,
        threshold,
        high_strength,
    ) in sorted(active.items()):
        selected = (df["horizon"] == 14) & (df["hour"] == hour)
        q05 = df.loc[selected, "q05"].to_numpy(dtype="float64")
        q50 = df.loc[selected, "q50"].to_numpy(dtype="float64")
        q95 = df.loc[selected, "q95"].to_numpy(dtype="float64")
        if (
            not np.all(np.isfinite(q05))
            or not np.all(np.isfinite(q50))
            or not np.all(np.isfinite(q95))
            or np.any(q05 > q50)
            or np.any(q50 > q95)
        ):
            raise ValueError("Invalid speed interval before d14 endpoint policy")
        strength = np.where(q50 < threshold, 3.0, high_strength)
        ratio = strength / 3.0
        local_lower_factor = 1.0 + ratio * (lower_factor - 1.0)
        local_upper_factor = 1.0 + ratio * (upper_factor - 1.0)
        lower = np.maximum(
            0.0, q50 - local_lower_factor * (q50 - q05)
        )
        upper = q50 + local_upper_factor * (q95 - q50)
        if np.any(lower > q50) or np.any(q50 > upper):
            raise RuntimeError("d14 speed endpoint policy broke quantile ordering")
        df.loc[selected, "q05"] = lower.astype("float32")
        df.loc[selected, "q95"] = upper.astype("float32")
        changed += int(selected.sum())
    print(
        f"[infer] d14 speed endpoint policy {issue.date()} "
        f"hours={sorted(active)} rows={changed:,}",
        flush=True,
    )
    return df


def apply_fine_speed_residual_policy(df, issue_date, artifact: dict):
    """Apply strictly cross-fitted signed fine-grid speed endpoint lookups."""
    payload = artifact.get("fine_speed_residual_policy")
    if payload is None or not payload.get("rules"):
        return df
    if payload.get("input_only_training") is not True:
        raise ValueError("Fine speed residual policy lacks input-only provenance")
    if payload.get("previous_submission_inputs") != []:
        raise ValueError("Fine speed residual policy used a previous submission")
    if payload.get("new_models") != 0:
        raise ValueError("Fine speed residual policy unexpectedly adds models")
    if payload.get("gate", {}).get("passed") is not True:
        raise ValueError("Fine speed residual policy did not pass its strict gate")

    issue = pd.Timestamp(issue_date)
    active = [
        rule
        for rule in payload["rules"]
        if (int(rule["month"]), int(rule["day"]))
        == (issue.month, issue.day)
    ]
    if not active:
        return df
    touched = np.zeros(len(df), dtype=bool)
    changed = 0
    for rule in active:
        lead = int(rule["lead"])
        hour = int(rule["hour"])
        low = float(rule["lower_edge"])
        high = float(rule["upper_edge"])
        lower_offset = float(rule["lower_offset"])
        upper_offset = float(rule["upper_offset"])
        lower_blend = float(rule["lower_blend"])
        upper_blend = float(rule["upper_blend"])
        if not (
            lead in LEADS
            and hour in HOURS
            and 0.0 <= low < high
            and 0.0 < lower_blend <= 1.0
            and 0.0 < upper_blend <= 1.0
            and lower_offset <= upper_offset
        ):
            raise ValueError("Invalid fine speed residual rule")
        q50_all = df["q50"].to_numpy(dtype="float64")
        selected = (
            (df["horizon"].to_numpy() == lead)
            & (df["hour"].to_numpy() == hour)
            & (q50_all >= low)
            & (q50_all < high)
        )
        if np.any(touched & selected):
            raise ValueError("Overlapping fine speed residual rules")
        if not selected.any():
            continue
        q05 = df.loc[selected, "q05"].to_numpy(dtype="float64")
        q50 = df.loc[selected, "q50"].to_numpy(dtype="float64")
        q95 = df.loc[selected, "q95"].to_numpy(dtype="float64")
        if (
            not np.all(np.isfinite(q05))
            or not np.all(np.isfinite(q50))
            or not np.all(np.isfinite(q95))
            or np.any(q05 > q50)
            or np.any(q50 > q95)
        ):
            raise ValueError(
                "Invalid interval before fine speed residual policy"
            )
        lower_target = np.minimum(
            q50, np.maximum(0.0, q50 + lower_offset)
        )
        upper_target = np.maximum(q50, q50 + upper_offset)
        lower = np.maximum(
            0.0, q05 + lower_blend * (lower_target - q05)
        )
        upper = q95 + upper_blend * (upper_target - q95)
        lower = np.minimum(lower, q50)
        upper = np.maximum(upper, q50)
        if np.any(lower > q50) or np.any(q50 > upper):
            raise RuntimeError("Fine speed residual policy broke ordering")
        df.loc[selected, "q05"] = lower.astype("float32")
        df.loc[selected, "q95"] = upper.astype("float32")
        touched |= selected
        changed += int(selected.sum())
    print(
        f"[infer] fine speed residual policy {issue.date()} "
        f"rules={len(active)} rows={changed:,}",
        flush=True,
    )
    return df


def write_submission_frugal(
    df,
    output: Path,
    chunk_rows: int = 100_000,
    append: bool = False,
) -> None:
    """Write predictions without pandas' one-shot CSV memory spike."""
    if output.suffix.lower() != ".csv":
        raise ValueError("Phase 2 forecast submissions must be written as predictions.csv")
    with output.open("a" if append else "w", encoding="utf-8", newline="") as f:
        for start in range(0, len(df), chunk_rows):
            stop = min(start + chunk_rows, len(df))
            df.iloc[start:stop].to_csv(
                f,
                header=(start == 0 and not append),
                index=False,
                float_format="%.6f",
            )


def count_csv_rows(path: Path) -> int:
    rows = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        header = f.readline()
        if not header:
            return 0
        for _ in f:
            rows += 1
    return rows


def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def verify_training_provenance(artifacts_dir: Path) -> dict:
    manifest_path = artifacts_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing training manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("input_only_training") is not True:
        raise ValueError(
            "Artifact manifest does not certify input-only training. Run train.py "
            "from the supplied data before inference."
        )
    if manifest.get("previous_submission_inputs") != []:
        raise ValueError("Training manifest lists a previous submission input")
    return manifest


def package_submission(
    csv_path: Path,
    submission_json_path: Path,
    archive_path: Path,
) -> None:
    """Package the immutable forecast and the generated siting deliverable."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = archive_path.with_suffix(archive_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9, allowZip64=True) as zf:
        zf.write(csv_path, arcname="predictions.csv")
        zf.write(submission_json_path, arcname="submission.json")
    tmp.replace(archive_path)


def copy_auxiliary_outputs(artifacts_dir: Path, output_dir: Path) -> dict:
    """Publish train-generated non-forecast deliverables beside predictions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for name in AUXILIARY_OUTPUT_FILES:
        source = (artifacts_dir / name).resolve()
        if not source.exists():
            raise FileNotFoundError(
                f"Missing train-generated Phase 2 deliverable: {source}"
            )
        destination = (output_dir / name).resolve()
        if source != destination:
            shutil.copy2(source, destination)
        copied[name] = str(destination)
    return copied


def _default_siting_power_forecast_policy() -> dict:
    """Fallback for artifacts trained before the optional bidding output."""
    return {
        "source_height_m": 125.0,
        "hub_height_m": 170.0,
        "shear_alpha": 0.11,
        "step_hours": 6.0,
        "capacity_mw": 1210.0,
        "expected_steps": 8 * 3 * 4,
        "ordering": ["window", "horizon", "hour"],
        "output_unit": "MWh per six-hour forecast step",
        "input_only": True,
        "previous_submission_inputs": [],
    }


def _site_forecast_rows(
    predictions_path: Path,
    centre_lat: float,
    centre_lon: float,
    expected_steps: int,
) -> pd.DataFrame:
    """Read the nearest selected-centre row from every forecast block."""
    columns = [
        "window",
        "horizon",
        "hour",
        "latitude",
        "longitude",
        "q05",
        "q50",
        "q95",
        "dir_50",
    ]
    rows = []
    reader = pd.read_csv(
        predictions_path,
        usecols=columns,
        chunksize=FOOTPRINT_ROWS,
    )
    for block_index, block in enumerate(reader):
        if len(block) != FOOTPRINT_ROWS:
            raise RuntimeError(
                f"Forecast block {block_index} has {len(block)} rows; "
                f"expected {FOOTPRINT_ROWS}"
            )
        group = block[["window", "horizon", "hour"]].drop_duplicates()
        if len(group) != 1:
            raise RuntimeError(
                f"Forecast block {block_index} contains multiple time keys"
            )
        distance2 = (
            (block["latitude"].to_numpy(dtype="float64") - centre_lat) ** 2
            + (block["longitude"].to_numpy(dtype="float64") - centre_lon) ** 2
        )
        nearest = int(np.argmin(distance2))
        distance_deg = float(np.sqrt(distance2[nearest]))
        if distance_deg > 0.02:
            raise RuntimeError(
                f"Selected farm centre is {distance_deg:.5f} degrees from "
                f"the nearest forecast cell in block {block_index}"
            )
        row = block.iloc[nearest].copy()
        row["cell_distance_deg"] = distance_deg
        rows.append(row)
    site = pd.DataFrame(rows).reset_index(drop=True)
    if len(site) != expected_steps:
        raise RuntimeError(
            f"Expected {expected_steps} farm forecast steps, found {len(site)}"
        )
    keys = site[["window", "horizon", "hour"]].astype(int)
    if keys.duplicated().any():
        raise RuntimeError("Farm power forecast contains duplicate time keys")
    windows = sorted(keys["window"].unique().tolist())
    expected_keys = {
        (window, lead, hour)
        for window in windows
        for lead in LEADS
        for hour in HOURS
    }
    actual_keys = set(map(tuple, keys.to_numpy().tolist()))
    if len(windows) != 8 or actual_keys != expected_keys:
        raise RuntimeError("Farm power forecast does not cover the required 8x3x4 keys")
    quantiles = site[["q05", "q50", "q95"]].to_numpy(dtype="float64")
    if (
        not np.isfinite(quantiles).all()
        or np.any(quantiles[:, 0] > quantiles[:, 1])
        or np.any(quantiles[:, 1] > quantiles[:, 2])
        or not np.isfinite(site["dir_50"].to_numpy(dtype="float64")).all()
    ):
        raise RuntimeError("Invalid wind quantiles at the selected farm centre")
    return site


def generate_power_augmented_siting_submission(
    predictions_path: Path,
    artifacts_dir: Path,
    output_path: Path,
    kit_root: Path,
) -> dict:
    """Generate optional farm-power quantiles from the final forecast itself."""
    add_kit_paths(kit_root)
    from turbines_catalog import load_turbine
    from wind_farm_simulator import FarmLayout, WindSeries, simulate

    base_path = artifacts_dir / "siting_submission.json"
    if not base_path.exists():
        raise FileNotFoundError(f"Missing train-generated siting JSON: {base_path}")
    submission = json.loads(base_path.read_text(encoding="utf-8"))
    required = {
        "team",
        "farm_centre_lat",
        "farm_centre_lon",
        "turbine_key",
        "layout_x_m",
        "layout_y_m",
    }
    if not required.issubset(submission):
        raise RuntimeError("Train-generated siting JSON is missing required fields")
    if len(submission["layout_x_m"]) != 55 or len(submission["layout_y_m"]) != 55:
        raise RuntimeError("Siting power forecast requires exactly 55 turbines")

    policy = _default_siting_power_forecast_policy()
    artifact_path = artifacts_dir / "phase2_forecast_artifacts.joblib"
    artifact = joblib.load(artifact_path)
    policy.update(artifact.get("siting_power_forecast_policy", {}))
    if policy.get("input_only") is not True or policy.get("previous_submission_inputs"):
        raise RuntimeError("Siting power policy failed the input-only provenance gate")

    site = _site_forecast_rows(
        predictions_path,
        float(submission["farm_centre_lat"]),
        float(submission["farm_centre_lon"]),
        int(policy["expected_steps"]),
    )
    step_hours = float(policy["step_hours"])
    shear = (
        float(policy["hub_height_m"]) / float(policy["source_height_m"])
    ) ** float(policy["shear_alpha"])
    times = pd.Timestamp("2000-01-01") + pd.to_timedelta(
        np.arange(len(site), dtype="float64") * step_hours,
        unit="h",
    )
    directions = site["dir_50"].to_numpy(dtype="float64") % 360.0
    turbine = load_turbine(str(submission["turbine_key"]))
    layout = FarmLayout(
        np.asarray(submission["layout_x_m"], dtype="float64"),
        np.asarray(submission["layout_y_m"], dtype="float64"),
        turbine,
    )
    power_paths = []
    for column in ("q05", "q50", "q95"):
        wind = WindSeries(
            pd.DataFrame(
                {
                    "time": times,
                    "ws": np.maximum(
                        0.0,
                        site[column].to_numpy(dtype="float64") * shear,
                    ),
                    "wd": directions,
                }
            )
        )
        result = simulate(layout, wind, annualisation=False)
        power_paths.append(result.farm_power_mw * step_hours)
    power = np.sort(np.column_stack(power_paths), axis=1)
    capacity_per_step = float(policy["capacity_mw"]) * step_hours
    power = np.clip(power, 0.0, capacity_per_step)
    if (
        not np.isfinite(power).all()
        or np.any(power[:, 0] > power[:, 1])
        or np.any(power[:, 1] > power[:, 2])
    ):
        raise RuntimeError("Generated farm-power quantiles are invalid")

    submission["predicted_q05"] = np.round(power[:, 0], 3).tolist()
    submission["predicted_q50"] = np.round(power[:, 1], 3).tolist()
    submission["predicted_q95"] = np.round(power[:, 2], 3).tolist()
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(submission, indent=2), encoding="utf-8")
    tmp.replace(output_path)
    return {
        "path": str(output_path),
        "steps": int(len(site)),
        "unit": policy["output_unit"],
        "nearest_cell_max_distance_deg": float(site["cell_distance_deg"].max()),
        "q05_mean": float(power[:, 0].mean()),
        "q50_mean": float(power[:, 1].mean()),
        "q95_mean": float(power[:, 2].mean()),
    }


def run_lightweight_coordinator(args: argparse.Namespace) -> None:
    """Run isolated workers without importing the modelling stack in the parent."""
    t0 = time.time()
    args.output = args.output.expanduser().resolve()
    args.artifacts_dir = args.artifacts_dir.expanduser().resolve()
    final_metadata = locate_final_inference_metadata(args.data_root, args.eval_year)
    args.eval_year = int(final_metadata["eval_year"])
    archive = (args.archive or args.output.with_suffix(".zip")).expanduser().resolve()
    artifact_path = args.artifacts_dir / "phase2_forecast_artifacts.joblib"
    clim_path = args.artifacts_dir / "climatology_coarse.npz"
    if not artifact_path.exists() or not clim_path.exists():
        raise FileNotFoundError(
            f"Missing trained artifacts under {args.artifacts_dir}; run train.py first"
        )
    train_manifest = verify_training_provenance(args.artifacts_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_windows = 8
    chunk_root = args.output.parent / f".{args.output.stem}_windows"
    chunk_root.mkdir(parents=True, exist_ok=True)
    cache_manifest = chunk_root / "cache_manifest.json"
    cache_spec = {
        "artifact_path": str(artifact_path),
        "artifact_size": artifact_path.stat().st_size,
        "artifact_mtime_ns": artifact_path.stat().st_mtime_ns,
        "inference_code_sha256": code_sha256(),
        "kit_dir": str(args.kit_dir) if args.kit_dir is not None else None,
        "data_root": str(args.data_root) if args.data_root is not None else None,
        "eval_year": args.eval_year,
        "final_inference_metadata_sha256": final_metadata["metadata_sha256"],
        "window_base": args.window_base,
        "speed_width_scale": args.speed_width_scale,
        "dir_halfwidth_scale": args.dir_halfwidth_scale,
        "dir_halfwidth_deg": args.dir_halfwidth_deg,
        "d7_center_policy_max_weight": args.d7_center_policy_max_weight,
        "d1_context_blend_scale": args.d1_context_blend_scale,
        "dir_halfwidth_cap_deg": args.dir_halfwidth_cap_deg,
    }
    cached_spec = None
    if cache_manifest.exists():
        try:
            cached_spec = json.loads(cache_manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached_spec = None
    if cached_spec != cache_spec:
        for stale_chunk in chunk_root.glob("window_*.csv"):
            stale_chunk.unlink()
        cache_manifest.write_text(
            json.dumps(cache_spec, indent=2), encoding="utf-8"
        )

    worker_files: list[Path] = []
    expected_worker_rows = 3 * 4 * FOOTPRINT_ROWS
    print(f"[infer] spawning {n_windows} isolated window workers", flush=True)
    for idx in range(n_windows):
        worker_output = chunk_root / f"window_{idx:02d}.csv"
        if worker_output.exists():
            cached_rows = count_csv_rows(worker_output)
            if cached_rows == expected_worker_rows:
                print(
                    f"[infer] reusing validated window={idx + args.window_base} "
                    f"rows={cached_rows:,}",
                    flush=True,
                )
                worker_files.append(worker_output)
                continue
            worker_output.unlink()

        cmd = [
            sys.executable, "-u", str(Path(__file__).resolve()),
            "--artifacts-dir", str(args.artifacts_dir),
            "--output", str(worker_output),
            "--window-base", str(args.window_base),
            "--worker-window-index", str(idx),
            "--speed-width-scale", args.speed_width_scale,
            "--dir-halfwidth-scale", args.dir_halfwidth_scale,
            "--dir-halfwidth-deg", args.dir_halfwidth_deg,
            "--d7-center-policy-max-weight",
            str(args.d7_center_policy_max_weight),
            "--d1-context-blend-scale",
            str(args.d1_context_blend_scale),
            "--dir-halfwidth-cap-deg",
            args.dir_halfwidth_cap_deg,
        ]
        if args.kit_dir is not None:
            cmd.extend(["--kit-dir", str(args.kit_dir)])
        if args.data_root is not None:
            cmd.extend(["--data-root", str(args.data_root)])
        if args.eval_year is not None:
            cmd.extend(["--eval-year", str(args.eval_year)])
        last_error = None
        for attempt in range(1, max(1, args.worker_retries) + 1):
            if worker_output.exists():
                worker_output.unlink()
            try:
                subprocess.run(cmd, check=True)
                last_error = None
                break
            except subprocess.CalledProcessError as exc:
                last_error = exc
                print(
                    f"[infer] window={idx + args.window_base} failed "
                    f"attempt {attempt}/{args.worker_retries} exit={exc.returncode}",
                    flush=True,
                )
        if last_error is not None:
            raise last_error
        if count_csv_rows(worker_output) != expected_worker_rows:
            raise RuntimeError(f"Incomplete worker output: {worker_output}")
        worker_files.append(worker_output)

    tmp_csv = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp_csv.open("wb") as dst:
        for idx, worker_file in enumerate(worker_files):
            with worker_file.open("rb") as src:
                if idx:
                    src.readline()
                shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
    expected_rows = n_windows * expected_worker_rows
    rows = count_csv_rows(tmp_csv)
    if rows != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, assembled {rows}")
    tmp_csv.replace(args.output)
    auxiliary_outputs = copy_auxiliary_outputs(
        args.artifacts_dir, args.output.parent
    )
    submission_json_path = args.output.parent / "submission.json"
    power_forecast = generate_power_augmented_siting_submission(
        args.output,
        args.artifacts_dir,
        submission_json_path,
        resolve_kit_root(args.kit_dir),
    )
    auxiliary_outputs["submission.json"] = str(submission_json_path)
    package_submission(args.output, submission_json_path, archive)
    run_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_only_pipeline": True,
        "previous_submission_inputs": [],
        "train_manifest": train_manifest,
        "inference_code_sha256": code_sha256(),
        "artifact_path": str(artifact_path),
        "output": str(args.output),
        "archive": str(archive),
        "eval_year": args.eval_year,
        "final_inference_metadata": final_metadata,
        "speed_width_scale": parse_lead_float_map(args.speed_width_scale),
        "dir_halfwidth_scale": parse_lead_float_map(args.dir_halfwidth_scale),
        "dir_halfwidth_deg": parse_lead_float_map(args.dir_halfwidth_deg),
        "d7_center_policy_max_weight": args.d7_center_policy_max_weight,
        "d1_context_blend_scale": args.d1_context_blend_scale,
        "dir_halfwidth_cap_deg": parse_lead_float_map(
            args.dir_halfwidth_cap_deg
        ),
        "rows": rows,
        "windows": n_windows,
        "siting_power_forecast": power_forecast,
        "auxiliary_outputs": auxiliary_outputs,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    for worker_file in worker_files:
        worker_file.unlink()
    cache_manifest.unlink(missing_ok=True)
    try:
        chunk_root.rmdir()
    except OSError:
        pass
    print(f"[infer] wrote {args.output}")
    print(f"[infer] wrote {archive}")
    for path in auxiliary_outputs.values():
        print(f"[infer] wrote {path}")
    print(f"[infer] rows={rows:,}")


def constrain_model_threads(obj) -> None:
    """Force loaded sklearn/LightGBM-style estimators to one prediction thread."""
    if isinstance(obj, dict):
        for value in obj.values():
            constrain_model_threads(value)
        return
    if isinstance(obj, (list, tuple)):
        for value in obj:
            constrain_model_threads(value)
        return
    if hasattr(obj, "set_params") and hasattr(obj, "get_params"):
        try:
            params = obj.get_params()
            updates = {}
            if "n_jobs" in params:
                updates["n_jobs"] = 1
            if "num_threads" in params:
                updates["num_threads"] = 1
            if updates:
                obj.set_params(**updates)
        except Exception:
            pass


def inference_speed_inflation(artifact: dict) -> dict:
    """Use the training-gated d1 interval factor from the accepted speed center."""
    values = dict(artifact["speed_inflation"])
    values.update(PROTECTED_SPEED_INFLATION)
    return values


def apply_masked_qmos_refit_to_blocks(
    pipeline, submission, artifact, fields, blocks, window_id, speed_inflation
):
    """Replace speed rows only where the strict all-years refit mask is active."""
    for lead in LEADS:
        for hour in HOURS:
            refit_key = (lead, hour, "qmos_refit_spd")
            rule_key = (lead, hour, "qmos_refit_rule")
            if refit_key not in fields or rule_key not in fields:
                continue
            rule = fields[rule_key]
            base_stack = np.asarray(fields[(lead, hour, "spd")], dtype="float64")
            refit_stack = np.asarray(fields[refit_key], dtype="float64")
            weight = float(rule["weight"])
            blended_stack = np.sort(
                base_stack + weight * (refit_stack - base_stack), axis=0
            ).astype("float32")

            base_u, base_v = fields.get(
                (lead, hour, "speed_det"), fields[(lead, hour, "det")]
            )
            direction = np.arctan2(-base_u, -base_v)
            median = blended_stack[1].astype("float64")
            candidate_u = -median * np.sin(direction)
            candidate_v = -median * np.cos(direction)
            fine_u, fine_v = pipeline.dn.downscale(
                artifact["downscaler"], candidate_u, candidate_v
            )
            q50 = np.hypot(fine_u, fine_v)
            candidate_fields = dict(fields)
            candidate_fields[(lead, hour, "spd")] = blended_stack
            q05, q95 = pipeline._speed_interval(
                candidate_fields,
                lead,
                hour,
                q50,
                k=float(speed_inflation.get(lead, 1.0)),
            )
            zeros = np.zeros_like(q50)
            candidate_rows = submission.field_to_rows(
                window_id,
                lead,
                hour,
                q05,
                q50,
                q95,
                zeros,
                zeros,
                zeros,
            )
            block_index = LEADS.index(lead) * len(HOURS) + HOURS.index(hour)
            block = blocks[block_index]
            latitude = block["latitude"].to_numpy(dtype="float64")
            longitude = block["longitude"].to_numpy(dtype="float64")
            spatial = (
                (latitude >= np.median(latitude)).astype("int16") * 2
                + (longitude >= np.median(longitude)).astype("int16")
            )
            width = (
                block["q95"].to_numpy(dtype="float64")
                - block["q05"].to_numpy(dtype="float64")
            )
            edges = np.quantile(width, (0.0, 0.25, 0.5, 0.75, 1.0))
            edges[0], edges[-1] = -np.inf, np.inf
            width_bin = np.clip(np.digitize(width, edges) - 1, 0, 3)
            selected = np.isin(
                spatial,
                np.asarray(rule["spatial_bins"], dtype="int16"),
            )
            for spatial_bin, width_value in rule.get(
                "exclude_spatial_width", ()
            ):
                selected &= ~(
                    (spatial == int(spatial_bin))
                    & (width_bin == int(width_value))
                )
            speed_columns = ["q05", "q50", "q95"]
            block.loc[selected, speed_columns] = candidate_rows.loc[
                selected, speed_columns
            ].to_numpy()
            print(
                f"[infer] qMOS refit applied window={window_id} d{lead} "
                f"h{hour:02d} rows={int(selected.sum()):,}",
                flush=True,
            )
    return blocks


def apply_d1_speed_context_to_blocks(
    pipeline, submission, artifact, fields, blocks, window_id
):
    """Apply the held-year-gated February upper-endpoint challenger."""
    policy = artifact.get("d1_speed_context")
    if policy is None:
        return blocks
    for hour in HOURS:
        key = (1, hour, "d1_context_spd")
        if key not in fields:
            continue
        candidate_stack = np.asarray(fields[key], dtype="float64")
        base_u, base_v = fields.get(
            (1, hour, "speed_det"), fields[(1, hour, "det")]
        )
        direction = np.arctan2(-base_u, -base_v)
        candidate_median = candidate_stack[1]
        candidate_u = -candidate_median * np.sin(direction)
        candidate_v = -candidate_median * np.cos(direction)
        fine_u, fine_v = pipeline.dn.downscale(
            artifact["downscaler"], candidate_u, candidate_v
        )
        candidate_q50 = np.hypot(fine_u, fine_v)
        candidate_fields = dict(fields)
        candidate_fields[(1, hour, "spd")] = candidate_stack
        _, candidate_q95 = pipeline._speed_interval(
            candidate_fields,
            1,
            hour,
            candidate_q50,
            k=float(policy["candidate_inflation"]),
        )
        zeros = np.zeros_like(candidate_q50)
        candidate_rows = submission.field_to_rows(
            window_id,
            1,
            hour,
            zeros,
            zeros,
            candidate_q95,
            zeros,
            zeros,
            zeros,
        )
        block_index = HOURS.index(hour)
        block = blocks[block_index]
        base_q05 = block["q05"].to_numpy(dtype="float64")
        base_q50 = block["q50"].to_numpy(dtype="float64")
        base_q95 = block["q95"].to_numpy(dtype="float64")
        width = base_q95 - base_q05
        edges = np.quantile(width, (0.0, 0.25, 0.5, 0.75, 1.0))
        edges[0], edges[-1] = -np.inf, np.inf
        width_bin = np.clip(np.digitize(width, edges) - 1, 0, 3)
        proposed_q95 = candidate_rows["q95"].to_numpy(dtype="float64")
        active = np.isin(
            width_bin, np.asarray(policy["width_bins"], dtype="int8")
        ) & (proposed_q95 < base_q95)
        blended_q95 = base_q95.copy()
        blended_q95[active] += float(policy["upper_blend"]) * (
            proposed_q95[active] - blended_q95[active]
        )
        block.loc[:, "q95"] = np.maximum(blended_q95, base_q50).astype(
            block["q95"].dtype, copy=False
        )
        print(
            f"[infer] d1 context upper endpoint window={window_id} "
            f"h{hour:02d} rows={int(active.sum()):,}",
            flush=True,
        )
    return blocks


def downscale_window_decoupled(pipeline, submission, artifact, fields, window_id):
    """Return candidate blocks and an untouched protected direction path."""
    speed_inflation = inference_speed_inflation(artifact)
    has_refit = any(
        (lead, hour, "qmos_refit_spd") in fields
        for lead in LEADS
        for hour in HOURS
    ) or any((1, hour, "d1_context_spd") in fields for hour in HOURS)
    replacements = [
        (lead, hour)
        for lead in LEADS
        for hour in HOURS
        if (lead, hour, "speed_det") in fields
    ]
    if not replacements:
        blocks = pipeline.downscale_window(
            artifact["downscaler"],
            fields,
            artifact["dir_offsets"],
            window_id,
            spd_infl=speed_inflation,
            dir_off=artifact["fine_dir_offsets"],
        )
        protected_blocks = (
            [block.copy(deep=True) for block in blocks] if has_refit else None
        )
        candidate_blocks = apply_masked_qmos_refit_to_blocks(
            pipeline, submission, artifact, fields, blocks, window_id,
            speed_inflation,
        )
        candidate_blocks = apply_d1_speed_context_to_blocks(
            pipeline,
            submission,
            artifact,
            fields,
            candidate_blocks,
            window_id,
        )
        return candidate_blocks, protected_blocks

    baseline_fields = dict(fields)
    for lead, hour in replacements:
        baseline_fields[(lead, hour, "det")] = fields[(lead, hour, "speed_det")]
    blocks = pipeline.downscale_window(
        artifact["downscaler"],
        baseline_fields,
        artifact["dir_offsets"],
        window_id,
        spd_infl=speed_inflation,
        dir_off=artifact["fine_dir_offsets"],
    )

    direction_columns = ["dir_05", "dir_50", "dir_95"]
    for lead, hour in replacements:
        direction_offset = artifact["fine_dir_offsets"].get(
            lead, artifact["dir_offsets"][lead]
        )
        candidate_u, candidate_v = fields[(lead, hour, "det")]
        fine_u, fine_v = pipeline.dn.downscale(
            artifact["downscaler"], candidate_u, candidate_v
        )
        direction_50 = np.degrees(np.arctan2(-fine_u, -fine_v)) % 360.0
        zeros = np.zeros_like(direction_50)
        direction_rows = submission.field_to_rows(
            window_id,
            lead,
            hour,
            zeros,
            zeros,
            zeros,
            direction_50 - direction_offset,
            direction_50,
            direction_50 + direction_offset,
        )
        block_index = LEADS.index(lead) * len(HOURS) + HOURS.index(hour)
        blocks[block_index].loc[:, direction_columns] = direction_rows[
            direction_columns
        ].to_numpy()
    protected_blocks = (
        [block.copy(deep=True) for block in blocks] if has_refit else None
    )
    candidate_blocks = apply_masked_qmos_refit_to_blocks(
        pipeline, submission, artifact, fields, blocks, window_id,
        speed_inflation,
    )
    candidate_blocks = apply_d1_speed_context_to_blocks(
        pipeline,
        submission,
        artifact,
        fields,
        candidate_blocks,
        window_id,
    )
    return candidate_blocks, protected_blocks


def apply_d1_dense_daily_spatial_policy(
    pipeline,
    submission,
    artifact,
    fields,
    frame,
    issue_date,
    window_id,
    speed_width_scale,
):
    """Blend dense-daily endpoints only in the 16 independently confirmed cells."""
    policy = artifact.get("d1_dense_daily")
    if policy is None:
        return frame
    issue_date = pd.Timestamp(issue_date)
    rules = [
        rule
        for rule in policy.get("rules", ())
        if (int(rule["month"]), int(rule["day"]))
        == (issue_date.month, issue_date.day)
    ]
    if not rules:
        return frame
    if not policy.get("gate", {}).get("passed", False):
        raise RuntimeError("Dense-daily d1 policy lacks a passed strict gate")
    expected_scale = float(policy["post_width_scale"])
    actual_scale = float(speed_width_scale.get(1, 1.0))
    if not np.isclose(actual_scale, expected_scale, atol=1e-12):
        print(
            f"[infer] dense-daily d1 skipped: validated width scale "
            f"{expected_scale:.3f}, requested {actual_scale:.3f}",
            flush=True,
        )
        return frame

    inflation = float(inference_speed_inflation(artifact)[1])
    total_active = 0
    for hour in HOURS:
        hour_rules = [rule for rule in rules if int(rule["hour"]) == hour]
        candidate_key = (1, hour, "d1_dense_daily_spd")
        if not hour_rules:
            continue
        if candidate_key not in fields:
            raise RuntimeError(
                f"Missing dense-daily d1 candidate at {issue_date.date()} h{hour:02d}"
            )
        candidate_fields = dict(fields)
        candidate_fields[(1, hour, "spd")] = fields[candidate_key]
        base_u, base_v = fields.get(
            (1, hour, "speed_det"), fields[(1, hour, "det")]
        )
        fine_u, fine_v = pipeline.dn.downscale(
            artifact["downscaler"], base_u, base_v
        )
        fine_center = np.hypot(fine_u, fine_v)
        lower_grid, upper_grid = pipeline._speed_interval(
            candidate_fields,
            1,
            hour,
            fine_center,
            k=inflation,
        )
        zeros = np.zeros_like(fine_center)
        candidate_rows = submission.field_to_rows(
            window_id,
            1,
            hour,
            lower_grid,
            fine_center,
            upper_grid,
            zeros,
            zeros,
            zeros,
        )
        selected_rows = (frame["horizon"] == 1) & (frame["hour"] == hour)
        block = frame.loc[selected_rows]
        if len(block) != len(candidate_rows):
            raise RuntimeError("Dense-daily d1 footprint row order changed")
        center = block["q50"].to_numpy(dtype="float64")
        base_lower = block["q05"].to_numpy(dtype="float64")
        base_upper = block["q95"].to_numpy(dtype="float64")
        candidate_lower = candidate_rows["q05"].to_numpy(dtype="float64")
        candidate_upper = candidate_rows["q95"].to_numpy(dtype="float64")
        candidate_lower = center - expected_scale * (center - candidate_lower)
        candidate_upper = center + expected_scale * (candidate_upper - center)
        candidate_lower = np.minimum(np.maximum(candidate_lower, 0.0), center)
        candidate_upper = np.maximum(candidate_upper, center)

        latitude = block["latitude"].to_numpy(dtype="float64")
        longitude = block["longitude"].to_numpy(dtype="float64")
        spatial = (
            (latitude >= np.median(latitude)).astype("int8") * 2
            + (longitude >= np.median(longitude)).astype("int8")
        )
        weights = np.zeros(len(block), dtype="float64")
        for rule in hour_rules:
            spatial_bin = int(rule["spatial_bin"])
            if np.any(weights[spatial == spatial_bin] > 0.0):
                raise RuntimeError("Duplicate dense-daily d1 activation rule")
            weights[spatial == spatial_bin] = float(rule["weight"])
        active = weights > 0.0
        deployed_lower = base_lower.copy()
        deployed_upper = base_upper.copy()
        deployed_lower[active] += weights[active] * (
            candidate_lower[active] - deployed_lower[active]
        )
        deployed_upper[active] += weights[active] * (
            candidate_upper[active] - deployed_upper[active]
        )
        deployed_lower = np.minimum(np.maximum(deployed_lower, 0.0), center)
        deployed_upper = np.maximum(deployed_upper, center)
        frame.loc[selected_rows, "q05"] = deployed_lower.astype(
            frame["q05"].dtype, copy=False
        )
        frame.loc[selected_rows, "q95"] = deployed_upper.astype(
            frame["q95"].dtype, copy=False
        )
        total_active += int(active.sum())
    print(
        f"[infer] dense-daily d1 spatial policy window={window_id} "
        f"rows={total_active:,}",
        flush=True,
    )
    return frame


def run_window_worker(
    args: argparse.Namespace,
    windows,
    splits,
    pipeline,
    fh,
    config,
    artifact,
    submission,
) -> dict:
    idx = int(args.worker_window_index)
    if idx < 0 or idx >= len(windows):
        raise ValueError(f"worker-window-index out of range: {idx}")
    window = windows[idx]
    issue_date = pipeline.issue_date_of(window)
    window_id = idx + args.window_base
    speed_width_scale = parse_lead_float_map(args.speed_width_scale)
    dir_halfwidth_scale = parse_lead_float_map(args.dir_halfwidth_scale)
    dir_halfwidth_deg = parse_lead_float_map(args.dir_halfwidth_deg)
    dir_halfwidth_caps = parse_lead_float_map(args.dir_halfwidth_cap_deg)

    d1_context_blend_scale = float(args.d1_context_blend_scale)
    if not 0.0 < d1_context_blend_scale <= 2.0:
        raise ValueError("d1 context blend scale must be in (0, 2]")
    if d1_context_blend_scale != 1.0:
        d1_context = artifact.get("d1_speed_context")
        if not isinstance(d1_context, dict):
            raise ValueError("D1 context scaling requires the trained policy")
        base_blend = float(d1_context["upper_blend"])
        d1_context["upper_blend"] = min(
            1.0, base_blend * d1_context_blend_scale
        )
        print(
            f"[infer] d1 context upper blend "
            f"{base_blend:.4f}->{d1_context['upper_blend']:.4f}",
            flush=True,
        )

    maximum_center_weight = float(args.d7_center_policy_max_weight)
    if not 0.0 <= maximum_center_weight <= 1.0:
        raise ValueError("d7 center-policy maximum weight must be in [0, 1]")
    if maximum_center_weight < 1.0:
        direction_models = artifact.get("direction_models")
        if not isinstance(direction_models, dict):
            raise ValueError("D7 center-policy gating requires direction models")
        policy = direction_models.get("d7_center_policy", {})
        retained = {
            slot: rule
            for slot, rule in policy.items()
            if float(rule[1]) <= maximum_center_weight
        }
        direction_models["d7_center_policy"] = retained
        print(
            f"[infer] d7 center-transfer policy gate window={window_id} "
            f"retained={len(retained)}/{len(policy)} "
            f"maximum_weight={maximum_center_weight:.3f}",
            flush=True,
        )

    if "qmos" in artifact and "direction_models" in artifact:
        fields = coarse_fields_hybrid(fh, config, artifact, issue_date)
    else:
        fields = pipeline.coarse_fields(
            artifact["mos"],
            artifact["qmos"],
            artifact["conformal_adjust"],
            issue_date,
        )
    window_blocks, protected_blocks = downscale_window_decoupled(
        pipeline, submission, artifact, fields, window_id
    )
    df = normalize_directions(submission.assemble(window_blocks))
    direction_df = (
        normalize_directions(submission.assemble(protected_blocks))
        if protected_blocks is not None
        else None
    )
    direction_source = direction_df if direction_df is not None else df
    d7_mask = direction_source["horizon"] == 7
    raw_d7_center = pd.Series(
        direction_source.loc[d7_mask, "dir_50"].to_numpy(dtype="float64"),
        index=direction_source.index[d7_mask],
        dtype="float64",
    )
    if artifact.get("fine_d7_direction_models") is None:
        fine_d7_correction = None
    else:
        raw_d10_directions = (
            fine_raw_d10_directions(pipeline, artifact, fields)
            if artifact.get("d7_d10_tendency_policy")
            else None
        )
        fine_d7_correction = predict_fine_d7_direction_correction(
            direction_source,
            issue_date,
            artifact,
            config=config,
            raw_d10_directions=raw_d10_directions,
        )
    if (
        speed_width_scale
        or dir_halfwidth_scale
        or dir_halfwidth_deg
        or artifact.get("d1_direction_speed_interval") is not None
        or artifact.get("d7_direction_interval_policy") is not None
        or artifact.get("d14_direction_speed_interval") is not None
    ):
        if direction_df is not None:
            # Complete the protected path before evaluating any candidate-only
            # postprocessing. This prevents shared estimator state from
            # influencing a direction decision at a numerical threshold.
            direction_df = apply_interval_postprocess(
                direction_df,
                speed_width_scale=speed_width_scale,
                dir_halfwidth_scale=dir_halfwidth_scale,
                dir_halfwidth_deg=dir_halfwidth_deg,
                d1_direction_speed_interval=artifact.get(
                    "d1_direction_speed_interval"
                ),
                d7_direction_interval_policy=artifact.get(
                    "d7_direction_interval_policy"
                ),
                d14_direction_speed_interval=artifact.get(
                    "d14_direction_speed_interval"
                ),
                issue_date=issue_date,
            )
            if speed_width_scale:
                df = apply_interval_postprocess(
                    df,
                    speed_width_scale=speed_width_scale,
                    dir_halfwidth_scale={},
                    dir_halfwidth_deg={},
                    d1_direction_speed_interval=None,
                    d7_direction_interval_policy=None,
                    d14_direction_speed_interval=None,
                    issue_date=issue_date,
                )
        else:
            df = apply_interval_postprocess(
                df,
                speed_width_scale=speed_width_scale,
                dir_halfwidth_scale=dir_halfwidth_scale,
                dir_halfwidth_deg=dir_halfwidth_deg,
                d1_direction_speed_interval=artifact.get(
                    "d1_direction_speed_interval"
                ),
                d7_direction_interval_policy=artifact.get(
                    "d7_direction_interval_policy"
                ),
                d14_direction_speed_interval=artifact.get(
                    "d14_direction_speed_interval"
                ),
                issue_date=issue_date,
            )
    df = apply_d1_dense_daily_spatial_policy(
        pipeline,
        submission,
        artifact,
        fields,
        df,
        issue_date,
        window_id,
        speed_width_scale,
    )
    if direction_df is not None:
        direction_df = apply_d7_speed_endpoint_policy(
            direction_df, issue_date, artifact
        )
        direction_df = apply_d14_speed_endpoint_policy(
            direction_df, issue_date, artifact
        )
        direction_df = apply_fine_speed_residual_policy(
            direction_df, issue_date, artifact
        )
    else:
        df = apply_d7_speed_endpoint_policy(df, issue_date, artifact)
        df = apply_d14_speed_endpoint_policy(df, issue_date, artifact)
        df = apply_fine_speed_residual_policy(df, issue_date, artifact)
    print(f"[infer-worker] window={window_id} applying direction policies", flush=True)
    direction_target = direction_df if direction_df is not None else df
    direction_target = apply_fine_d14_climatology(
        direction_target, issue_date, artifact
    )
    if fine_d7_correction is not None:
        direction_target = apply_fine_d7_direction_correction(
            direction_target, fine_d7_correction
        )
    direction_target = apply_fine_d7_climatology(
        direction_target, issue_date, artifact
    )
    direction_target = apply_d7_conditional_endpoint(
        direction_target, issue_date, artifact, config
    )
    direction_target = apply_d7_pressure_policy(
        direction_target, issue_date, artifact, raw_d7_center
    )
    direction_target = cap_direction_intervals(
        direction_target, dir_halfwidth_caps
    )
    if direction_df is not None:
        df = apply_d7_speed_endpoint_policy(df, issue_date, artifact)
        df = apply_d14_speed_endpoint_policy(df, issue_date, artifact)
        df = apply_fine_speed_residual_policy(df, issue_date, artifact)
        direction_columns = ["dir_05", "dir_50", "dir_95"]
        df.loc[:, direction_columns] = direction_target[
            direction_columns
        ].to_numpy()
    else:
        df = direction_target
    df = apply_external_trajectory_policy(df, issue_date, artifact)
    df = apply_hres_analog_policy(df, issue_date, artifact)
    print(f"[infer-worker] window={window_id} validating rows", flush=True)
    df = normalize_directions(df)
    validate_submission_rows(df)
    print(f"[infer-worker] window={window_id} writing rows", flush=True)
    write_submission_frugal(df, args.output, append=args.append_output)
    summary = {
        "window": window_id,
        "issue_date": str(issue_date.date()),
        "rows": int(len(df)),
        "q50_sum": float(df["q50"].sum()),
    }
    print(
        f"[infer-worker] window={window_id} issue={issue_date.date()} "
        f"rows={summary['rows']} written"
    )
    return summary


def run_window_subprocesses(args: argparse.Namespace, windows, splits) -> dict:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    print(f"[infer] spawning {len(windows)} per-window workers")
    for idx, _window in enumerate(windows):
        cmd = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--artifacts-dir",
            str(args.artifacts_dir),
            "--output",
            str(args.output),
            "--window-base",
            str(args.window_base),
            "--worker-window-index",
            str(idx),
        ]
        if args.kit_dir is not None:
            cmd.extend(["--kit-dir", str(args.kit_dir)])
        if args.data_root is not None:
            cmd.extend(["--data-root", str(args.data_root)])
        if args.eval_year is not None:
            cmd.extend(["--eval-year", str(args.eval_year)])
        if args.speed_width_scale:
            cmd.extend(["--speed-width-scale", args.speed_width_scale])
        if args.dir_halfwidth_scale:
            cmd.extend(["--dir-halfwidth-scale", args.dir_halfwidth_scale])
        if args.dir_halfwidth_deg:
            cmd.extend(["--dir-halfwidth-deg", args.dir_halfwidth_deg])
        if idx > 0:
            cmd.append("--append-output")
        last_error = None
        for attempt in range(1, max(1, args.worker_retries) + 1):
            try:
                subprocess.run(cmd, check=True)
                last_error = None
                break
            except subprocess.CalledProcessError as exc:
                last_error = exc
                print(
                    f"[infer] worker window={idx + args.window_base} "
                    f"failed attempt {attempt}/{args.worker_retries} "
                    f"exit={exc.returncode}",
                    flush=True,
                )
        if last_error is not None:
            raise last_error

    rows = count_csv_rows(args.output)
    expected_rows = len(windows) * 3 * 4 * FOOTPRINT_ROWS
    if rows != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} prediction rows, wrote {rows}")
    return {
        "rows": rows,
        "windows": len(windows),
        "footprint_rows": FOOTPRINT_ROWS,
        "q50_mean": None,
    }


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    kit_root = resolve_kit_root(args.kit_dir)
    add_kit_paths(kit_root)
    configure_data_root(args.data_root)

    import config
    import forecast_hres as fh
    import forecast_pipeline as pipeline
    import build_forecast_submission as submission
    import splits

    eval_year = resolve_final_eval_year(config, args.eval_year)
    args.eval_year = eval_year
    final_metadata = inspect_final_inference_metadata(
        config.inference_root(), eval_year
    )

    artifact_path = args.artifacts_dir / "phase2_forecast_artifacts.joblib"
    clim_path = args.artifacts_dir / "climatology_coarse.npz"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing artifact bundle: {artifact_path}")
    if not clim_path.exists():
        raise FileNotFoundError(f"Missing climatology artifact: {clim_path}")

    windows = splits.eval_windows(eval_year)
    print(config.describe())
    print(f"[infer] artifact: {artifact_path}")
    print(f"[infer] windows: {len(windows)} eval_year={eval_year}")

    if args.worker_window_index is not None or args.single_process:
        artifact = joblib.load(artifact_path)
        constrain_model_threads(artifact)
        fh.CLIM_CACHE = clim_path
        fh._climatology.cache_clear()
        print(f"[infer] loaded artifact: {artifact_path}")

    if args.worker_window_index is not None:
        run_window_worker(
            args,
            windows,
            splits,
            pipeline,
            fh,
            config,
            artifact,
            submission,
        )
        return

    if not args.single_process:
        summary = run_window_subprocesses(args, windows, splits)
        run_manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "kit_root": str(kit_root),
            "data_root": os.environ.get("PHASE2_DATA_ROOT"),
            "artifact_path": str(artifact_path),
            "output": str(args.output),
            "eval_year": eval_year,
            "final_inference_metadata": final_metadata,
            "window_base": args.window_base,
            "speed_width_scale": parse_lead_float_map(args.speed_width_scale),
            "dir_halfwidth_scale": parse_lead_float_map(args.dir_halfwidth_scale),
            "dir_halfwidth_deg": parse_lead_float_map(args.dir_halfwidth_deg),
            "elapsed_seconds": round(time.time() - t0, 2),
            "submission": summary,
            "execution": "per-window subprocess workers",
        }
        manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
        manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
        print(f"[infer] wrote {args.output}")
        print(f"[infer] wrote {manifest_path}")
        print(f"[infer] rows={summary['rows']} footprint_rows={summary['footprint_rows']}")
        return

    speed_width_scale = parse_lead_float_map(args.speed_width_scale)
    dir_halfwidth_scale = parse_lead_float_map(args.dir_halfwidth_scale)
    dir_halfwidth_deg = parse_lead_float_map(args.dir_halfwidth_deg)
    if speed_width_scale or dir_halfwidth_scale or dir_halfwidth_deg:
        print(
            "[infer] interval postprocess "
            f"speed={speed_width_scale} "
            f"dir_scale={dir_halfwidth_scale} "
            f"dir_deg={dir_halfwidth_deg}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    total_rows = 0
    q50_sum = 0.0
    windows_written = 0
    for idx, window in enumerate(windows):
        issue_date = pipeline.issue_date_of(window)
        if "qmos" in artifact and "direction_models" in artifact:
            fields = coarse_fields_hybrid(fh, config, artifact, issue_date)
        else:
            fields = pipeline.coarse_fields(
                artifact["mos"],
                artifact["qmos"],
                artifact["conformal_adjust"],
                issue_date,
            )
        window_id = idx + args.window_base
        window_blocks, protected_blocks = downscale_window_decoupled(
            pipeline, submission, artifact, fields, window_id
        )
        df = normalize_directions(submission.assemble(window_blocks))
        direction_df = (
            normalize_directions(submission.assemble(protected_blocks))
            if protected_blocks is not None
            else None
        )
        direction_source = direction_df if direction_df is not None else df
        d7_mask = direction_source["horizon"] == 7
        raw_d7_center = pd.Series(
            direction_source.loc[d7_mask, "dir_50"].to_numpy(dtype="float64"),
            index=direction_source.index[d7_mask],
            dtype="float64",
        )
        if artifact.get("fine_d7_direction_models") is None:
            fine_d7_correction = np.zeros(len(df), dtype="float32")
        else:
            raw_d10_directions = (
                fine_raw_d10_directions(pipeline, artifact, fields)
                if artifact.get("d7_d10_tendency_policy")
                else None
            )
            fine_d7_correction = predict_fine_d7_direction_correction(
                direction_source,
                issue_date,
                artifact,
                config=config,
                raw_d10_directions=raw_d10_directions,
            )
        if (
            speed_width_scale
            or dir_halfwidth_scale
            or dir_halfwidth_deg
            or artifact.get("d1_direction_speed_interval") is not None
            or artifact.get("d7_direction_interval_policy") is not None
            or artifact.get("d14_direction_speed_interval") is not None
        ):
            if direction_df is not None:
                direction_df = apply_interval_postprocess(
                    direction_df,
                    speed_width_scale=speed_width_scale,
                    dir_halfwidth_scale=dir_halfwidth_scale,
                    dir_halfwidth_deg=dir_halfwidth_deg,
                    d1_direction_speed_interval=artifact.get(
                        "d1_direction_speed_interval"
                    ),
                    d7_direction_interval_policy=artifact.get(
                        "d7_direction_interval_policy"
                    ),
                    d14_direction_speed_interval=artifact.get(
                        "d14_direction_speed_interval"
                    ),
                    issue_date=issue_date,
                )
                if speed_width_scale:
                    df = apply_interval_postprocess(
                        df,
                        speed_width_scale=speed_width_scale,
                        dir_halfwidth_scale={},
                        dir_halfwidth_deg={},
                        d1_direction_speed_interval=None,
                        d7_direction_interval_policy=None,
                        d14_direction_speed_interval=None,
                        issue_date=issue_date,
                    )
            else:
                df = apply_interval_postprocess(
                    df,
                    speed_width_scale=speed_width_scale,
                    dir_halfwidth_scale=dir_halfwidth_scale,
                    dir_halfwidth_deg=dir_halfwidth_deg,
                    d1_direction_speed_interval=artifact.get(
                        "d1_direction_speed_interval"
                    ),
                    d7_direction_interval_policy=artifact.get(
                        "d7_direction_interval_policy"
                    ),
                    d14_direction_speed_interval=artifact.get(
                        "d14_direction_speed_interval"
                    ),
                    issue_date=issue_date,
                )
        df = apply_d1_dense_daily_spatial_policy(
            pipeline,
            submission,
            artifact,
            fields,
            df,
            issue_date,
            window_id,
            speed_width_scale,
        )
        if direction_df is not None:
            direction_df = apply_d7_speed_endpoint_policy(
                direction_df, issue_date, artifact
            )
            direction_df = apply_d14_speed_endpoint_policy(
                direction_df, issue_date, artifact
            )
            direction_df = apply_fine_speed_residual_policy(
                direction_df, issue_date, artifact
            )
        else:
            df = apply_d7_speed_endpoint_policy(df, issue_date, artifact)
            df = apply_d14_speed_endpoint_policy(df, issue_date, artifact)
            df = apply_fine_speed_residual_policy(df, issue_date, artifact)
        direction_target = direction_df if direction_df is not None else df
        direction_target = apply_fine_d14_climatology(
            direction_target, issue_date, artifact
        )
        direction_target = apply_fine_d7_direction_correction(
            direction_target, fine_d7_correction
        )
        direction_target = apply_fine_d7_climatology(
            direction_target, issue_date, artifact
        )
        direction_target = apply_d7_conditional_endpoint(
            direction_target, issue_date, artifact, config
        )
        direction_target = apply_d7_pressure_policy(
            direction_target, issue_date, artifact, raw_d7_center
        )
        if direction_df is not None:
            df = apply_d7_speed_endpoint_policy(df, issue_date, artifact)
            df = apply_d14_speed_endpoint_policy(df, issue_date, artifact)
            df = apply_fine_speed_residual_policy(df, issue_date, artifact)
            direction_columns = ["dir_05", "dir_50", "dir_95"]
            df.loc[:, direction_columns] = direction_target[
                direction_columns
            ].to_numpy()
        else:
            df = direction_target
        df = apply_external_trajectory_policy(df, issue_date, artifact)
        df = apply_hres_analog_policy(df, issue_date, artifact)
        df = normalize_directions(df)
        validate_submission_rows(df)
        write_submission_frugal(df, args.output, append=(idx > 0))
        total_rows += int(len(df))
        q50_sum += float(df["q50"].sum())
        windows_written += 1
        print(
            f"[infer] window={window_id} issue={issue_date.date()} "
            f"rows={len(df)} written"
        )
        del fields, window_blocks, df
        gc.collect()

    if windows_written != len(windows):
        raise RuntimeError(f"Expected {len(windows)} windows, wrote {windows_written}")
    summary = {
        "rows": total_rows,
        "windows": windows_written,
        "footprint_rows": int(total_rows // (windows_written * 3 * 4)),
        "q50_mean": float(q50_sum / total_rows),
    }
    archive = (args.archive or args.output.with_suffix(".zip")).expanduser().resolve()
    auxiliary_outputs = copy_auxiliary_outputs(
        args.artifacts_dir, args.output.parent
    )
    submission_json_path = args.output.parent / "submission.json"
    power_forecast = generate_power_augmented_siting_submission(
        args.output,
        args.artifacts_dir,
        submission_json_path,
        kit_root,
    )
    auxiliary_outputs["submission.json"] = str(submission_json_path)
    package_submission(args.output, submission_json_path, archive)

    run_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kit_root": str(kit_root),
        "data_root": os.environ.get("PHASE2_DATA_ROOT"),
        "artifact_path": str(artifact_path),
        "output": str(args.output),
        "archive": str(archive),
        "eval_year": eval_year,
        "final_inference_metadata": final_metadata,
        "window_base": args.window_base,
        "speed_width_scale": speed_width_scale,
        "dir_halfwidth_scale": dir_halfwidth_scale,
        "dir_halfwidth_deg": dir_halfwidth_deg,
        "elapsed_seconds": round(time.time() - t0, 2),
        "submission": summary,
        "siting_power_forecast": power_forecast,
        "auxiliary_outputs": auxiliary_outputs,
        "execution": "single-process",
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    print(f"[infer] wrote {args.output}")
    print(f"[infer] wrote {archive}")
    print(f"[infer] wrote {manifest_path}")
    for path in auxiliary_outputs.values():
        print(f"[infer] wrote {path}")
    print(f"[infer] rows={summary['rows']} footprint_rows={summary['footprint_rows']}")


def main() -> None:
    args = parse_args()
    if args.worker_window_index is None and not args.single_process:
        run_lightweight_coordinator(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
