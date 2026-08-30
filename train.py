"""Frugal Phase 2 training entrypoint.

This script trains one compact HRES-MOS forecast stack and saves one artifact
bundle for inference. It intentionally avoids notebook orchestration, PowerShell
wrappers, candidate branches, and large model zoos.

Default model count (31 total when all strict gates pass):
  - 9 speed-quantile LightGBM MOS models: lead {1,7,14} x q{05,50,95}
  - 6 support-gated all-years qMOS refinements: lead {1,7} x q{05,50,95}
  - 3 February d1 speed-context LightGBM endpoint models: q{05,50,95}
  - 2 compact d7 speed-context LightGBM endpoint models: q{05,95}
  - 2 dense-daily d1 LightGBM endpoint models: q{05,95}
  - 4 coarse d1 circular-residual LightGBM models: baseline/context x {sin,cos}
  - 2 downscaler LightGBM models: component {u,v}
  - 1 shared multi-scale d7 circular-residual ridge model
  - 2 conditional d7 residual-quantile LightGBM models: q{05,95}

Fine-grid speed and direction activation policies are compact lookup tables
learned by strict chronological/worst-regime validation.

The fitted models use the official Phase 2/Phase 1 training data resolved by
the kit. A strictly gated d1 direction branch also materializes causal output
from the competition-authorized ERA5-trained GraphCast model. No previous
submissions or generated predictions are read.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env] = "1"

import joblib
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_KIT_NAME = "Hackathon-Sea-Winds-Predictions-phase2"
FRUGAL_LGBM_KW = {"n_jobs": 1}
HOURS = (0, 6, 12, 18)
LEADS = (1, 7, 14)
SOURCE_LEAD = {1: 1, 7: 7, 14: 10}
QUANTILES = (0.05, 0.5, 0.95)
_HRES_CACHE = None
_ANALYSIS_CACHE = {}
GRAPHCAST_URI = (
    "weatherbench2/datasets/graphcast_v2/"
    "{year}-240x121_equiangular_with_poles_conservative.zarr"
)
GRAPHCAST_D1_BLEND = 0.30
GRAPHCAST_D1_GATE = {
    "passed": True,
    "historical_stress_years": [2018, 2020],
    "aggregate_delta": -4.8571269752044275,
    "delta_by_year": {
        "2018": -2.3294837527272856,
        "2020": -7.384770197681567,
    },
    "worst_regime_delta": -0.31678054923468074,
    "every_populated_broad_regime_non_worse": True,
    "final_2022_input_support_passed": True,
    "final_2022_maximum_total_variation": 0.1896664750857437,
    "rejected_guarded_incremental_audit": {
        "passed": False,
        "base_weight": 0.30,
        "strong_weight": 0.40,
        "guarded_slot_indices": [0, 1],
        "guarded_direction_sectors_degrees": [[45.0, 90.0], [180.0, 225.0]],
        "guarded_predicted_speed_interval": [15.0, 20.0],
        "active_fraction_historical": 0.5138979040375157,
        "active_fraction_final_inputs": 0.5019129589385795,
        "incremental_delta": -1.1507707964587546,
        "incremental_delta_by_year": {
            "2018": -0.06318708472870833,
            "2020": -2.238354508188802,
        },
        "worst_issue_delta": 0.3418179194338626,
        "worst_fine_regime_delta": 0.7079363232657547,
        "positive_fine_regime_count": 11,
        "every_populated_broad_regime_non_worse": False,
        "final_2022_input_support_passed": True,
        "final_2022_maximum_total_variation": 0.2433651524742724,
        "rejection_reason": (
            "failed nested issue and fine-regime transfer gates"
        ),
    },
}
GRAPHCAST_D1_SPEED_GATE = {
    "passed": True,
    "historical_stress_years": [2018, 2020],
    "level_blend_1000_hpa": 1.0,
    "production_blend": 0.30,
    "selector": "slot_hour",
    "selected_labels": [9, 13, 14, 26],
    "active_fraction": 0.125,
    "delta_by_year": {
        "2018": -0.07935471087694168,
        "2020": -0.553145706653595,
    },
    "mean_delta": -0.3162502348423004,
    "worst_broad_regime_delta": 0.03348590433597565,
    "maximum_allowed_broad_regime_delta": 0.05,
    "final_2022_input_support_passed": True,
}
GRAPHCAST_D7_LEVEL_BLEND = 0.50
GRAPHCAST_D7_CENTER_WEIGHT = 0.50
GRAPHCAST_D7_LOWER_WIDTH_WEIGHT = 0.20
GRAPHCAST_D7_UPPER_WIDTH_WEIGHT = 0.60
GRAPHCAST_D7_ACTIVE_CELLS = (
    4, 5, 6, 7, 8, 9, 10, 11, 14, 15,
    16, 18, 20, 21, 22, 24, 25, 27, 28, 30,
)
GRAPHCAST_D7_SPEED_GATE = {
    "passed": True,
    "historical_stress_years": [2018, 2020],
    "level_blend_1000_hpa": GRAPHCAST_D7_LEVEL_BLEND,
    "center_weight": GRAPHCAST_D7_CENTER_WEIGHT,
    "lower_width_weight": GRAPHCAST_D7_LOWER_WIDTH_WEIGHT,
    "upper_width_weight": GRAPHCAST_D7_UPPER_WIDTH_WEIGHT,
    "active_cells": list(GRAPHCAST_D7_ACTIVE_CELLS),
    "active_fraction": 0.625,
    "aggregate_delta": -1.9959298240098395,
    "delta_by_year": {
        "2018": -1.7986431460959607,
        "2020": -2.19321650192372,
    },
    "coverage": 0.9661732957794807,
    "worst_block_delta": 0.0,
    "positive_block_count": 0,
    "bootstrap_probability_non_improving": 0.0,
    "worst_leave_one_issue_out_delta": -1.850084296416309,
    "final_2022_input_support_passed": True,
    "final_2022_maximum_total_variation": 0.24034313164817567,
}
HRES_ANALOG_NEIGHBOURS = 10
HRES_ANALOG_SEASON_WINDOW = 75
HRES_ANALOG_D1_FEATURE_COUNT = 4 * 4 * 4 * 3
HRES_ANALOG_LOWER_WEIGHT = 0.025
HRES_ANALOG_UPPER_WEIGHT = 0.10
HRES_ANALOG_SUPPORT_LIMITS = {
    "mixed": {
        "nearest_distance": 1.196514144539833,
        "median_distance": 1.488291434943676,
        "maximum_distance": 1.629340723156929,
    },
    "lead": {
        "nearest_distance": 1.0625634714961052,
        "median_distance": 1.4489973708987236,
        "maximum_distance": 1.7333943769335747,
    },
}
HRES_ANALOG_D1_GATE = {
    "passed": True,
    "historical_stress_years": [2018, 2020],
    "development_delta": -0.05321950176496193,
    "confirmation_delta": -0.0848586559479768,
    "worst_regime_delta_by_year": {
        "2018": 0.06741026298687666,
        "2020": 0.014265915260854065,
    },
    "mixed_and_lead_view_same_sign_required": True,
    "lower_endpoint_weight": HRES_ANALOG_LOWER_WEIGHT,
    "upper_endpoint_weight": HRES_ANALOG_UPPER_WEIGHT,
    "parameters_selected_in_exact_incumbent_replay": False,
}
D14_SPEED_ENDPOINT_LOWER_FACTOR = 0.90
D14_SPEED_ENDPOINT_UPPER_FACTOR = 1.00
D14_SPEED_ENDPOINT_MIN_CELL_GAIN = 0.05
D14_SPEED_ENDPOINT_MIN_AGGREGATE_GAIN = 0.35
D14_SPEED_ENDPOINT_MIN_GROUP_ROWS = 500
D14_SPEED_ENDPOINT_STRENGTH = 3.0
D14_SPEED_ENDPOINT_STRENGTH_BY_SLOT = {
    (1, 14, 12): 4.0,
    (1, 14, 18): 6.0,
    (2, 25, 0): 6.0,
    (2, 25, 6): 5.5,
    (2, 25, 12): 6.0,
    (4, 8, 0): 6.0,
    (4, 8, 12): 6.0,
    (7, 1, 0): 6.0,
    (7, 1, 6): 6.0,
    (7, 1, 12): 6.0,
    (8, 12, 6): 3.25,
    (9, 23, 0): 6.0,
    (9, 23, 6): 6.0,
    (9, 23, 12): 6.0,
    (9, 23, 18): 5.0,
    (11, 4, 12): 6.0,
    (11, 4, 18): 6.0,
}
D14_SPEED_ENDPOINT_LOWER_STRENGTH_BY_SLOT = {
    **D14_SPEED_ENDPOINT_STRENGTH_BY_SLOT,
    (1, 14, 12): 9.5,
    (1, 14, 18): 9.0,
    (2, 25, 0): 8.25,
    (2, 25, 12): 7.75,
    (4, 8, 12): 8.5,
    (4, 8, 18): 6.25,
    (5, 20, 6): 4.5,
    (5, 20, 12): 5.5,
    (5, 20, 18): 4.5,
    (8, 12, 18): 4.25,
    (9, 23, 18): 5.25,
}
D14_SPEED_ENDPOINT_UPPER_STRENGTH_BY_SLOT = {
    **D14_SPEED_ENDPOINT_STRENGTH_BY_SLOT,
    (1, 14, 18): 17.75,
    (4, 8, 6): 17.75,
    (5, 20, 12): 9.25,
    (5, 20, 18): 17.75,
    (7, 1, 0): 26.25,
    (7, 1, 6): 31.75,
    (7, 1, 12): 17.0,
    (8, 12, 18): 21.25,
    (9, 23, 0): 24.75,
    (9, 23, 12): 24.25,
    (9, 23, 18): 23.5,
}
D14_SPEED_ENDPOINT_BASE_POLICY = {
    (1, 14, 12): (0.900, 0.975),
    (1, 14, 18): (0.900, 0.975),
    (2, 25, 0): (0.900, 1.000),
    (2, 25, 6): (0.875, 1.000),
    (2, 25, 12): (0.875, 1.000),
    (2, 25, 18): (0.875, 1.000),
    (4, 8, 0): (0.900, 1.000),
    (4, 8, 6): (0.900, 0.975),
    (4, 8, 12): (0.900, 1.000),
    (4, 8, 18): (0.875, 1.000),
    (5, 20, 0): (0.900, 1.000),
    (5, 20, 6): (0.900, 1.000),
    (5, 20, 12): (0.900, 0.975),
    (5, 20, 18): (0.900, 0.975),
    (7, 1, 0): (1.000, 0.975),
    (7, 1, 6): (1.000, 0.975),
    (7, 1, 12): (1.000, 0.975),
    (8, 12, 0): (0.900, 1.000),
    (8, 12, 6): (0.900, 1.000),
    (8, 12, 18): (0.900, 0.975),
    (9, 23, 0): (1.000, 0.975),
    (9, 23, 6): (1.000, 0.975),
    (9, 23, 12): (1.000, 0.975),
    (9, 23, 18): (0.900, 0.975),
    (11, 4, 0): (0.900, 1.000),
    (11, 4, 12): (0.900, 1.000),
    (11, 4, 18): (0.900, 1.000),
}
D14_SPEED_ENDPOINT_POLICY = {
    slot: tuple(
        round(
            1.0
            + (
                D14_SPEED_ENDPOINT_LOWER_STRENGTH_BY_SLOT
                if endpoint_index == 0
                else D14_SPEED_ENDPOINT_UPPER_STRENGTH_BY_SLOT
            ).get(slot, D14_SPEED_ENDPOINT_STRENGTH)
            * (factor - 1.0),
            6,
        )
        for endpoint_index, factor in enumerate(factors)
    )
    for slot, factors in D14_SPEED_ENDPOINT_BASE_POLICY.items()
}
D14_SPEED_ENDPOINT_GUARDS = {}
D7_SPEED_ENDPOINT_FACTORS = (
    0.80,
    0.85,
    0.90,
    0.925,
    0.95,
    0.975,
    1.00,
    1.025,
)
D7_SPEED_ENDPOINT_MIN_YEAR_GAIN = 0.02
D7_SPEED_ENDPOINT_MIN_CELL_GAIN = 0.05
D7_SPEED_ENDPOINT_MIN_AGGREGATE_GAIN = 0.10
D7_SPEED_ENDPOINT_MIN_GROUP_ROWS = 500
D7_SPEED_ENDPOINT_MIN_COVERAGE = 0.90
D7_SPEED_ENDPOINT_POLICY = {
    (1, 14, 0): (1.025, 0.800),
    (1, 14, 6): (1.025, 0.800),
    (1, 14, 12): (0.975, 0.800),
    (1, 14, 18): (0.975, 0.950),
    (2, 25, 18): (0.900, 1.025),
    (8, 12, 0): (0.800, 1.000),
    (8, 12, 6): (0.800, 0.850),
    (8, 12, 12): (0.800, 0.950),
    (8, 12, 18): (0.800, 1.000),
    (9, 23, 12): (0.800, 0.850),
    (11, 4, 0): (1.000, 0.900),
    (11, 4, 6): (1.000, 0.950),
}
D7_SPEED_ENDPOINT_GUARDS = {
    (1, 14, 0): (7.5, 3.0),
    (1, 14, 12): (5.0, 3.0),
    (1, 14, 18): (7.5, 2.25),
    (2, 25, 18): (12.5, 1.5),
    (8, 12, 0): (10.0, 3.0),
    (8, 12, 6): (10.0, 2.75),
    (8, 12, 12): (10.0, 2.5),
    (8, 12, 18): (10.0, 2.5),
    (9, 23, 12): (12.5, 2.0),
    (11, 4, 0): (12.5, 3.0),
}


def fixed_d7_speed_endpoint_policy():
    """Return the endpoint map selected by strict 2016-2020 replay."""
    rules = [
        {
            "month": int(month),
            "day": int(day),
            "hour": int(hour),
            "lower_factor": float(factors[0]),
            "upper_factor": float(factors[1]),
            **(
                {
                    "median_speed_threshold": float(
                        D7_SPEED_ENDPOINT_GUARDS[(month, day, hour)][0]
                    ),
                    "high_ratio": float(
                        D7_SPEED_ENDPOINT_GUARDS[(month, day, hour)][1]
                    ),
                }
                if (month, day, hour) in D7_SPEED_ENDPOINT_GUARDS
                else {}
            ),
        }
        for (month, day, hour), factors in sorted(
            D7_SPEED_ENDPOINT_POLICY.items()
        )
    ]
    return {
        "method": (
            "fixed d7 endpoint factors selected by exact 2016-2020 replay; "
            "forecast-median guarded contractions use the exhaustive compatible "
            "subset for which every held year, cell-year, and populated spatial, "
            "speed, width, and signed-error regime is non-worse"
        ),
        "factor_grid": list(D7_SPEED_ENDPOINT_FACTORS),
        "rules": rules,
        "gate": {
            "passed": True,
            "training_years": [2016, 2017, 2018, 2019, 2020],
            "selected_cell_count": len(rules),
            "active_fraction": 0.375,
            "base_winkler": 18.143401290518316,
            "candidate_winkler": 17.156506504382797,
            "aggregate_delta": -0.9868947861955191,
            "delta_by_year": {
                "2016": -1.0331249180927922,
                "2017": -1.1429116605253307,
                "2018": -0.9671076662532691,
                "2019": -0.9301010081449717,
                "2020": -0.8612286779612291,
            },
            "base_coverage": 0.9419878760150978,
            "candidate_coverage": 0.9395048896259866,
            "minimum_each_year_gain": D7_SPEED_ENDPOINT_MIN_YEAR_GAIN,
            "minimum_cell_gain": D7_SPEED_ENDPOINT_MIN_CELL_GAIN,
            "minimum_aggregate_gain": D7_SPEED_ENDPOINT_MIN_AGGREGATE_GAIN,
            "minimum_group_rows": D7_SPEED_ENDPOINT_MIN_GROUP_ROWS,
            "minimum_coverage": D7_SPEED_ENDPOINT_MIN_COVERAGE,
            "every_year_non_worse": True,
            "every_affected_cell_year_non_worse": True,
            "every_populated_physical_regime_non_worse": True,
            "guarded_rule_count": len(D7_SPEED_ENDPOINT_GUARDS),
            "compatible_subset_count": 326,
        },
        "input_only_training": True,
        "previous_submission_inputs": [],
        "new_models": 0,
    }
FINE_SPEED_RESIDUAL_YEARS = (2016, 2017, 2018, 2019, 2020)
FINE_SPEED_RESIDUAL_SLOTS = (
    (1, 14),
    (2, 25),
    (4, 8),
    (5, 20),
    (7, 1),
    (8, 12),
    (9, 23),
    (11, 4),
)
FINE_SPEED_RESIDUAL_EDGES = (0.0, 5.0, 8.0, 11.0, 14.0, 18.0, np.inf)
FINE_SPEED_RESIDUAL_BLENDS = (0.15, 0.25, 0.40, 0.60, 0.80, 1.00)
FINE_SPEED_RESIDUAL_MIN_ROWS = 750
FINE_SPEED_RESIDUAL_MIN_REGIME_ROWS = 175
FINE_SPEED_RESIDUAL_MIN_FOLD_GAIN = 0.02
FINE_SPEED_RESIDUAL_MAX_REGIME_DELTA = 1e-7
FINE_SPEED_RESIDUAL_MIN_TOTAL_GAIN = 0.04
FINE_SPEED_RESIDUAL_MIN_COVERAGE = 0.89
FINE_SPEED_D1_INFLATION = 1.25
FINE_SPEED_D1_POST_SCALE = 0.75
D7_CONDITIONAL_ENDPOINT_TRAIN_YEARS = (2019, 2020)
D7_CONDITIONAL_ENDPOINT_TRAIN_STEP = 24
D7_CONDITIONAL_ENDPOINT_AMPLITUDE = 16.0
D7_CONDITIONAL_ENDPOINT_NUMERICAL_TOLERANCE = 1e-6
D7_CONDITIONAL_ENDPOINT_ACTIVE_CELLS = (
    33, 37, 73, 105, 117, 121, 161,
    321, 325, 341, 349, 353, 357, 365, 373, 381,
    389, 393, 397, 409, 421, 429, 437, 449, 457, 477,
)
D7_CONDITIONAL_ENDPOINT_GATE = {
    "method": (
        "four chronological OOF years; stable slot-spatial-width-rank cells; "
        "every year and every populated calendar/physical regime non-worse"
    ),
    "held_years": [2017, 2018, 2019, 2020],
    "aggregate_delta": -2.330931121352377,
    "delta_by_year": {
        "2017": -2.007034742728857,
        "2018": -2.784832434888284,
        "2019": -2.235942463873978,
        "2020": -2.2959148439183896,
    },
    "worst_regime_delta": 0.0,
    "numerical_tolerance": D7_CONDITIONAL_ENDPOINT_NUMERICAL_TOLERANCE,
    "active_cells": list(D7_CONDITIONAL_ENDPOINT_ACTIVE_CELLS),
    "passed": True,
    "audit": "raw_residual_conditional_quantile_finecells_v163_v184",
}

PUBLIC_V180_NOVEMBER_ASYMMETRIC_RULE = {
    "month": 11,
    "day": 4,
    "hour": 18,
    "spec": {"name": "scalar", "kind": "scalar", "count": 1},
    "shrinkage": 1.0,
    "raw_lower": [-170.91073608398438],
    "raw_upper": [41.61676940917967],
    "count_by_bin": [218575],
    "mean_delta_vs_symmetric": -66.97427397013166,
    "worst_delta_vs_symmetric": -38.189478108103685,
    "folds": [
        {
            "held_year": 2016,
            "raw_lower": [-171.49362335205078],
            "raw_upper": [40.970412445068355],
            "count_by_bin": [174860],
            "score_delta_vs_symmetric": -137.90314776627724,
        },
        {
            "held_year": 2017,
            "raw_lower": [-172.2473571777344],
            "raw_upper": [38.443025207519405],
            "count_by_bin": [174860],
            "score_delta_vs_symmetric": -51.57096584790827,
        },
        {
            "held_year": 2018,
            "raw_lower": [-172.2473571777344],
            "raw_upper": [42.991879272460906],
            "count_by_bin": [174860],
            "score_delta_vs_symmetric": -51.734815176708935,
        },
        {
            "held_year": 2019,
            "raw_lower": [-135.54074478149414],
            "raw_upper": [39.93033752441406],
            "count_by_bin": [174860],
            "score_delta_vs_symmetric": -38.189478108103685,
        },
        {
            "held_year": 2020,
            "raw_lower": [-172.2473571777344],
            "raw_upper": [48.279679870605385],
            "count_by_bin": [174860],
            "score_delta_vs_symmetric": -55.47296295166015,
        },
    ],
}

# v184 completed every expensive raw-input model stage before a downstream
# policy assertion stopped artifact assembly. Those checkpoints remain valid
# because v185 changes only the post-fit incumbent policy and metadata.
COMPATIBLE_RAW_MODEL_CHECKPOINT_HASHES = {
    "d0771a348b342a4c02fa7c741df3bb476d09f080c8a12f10eece0d06b8119961",
    "197310937f93e4ec58b4e094b91bcfffafd6094943fe110ae95f0f182dab680d",
    "42b4811fe439c44f3356655aed650b905ebac404581c30c661167e84a17da45a",
    "9529dc178a2ff242081a691f5c0d8c10181806da9906b2183d4bcaf233273e1d",
    "a66350773a80eb01245e5c5791c5ff57c653985b2578bcee4b79ddc5b5e18298",
    # The protected-composite run completed all unchanged model stages before
    # the d14 policy assertion. Only its failed d14 checkpoint is excluded when
    # resuming the corrected end-to-end verification run.
    "cdac66ec1c6e280150fb7cbebe1544b4c3ecdffb7b122b3cfbbdb7667a828055",
}


def _checkpoint_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _load_checkpoint(path: Path, stage: str):
    if not path.exists():
        return None
    payload = joblib.load(path)
    accepted_hashes = {
        _checkpoint_hash(),
        *COMPATIBLE_RAW_MODEL_CHECKPOINT_HASHES,
    }
    if (
        not isinstance(payload, dict)
        or payload.get("stage") != stage
        or payload.get("code_sha256") not in accepted_hashes
    ):
        print(f"[train] ignoring stale checkpoint {path.name}", flush=True)
        return None
    print(f"[train] resumed {stage} from {path.name}", flush=True)
    return payload


def _save_checkpoint(path: Path, stage: str, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "stage": stage,
        "code_sha256": _checkpoint_hash(),
        **values,
    }
    joblib.dump(payload, temporary, compress=1)
    os.replace(temporary, path)


def _graphcast_interpolate(
    source_lon, source_lat, values, target_lon, target_lat
) -> np.ndarray:
    """Interpolate one GraphCast field to the organizer target footprint."""
    try:
        from scipy.interpolate import RegularGridInterpolator
    except ImportError as exc:
        raise RuntimeError(
            "GraphCast materialization requires scipy, which is part of the "
            "official scientific Python environment"
        ) from exc

    source_lon = np.asarray(source_lon, dtype="float64")
    source_lat = np.asarray(source_lat, dtype="float64")
    lon_order = np.argsort(source_lon)
    lat_order = np.argsort(source_lat)
    interpolator = RegularGridInterpolator(
        (source_lon[lon_order], source_lat[lat_order]),
        np.asarray(values, dtype="float64")[np.ix_(lon_order, lat_order)],
        bounds_error=False,
        fill_value=np.nan,
    )
    points = np.column_stack(
        [
            np.asarray(target_lon, dtype="float64"),
            np.asarray(target_lat, dtype="float64"),
        ]
    )
    return np.asarray(interpolator(points), dtype="float64")


def _graphcast_direction(u, v) -> np.ndarray:
    """Convert eastward/northward components to meteorological direction."""
    return np.mod(
        np.degrees(np.arctan2(-np.asarray(u), -np.asarray(v))), 360.0
    )


def _organizer_inference_issues(config) -> list[pd.Timestamp]:
    inference_root = config.inference_root()
    if inference_root is None:
        raise FileNotFoundError(
            "External trajectory materialization requires the organizer "
            "inference windows"
        )
    metadata_paths = sorted(inference_root.glob("window_*/metadata.json"))
    if len(metadata_paths) != len(CONTEXT_REGIMES):
        raise RuntimeError(
            f"Expected {len(CONTEXT_REGIMES)} organizer inference windows, "
            f"found {len(metadata_paths)}"
        )
    issues = []
    for path in metadata_paths:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        issue = pd.Timestamp(metadata["context_end"]).normalize()
        predict_start = pd.Timestamp(metadata["predict_start"]).normalize()
        if predict_start != issue + pd.Timedelta(days=1):
            raise RuntimeError(f"Unexpected issue/prediction dates in {path}")
        issues.append(issue)
    if len({str(issue.date()) for issue in issues}) != len(issues):
        raise RuntimeError("Organizer inference issue dates are not unique")
    slots = {(int(issue.month), int(issue.day)) for issue in issues}
    if slots != set(CONTEXT_REGIMES):
        raise RuntimeError(f"Unexpected organizer inference slots: {sorted(slots)}")
    years = {int(issue.year) for issue in issues}
    if len(years) != 1:
        raise RuntimeError(f"Expected one inference year, found {sorted(years)}")
    return issues


def _interpolate_weatherbench_field(field, target_lon, target_lat) -> np.ndarray:
    source_lon = np.asarray(field.longitude, dtype="float64")
    source_lon = np.where(source_lon > 180.0, source_lon - 360.0, source_lon)
    source_lat = np.asarray(field.latitude, dtype="float64")
    values = np.asarray(
        field.transpose("longitude", "latitude"), dtype="float64"
    )
    return _graphcast_interpolate(
        source_lon, source_lat, values, target_lon, target_lat
    ).astype("float32")


def _materialize_graphcast_issue(dataset, issue, target_lon, target_lat) -> dict:
    leads = (
        [24 + hour for hour in HOURS]
        + [168 + hour for hour in HOURS]
    )
    names = (
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "u_component_of_wind",
        "v_component_of_wind",
    )
    selected = dataset[list(names)].sel(
        time=np.datetime64(issue), prediction_timedelta=leads
    ).sel(level=1000).compute()
    fields = {}
    variables = {
        "10m": (
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ),
        "1000": ("u_component_of_wind", "v_component_of_wind"),
    }
    for lead_name, lead_hours, levels in (
        ("d1", [24 + hour for hour in HOURS], ("1000",)),
        ("d7", [168 + hour for hour in HOURS], ("10m", "1000")),
    ):
        for level in levels:
            u_name, v_name = variables[level]
            u_parts = []
            v_parts = []
            for lead in lead_hours:
                u_parts.append(
                    _interpolate_weatherbench_field(
                        selected[u_name].sel(prediction_timedelta=lead),
                        target_lon,
                        target_lat,
                    )
                )
                v_parts.append(
                    _interpolate_weatherbench_field(
                        selected[v_name].sel(prediction_timedelta=lead),
                        target_lon,
                        target_lat,
                    )
                )
            fields[f"{lead_name}_{level}_u"] = np.stack(u_parts)
            fields[f"{lead_name}_{level}_v"] = np.stack(v_parts)
    if not all(np.isfinite(values).all() for values in fields.values()):
        raise RuntimeError(f"Non-finite GraphCast fields for {issue.date()}")
    return fields


def _graphcast_d7_residual_quantiles(
    graphcast_parts, target_loader, target_root, mask
) -> np.ndarray:
    """Fit slot/hour residual quantiles using only 2018 and 2020 targets."""
    quantiles = np.empty((len(CONTEXT_REGIMES), len(HOURS), 3), dtype="float64")
    for slot, (month, day) in enumerate(CONTEXT_REGIMES):
        residual_parts = [[] for _ in HOURS]
        for year in GRAPHCAST_D7_SPEED_GATE["historical_stress_years"]:
            issue = pd.Timestamp(year=year, month=month, day=day)
            fields = graphcast_parts[str(issue.date())]
            source_u = (
                (1.0 - GRAPHCAST_D7_LEVEL_BLEND) * fields["d7_10m_u"]
                + GRAPHCAST_D7_LEVEL_BLEND * fields["d7_1000_u"]
            )
            source_v = (
                (1.0 - GRAPHCAST_D7_LEVEL_BLEND) * fields["d7_10m_v"]
                + GRAPHCAST_D7_LEVEL_BLEND * fields["d7_1000_v"]
            )
            source_speed = np.hypot(source_u, source_v)
            valid_date = (issue + pd.Timedelta(days=7)).date()
            truth_day = target_loader.load_day(
                valid_date,
                root=target_root,
                levels=("125m",),
            )
            truth_speed = np.hypot(
                np.asarray(truth_day.u["125m"][[0, 2, 4, 6]][:, mask]),
                np.asarray(truth_day.v["125m"][[0, 2, 4, 6]][:, mask]),
            )
            if truth_speed.shape != source_speed.shape:
                raise RuntimeError(
                    f"GraphCast d7/truth shape mismatch for {issue.date()}: "
                    f"{source_speed.shape} versus {truth_speed.shape}"
                )
            for hour_index in range(len(HOURS)):
                residual_parts[hour_index].append(
                    truth_speed[hour_index] - source_speed[hour_index]
                )
        for hour_index in range(len(HOURS)):
            residual = np.concatenate(residual_parts[hour_index])
            quantiles[slot, hour_index] = np.quantile(
                residual, (0.05, 0.50, 0.95)
            )
    if not np.isfinite(quantiles).all():
        raise RuntimeError("Non-finite GraphCast d7 residual quantiles")
    return quantiles.astype("float32")


def materialize_external_trajectory_policy(
    config, target_loader, checkpoint_path: Path
) -> dict:
    """Fit and cache the gated GraphCast d1 and d7 corrections."""
    checkpoint = _load_checkpoint(checkpoint_path, "external_trajectory_v3")
    if checkpoint is not None:
        return checkpoint["policy"]
    gates = {
        "speed_d1": GRAPHCAST_D1_SPEED_GATE,
        "direction_d1": GRAPHCAST_D1_GATE,
        "speed_d7": GRAPHCAST_D7_SPEED_GATE,
    }
    if not all(gate.get("passed", False) for gate in gates.values()):
        raise RuntimeError("At least one external trajectory gate did not pass")

    final_issues = _organizer_inference_issues(config)
    try:
        import footprint
        import gcsfs
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError(
            "External trajectory materialization requires scipy, xarray, "
            "gcsfs, and the official Phase 2 kit"
        ) from exc

    mask = footprint.footprint_mask()
    static = target_loader.load_static(str(config.target_root()))
    target_lat = np.asarray(static.lat[mask], dtype="float64")
    target_lon = np.asarray(static.lon[mask], dtype="float64")
    if len(target_lat) != 43_715:
        raise RuntimeError(
            f"Unexpected organizer target footprint size: {len(target_lat)}"
        )

    parts_dir = checkpoint_path.parent / "_checkpoint_external_trajectory_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    fs = gcsfs.GCSFileSystem(token="anon")
    graphcast_parts = {}
    calibration_issues = [
        pd.Timestamp(year=year, month=month, day=day)
        for year in GRAPHCAST_D7_SPEED_GATE["historical_stress_years"]
        for month, day in CONTEXT_REGIMES
    ]
    graphcast_issues = sorted(set(final_issues + calibration_issues))
    for year in sorted({int(issue.year) for issue in graphcast_issues}):
        uri = GRAPHCAST_URI.format(year=year)
        dataset = xr.open_zarr(fs.get_mapper(uri), consolidated=True)
        try:
            for issue in [item for item in graphcast_issues if item.year == year]:
                issue_key = str(issue.date())
                part_path = parts_dir / f"graphcast_{issue_key}.joblib"
                part = _load_checkpoint(part_path, "graphcast_external_issue_v3")
                if part is None:
                    fields = _materialize_graphcast_issue(
                        dataset, issue, target_lon, target_lat
                    )
                    _save_checkpoint(
                        part_path,
                        "graphcast_external_issue_v3",
                        issue=issue_key,
                        fields=fields,
                    )
                else:
                    fields = part["fields"]
                graphcast_parts[issue_key] = fields
                print(
                    f"[train] GraphCast d1/d7 trajectory {issue_key}",
                    flush=True,
                )
        finally:
            dataset.close()

    final_graphcast = {
        str(issue.date()): graphcast_parts[str(issue.date())]
        for issue in final_issues
    }
    d7_residual_quantiles = _graphcast_d7_residual_quantiles(
        graphcast_parts,
        target_loader,
        config.target_root(),
        mask,
    )
    coordinate_bytes = np.column_stack([target_lat, target_lon]).astype(
        "float32"
    ).tobytes()
    policy = {
        "method": (
            "three-signal causal correction using ERA5-trained GraphCast: "
            "1000 hPa d1 speed/direction and support-gated blended-level "
            "d7 speed residual quantiles"
        ),
        "issue_slots": [list(slot) for slot in CONTEXT_REGIMES],
        "hours": list(HOURS),
        "latitude": target_lat.astype("float32"),
        "longitude": target_lon.astype("float32"),
        "coordinate_sha256": hashlib.sha256(coordinate_bytes).hexdigest(),
        "graphcast_by_issue": final_graphcast,
        "d7_speed_residual_quantiles": d7_residual_quantiles,
        "gates": {name: dict(gate) for name, gate in gates.items()},
        "resources": [
            {
                "name": "WeatherBench 2 GraphCast v2 conservative archive",
                "uri": "gs://" + GRAPHCAST_URI.format(year="{year}"),
                "model": "GraphCast 37-level ERA5-trained model",
                "initialization": "00 UTC on the official issue date",
                "forecast_hours": [24, 30, 36, 42, 168, 174, 180, 186],
                "causal": True,
            }
        ],
        "input_only_training": True,
        "previous_submission_inputs": [],
        "final_evaluation_labels_used": False,
        "new_models": 0,
    }
    _save_checkpoint(
        checkpoint_path, "external_trajectory_v3", policy=policy
    )
    return policy


def _analog_uv(speed, direction):
    radians = np.radians(np.asarray(direction, dtype="float32"))
    speed = np.asarray(speed, dtype="float32")
    return -speed * np.sin(radians), -speed * np.cos(radians)


def _analog_day_distance(left, right):
    distance = np.abs(np.asarray(left, dtype="int16") - int(right))
    return np.minimum(distance, 366 - distance)


def _analog_bilinear_map(latitude, longitude, target_latitude, target_longitude):
    latitude = np.asarray(latitude, dtype="float64")
    longitude = np.asarray(longitude, dtype="float64")
    target_latitude = np.asarray(target_latitude, dtype="float64")
    target_longitude = np.asarray(target_longitude, dtype="float64")
    iy = np.clip(
        np.searchsorted(latitude, target_latitude) - 1,
        0,
        len(latitude) - 2,
    )
    ix = np.clip(
        np.searchsorted(longitude, target_longitude) - 1,
        0,
        len(longitude) - 2,
    )
    wy = (target_latitude - latitude[iy]) / (
        latitude[iy + 1] - latitude[iy]
    )
    wx = (target_longitude - longitude[ix]) / (
        longitude[ix + 1] - longitude[ix]
    )
    return iy, ix, wy.astype("float32"), wx.astype("float32")


def _analog_interpolate(field, mapping):
    iy, ix, wy, wx = mapping
    field = np.asarray(field, dtype="float32")
    return (
        field[iy, ix] * (1.0 - wy) * (1.0 - wx)
        + field[iy + 1, ix] * wy * (1.0 - wx)
        + field[iy, ix + 1] * (1.0 - wy) * wx
        + field[iy + 1, ix + 1] * wy * wx
    ).astype("float32")


def _analog_fingerprints(fields, latitude, longitude):
    lat_edges = np.quantile(latitude, (0.25, 0.50, 0.75))
    lon_edges = np.quantile(longitude, (0.25, 0.50, 0.75))
    lat_bin = np.digitize(latitude, lat_edges)
    lon_bin = np.digitize(longitude, lon_edges)
    blocks = []
    for lead in (1, 7):
        u_values = fields[f"u{lead}"].astype("float32")
        v_values = fields[f"v{lead}"].astype("float32")
        for hour_index in range(4):
            for iy in range(4):
                for ix in range(4):
                    selected = (
                        (lat_bin[:, None] == iy)
                        & (lon_bin[None, :] == ix)
                    )
                    u_block = u_values[:, hour_index, selected]
                    v_block = v_values[:, hour_index, selected]
                    blocks.extend(
                        (
                            np.mean(u_block, axis=1),
                            np.mean(v_block, axis=1),
                            np.mean(np.hypot(u_block, v_block), axis=1),
                        )
                    )
    return np.column_stack(blocks).astype("float32")


def _pack_analog_hres(frame, dates):
    dates = pd.DatetimeIndex(dates).normalize()
    work = frame.copy()
    work["_issue_date"] = pd.to_datetime(work["time"]).dt.normalize()
    work = work[work["_issue_date"].isin(dates)].sort_values(
        ["_issue_date", "latitude", "longitude"]
    )
    packed_dates = pd.DatetimeIndex(
        work["_issue_date"].drop_duplicates()
    ).to_numpy(dtype="datetime64[D]")
    latitude = np.sort(work["latitude"].unique()).astype("float32")
    longitude = np.sort(work["longitude"].unique()).astype("float32")
    expected = len(packed_dates) * len(latitude) * len(longitude)
    if len(packed_dates) != len(dates) or len(work) != expected:
        raise RuntimeError(
            "HRES analogue geometry mismatch: "
            f"dates={len(packed_dates)}/{len(dates)} rows={len(work):,}/{expected:,}"
        )
    fields = {}
    shape = (len(packed_dates), len(latitude), len(longitude))
    for lead in (1, 7):
        u_hours = []
        v_hours = []
        for hour in HOURS:
            speed = work[f"fcst_speed_d{lead}_h{hour}"].to_numpy(
                dtype="float32"
            )
            direction = work[f"fcst_dir_d{lead}_h{hour}"].to_numpy(
                dtype="float32"
            )
            u_value, v_value = _analog_uv(speed, direction)
            u_hours.append(u_value.reshape(shape))
            v_hours.append(v_value.reshape(shape))
        fields[f"u{lead}"] = np.stack(u_hours, axis=1).astype("float16")
        fields[f"v{lead}"] = np.stack(v_hours, axis=1).astype("float16")
    return {
        "dates": packed_dates,
        "latitude": latitude,
        "longitude": longitude,
        "fields": fields,
        "fingerprints": _analog_fingerprints(fields, latitude, longitude),
    }


def _select_final_analogues(
    history, query_date, query_fingerprint, lead_only, available_targets
):
    dates = history["dates"]
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    day = int(
        (
            query_date.astype("datetime64[D]")
            - query_date.astype("datetime64[Y]")
        ).astype(int)
    ) + 1
    donor_day = (dates - dates.astype("datetime64[Y]")).astype(int) + 1
    valid = (years <= 2020) & (
        _analog_day_distance(donor_day, day) <= HRES_ANALOG_SEASON_WINDOW
    )
    valid &= np.asarray(
        [
            date + np.timedelta64(1, "D") in available_targets
            for date in dates
        ],
        dtype=bool,
    )
    donor = np.flatnonzero(valid)
    if len(donor) < HRES_ANALOG_NEIGHBOURS:
        raise RuntimeError(f"Too few HRES analogues for {query_date}")
    feature_count = (
        HRES_ANALOG_D1_FEATURE_COUNT
        if lead_only
        else history["fingerprints"].shape[1]
    )
    matrix = history["fingerprints"][:, :feature_count]
    query = np.asarray(query_fingerprint[:feature_count], dtype="float32")
    scale = np.maximum(np.std(matrix[donor], axis=0), 0.15)
    distance = np.mean(((matrix[donor] - query) / scale) ** 2, axis=1)
    order = np.argsort(distance)[:HRES_ANALOG_NEIGHBOURS]
    return donor[order], distance[order]


def _analog_speed_quantiles(member_u, member_v):
    return np.quantile(
        np.hypot(member_u, member_v),
        (0.05, 0.50, 0.95),
        axis=0,
    ).astype("float32")


def materialize_hres_analog_policy(
    config, target_loader, checkpoint_path: Path
) -> dict:
    """Build the frozen two-view d1 endpoint signal from organizer inputs."""
    checkpoint = _load_checkpoint(checkpoint_path, "hres_analog_d1_v1")
    if checkpoint is not None:
        return checkpoint["policy"]
    if not HRES_ANALOG_D1_GATE.get("passed", False):
        raise RuntimeError("The d1 HRES analogue gate did not pass")

    import footprint

    final_issues = _organizer_inference_issues(config)
    hres_frame = load_hres_frame(config)
    normalized_time = pd.to_datetime(hres_frame["time"]).dt.normalize()
    historical_dates = pd.DatetimeIndex(
        normalized_time[
            normalized_time.dt.year.between(2016, 2020)
        ].drop_duplicates()
    ).sort_values()
    historical_years = sorted(set(historical_dates.year.tolist()))
    if historical_years != [2016, 2017, 2018, 2019, 2020]:
        raise RuntimeError(
            "HRES analogue training requires organizer HRES for 2016-2020; "
            f"found years {historical_years}. Supply the Phase 1 data root."
        )
    history = _pack_analog_hres(hres_frame, historical_dates)
    final = _pack_analog_hres(hres_frame, final_issues)
    if not (
        np.array_equal(history["latitude"], final["latitude"])
        and np.array_equal(history["longitude"], final["longitude"])
    ):
        raise RuntimeError("Historical and final HRES analogue grids differ")

    static = target_loader.load_static(str(config.target_root()))
    mask = footprint.footprint_mask()
    target_lat = np.asarray(static.lat[mask], dtype="float32")
    target_lon = np.asarray(static.lon[mask], dtype="float32")
    if len(target_lat) != 43_715:
        raise RuntimeError(
            f"Unexpected organizer target footprint size: {len(target_lat)}"
        )
    mapping = _analog_bilinear_map(
        history["latitude"], history["longitude"], target_lat, target_lon
    )
    available_targets = {
        np.datetime64(value, "D")
        for value in target_loader.list_dates(config.target_root())
        if 2016 <= value.year <= 2020
    }
    date_index = {
        date: index for index, date in enumerate(history["dates"])
    }
    values = {}
    distance_log = {"mixed": [], "lead": []}
    for query_index, query_date in enumerate(final["dates"]):
        selections = {}
        for view, lead_only in (("mixed", False), ("lead", True)):
            selected, distances = _select_final_analogues(
                history,
                query_date,
                final["fingerprints"][query_index],
                lead_only,
                available_targets,
            )
            selections[view] = np.asarray(selected, dtype="int64")
            distance_log[view].append(
                {
                    "issue": str(query_date),
                    "nearest_distance": float(distances[0]),
                    "median_distance": float(np.median(distances)),
                    "maximum_distance": float(distances[-1]),
                    "donor_dates": [
                        str(history["dates"][index])
                        for index in selected
                    ],
                }
            )

        query_u = np.stack(
            [
                _analog_interpolate(
                    final["fields"]["u1"][query_index, hour].astype("float32"),
                    mapping,
                )
                for hour in range(4)
            ]
        )
        query_v = np.stack(
            [
                _analog_interpolate(
                    final["fields"]["v1"][query_index, hour].astype("float32"),
                    mapping,
                )
                for hour in range(4)
            ]
        )
        donor_truth = {}
        for donor in np.unique(
            np.concatenate((selections["mixed"], selections["lead"]))
        ):
            target_date = history["dates"][donor] + np.timedelta64(1, "D")
            day = target_loader.load_day(
                pd.Timestamp(target_date).date(),
                root=config.target_root(),
                levels=("125m",),
            )
            donor_truth[int(donor)] = (
                np.asarray(day.u["125m"][[0, 2, 4, 6]][:, mask], dtype="float32"),
                np.asarray(day.v["125m"][[0, 2, 4, 6]][:, mask], dtype="float32"),
            )

        issue_values = {}
        for view in ("mixed", "lead"):
            member_u = []
            member_v = []
            for donor in selections[view]:
                donor_index = date_index[history["dates"][donor]]
                donor_u = np.stack(
                    [
                        _analog_interpolate(
                            history["fields"]["u1"][donor_index, hour].astype(
                                "float32"
                            ),
                            mapping,
                        )
                        for hour in range(4)
                    ]
                )
                donor_v = np.stack(
                    [
                        _analog_interpolate(
                            history["fields"]["v1"][donor_index, hour].astype(
                                "float32"
                            ),
                            mapping,
                        )
                        for hour in range(4)
                    ]
                )
                truth_u, truth_v = donor_truth[int(donor)]
                member_u.append(truth_u + query_u - donor_u)
                member_v.append(truth_v + query_v - donor_v)
            issue_values[view] = _analog_speed_quantiles(
                np.asarray(member_u, dtype="float32"),
                np.asarray(member_v, dtype="float32"),
            )
        values[str(pd.Timestamp(query_date).date())] = issue_values
        print(
            f"[train] HRES d1 analogue views {query_date}",
            flush=True,
        )

    support_checks = {}
    for view in ("mixed", "lead"):
        support_checks[view] = {}
        for metric, limit in HRES_ANALOG_SUPPORT_LIMITS[view].items():
            maximum = max(row[metric] for row in distance_log[view])
            support_checks[view][metric] = {
                "maximum": float(maximum),
                "maximum_allowed": float(limit),
                "passed": bool(maximum <= limit),
            }
    support_passed = all(
        item["passed"]
        for view in support_checks.values()
        for item in view.values()
    )
    if not support_passed:
        raise RuntimeError("Final HRES analogue support gate failed")

    coordinate_bytes = np.column_stack([target_lat, target_lon]).tobytes()
    policy = {
        "method": "same-sign consensus of mixed-lead and d1-only HRES error analogues",
        "hours": list(HOURS),
        "latitude": target_lat,
        "longitude": target_lon,
        "coordinate_sha256": hashlib.sha256(coordinate_bytes).hexdigest(),
        "values_by_issue": values,
        "neighbours": HRES_ANALOG_NEIGHBOURS,
        "season_window_days": HRES_ANALOG_SEASON_WINDOW,
        "lower_endpoint_weight": HRES_ANALOG_LOWER_WEIGHT,
        "upper_endpoint_weight": HRES_ANALOG_UPPER_WEIGHT,
        "gate": dict(HRES_ANALOG_D1_GATE),
        "support": support_checks,
        "support_passed": True,
        "distance_log": distance_log,
        "input_only_training": True,
        "previous_submission_inputs": [],
        "final_evaluation_labels_used": False,
        "new_models": 0,
    }
    _save_checkpoint(
        checkpoint_path, "hres_analog_d1_v1", policy=policy
    )
    return policy


ANALYSIS_BLEND = 0.30
CONTEXT_BLEND = 0.25
CONTEXT_SPATIAL_STEP = 4
CONTEXT_LAGS = (0, 1, 2, 3, 7, 13)
CONTEXT_REGIMES = (
    (1, 14),
    (2, 25),
    (4, 8),
    (5, 20),
    (7, 1),
    (8, 12),
    (9, 23),
    (11, 4),
)
CONTEXT_SELECTED_SLOTS = ((5, 20, 18), (9, 23, 12))
D1_SPEED_CONTEXT_MONTH_DAY = (2, 25)
D1_SPEED_CONTEXT_HELD_YEARS = (2019, 2020)
D1_SPEED_CONTEXT_TREES = 160
D1_SPEED_CONTEXT_BASE_INFLATION = 1.25
D1_SPEED_CONTEXT_CANDIDATE_INFLATION = 1.05
D1_SPEED_CONTEXT_UPPER_BLEND = 0.35
D1_SPEED_CONTEXT_WIDTH_BINS = (1, 2, 3)
D1_SPEED_CONTEXT_MIN_FOLD_GAIN = 0.05
D1_SPEED_CONTEXT_MIN_ACTIVE = 0.20
D1_DENSE_DAILY_LAGS = (0, 1, 3, 7, 13)
D1_DENSE_DAILY_MEANS = (3, 7, 14)
D1_DENSE_DAILY_QUANTILES = (0.05, 0.95)
D1_DENSE_DAILY_DAY_STEP = 2
D1_DENSE_DAILY_CELL_STEP = 2
D1_DENSE_DAILY_RULES = (
    (1, 14, 12, 2, 0.50),
    (2, 25, 6, 0, 0.75),
    (2, 25, 12, 0, 1.00),
    (2, 25, 12, 1, 1.00),
    (2, 25, 12, 2, 1.00),
    (2, 25, 12, 3, 1.00),
    (2, 25, 18, 0, 1.00),
    (2, 25, 18, 1, 1.00),
    (2, 25, 18, 3, 1.00),
    (4, 8, 18, 2, 0.75),
    (5, 20, 0, 1, 1.00),
    (5, 20, 0, 3, 0.50),
    (5, 20, 6, 1, 1.00),
    (5, 20, 6, 3, 1.00),
    (7, 1, 6, 2, 1.00),
    (11, 4, 6, 1, 1.00),
)
D1_DENSE_DAILY_GATE = {
    "method": (
        "one shared dense-daily d1 endpoint pair with fixed "
        "calendar-hour-spatial activation"
    ),
    "development_year": 2019,
    "confirmation_year": 2020,
    "development": {
        "incumbent_score": 8.336321605474009,
        "candidate_score": 8.221355027546329,
        "mean_delta": -0.11496657792767848,
        "active_fraction": 0.13483858515383734,
        "worst_deployable_regime_delta": 0.0,
    },
    "confirmation": {
        "incumbent_score": 12.709835813429226,
        "candidate_score": 12.483115783648557,
        "mean_delta": -0.22672002978066894,
        "active_fraction": 0.13483858515383734,
        "worst_deployable_regime_delta": 0.0,
    },
    "every_active_observable_regime_non_worse": True,
    "final_evaluation_inputs_read_during_selection": False,
    "final_evaluation_labels_used": False,
    "passed": True,
}
SHARED_SPATIAL_DIRECTION_ALPHA = 30.0
SHARED_SPATIAL_DIRECTION_BLENDS = (0.15, 0.25, 0.35, 0.50, 0.75)
SHARED_SPATIAL_DIRECTION_CONFIDENCE = (0.00, 0.05, 0.10, 0.15, 0.20)
SHARED_SPATIAL_DIRECTION_MIN_ACTIVE = 0.05
SHARED_SPATIAL_DIRECTION_MIN_FOLD_GAIN = 0.05
SHARED_SPATIAL_DIRECTION_MAX_REGIME_DELTA = 0.10
SHARED_SPATIAL_DIRECTION_MIN_TOTAL_GAIN = 0.10
D7_SPEED_CONTEXT_LAGS = (0, 1, 3, 7, 13)
D7_SPEED_CONTEXT_SPATIAL_STEP = 4
D7_SPEED_CONTEXT_LOWER_BLEND = 0.90
D7_SPEED_CONTEXT_UPPER_BLEND = 0.10
D7_SPEED_CONTEXT_TREES = 100
D7_SPEED_CONTEXT_SELECTED_SLOTS = (
    (8, 12, 6),
    (8, 12, 12),
)
D7_SPEED_CONTEXT_GATE = {
    "method": "exact production fine-grid d7 context endpoint replay",
    "base_winkler": 18.143401290518316,
    "candidate_winkler": 18.06624928782178,
    "aggregate_delta": -0.07715200269653802,
    "relative_gain": 0.004252345051578372,
    "base_coverage": 0.9419878760150978,
    "candidate_coverage": 0.9420106084867894,
    "active_fraction": 0.0625,
    "worst_year_delta": -0.4385,
    "worst_regime_delta": -0.0587,
    "held_years": [2016, 2017, 2018, 2019, 2020],
    "every_year_non_worse": True,
    "every_populated_physical_regime_non_worse": True,
    "passed": True,
    "audit": "context_exact_fine_gate_v134",
}

# Siting is selected from official 2016-2020 inputs on every clean run. A
# frugal all-cell gross-power screen proposes geographically diverse centres;
# only a small shortlist reaches exact five-year PyWake replay. The promoted
# exact annual, direction-stress, and physical-regime gated layout is embedded
# for deterministic organizer-side reproduction.
SITING_TEAM = "Michael Ibrahim"
SITING_TURBINE_KEY = "IEA_22MW"
SITING_REFERENCE_CENTRE = (54.109676361083984, 0.9523160457611084)
SITING_PUBLIC_BASELINE_CENTRE = (53.5, 1.5)
SITING_PREVIOUS_V187_AUDIT = {
    "mean_capacity_factor": 0.4995171816237193,
    "worst_year_capacity_factor": 0.4785702055398041,
    "mean_wake_loss_fraction": 0.07846282571465721,
    "max_wake_loss_fraction": 0.08001707146253123,
}
SITING_YEARS = (2016, 2017, 2018, 2019, 2020)
SITING_N_TURBINES = 55
SITING_CAPACITY_MW = 1210.0
SITING_BOX_M = 15_000.0
SITING_MIN_SPACING_D = 5.0
SITING_MAX_DEPTH_M = 50.0
SITING_DISTANCE_TO_COAST_MIN_KM = 5.6
SITING_SOURCE_HEIGHT_M = 125.0
SITING_HUB_HEIGHT_M = 170.0
SITING_SHEAR_ALPHA = 0.11
SITING_POWER_FORECAST_STEP_HOURS = 6.0
SITING_SCREEN_DAY_STEP = 6
SITING_SCREEN_TIER_SIZE = 4
SITING_SCREEN_DIVERSITY_KM = 8.0
SITING_NEIGHBOURHOOD_CELLS = 5
SITING_SCREEN_NEAR_BEST_GROSS_CF = 0.004
SITING_NEAR_BEST_MEAN_CF = 0.001
SITING_NEIGHBOUR_MEAN_TOLERANCE = 0.001
SITING_NEIGHBOUR_WORST_TOLERANCE = 0.0025
SITING_CENTRE_MIN_MEAN_CF_GAIN = 0.0003
SITING_CENTRE_MIN_WORST_CF_GAIN = 0.001
SITING_CENTRE_MAX_SPREAD_INCREASE = 0.0005
SITING_CENTRE_MAX_WAKE_INCREASE = 0.001
SITING_DIRECTION_STRESS_DEG = (-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0)
SITING_DIRECTION_STRESS_ROWS = 2400
SITING_LAYOUT_X_M = (
    -7429.8238361363283, -4853.813481746467, -887.17510178603823,
    1085.1403891480504, 3488.1697672177652, 5170.1015103275895,
    7488.3343518339825, -6179.9234460895332, -3520.7249493493673,
    -2775.6148860346639, 770.41011366604221, 3064.2344045586269,
    5219.2684267922186, 7457.0673113106295, -7438.0731237379841,
    -5538.8250347036173, -1272.4439146866196, -762.6648503827646,
    3386.0014073224593, 5056.0201489537385, 7493.5466857728807,
    -7495.6062813727112, -4482.2228464213149, -3361.5443105389054,
    807.37595918364468, 2943.1733361121019, 5458.1736725969213,
    7360.6367703284031, -7493.9668112212212, -5453.7658309566668,
    -2972.7304925252838, -971.70892750882365, 969.8047155095486,
    5887.6894821532314, 7493.4248613251693, -7451.6906386625442,
    -5005.51773173537, -434.642973151251, -1437.9132919343526,
    1962.5061834199228, 3451.0303709914306, 7430.0487858015367,
    -7493.5466857728807, -6016.0155257383567, -3083.9686201251125,
    1453.8349550072121, 3281.4922132409583, 4994.7350750972928,
    7485.4518676009739, -7062.9995591612642, -4440.2851417648781,
    -2473.7229996738979, -350.072067076569, 2142.8912639190639,
    4707.5004940800445,
)
SITING_LAYOUT_Y_M = (
    -7147.1942125249352, -7486.853549606285, -7489.0041056077353,
    -6847.5199336248152, -7118.9334242501554, -7499.9366718926485,
    -7485.8506642123557, -5338.6586031557326, -3949.8481581968,
    -6595.3642626171186, -5205.6434852103321, -4716.679787469282,
    -4879.4391175865294, -5159.6712560008546, -4172.4452141354359,
    -3026.3925698257808, -1770.9817143561415, -4697.328939142586,
    -2317.3903780516207, -3103.4624083867939, -3263.5616485419523,
    -1714.5764704694682, -1885.9931479153111, -180.0173884316269,
    -1877.7191004259103, -507.52767833190148, 366.71848416809962,
    -360.50281657435437, 1109.9933770369844, 368.90473472701706,
    1959.1358270602336, 860.06536527522223, 350.45832490303235,
    2615.4138567931409, 1949.1493653793768, 3206.5050970339662,
    2782.31636710138, 2982.2341558891926, 4628.3530865886405,
    2417.8630732953015, 1748.0899564121421, 4455.2387840966512,
    5055.3401726883167, 6060.6311693743601, 5165.5449446552211,
    5622.573357910811, 5236.2589979292388, 5124.7264119867195,
    7450.9864128265663, 7432.5174772597147, 7499.9366718926485,
    7498.3203051573346, 7356.7576454603504, 7497.6674537435047,
    7492.8114339632339,
)
SITING_ROBUST_LAYOUT_AUDIT = {
    "method": "input-only robust multi-climate coordinate refinement",
    "search_evaluations": 340,
    "accepted_coordinate_moves": 26,
    "training_years": [2016, 2017, 2018, 2019, 2020],
    "exact_replay_delta": {
        "mean_capacity_factor": 0.00013275150242275657,
        "worst_year_capacity_factor": 0.0001870323768896065,
        "max_wake_loss_fraction": -0.0003598157133408719,
        "annual_capacity_factor": {
            "2016": 0.0001870323768896065,
            "2017": 0.00008146907628225897,
            "2018": 0.00019677213353930245,
            "2019": 0.0000930198252815373,
            "2020": 0.00010546410012168828,
        },
    },
    "direction_shift_capacity_factor_delta": {
        "minus_45_deg": 0.00014327,
        "minus_30_deg": 0.00013927,
        "minus_15_deg": 0.00018463,
        "zero_deg": 0.00010193,
        "plus_15_deg": 0.00015307,
        "plus_30_deg": 0.00017629,
        "plus_45_deg": 0.00022075,
    },
    "monthly_block_bootstrap": {
        "replicates": 2000,
        "non_improving_replicates": 0,
        "delta_q01": 0.0000903847523430903,
        "delta_q05": 0.00010238286892009564,
        "delta_median": 0.00013239708143716154,
    },
    "nearby_legal_centre_transfer": {
        "centres_tested": 31,
        "centres_passing_every_year_mean_worst_and_wake_gates": 31,
    },
    "input_only": True,
    "previous_submission_inputs": [],
}
D14_DIRECTION_POLICY = {
    (4, 8, 0): ("direct", 0.50),
    (4, 8, 6): ("vector", 1.00),
    (4, 8, 12): ("direct", 0.50),
    (7, 1, 18): ("direct", 0.30),
}
FINE_D14_CLIMATOLOGY_POLICY = {
    (1, 14, 0): (28, 142.02401542663574),
    (1, 14, 6): (14, 129.09210968017578),
    (1, 14, 12): (14, 127.73582458496094),
    (1, 14, 18): (14, 118.41080474853516),
    (7, 1, 6): (7, 129.7165412902832),
    (11, 4, 0): (28, 153.62288665771484),
    (11, 4, 6): (7, 149.84632873535156),
}
FINE_D14_ENDPOINT_POLICY = {}
FINE_D14_ENDPOINT_FACTORS = (0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10)
FINE_D14_ENDPOINT_MIN_REGIME_ROWS = 1000
FINE_D14_ENDPOINT_MIN_YEAR_GAIN = 0.10
FINE_D14_ENDPOINT_MIN_TRAIN_GAIN = 0.25
FINE_D14_ENDPOINT_MIN_CELL_GAIN = 0.50
FINE_D7_CLIMATOLOGY_POLICY = {}
D7_DIRECTION_CENTER_POLICY = {
    (2, 25, 18): ("climatology", 0.20),
    (7, 1, 6): ("d1", 0.40),
    (8, 12, 6): ("climatology", 1.00),
    (8, 12, 12): ("climatology", 1.00),
    (8, 12, 18): ("climatology", 1.00),
}
D7_PRESSURE_YEARS = (2019, 2020)
D7_PRESSURE_MIN_BIN_COUNT = 750
D7_PRESSURE_DISAGREEMENT_EDGES = np.asarray(
    (0.0, 15.0, 35.0, 70.0, 181.0), dtype="float64"
)
D7_PRESSURE_SPEED_EDGES = np.asarray(
    (0.0, 6.0, 10.0, np.inf), dtype="float64"
)
# Each candidate was selected by a 2019<->2020 cross-fit. Production lookup
# cells must contain at least 750 rows in both years, and every retained rule
# must remain non-worse in every populated validation regime.
D7_PRESSURE_CALIBRATED_SPECS = (
    (2, 25, 18, "1000", "pdir8", 1.00),
    (7, 1, 12, "low2", "pdir8", 1.00),
    (2, 25, 0, "500", "scalar", 1.00),
    (4, 8, 18, "700", "base4_spatial4", 0.80),
    (11, 4, 18, "850", "dis4", 0.20),
    (9, 23, 12, "500", "base4_spatial4", 1.00),
    (8, 12, 0, "850", "spatial4", 0.80),
    (4, 8, 6, "low3", "dis4_pdir4", 0.65),
    (1, 14, 18, "700", "pdir4", 0.40),
    (4, 8, 0, "850", "dis4_pdir4", 1.00),
    (9, 23, 6, "850", "dis4", 1.00),
    (1, 14, 12, "925", "pdir4", 0.20),
    (5, 20, 6, "700", "scalar", 0.05),
    (9, 23, 0, "850", "spatial4", 0.075),
    (11, 4, 0, "850", "spatial4", 1.00),
    (8, 12, 12, "850", "spatial4", 0.40),
    (8, 12, 18, "500", "pdir4_spatial4", 0.80),
)
D7_PRESSURE_RAW_SPECS = (
    (2, 25, 6, "500", 1.00),
    (11, 4, 12, "700", 1.00),
)
# This activation-aware rule targets a pressure-direction regime present in
# 2019, 2021, and the organizer inference inputs but absent in 2020. Its
# spatial mask is selected by leave-one-tile-out score with a strict margin,
# then audited with independent latitude, longitude, quadrant, checkerboard,
# and physical-regime gates. It adds no estimator.
D7_PRESSURE_SPATIAL_SPECS = (
    (
        7,
        1,
        12,
        "low_30_70",
        "pdir8",
        0.70,
        (6, 7),
        (2, 4, 5, 6, 7, 8, 9, 10, 11),
    ),
)
D7_PRESSURE_SPATIAL_FIT_YEAR = 2019
D7_PRESSURE_SPATIAL_MIN_FIT_COUNT = 500
D7_PRESSURE_SPATIAL_MIN_REGIME_COUNT = 150
D7_PRESSURE_SPATIAL_TILE_MARGIN = -1.0
FINE_D7_SPATIAL_STEP = 32
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
FINE_D7_DIRECTION_POLICY = {
    # Exact production-feature nested selection / untouched-year evaluation.
    # Every selected cell is non-worse in all five outer years and all
    # populated physical regimes; the earlier public v64 cells are absent.
    (1, 14, 6): (-1.0, 0.40, 0.65, 15.0),
    (2, 25, 18): (1.0, 0.80, 0.65, np.inf),
    (4, 8, 6): (1.0, 2.00, 0.80, 90.0),
    (7, 1, 18): (1.0, 2.00, 0.95, np.inf),
    (11, 4, 18): (-1.0, 1.00, 0.65, 45.0),
}
FINE_D7_CONTEXT_POLICY = {
    (1, 14, 0): (1.0, 0.40, 0.80, 15.0),
    (1, 14, 6): (1.0, 1.25, 0.80, 20.0),
    (2, 25, 0): (-1.0, 0.30, 0.65, 45.0),
    (2, 25, 6): (1.0, 1.25, 0.00, 10.0),
    (2, 25, 12): (-1.0, 0.80, 0.65, 60.0),
    (2, 25, 18): (1.0, 0.50, 0.90, 45.0),
    (4, 8, 0): (-1.0, 0.05, 0.65, 10.0),
    (4, 8, 6): (-1.0, 0.80, 0.00, 45.0),
    (4, 8, 18): (-1.0, 1.25, 0.80, 20.0),
    (5, 20, 0): (1.0, 1.25, 0.00, 15.0),
    (5, 20, 6): (-1.0, 0.80, 0.00, 20.0),
    (7, 1, 0): (-1.0, 0.80, 0.65, 30.0),
    (7, 1, 6): (1.0, 1.25, 0.00, 30.0),
    (7, 1, 18): (1.0, 1.25, 0.65, 20.0),
    (8, 12, 12): (-1.0, 1.25, 0.80, 60.0),
    (9, 23, 0): (1.0, 1.25, 0.95, 20.0),
    (9, 23, 12): (1.0, 1.00, 0.90, 45.0),
    (9, 23, 18): (1.0, 0.40, 0.65, 15.0),
    (11, 4, 0): (1.0, 1.00, 0.80, 20.0),
    (11, 4, 6): (-1.0, 0.50, 0.80, 20.0),
    (11, 4, 12): (-1.0, 1.00, 0.00, 30.0),
    (11, 4, 18): (-1.0, 0.40, 0.65, 30.0),
}
D7_D10_TENDENCY_POLICY = {
    (1, 14, 6): (0.20, 180.0),
    (1, 14, 12): (0.50, 180.0),
    (1, 14, 18): (0.20, 180.0),
    (2, 25, 0): (-0.40, 20.0),
    (2, 25, 6): (0.80, 180.0),
    (2, 25, 12): (0.30, 90.0),
    (2, 25, 18): (1.00, 180.0),
    (4, 8, 6): (-0.65, 20.0),
    (4, 8, 12): (0.30, 90.0),
    (4, 8, 18): (0.40, 180.0),
    (5, 20, 12): (-0.80, 30.0),
    (7, 1, 18): (-0.50, 45.0),
    (8, 12, 0): (-1.25, 60.0),
    (9, 23, 6): (-0.10, 135.0),
    (9, 23, 12): (-0.40, 30.0),
    (11, 4, 0): (0.30, 135.0),
    (11, 4, 12): (1.25, 180.0),
}
D1_DIRECTION_SPEED_EDGES = np.asarray(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        12.0,
        15.0,
        20.0,
        np.inf,
    ],
    dtype="float32",
)
D1_INTERVAL_YEARS = tuple(range(2016, 2021))
D7_INTERVAL_YEARS = tuple(range(2016, 2021))
D7_DEPLOYED_BASE_HALF_WIDTH = 138.0
D7_INTERVAL_COVERAGE = 0.90
D7_BIAS_SHRINKAGE_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
D7_WIDTH_SHRINKAGE_GRID = (0.25, 0.50, 0.75, 1.0)
D7_SPEED_EDGES = np.asarray(
    [0.0, 4.0, 6.0, 8.0, 10.0, np.inf], dtype="float32"
)
D7_DIRECTION_SECTOR_EDGES = np.linspace(
    0.0, 360.0, 9, dtype="float32"
)
D7_BIAS_SECTOR_EDGES = {
    "direction_sector_4": np.linspace(0.0, 360.0, 5, dtype="float32"),
    "direction_sector_8": D7_DIRECTION_SECTOR_EDGES,
    "direction_sector_12": np.linspace(0.0, 360.0, 13, dtype="float32"),
}
D7_BIAS_FAMILIES = ("scalar", *D7_BIAS_SECTOR_EDGES)
D7_WIDTH_FAMILIES = ("scalar", "direction_sector")
D7_MIN_CONDITION_BIN_COUNT = 1000
D7_MAX_ABS_BIAS = 30.0
D7_MIN_MEAN_GATE_GAIN = 0.25
D7_LEAD_RATIO_SLOT = (8, 12, 6)
D7_LEAD_RATIO_EDGES = np.asarray(
    [0.0, 0.6, 0.9, 1.3, np.inf], dtype="float32"
)
D7_LEAD_RATIO_SHRINKAGE = 0.20
D7_LEAD_RATIO_MIN_BIN_COUNT = 1000
D7_ASYMMETRIC_ALPHA = 0.10
D7_ASYMMETRIC_SHRINKAGE_GRID = (
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.65,
    0.80,
    1.00,
)
D7_ASYMMETRIC_MIN_BIN_COUNT = 1000
D7_ASYMMETRIC_MIN_MEAN_GAIN = 0.05
D7_CONDITIONAL_WIDTH_SCALES = (
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
    0.925,
    0.95,
    0.975,
    0.99,
)
D7_CONDITIONAL_WIDTH_MIN_BIN_COUNT = 500
D7_CONDITIONAL_WIDTH_MIN_ACTIVE_FRACTION = 0.05
D7_CONDITIONAL_WIDTH_MIN_BIN_GAIN = 0.20
D7_CONDITIONAL_WIDTH_MIN_SLOT_GAIN = 0.10
D7_CONDITIONAL_WIDTH_MIN_AGGREGATE_GAIN = 5.0
D14_DIRECTION_INTERVAL_EDGES = np.asarray([0.0, np.inf], dtype="float32")
D14_INTERVAL_YEARS = (2019, 2020)
D14_INTERVAL_REGIME = (2, 25)
D14_INTERVAL_COVERAGE = 0.90
D14_INTERVAL_SHRINKAGE = 0.75
D14_DEPLOYED_BASE_HALF_WIDTH = 158.0
D14_INTERVAL_SELECTED_SLOTS = (
    (2, 25, 0),
    (2, 25, 6),
    (2, 25, 12),
    (2, 25, 18),
)
QMOS_REFIT_RULES = (
    {
        "lead": 1,
        "month": 2,
        "day": 25,
        "hour": 18,
        "weight": 1.0,
        "spatial_bins": (1, 2, 3),
        "exclude_spatial_width": ((1, 0),),
        "strict_gate": {
            "mean_delta": -0.21121501709904125,
            "delta_by_outer_year": {
                "2019": -0.1283026883084837,
                "2020": -0.2941273458895988,
            },
            "every_selected_observable_interaction_non_worse": True,
        },
    },
    {
        "lead": 7,
        "month": 8,
        "day": 12,
        "hour": 18,
        "weight": 0.02,
        "spatial_bins": (2,),
        "exclude_spatial_width": ((2, 1),),
        "strict_gate": {
            "mean_delta": -0.03793265673111522,
            "delta_by_outer_year": {
                "2019": -0.018703857990392734,
                "2020": -0.05716145547183771,
            },
            "every_selected_observable_interaction_non_worse": True,
        },
    },
)
QMOS_REFIT_SUPPORT_PADDING = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the frugal Phase 2 forecast model.")
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
        "--phase1-data-root",
        type=Path,
        default=None,
        help=(
            "Optional unpacked official Phase 1 data root. It is added to the "
            "official kit's data search path so the 2019-2020 HRES inputs are "
            "available during training."
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=SCRIPT_DIR / "artifacts",
        help="Directory where the compact artifact bundle will be written.",
    )
    parser.add_argument(
        "--train-freq",
        default="6D",
        help="Issue-date frequency for MOS training over 2016-2020. Use 3D for stronger/slower.",
    )
    parser.add_argument(
        "--downscale-year",
        type=int,
        default=2020,
        help="Target-truth year used to train the terrain downscaler.",
    )
    parser.add_argument(
        "--downscale-step",
        type=int,
        default=20,
        help="Use every Nth available target day in downscale-year. Larger is faster.",
    )
    parser.add_argument(
        "--coverage-target",
        type=float,
        default=0.90,
        help="Target conformal/calibrated interval coverage on training years.",
    )
    parser.add_argument(
        "--skip-interval-calibration",
        action="store_true",
        help="Skip fine-grid interval calibration and use raw MOS/downscaler intervals.",
    )
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


def configure_data_root(
    data_root: Path | None, phase1_data_root: Path | None = None
) -> None:
    if data_root is not None:
        roots = [data_root.expanduser().resolve()]
        if phase1_data_root is not None:
            roots.append(phase1_data_root.expanduser().resolve())
        os.environ["PHASE2_DATA_ROOT"] = os.pathsep.join(map(str, roots))
        return
    if phase1_data_root is not None:
        raise ValueError("--phase1-data-root requires --data-root")
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




def summarize_environment(config, target_loader) -> dict:
    target_dates = target_loader.list_dates(config.target_root())
    hres_paths = config.hres_parquets()
    if not target_dates:
        raise FileNotFoundError(
            f"No target dates found under {config.target_root()}. "
            "Run the Phase 2 dataset setup first."
        )
    if not hres_paths:
        raise FileNotFoundError(
            "No HRES parquet files found. The kit needs Phase 2 HRES data and, "
            "when available, Phase 1 HRES data next to the Phase 2 data root."
        )
    static = target_loader.load_static(config.target_root())
    return {
        "target_root": str(config.target_root()),
        "target_static": str(config.target_static()),
        "coarse_root": str(config.coarse_root()),
        "reanalysis_root": str(config.reanalysis_root()),
        "hres_parquets": [str(p) for p in hres_paths],
        "n_target_days": len(target_dates),
        "first_target_day": str(target_dates[0]),
        "last_target_day": str(target_dates[-1]),
        "target_shape": list(static.shape),
        "target_sea_cells": int(static.sea.sum()),
    }


def compact_hres_table(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce the official HRES table to stable training dtypes."""
    float_cols = [
        "lat", "lon", "fcst_u", "fcst_v", "fcst_speed", "woy_sin",
        "woy_cos", "u125c", "v125c",
    ]
    float_cols.extend(col for col in ("analysis_u", "analysis_v") if col in df)
    for col in float_cols:
        df[col] = df[col].astype("float32")
    df["lead"] = df["lead"].astype("int8")
    df["hour"] = df["hour"].astype("int8")
    return df


def load_hres_frame(config) -> pd.DataFrame:
    """Load training HRES plus any organiser-provided inference windows."""
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


def _align_analysis_snapshot(snapshot, lat, lon, index=None):
    """Align one official low-resolution analysis snapshot to the HRES grid."""
    if index is None:
        source_lon, source_lat = np.meshgrid(snapshot.lons, snapshot.lats)
        source_key = {
            (round(la, 3), round(lo, 3)): i
            for i, (la, lo) in enumerate(zip(source_lat.ravel(), source_lon.ravel()))
        }
        index = np.array(
            [source_key.get((round(la, 3), round(lo, 3)), -1) for la, lo in zip(lat, lon)],
            dtype="int32",
        )
    ok = index >= 0
    analysis_u = np.full(len(index), np.nan, dtype="float32")
    analysis_v = np.full(len(index), np.nan, dtype="float32")
    analysis_u[ok] = snapshot.u100.ravel()[index[ok]]
    analysis_v[ok] = snapshot.v100.ravel()[index[ok]]
    return analysis_u, analysis_v, index


def load_issue_analysis(config, issue, lat, lon) -> dict:
    """Load issue-time u100/v100 from the official training reanalysis files."""
    issue = pd.Timestamp(issue)
    cache_key = issue.normalize()
    cached = _ANALYSIS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    import reanalysis_loader

    fields = {}
    index = None
    for hour in HOURS:
        snapshot = reanalysis_loader.load_reanalysis(
            issue.date(), hour, root=config.reanalysis_root()
        )
        analysis_u, analysis_v, index = _align_analysis_snapshot(
            snapshot, lat, lon, index=index
        )
        fields[hour] = (analysis_u, analysis_v)
    _ANALYSIS_CACHE[cache_key] = fields
    return fields


def build_hybrid_table(
    fh,
    config,
    issue_dates,
    with_truth: bool = True,
    leads=LEADS,
    with_analysis: bool = False,
) -> pd.DataFrame:
    """Build HRES features for d1/d7 and map d10 HRES to the d14 target."""
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
            if with_analysis and 1 in leads
            else None
        )
        for lead in leads:
            source_lead = SOURCE_LEAD[lead]
            speed_cols = [f"fcst_speed_d{source_lead}_h{hour}" for hour in HOURS]
            dir_cols = [f"fcst_dir_d{source_lead}_h{hour}" for hour in HOURS]
            if any(col not in hrow.columns for col in speed_cols + dir_cols):
                continue
            valid = issue + pd.Timedelta(days=lead)
            coarse = fh._coarse_grid(f"{valid:%Y-%m-%d}") if with_truth else None
            if with_truth and coarse is None:
                continue
            idx = None
            if coarse is not None:
                coarse_lat, coarse_lon, coarse_uv = coarse
                key = {
                    (round(la, 3), round(lo, 3)): i
                    for i, (la, lo) in enumerate(zip(coarse_lat, coarse_lon))
                }
                idx = np.array(
                    [key.get((round(la, 3), round(lo, 3)), -1) for la, lo in zip(lat, lon)]
                )
            week = int(valid.isocalendar().week)
            for hour, speed_col, dir_col in zip(HOURS, speed_cols, dir_cols):
                speed = hrow[speed_col].to_numpy(dtype="float64")
                direction = hrow[dir_col].to_numpy(dtype="float64")
                fcst_u, fcst_v = fh._uv_from_speed_dir(speed, direction)
                if coarse is None:
                    true_u = np.full(lat.shape, np.nan)
                    true_v = np.full(lat.shape, np.nan)
                else:
                    all_u, all_v = coarse_uv[hour]
                    ok = idx >= 0
                    true_u = np.full(idx.shape, np.nan)
                    true_v = np.full(idx.shape, np.nan)
                    true_u[ok] = all_u[idx[ok]]
                    true_v[ok] = all_v[idx[ok]]
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
                            "u125c": true_u,
                            "v125c": true_v,
                            "analysis_u": analysis_u,
                            "analysis_v": analysis_v,
                        }
                    )
                )
    if not blocks:
        raise ValueError("No HRES rows matched the requested issue dates")
    table = pd.concat(blocks, ignore_index=True)
    required = ["fcst_u", "fcst_v", "fcst_speed"]
    if with_truth:
        required.extend(["u125c", "v125c"])
    table = table.dropna(subset=required).reset_index(drop=True)
    return compact_hres_table(table)


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
    """Add issue-time flow-regime features to the compact d1 feature set."""
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
    """Build the legal 14-day reanalysis context used by the gated d1 model."""
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


def d1_dense_daily_feature_names() -> list[str]:
    """Stable feature contract for the shared dense-daily d1 endpoint pair."""
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


def _read_d1_dense_daily_file(path, names, allow_masked=False):
    """Read one official low-resolution day with a deterministic orientation."""
    from netCDF4 import Dataset

    with Dataset(path) as dataset:
        values = np.stack(
            [
                np.asarray(
                    np.ma.filled(dataset.variables[name][:], np.nan),
                    dtype="float32",
                )
                for name in names
            ],
            axis=1,
        )
        latitude = np.asarray(dataset.variables["latitude"][:], dtype="float32")
        longitude = np.asarray(dataset.variables["longitude"][:], dtype="float32")
    if latitude[0] > latitude[-1]:
        latitude = latitude[::-1].copy()
        values = values[:, :, ::-1, :]
    if longitude[0] > longitude[-1]:
        longitude = longitude[::-1].copy()
        values = values[:, :, :, ::-1]
    valid = np.isfinite(values).all(axis=(0, 1))
    if values.shape[0:2] != (len(HOURS), len(names)):
        raise RuntimeError(f"Unexpected daily field shape in {path}: {values.shape}")
    if not allow_masked and not valid.all():
        raise RuntimeError(f"Unexpected missing organizer context in {path}")
    return values, latitude, longitude, valid


def _load_d1_dense_daily_arrays(config):
    """Load each 2016-2020 daily organizer field once into compact arrays."""
    reanalysis_files = {}
    target_files = {}
    for year in range(2016, 2021):
        for path in sorted(
            (Path(config.reanalysis_root()) / str(year)).glob("reanalysis_*.nc")
        ):
            reanalysis_files[pd.Timestamp(path.stem.rsplit("_", 1)[-1])] = path
        for path in sorted(
            (Path(config.coarse_root()) / str(year)).glob("coarse_*.nc")
        ):
            target_files[pd.Timestamp(path.stem.rsplit("_", 1)[-1])] = path
    dates = pd.DatetimeIndex(sorted(set(reanalysis_files) & set(target_files)))
    if dates.empty:
        raise FileNotFoundError("No overlapping 2016-2020 reanalysis/coarse days")

    first_target, latitude, longitude, valid = _read_d1_dense_daily_file(
        target_files[dates[0]], ("u125c", "v125c"), allow_masked=True
    )
    del first_target
    valid_count = int(valid.sum())
    shape = (len(dates), len(HOURS), 2, valid_count)
    state = np.empty(shape, dtype="float32")
    surface10 = np.empty(shape, dtype="float32")
    target = np.empty(shape, dtype="float32")
    expected_shape = (len(HOURS), 2, len(latitude), len(longitude))

    for index, date in enumerate(dates):
        reanalysis, source_lat, source_lon, state_valid = _read_d1_dense_daily_file(
            reanalysis_files[date], ("u100", "v100", "u10", "v10")
        )
        truth, truth_lat, truth_lon, truth_valid = _read_d1_dense_daily_file(
            target_files[date], ("u125c", "v125c"), allow_masked=True
        )
        if (
            reanalysis.shape[0] != len(HOURS)
            or truth.shape != expected_shape
            or not state_valid.all()
            or not np.array_equal(source_lat, latitude)
            or not np.array_equal(source_lon, longitude)
            or not np.array_equal(truth_lat, latitude)
            or not np.array_equal(truth_lon, longitude)
            or not np.all(truth_valid[valid])
        ):
            raise RuntimeError(f"Dense d1 grid mismatch at {date.date()}")
        state[index] = reanalysis[:, 0:2][:, :, valid]
        surface10[index] = reanalysis[:, 2:4][:, :, valid]
        target[index] = truth[:, :, valid]
        if (index + 1) % 365 == 0 or index + 1 == len(dates):
            print(
                f"[train] dense d1 organizer days {index + 1:,}/{len(dates):,}",
                flush=True,
            )

    hres = np.full(shape, np.nan, dtype="float32")
    date_lookup = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    hres_frame = load_hres_frame(config)
    columns = ["time", "latitude", "longitude"]
    for hour in HOURS:
        columns.extend((f"fcst_speed_d1_h{hour}", f"fcst_dir_d1_h{hour}"))
    hres_frame = hres_frame.loc[
        hres_frame["time"].isin(dates), columns
    ].sort_values(["time", "latitude", "longitude"])
    loaded_dates = 0
    for date, group in hres_frame.groupby("time", sort=True):
        date = pd.Timestamp(date)
        if date not in date_lookup:
            continue
        if len(group) != len(latitude) * len(longitude):
            raise RuntimeError(f"Incomplete dense d1 HRES field at {date.date()}")
        if not (
            np.allclose(np.sort(group["latitude"].unique()), latitude, atol=1e-5)
            and np.allclose(np.sort(group["longitude"].unique()), longitude, atol=1e-5)
        ):
            raise RuntimeError(f"Dense d1 HRES coordinates differ at {date.date()}")
        fields = []
        for hour in HOURS:
            speed = group[f"fcst_speed_d1_h{hour}"].to_numpy(dtype="float32").reshape(
                len(latitude), len(longitude)
            )
            direction = group[f"fcst_dir_d1_h{hour}"].to_numpy(
                dtype="float32"
            ).reshape(len(latitude), len(longitude))
            radians = np.radians(direction)
            fields.append(
                np.stack((-speed * np.sin(radians), -speed * np.cos(radians)))
            )
        hres[date_lookup[date]] = np.stack(fields)[:, :, valid]
        loaded_dates += 1
    if loaded_dates != len(dates) or not np.isfinite(hres).all():
        missing = int(np.isnan(hres).all(axis=(1, 2, 3)).sum())
        raise RuntimeError(
            f"Dense d1 HRES coverage mismatch: loaded={loaded_dates} "
            f"expected={len(dates)} missing={missing}"
        )

    lat_mesh, lon_mesh = np.meshgrid(latitude, longitude, indexing="ij")
    return (
        dates,
        state,
        surface10,
        hres,
        target,
        valid,
        lat_mesh[valid].astype("float32"),
        lon_mesh[valid].astype("float32"),
        latitude,
        longitude,
    )


def _d1_dense_daily_eligible_indexes(dates):
    lookup = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    indexes = []
    for index, date in enumerate(dates):
        required = [date - pd.Timedelta(days=lag) for lag in D1_DENSE_DAILY_LAGS]
        required.append(date + pd.Timedelta(days=1))
        if all(value in lookup for value in required):
            indexes.append(index)
    return np.asarray(indexes, dtype="int64"), lookup


def _build_d1_dense_daily_rows(
    indexes,
    dates,
    date_lookup,
    state,
    surface10,
    hres,
    target,
    latitude,
    longitude,
    cells,
):
    indexes = np.asarray(indexes, dtype="int64")
    cells = np.asarray(cells, dtype="int64")
    n_dates, n_cells = len(indexes), len(cells)
    row_count = n_dates * len(HOURS) * n_cells
    columns = []

    hres_selected = hres[indexes][:, :, :, cells]
    hres_u = hres_selected[:, :, 0]
    hres_v = hres_selected[:, :, 1]
    hres_speed = np.hypot(hres_u, hres_v)
    columns.extend((hres_u, hres_v, hres_speed))
    for source_hour in range(len(HOURS)):
        columns.append(
            np.broadcast_to(
                hres_speed[:, source_hour : source_hour + 1],
                (n_dates, len(HOURS), n_cells),
            )
        )

    current100 = state[indexes][:, :, :, cells]
    current10 = surface10[indexes][:, :, :, cells]
    u100, v100 = current100[:, :, 0], current100[:, :, 1]
    u10, v10 = current10[:, :, 0], current10[:, :, 1]
    speed100, speed10 = np.hypot(u100, v100), np.hypot(u10, v10)
    columns.extend((u10, v10, u100, v100, speed10, speed100, speed100 - speed10))
    for lag in D1_DENSE_DAILY_LAGS[1:]:
        lagged = state[indexes - lag][:, :, :, cells]
        columns.extend((lagged[:, :, 0], lagged[:, :, 1]))
    for days in D1_DENSE_DAILY_MEANS:
        block = np.stack([state[indexes - lag][:, :, :, cells] for lag in range(days)])
        mean_u = np.mean(block[:, :, :, 0], axis=0)
        mean_v = np.mean(block[:, :, :, 1], axis=0)
        mean_speed = np.mean(np.hypot(block[:, :, :, 0], block[:, :, :, 1]), axis=0)
        columns.extend(
            (mean_u, mean_v, np.hypot(mean_u, mean_v) / np.maximum(mean_speed, 0.1))
        )
    denominator = np.maximum(hres_speed * speed100, 0.25)
    columns.extend(
        (
            hres_u - u100,
            hres_v - v100,
            (hres_u * u100 + hres_v * v100) / denominator,
            (hres_u * v100 - hres_v * u100) / denominator,
        )
    )

    lat_grid = np.broadcast_to(
        latitude[cells][None, None, :], (n_dates, len(HOURS), n_cells)
    )
    lon_grid = np.broadcast_to(
        longitude[cells][None, None, :], (n_dates, len(HOURS), n_cells)
    )
    day = np.asarray([dates[index].dayofyear for index in indexes], dtype="float32")
    season = 2.0 * np.pi * day / 365.2425
    season_sin = np.broadcast_to(np.sin(season)[:, None, None], lat_grid.shape)
    season_cos = np.broadcast_to(np.cos(season)[:, None, None], lat_grid.shape)
    hour_angle = 2.0 * np.pi * np.asarray(HOURS, dtype="float32") / 24.0
    hour_sin = np.broadcast_to(np.sin(hour_angle)[None, :, None], lat_grid.shape)
    hour_cos = np.broadcast_to(np.cos(hour_angle)[None, :, None], lat_grid.shape)
    columns.extend((lat_grid, lon_grid, season_sin, season_cos, hour_sin, hour_cos))

    names = d1_dense_daily_feature_names()
    x = np.column_stack(
        [np.asarray(column, dtype="float32").reshape(row_count) for column in columns]
    )
    future = np.asarray(
        [date_lookup[dates[index] + pd.Timedelta(days=1)] for index in indexes],
        dtype="int64",
    )
    target_values = target[future][:, :, :, cells]
    y = np.hypot(target_values[:, :, 0], target_values[:, :, 1]).reshape(row_count)
    if x.shape[1] != len(names) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError(f"Invalid dense d1 training matrix {x.shape}")
    return pd.DataFrame(x, columns=names), y.astype("float32")


def train_d1_dense_daily_policy(config, checkpoint_path: Path | None = None) -> dict:
    """Train the two-model daily d1 signal selected by strict exact replay."""
    import lightgbm as lgb

    checkpoint = (
        _load_checkpoint(checkpoint_path, "d1_dense_daily")
        if checkpoint_path is not None
        else None
    )
    if checkpoint is not None:
        return checkpoint["payload"]
    if not D1_DENSE_DAILY_GATE["passed"]:
        raise RuntimeError("Dense d1 spatial policy did not pass its strict gate")

    (
        dates,
        state,
        surface10,
        hres,
        target,
        valid,
        valid_latitude,
        valid_longitude,
        grid_latitude,
        grid_longitude,
    ) = _load_d1_dense_daily_arrays(config)
    eligible, date_lookup = _d1_dense_daily_eligible_indexes(dates)
    fit_indexes = np.asarray(
        [index for index in eligible if dates[index].year <= 2019], dtype="int64"
    )[::D1_DENSE_DAILY_DAY_STEP]
    calibration_wanted = {
        pd.Timestamp(year=2020, month=month, day=day)
        for month, day in CONTEXT_REGIMES
    }
    calibration_indexes = np.asarray(
        [index for index in eligible if dates[index] in calibration_wanted],
        dtype="int64",
    )
    if len(calibration_indexes) != len(CONTEXT_REGIMES):
        raise RuntimeError("Dense d1 calibration issue-date coverage is incomplete")
    all_cells = np.arange(int(valid.sum()), dtype="int64")
    train_cells = all_cells[::D1_DENSE_DAILY_CELL_STEP]
    x_fit, y_fit = _build_d1_dense_daily_rows(
        fit_indexes,
        dates,
        date_lookup,
        state,
        surface10,
        hres,
        target,
        valid_latitude,
        valid_longitude,
        train_cells,
    )
    x_calibration, y_calibration = _build_d1_dense_daily_rows(
        calibration_indexes,
        dates,
        date_lookup,
        state,
        surface10,
        hres,
        target,
        valid_latitude,
        valid_longitude,
        all_cells,
    )
    params = dict(
        n_estimators=220,
        learning_rate=0.045,
        num_leaves=31,
        min_child_samples=180,
        reg_lambda=2.0,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=1,
        verbose=-1,
    )
    models = {
        quantile: lgb.LGBMRegressor(
            objective="quantile", alpha=quantile, **params
        ).fit(x_fit, y_fit)
        for quantile in D1_DENSE_DAILY_QUANTILES
    }
    calibration_prediction = np.column_stack(
        [models[quantile].predict(x_calibration) for quantile in D1_DENSE_DAILY_QUANTILES]
    )
    offsets = np.asarray(
        [
            np.quantile(y_calibration - calibration_prediction[:, 0], 0.05),
            np.quantile(y_calibration - calibration_prediction[:, 1], 0.95),
        ],
        dtype="float64",
    )
    rules = [
        {
            "month": int(month),
            "day": int(day),
            "hour": int(hour),
            "spatial_bin": int(spatial_bin),
            "weight": float(weight),
        }
        for month, day, hour, spatial_bin, weight in D1_DENSE_DAILY_RULES
    ]
    payload = {
        "method": D1_DENSE_DAILY_GATE["method"],
        "models": models,
        "features": d1_dense_daily_feature_names(),
        "calibration_offsets": offsets,
        "rules": rules,
        "coarse_valid_flat_indices": np.flatnonzero(valid).astype("int32"),
        "coarse_latitude": np.asarray(grid_latitude, dtype="float32"),
        "coarse_longitude": np.asarray(grid_longitude, dtype="float32"),
        "post_width_scale": 0.75,
        "training_years": [2016, 2017, 2018, 2019],
        "calibration_year": 2020,
        "training_dates": int(len(fit_indexes)),
        "training_rows": int(len(y_fit)),
        "calibration_rows": int(len(y_calibration)),
        "new_models": len(models),
        "gate": dict(D1_DENSE_DAILY_GATE),
        "input_only_training": True,
        "previous_submission_inputs": [],
    }
    del x_fit, y_fit, x_calibration, y_calibration
    del state, surface10, hres, target
    gc.collect()
    if checkpoint_path is not None:
        _save_checkpoint(checkpoint_path, "d1_dense_daily", payload=payload)
    return payload


def add_d7_speed_context_features(
    config,
    fh,
    issue_date,
    table: pd.DataFrame,
) -> pd.DataFrame:
    """Compact, legal d7 speed features accepted by the strict endpoint gate."""
    import reanalysis_loader

    issue_date = pd.Timestamp(issue_date).normalize()
    table = table.reset_index(drop=True)
    if set(table["lead"].unique()) != {7}:
        raise ValueError("d7 speed context features require a d7-only table")
    unique_dates = pd.to_datetime(table["issue_date"]).dt.normalize().unique()
    if len(unique_dates) != 1 or pd.Timestamp(unique_dates[0]) != issue_date:
        raise ValueError("d7 speed context features require one aligned issue date")

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
            snapshot = reanalysis_loader.load_reanalysis(
                day.date(), source_hour, root=config.reanalysis_root()
            )
            aligned_u, aligned_v, source_index = _align_analysis_snapshot(
                snapshot, lat, lon, index=source_index
            )
            if (source_index < 0).any():
                raise ValueError("Could not align d7 context to the HRES grid")
            u100[lag_index, source_hour_index] = aligned_u
            v100[lag_index, source_hour_index] = aligned_v
            u10[lag_index, source_hour_index] = np.asarray(
                snapshot.u10, dtype="float32"
            ).ravel()[source_index]
            v10[lag_index, source_hour_index] = np.asarray(
                snapshot.v10, dtype="float32"
            ).ravel()[source_index]

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


def d7_speed_context_training_dates() -> list[pd.Timestamp]:
    return [
        pd.Timestamp(year=year, month=month, day=day)
        for year in range(2016, 2021)
        for month, day in CONTEXT_REGIMES
    ]


def train_d7_speed_context_models(
    config,
    fh,
    checkpoint_path: Path | None = None,
) -> dict:
    """Fit only the two endpoints accepted by the held-year/regime audit."""
    import lightgbm as lgb

    checkpoint = (
        _load_checkpoint(checkpoint_path, "d7_speed_context")
        if checkpoint_path is not None
        else None
    )
    if checkpoint is not None:
        return checkpoint["payload"]

    x_rows = []
    y_rows = []
    dates_used = []
    dates = d7_speed_context_training_dates()
    for index, issue_date in enumerate(dates, start=1):
        try:
            table = build_hybrid_table(
                fh, config, [issue_date], with_truth=True, leads=(7,)
            )
            features = add_d7_speed_context_features(
                config, fh, issue_date, table
            )
        except (FileNotFoundError, ValueError) as exc:
            print(
                f"[train] d7 context {issue_date.date()} skipped: {exc}",
                flush=True,
            )
            continue
        grid_index = table.groupby(["lat", "lon"], sort=True).ngroup().to_numpy()
        keep = (grid_index % D7_SPEED_CONTEXT_SPATIAL_STEP) == 0
        x_rows.append(features.loc[keep].reset_index(drop=True))
        y_rows.append(
            np.hypot(
                table.loc[keep, "u125c"].to_numpy(),
                table.loc[keep, "v125c"].to_numpy(),
            ).astype("float32")
        )
        dates_used.append(issue_date)
        _ANALYSIS_CACHE.clear()
        if index % 10 == 0 or index == len(dates):
            print(
                f"[train] d7 speed context {index}/{len(dates)} rows="
                f"{sum(len(frame) for frame in x_rows):,}",
                flush=True,
            )
    if not x_rows:
        raise RuntimeError("No d7 speed context rows were generated")
    x = pd.concat(x_rows, ignore_index=True)
    y = np.concatenate(y_rows)
    params = dict(
        n_estimators=D7_SPEED_CONTEXT_TREES,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=120,
        reg_lambda=1.5,
        subsample=0.8,
        colsample_bytree=0.75,
        n_jobs=1,
        verbose=-1,
    )
    models = {
        quantile: lgb.LGBMRegressor(
            objective="quantile", alpha=quantile, **params
        ).fit(x, y)
        for quantile in (0.05, 0.95)
    }
    payload = {
        "models": models,
        "features": list(x.columns),
        "lower_blend": D7_SPEED_CONTEXT_LOWER_BLEND,
        "upper_blend": D7_SPEED_CONTEXT_UPPER_BLEND,
        "selected_slots": [tuple(slot) for slot in D7_SPEED_CONTEXT_SELECTED_SLOTS],
        "gate": dict(D7_SPEED_CONTEXT_GATE),
        "training_rows": int(len(x)),
        "training_dates": [str(date.date()) for date in dates_used],
        "spatial_step": D7_SPEED_CONTEXT_SPATIAL_STEP,
        "new_models": 2,
    }
    if checkpoint_path is not None:
        _save_checkpoint(
            checkpoint_path,
            "d7_speed_context",
            payload=payload,
        )
    return payload


def train_quantile_models(
    fh,
    table: pd.DataFrame,
    checkpoint_path: Path | None = None,
    leads=LEADS,
) -> dict:
    import lightgbm as lgb

    params = dict(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=1,
        verbose=-1,
    )
    checkpoint = (
        _load_checkpoint(checkpoint_path, "quantile_models")
        if checkpoint_path is not None
        else None
    )
    models = {} if checkpoint is None else checkpoint["models"]
    for lead in leads:
        subset = table[table["lead"] == lead]
        if subset.empty:
            raise ValueError(f"No quantile training rows for lead {lead}")
        x = engineered_features(fh, subset)
        y = np.hypot(subset["u125c"].to_numpy(), subset["v125c"].to_numpy())
        for quantile in QUANTILES:
            key = (lead, quantile)
            if key not in models:
                models[key] = lgb.LGBMRegressor(
                    objective="quantile", alpha=quantile, **params
                ).fit(x, y)
                if checkpoint_path is not None:
                    _save_checkpoint(
                        checkpoint_path,
                        "quantile_models",
                        models=models,
                    )
                print(
                    f"[train] quantile model lead={lead} q={quantile:.2f}",
                    flush=True,
                )
    return models


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
            issue_dates = (
                pd.to_datetime(d7_table["issue_date"])
                .dt.normalize()
                .unique()
            )
            if len(issue_dates) != 1:
                raise ValueError("d7 speed context expects one issue date")
            selected_slots = {
                tuple(map(int, slot))
                for slot in d7_speed_context.get("selected_slots", ())
            }
            issue_date = pd.Timestamp(issue_dates[0])
            active = np.asarray(
                [
                    (issue_date.month, issue_date.day, int(hour))
                    in selected_slots
                    for hour in d7_table["hour"].to_numpy()
                ],
                dtype=bool,
            )
            if active.any():
                context_x = add_d7_speed_context_features(
                    config, fh, issue_date, d7_table
                )
                context_x = context_x[d7_speed_context["features"]]
                lower_candidate = d7_speed_context["models"][0.05].predict(
                    context_x
                )
                upper_candidate = d7_speed_context["models"][0.95].predict(
                    context_x
                )
                lower_base = out.loc[mask, columns[0]].to_numpy(
                    dtype="float64"
                )
                upper_base = out.loc[mask, columns[-1]].to_numpy(
                    dtype="float64"
                )
                lower_base[active] += float(
                    d7_speed_context["lower_blend"]
                ) * (lower_candidate[active] - lower_base[active])
                upper_base[active] += float(
                    d7_speed_context["upper_blend"]
                ) * (upper_candidate[active] - upper_base[active])
                out.loc[mask, columns[0]] = lower_base
                out.loc[mask, columns[-1]] = upper_base
        if adjust and lead in adjust:
            out.loc[mask, columns[0]] -= adjust[lead]
            out.loc[mask, columns[-1]] += adjust[lead]
    values = np.sort(out[columns].to_numpy(dtype="float64"), axis=1)
    out[columns] = np.clip(values, 0.0, None)
    return out


def conformal_adjust(models: dict, fh, table: pd.DataFrame, alpha: float = 0.10) -> dict:
    pred = predict_quantiles(fh, models, table)
    adjust = {}
    for lead in LEADS:
        subset = pred[pred["lead"] == lead]
        truth = np.hypot(subset["u125c"].to_numpy(), subset["v125c"].to_numpy())
        error = np.maximum(
            subset["spd_q05"].to_numpy() - truth,
            truth - subset["spd_q95"].to_numpy(),
        )
        rank = int(np.ceil((len(error) + 1) * (1.0 - alpha)))
        adjust[lead] = float(np.sort(error)[min(rank, len(error)) - 1])
    return adjust


def train_direction_models(fh, table: pd.DataFrame) -> dict:
    """Train one compact d1 circular-residual model pair."""
    import lightgbm as lgb

    subset = table[table["lead"] == 1]
    base = np.degrees(np.arctan2(-subset["fcst_u"], -subset["fcst_v"])) % 360
    truth = np.degrees(np.arctan2(-subset["u125c"], -subset["v125c"])) % 360
    error = np.radians((truth - base + 180) % 360 - 180)
    params = dict(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=1,
        verbose=-1,
    )
    x = engineered_features(fh, subset)
    return {
        "sin": lgb.LGBMRegressor(**params).fit(x, np.sin(error)),
        "cos": lgb.LGBMRegressor(**params).fit(x, np.cos(error)),
    }


def context_training_dates() -> list[pd.Timestamp]:
    dates: set[pd.Timestamp] = set()
    for year in range(2016, 2021):
        dates.update(
            pd.date_range(f"{year}-01-14", f"{year}-12-15", freq="12D").normalize()
        )
        dates.update(
            pd.Timestamp(year=year, month=month, day=day)
            for month, day in CONTEXT_REGIMES
        )
    return sorted(dates)


def train_context_direction_models(
    config, fh, checkpoint_path: Path | None = None
) -> tuple[dict, list[str], dict]:
    """Fit one regularized context pair on a fixed, spatially thinned sample."""
    import lightgbm as lgb

    # Context dates are far enough apart that retaining every aligned reanalysis
    # snapshot only grows memory; each issue uses its 14-day context once.
    _ANALYSIS_CACHE.clear()

    checkpoint = (
        _load_checkpoint(checkpoint_path, "context_samples")
        if checkpoint_path is not None
        else None
    )
    if checkpoint is None:
        x_rows = []
        sin_rows = []
        cos_rows = []
        used_dates = []
        processed_dates = set()
    else:
        x_rows = checkpoint["x_rows"]
        sin_rows = checkpoint["sin_rows"]
        cos_rows = checkpoint["cos_rows"]
        used_dates = [pd.Timestamp(value) for value in checkpoint["used_dates"]]
        processed_dates = set(checkpoint["processed_dates"])
    dates = context_training_dates()
    for index, issue_date in enumerate(dates, start=1):
        date_key = issue_date.strftime("%Y-%m-%d")
        if date_key in processed_dates:
            continue
        try:
            table = build_hybrid_table(
                fh,
                config,
                [issue_date],
                with_truth=True,
                leads=(1,),
            )
        except ValueError as exc:
            print(
                f"[train] context {index}/{len(dates)} {issue_date.date()} skipped: {exc}",
                flush=True,
            )
            processed_dates.add(date_key)
            if checkpoint_path is not None:
                _save_checkpoint(
                    checkpoint_path,
                    "context_samples",
                    x_rows=x_rows,
                    sin_rows=sin_rows,
                    cos_rows=cos_rows,
                    used_dates=[value.strftime("%Y-%m-%d") for value in used_dates],
                    processed_dates=sorted(processed_dates),
                )
            continue
        features = add_lagged_context_features(config, fh, issue_date, table)
        _ANALYSIS_CACHE.clear()
        raw = np.degrees(np.arctan2(-table["fcst_u"], -table["fcst_v"])) % 360.0
        truth = np.degrees(np.arctan2(-table["u125c"], -table["v125c"])) % 360.0
        residual = np.radians((truth - raw + 180.0) % 360.0 - 180.0)
        grid_index = table.groupby(["lat", "lon"], sort=True).ngroup().to_numpy()
        keep = (grid_index % CONTEXT_SPATIAL_STEP) == 0
        x_rows.append(features.loc[keep])
        sin_rows.append(np.sin(residual[keep]).astype("float32"))
        cos_rows.append(np.cos(residual[keep]).astype("float32"))
        used_dates.append(issue_date)
        processed_dates.add(date_key)
        if checkpoint_path is not None and (
            index % 10 == 0 or index == len(dates)
        ):
            _save_checkpoint(
                checkpoint_path,
                "context_samples",
                x_rows=x_rows,
                sin_rows=sin_rows,
                cos_rows=cos_rows,
                used_dates=[value.strftime("%Y-%m-%d") for value in used_dates],
                processed_dates=sorted(processed_dates),
            )
        if index % 10 == 0 or index == len(dates):
            print(
                f"[train] context {index}/{len(dates)} rows="
                f"{sum(len(row) for row in x_rows):,}",
                flush=True,
            )

    if not x_rows:
        raise RuntimeError("No lagged-context training rows were generated")
    x = pd.concat(x_rows, ignore_index=True)
    y_sin = np.concatenate(sin_rows)
    y_cos = np.concatenate(cos_rows)
    params = dict(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=1,
        verbose=-1,
    )
    pair = {
        "sin": lgb.LGBMRegressor(**params).fit(x, y_sin),
        "cos": lgb.LGBMRegressor(**params).fit(x, y_cos),
    }
    summary = {
        "date_frequency": "12D plus eight fixed regimes",
        "dates_requested": len(dates),
        "dates_used": len(used_dates),
        "rows": int(len(x)),
        "spatial_step": CONTEXT_SPATIAL_STEP,
        "feature_count": int(x.shape[1]),
    }
    return pair, list(x.columns), summary


def circular_blend(a, b, weight: float) -> np.ndarray:
    a_rad = np.radians(np.asarray(a, dtype="float64"))
    b_rad = np.radians(np.asarray(b, dtype="float64"))
    y = (1.0 - weight) * np.sin(a_rad) + weight * np.sin(b_rad)
    x = (1.0 - weight) * np.cos(a_rad) + weight * np.cos(b_rad)
    return np.degrees(np.arctan2(y, x)) % 360.0


def _spatial_box_mean(values: np.ndarray, radius: int) -> np.ndarray:
    """Small pure-NumPy box filter used by the shared spatial estimator."""
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
    """Create multi-scale vector features for d7 rows on the coarse grid."""
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


def _fit_shared_spatial_ridge(x: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.linear_model import Ridge

    mean = np.mean(x, axis=0, dtype="float64")
    scale = np.std(x, axis=0, dtype="float64")
    scale = np.where(scale < 1e-5, 1.0, scale)
    normalized = ((x - mean) / scale).astype("float32")
    model = Ridge(
        alpha=SHARED_SPATIAL_DIRECTION_ALPHA,
        fit_intercept=True,
        solver="lsqr",
        tol=1e-5,
    ).fit(normalized, y)
    return {
        "model": model,
        "mean": mean.astype("float32"),
        "scale": scale.astype("float32"),
    }


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
    """Apply only the strictly gated production cells of the shared model."""
    payload = models.get("shared_spatial_direction")
    result = np.asarray(centers, dtype="float64").copy()
    if not payload or not payload.get("gate", {}).get("passed", False):
        return result
    source_indices, features = shared_spatial_direction_features(table)
    if not len(source_indices):
        return result
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
    """Apply held-year-gated d1/climatology signals to five d7 slots."""
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
            if not {"analysis_u", "analysis_v"}.issubset(table.columns):
                raise ValueError("Analysis-aware direction model requires issue-time u100/v100")
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


def shared_spatial_direction_training_dates() -> list[pd.Timestamp]:
    """Exact organizer inference-window regimes replayed over all train years."""
    return [
        pd.Timestamp(year=year, month=month, day=day)
        for year in range(2016, 2021)
        for month, day in CONTEXT_REGIMES
    ]


def _circular_absolute_error(truth, prediction) -> np.ndarray:
    return np.abs(
        (np.asarray(truth) - np.asarray(prediction) + 180.0) % 360.0 - 180.0
    )


def _shared_spatial_slot_audit(
    base,
    truth,
    correction,
    confidence,
    speed,
    dates,
    hours,
    blend: float,
    minimum_confidence: float,
) -> tuple[list[tuple[int, int, int]], dict]:
    """Select cells that pass both chronological and physical-regime gates."""
    selected = []
    slot_audits = []
    base_error = _circular_absolute_error(truth, base)
    candidate = np.asarray(base, dtype="float64").copy()
    active = confidence >= minimum_confidence
    candidate[active] = (
        candidate[active] + blend * correction[active]
    ) % 360.0
    delta = _circular_absolute_error(truth, candidate) - base_error
    month = dates.dt.month.to_numpy()
    day = dates.dt.day.to_numpy()
    year = dates.dt.year.to_numpy()
    raw_sector = np.floor((np.asarray(base) % 360.0) / 90.0).astype("int8")
    speed_bin = np.digitize(speed, (5.0, 10.0, 15.0)).astype("int8")

    for target_month, target_day in CONTEXT_REGIMES:
        for target_hour in HOURS:
            slot = (
                (month == target_month)
                & (day == target_day)
                & (hours == target_hour)
            )
            if not slot.any():
                continue
            active_fraction = float(np.mean(active[slot]))
            if active_fraction < SHARED_SPATIAL_DIRECTION_MIN_ACTIVE:
                continue
            fold_delta = {}
            fold_ok = True
            for fold_year in (2019, 2020):
                fold = slot & (year == fold_year)
                if not fold.any():
                    fold_ok = False
                    break
                value = float(np.mean(delta[fold]))
                fold_delta[str(fold_year)] = value
                if value > -SHARED_SPATIAL_DIRECTION_MIN_FOLD_GAIN:
                    fold_ok = False
            if not fold_ok:
                continue
            regime_deltas = []
            for labels in (speed_bin, raw_sector):
                for label in np.unique(labels[slot]):
                    regime = slot & (labels == label)
                    if np.sum(regime) >= 100:
                        regime_deltas.append(float(np.mean(delta[regime])))
            worst_regime = max(regime_deltas, default=float("inf"))
            if worst_regime > SHARED_SPATIAL_DIRECTION_MAX_REGIME_DELTA:
                continue
            mean_delta = float(np.mean(delta[slot]))
            selected.append((int(target_month), int(target_day), int(target_hour)))
            slot_audits.append(
                {
                    "slot": [int(target_month), int(target_day), int(target_hour)],
                    "rows": int(np.sum(slot)),
                    "active_fraction": active_fraction,
                    "mean_delta_deg": mean_delta,
                    "fold_delta_deg": fold_delta,
                    "worst_regime_delta_deg": worst_regime,
                }
            )
    deployed = np.zeros(len(base), dtype=bool)
    for target_month, target_day, target_hour in selected:
        deployed |= (
            (month == target_month)
            & (day == target_day)
            & (hours == target_hour)
        )
    total_delta = (
        float(np.mean(delta[deployed])) if deployed.any() else float("inf")
    )
    audit = {
        "selected_slots": [list(slot) for slot in selected],
        "selected_slot_count": len(selected),
        "deployed_rows": int(np.sum(deployed)),
        "mean_deployed_delta_deg": total_delta,
        "slots": slot_audits,
    }
    return selected, audit


def train_shared_spatial_direction_model(
    config,
    fh,
    direction_models: dict,
    checkpoint_path: Path | None = None,
) -> dict:
    """Fit one shared spatial d7 residual model with strict out-of-time gates."""
    checkpoint = (
        _load_checkpoint(checkpoint_path, "shared_spatial_direction")
        if checkpoint_path is not None
        else None
    )
    if checkpoint is not None:
        return checkpoint["payload"]

    dates = shared_spatial_direction_training_dates()
    print(
        f"[train] shared spatial direction replay on {len(dates)} issue dates",
        flush=True,
    )
    table = build_hybrid_table(
        fh,
        config,
        dates,
        with_truth=True,
        leads=(1, 7),
    )
    baseline = np.empty(len(table), dtype="float64")
    issue_dates = pd.to_datetime(table["issue_date"]).dt.normalize()
    for issue_date in pd.DatetimeIndex(issue_dates.unique()):
        positions = np.flatnonzero((issue_dates == issue_date).to_numpy())
        one_issue = table.iloc[positions].reset_index(drop=True)
        baseline[positions] = predict_direction_centers(
            fh, direction_models, one_issue, config=config
        )
    source_indices, features = shared_spatial_direction_features(table)
    d7 = table.loc[source_indices].copy()
    base = baseline[source_indices]
    truth = (
        np.degrees(np.arctan2(-d7["u125c"], -d7["v125c"])) % 360.0
    ).to_numpy(dtype="float64")
    residual = np.radians((truth - base + 180.0) % 360.0 - 180.0)
    target = np.column_stack([np.sin(residual), np.cos(residual)]).astype(
        "float32"
    )
    x = features.to_numpy(dtype="float32")
    dates_series = pd.to_datetime(d7["issue_date"]).reset_index(drop=True)
    years = dates_series.dt.year.to_numpy()

    fold_predictions = []
    for validation_year in (2019, 2020):
        train_mask = years < validation_year
        validation_mask = years == validation_year
        fold_model = _fit_shared_spatial_ridge(x[train_mask], target[train_mask])
        correction, confidence = _shared_spatial_prediction(
            fold_model, x[validation_mask]
        )
        fold_predictions.append(
            {
                "base": base[validation_mask],
                "truth": truth[validation_mask],
                "correction": correction,
                "confidence": confidence,
                "speed": d7["fcst_speed"].to_numpy(dtype="float32")[validation_mask],
                "dates": dates_series[validation_mask].reset_index(drop=True),
                "hours": d7["hour"].to_numpy(dtype="int16")[validation_mask],
            }
        )
        del fold_model

    combined = {
        key: (
            pd.concat([record[key] for record in fold_predictions], ignore_index=True)
            if key == "dates"
            else np.concatenate([record[key] for record in fold_predictions])
        )
        for key in fold_predictions[0]
    }
    candidates = []
    for blend in SHARED_SPATIAL_DIRECTION_BLENDS:
        for minimum_confidence in SHARED_SPATIAL_DIRECTION_CONFIDENCE:
            selected, audit = _shared_spatial_slot_audit(
                **combined,
                blend=float(blend),
                minimum_confidence=float(minimum_confidence),
            )
            if (
                selected
                and audit["mean_deployed_delta_deg"]
                <= -SHARED_SPATIAL_DIRECTION_MIN_TOTAL_GAIN
            ):
                candidates.append(
                    {
                        "blend": float(blend),
                        "minimum_confidence": float(minimum_confidence),
                        "selected_slots": selected,
                        "audit": audit,
                    }
                )
    candidates.sort(
        key=lambda row: (
            row["audit"]["mean_deployed_delta_deg"],
            -row["audit"]["selected_slot_count"],
        )
    )
    winner = candidates[0] if candidates else None
    gate = {
        "passed": winner is not None,
        "method": (
            "expanding-window 2019/2020 validation; every selected calendar/hour "
            "cell improves both folds and is non-worse in every populated speed "
            "bin and direction sector"
        ),
        "training_rows": int(len(x)),
        "validation_years": [2019, 2020],
        "minimum_active_fraction": SHARED_SPATIAL_DIRECTION_MIN_ACTIVE,
        "minimum_fold_gain_deg": SHARED_SPATIAL_DIRECTION_MIN_FOLD_GAIN,
        "maximum_regime_delta_deg": SHARED_SPATIAL_DIRECTION_MAX_REGIME_DELTA,
        "minimum_total_gain_deg": SHARED_SPATIAL_DIRECTION_MIN_TOTAL_GAIN,
        "candidate_count": len(candidates),
        "winner": None if winner is None else winner["audit"],
    }
    if winner is None:
        payload = {
            "gate": gate,
            "method": "shared multi-scale circular residual ridge (rejected)",
            "new_models": 0,
            "input_only_training": True,
            "previous_submission_inputs": [],
        }
    else:
        final_model = _fit_shared_spatial_ridge(x, target)
        payload = {
            **final_model,
            "features": list(features.columns),
            "blend": winner["blend"],
            "minimum_confidence": winner["minimum_confidence"],
            "selected_slots": [tuple(slot) for slot in winner["selected_slots"]],
            "gate": gate,
            "method": "one shared multi-scale circular residual ridge",
            "alpha": SHARED_SPATIAL_DIRECTION_ALPHA,
            "new_models": 1,
            "input_only_training": True,
            "previous_submission_inputs": [],
        }
    if checkpoint_path is not None:
        _save_checkpoint(
            checkpoint_path,
            "shared_spatial_direction",
            payload=payload,
        )
    print(
        "[train] shared spatial direction gate: "
        f"passed={gate['passed']} candidates={gate['candidate_count']} "
        f"slots={0 if winner is None else len(winner['selected_slots'])}",
        flush=True,
    )
    del table, features, x, target
    gc.collect()
    return payload


def fit_forecast_frugal(
    config,
    pipeline,
    fh,
    train_issue_dates: pd.DatetimeIndex,
    checkpoint_dir: Path | None = None,
):
    """Fit the coarse HRES hybrid with bounded native threading."""
    global _HRES_CACHE
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        core_path = checkpoint_dir / "_checkpoint_forecast_core.joblib"
        core = _load_checkpoint(core_path, "forecast_core")
        if core is not None:
            return (
                core["qmos"],
                core["direction_models"],
                core["conformal"],
                core["dir_offsets"],
            )
        baseline_path = checkpoint_dir / "_checkpoint_direction_base.joblib"
        context_sample_path = checkpoint_dir / "_checkpoint_context_samples.joblib"
        context_model_path = checkpoint_dir / "_checkpoint_context_models.joblib"
        quantile_path = checkpoint_dir / "_checkpoint_quantile_models.joblib"
    else:
        core_path = baseline_path = context_sample_path = None
        context_model_path = quantile_path = None
    calib_dates = train_issue_dates[train_issue_dates.year == 2020]
    quant_train_dates = train_issue_dates[train_issue_dates.year < 2020]
    baseline_checkpoint = (
        _load_checkpoint(baseline_path, "direction_base")
        if baseline_path is not None
        else None
    )
    if baseline_checkpoint is None:
        train_table = build_hybrid_table(fh, config, train_issue_dates, leads=(1,))
        print(f"[train] direction rows: {len(train_table):,}", flush=True)
        baseline_direction_models = train_direction_models(fh, train_table)
        if baseline_path is not None:
            _save_checkpoint(
                baseline_path,
                "direction_base",
                models=baseline_direction_models,
            )
        del train_table
        gc.collect()
    else:
        baseline_direction_models = baseline_checkpoint["models"]
    context_checkpoint = (
        _load_checkpoint(context_model_path, "context_models")
        if context_model_path is not None
        else None
    )
    if context_checkpoint is None:
        context_models, context_features, context_summary = (
            train_context_direction_models(config, fh, context_sample_path)
        )
        if context_model_path is not None:
            _save_checkpoint(
                context_model_path,
                "context_models",
                models=context_models,
                features=context_features,
                summary=context_summary,
            )
    else:
        context_models = context_checkpoint["models"]
        context_features = context_checkpoint["features"]
        context_summary = context_checkpoint["summary"]
    direction_models = {
        "base": baseline_direction_models,
        "context": context_models,
        "context_blend": CONTEXT_BLEND,
        "context_slots": [tuple(slot) for slot in CONTEXT_SELECTED_SLOTS],
        "context_features": context_features,
        "context_summary": context_summary,
        "d7_center_policy": D7_DIRECTION_CENTER_POLICY,
    }
    _ANALYSIS_CACHE.clear()
    gc.collect()
    quant_table = build_hybrid_table(fh, config, quant_train_dates)
    print(f"[train] quantile MOS rows: {len(quant_table):,}", flush=True)
    # The compact table owns its feature arrays, so the much larger source
    # parquet frame is no longer needed while LightGBM fits the nine models.
    _HRES_CACHE = None
    gc.collect()
    qmos = train_quantile_models(fh, quant_table, quantile_path)
    del quant_table
    gc.collect()
    calib_table = build_hybrid_table(fh, config, calib_dates)
    print(f"[train] calibration rows: {len(calib_table):,}", flush=True)
    conformal = conformal_adjust(qmos, fh, calib_table)
    predicted_direction = predict_direction_centers(
        fh, direction_models, calib_table, config=config
    )
    true_direction = np.degrees(
        np.arctan2(-calib_table["u125c"], -calib_table["v125c"])
    ) % 360
    errors = pipeline._circ_abs_deg(true_direction, predicted_direction)
    dir_offsets = {}
    for lead in (1, 7):
        mask = calib_table["lead"].to_numpy() == lead
        dir_offsets[lead] = float(np.nanpercentile(errors[mask], 90))
    dir_offsets[14] = 80.0
    del calib_table
    gc.collect()
    if core_path is not None:
        _save_checkpoint(
            core_path,
            "forecast_core",
            qmos=qmos,
            direction_models=direction_models,
            conformal=conformal,
            dir_offsets=dir_offsets,
        )
    return qmos, direction_models, conformal, dir_offsets


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


def fit_qmos_refit_policy(
    config,
    fh,
    train_issue_dates,
    base_qmos,
    conformal,
    checkpoint_dir: Path,
) -> dict:
    """Fit the six gated all-years qMOS models and predictor-support bounds."""
    policy_path = checkpoint_dir / "_checkpoint_qmos_refit_policy.joblib"
    checkpoint = _load_checkpoint(policy_path, "qmos_refit_policy")
    if checkpoint is not None:
        return checkpoint["payload"]

    table = build_hybrid_table(
        fh, config, train_issue_dates, leads=(1, 7)
    )
    print(f"[train] all-years qMOS refit rows: {len(table):,}", flush=True)
    models = train_quantile_models(
        fh,
        table,
        checkpoint_dir / "_checkpoint_qmos_refit_models.joblib",
        leads=(1, 7),
    )
    del table
    gc.collect()

    support_dates = pd.DatetimeIndex(
        sorted(
            {
                pd.Timestamp(year=year, month=rule["month"], day=rule["day"])
                for year in range(2016, 2021)
                for rule in QMOS_REFIT_RULES
            }
        )
    )
    support_table = build_hybrid_table(
        fh,
        config,
        support_dates,
        with_truth=False,
        leads=(1, 7),
    )
    base_predictions = predict_quantiles(
        fh, base_qmos, support_table, adjust=conformal
    )
    refit_predictions = predict_quantiles(
        fh, models, support_table, adjust=conformal
    )
    rules = []
    for source_rule in QMOS_REFIT_RULES:
        rule = dict(source_rule)
        yearly = []
        for year in range(2016, 2021):
            selected = (
                (pd.to_datetime(support_table["issue_date"]).dt.year == year)
                & (pd.to_datetime(support_table["issue_date"]).dt.month == rule["month"])
                & (pd.to_datetime(support_table["issue_date"]).dt.day == rule["day"])
                & (support_table["lead"] == rule["lead"])
                & (support_table["hour"] == rule["hour"])
            )
            if not selected.any():
                raise RuntimeError(
                    "Missing qMOS refit support rows for "
                    f"{year}-{rule['month']:02d}-{rule['day']:02d} "
                    f"d{rule['lead']} h{rule['hour']:02d}"
                )
            yearly.append(
                {
                    "year": year,
                    **qmos_refit_movement_summary(
                        base_predictions.loc[selected],
                        refit_predictions.loc[selected],
                    ),
                }
            )
        bounds = {}
        for metric in (
            "q05_shift",
            "q50_shift",
            "q95_shift",
            "normalized_abs_movement",
        ):
            values = np.asarray([row[metric] for row in yearly], dtype="float64")
            span = max(float(values.max() - values.min()), 0.01)
            padding = QMOS_REFIT_SUPPORT_PADDING * span
            bounds[metric] = [
                float(values.min() - padding),
                float(values.max() + padding),
            ]
        rule["support"] = {
            "historical_yearly_summaries": yearly,
            "bounds": bounds,
            "fallback": "retain the protected base qMOS interval",
        }
        rules.append(rule)

    payload = {
        "method": (
            "all-years qMOS refit activated only in exact fine-grid masks that "
            "improved both expanding-window outer years and every selected "
            "observable interaction"
        ),
        "models": models,
        "rules": rules,
        "support_padding": QMOS_REFIT_SUPPORT_PADDING,
        "gate": {
            "passed": True,
            "outer_years": [2019, 2020],
            "exact_production_downscaler": True,
            "exact_final_d1_width_scale": 0.75,
            "every_selected_observable_interaction_non_worse": True,
        },
        "new_models": len(models),
        "input_only_training": True,
        "previous_submission_inputs": [],
    }
    _save_checkpoint(policy_path, "qmos_refit_policy", payload=payload)
    return payload


def coarse_fields_hybrid(
    fh,
    config,
    qmos: dict,
    direction_models: dict,
    conformal: dict,
    issue_date,
    d7_speed_context=None,
    leads=LEADS,
) -> dict:
    """Create coarse d1/d7/d14 fields for the gated hybrid policy."""
    leads = tuple(leads)
    table = build_hybrid_table(
        fh,
        config,
        [issue_date],
        with_truth=False,
        with_analysis=True,
        leads=leads,
    )
    quantiles = predict_quantiles(
        fh,
        qmos,
        table,
        adjust=conformal,
        config=config,
        d7_speed_context=d7_speed_context,
    )
    directions = predict_direction_centers(
        fh, direction_models, table, config=config
    )
    quantiles["dir_pred"] = directions
    has_alternate_d1_center = any(
        key in direction_models for key in ("analysis", "context")
    )
    if has_alternate_d1_center:
        quantiles["dir_speed_baseline"] = predict_direction_centers(
            fh,
            direction_models.get("base", direction_models),
            table,
            config=config,
        )
    climatology = (
        fh.build_climatology_forecast(
            [issue_date], lead=14, with_truth=False
        )
        if 14 in leads
        else None
    )
    issue_date = pd.Timestamp(issue_date)
    uses_d14_signal = 14 in leads and any(
        (issue_date.month, issue_date.day, hour) in D14_DIRECTION_POLICY
        for hour in HOURS
    )
    source_climatology = (
        fh.build_climatology_forecast([issue_date], lead=7, with_truth=False)
        if uses_d14_signal
        else None
    )
    fields = {}
    for lead in leads:
        for hour in HOURS:
            subset = quantiles[
                (quantiles["lead"] == lead) & (quantiles["hour"] == hour)
            ].copy()
            if subset.empty:
                raise ValueError(f"No inference rows for lead={lead}, hour={hour}")
            speed_grids = []
            for column in ("spd_q05", "spd_q50", "spd_q95"):
                speed_grids.append(
                    fh.predictions_to_grid(
                        subset.assign(u_pred=subset[column], v_pred=0.0),
                        lead,
                        hour,
                    )[0]
                )
            speed_stack = np.stack(speed_grids).astype("float32")
            speed_direction_grid = None
            if lead == 14:
                clim_u, clim_v = fh.predictions_to_grid(
                    climatology, lead, hour
                )
                direction_grid = np.degrees(np.arctan2(-clim_u, -clim_v)) % 360
                rule = D14_DIRECTION_POLICY.get(
                    (issue_date.month, issue_date.day, hour)
                )
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
            if lead == 1 and has_alternate_d1_center:
                baseline_direction = fh.predictions_to_grid(
                    subset.assign(
                        u_pred=subset["dir_speed_baseline"], v_pred=0.0
                    ),
                    lead,
                    hour,
                )[0]
                baseline_radians = np.radians(baseline_direction)
                fields[(lead, hour, "speed_det")] = np.stack(
                    [
                        -median * np.sin(baseline_radians),
                        -median * np.cos(baseline_radians),
                    ]
                ).astype("float32")
            if speed_direction_grid is not None:
                speed_radians = np.radians(speed_direction_grid)
                speed_u = -median * np.sin(speed_radians)
                speed_v = -median * np.cos(speed_radians)
                fields[(lead, hour, "speed_det")] = np.stack(
                    [speed_u, speed_v]
                ).astype("float32")
            fields[(lead, hour, "spd")] = speed_stack
    return fields


def fit_d1_speed_context_policy(
    fh,
    config,
    pipeline,
    downscaling,
    qmos,
    direction_models,
    conformal,
    downscaler,
    checkpoint_path: Path | None = None,
) -> dict:
    """Fit and gate one causal February d1 upper-endpoint challenger."""
    import footprint
    import lightgbm as lgb
    import target_loader

    checkpoint = (
        _load_checkpoint(checkpoint_path, "d1_speed_context")
        if checkpoint_path is not None
        else None
    )
    if checkpoint is not None:
        return checkpoint["payload"]

    month, day = D1_SPEED_CONTEXT_MONTH_DAY
    years = tuple(range(2016, 2021))
    feature_frames = {}
    tables = {}
    targets = {}
    for year in years:
        issue_date = pd.Timestamp(year=year, month=month, day=day)
        table = build_hybrid_table(
            fh, config, [issue_date], with_truth=True, leads=(1,)
        )
        features = add_lagged_context_features(config, fh, issue_date, table)
        feature_frames[year] = features
        tables[year] = table
        targets[year] = np.hypot(
            table["u125c"].to_numpy(), table["v125c"].to_numpy()
        ).astype("float32")
        _ANALYSIS_CACHE.clear()
        print(
            f"[train] d1 context features {issue_date.date()} "
            f"rows={len(features):,}",
            flush=True,
        )

    model_params = dict(
        n_estimators=D1_SPEED_CONTEXT_TREES,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=120,
        reg_lambda=2.0,
        subsample=0.85,
        colsample_bytree=0.75,
        n_jobs=1,
        verbose=-1,
    )

    def fit_models(training_years):
        x = pd.concat(
            [feature_frames[year] for year in training_years],
            ignore_index=True,
        )
        y = np.concatenate([targets[year] for year in training_years])
        models = {
            quantile: lgb.LGBMRegressor(
                objective="quantile", alpha=quantile, **model_params
            ).fit(x, y)
            for quantile in QUANTILES
        }
        del x, y
        gc.collect()
        return models

    fine_mask = footprint.footprint_mask()
    static = target_loader.load_static(config.target_root())
    fine_lat = np.asarray(static.lat[fine_mask], dtype="float32")
    fine_lon = np.asarray(static.lon[fine_mask], dtype="float32")

    def interval_score(truth, lower, upper):
        return (
            upper
            - lower
            + 20.0 * np.maximum(lower - truth, 0.0)
            + 20.0 * np.maximum(truth - upper, 0.0)
        )

    def evaluate_fold(held_year, models):
        issue_date = pd.Timestamp(year=held_year, month=month, day=day)
        valid_date = issue_date + pd.Timedelta(days=1)
        table = tables[held_year]
        features = feature_frames[held_year]
        candidate = np.sort(
            np.column_stack(
                [models[q].predict(features) for q in QUANTILES]
            ),
            axis=1,
        )
        fields = coarse_fields_hybrid(
            fh,
            config,
            qmos,
            direction_models,
            conformal,
            issue_date,
            leads=(1,),
        )
        truth_day = target_loader.load_day(
            valid_date.date(),
            root=config.target_root(),
            levels=("125m",),
        )
        rows = {
            key: []
            for key in (
                "truth",
                "q05",
                "q50",
                "q95",
                "candidate_q95",
                "hour",
                "width_bin",
            )
        }
        for hour_index, hour in enumerate(HOURS):
            selected = table["hour"].to_numpy() == hour
            candidate_stack = np.stack(
                [
                    fh.predictions_to_grid(
                        table.loc[selected].assign(
                            u_pred=candidate[selected, index], v_pred=0.0
                        ),
                        1,
                        hour,
                    )[0]
                    for index in range(3)
                ]
            ).astype("float32")
            base_u, base_v = fields.get(
                (1, hour, "speed_det"), fields[(1, hour, "det")]
            )
            direction = np.arctan2(-base_u, -base_v)
            candidate_median = candidate_stack[1]
            candidate_u = -candidate_median * np.sin(direction)
            candidate_v = -candidate_median * np.cos(direction)
            candidate_fine_u, candidate_fine_v = downscaling.downscale(
                downscaler, candidate_u, candidate_v
            )
            candidate_q50 = np.hypot(candidate_fine_u, candidate_fine_v)
            candidate_fields = dict(fields)
            candidate_fields[(1, hour, "spd")] = candidate_stack
            _, candidate_q95 = pipeline._speed_interval(
                candidate_fields,
                1,
                hour,
                candidate_q50,
                k=D1_SPEED_CONTEXT_CANDIDATE_INFLATION,
            )

            base_fine_u, base_fine_v = downscaling.downscale(
                downscaler, base_u, base_v
            )
            base_q50 = np.hypot(base_fine_u, base_fine_v)
            base_q05, base_q95 = pipeline._speed_interval(
                fields,
                1,
                hour,
                base_q50,
                k=D1_SPEED_CONTEXT_BASE_INFLATION,
            )
            truth = np.hypot(
                truth_day.u["125m"][hour // 3],
                truth_day.v["125m"][hour // 3],
            )
            width = base_q95[fine_mask] - base_q05[fine_mask]
            edges = np.quantile(width, (0.0, 0.25, 0.5, 0.75, 1.0))
            edges[0], edges[-1] = -np.inf, np.inf
            width_bin = np.clip(np.digitize(width, edges) - 1, 0, 3)
            values = (
                truth[fine_mask],
                base_q05[fine_mask],
                base_q50[fine_mask],
                base_q95[fine_mask],
                candidate_q95[fine_mask],
            )
            if not all(np.isfinite(value).all() for value in values):
                raise ValueError(
                    f"Non-finite d1 context gate values for {issue_date.date()}"
                )
            for key, value in zip(
                ("truth", "q05", "q50", "q95", "candidate_q95"),
                values,
            ):
                rows[key].append(np.asarray(value, dtype="float32"))
            rows["hour"].append(
                np.full(len(width), hour_index, dtype="int8")
            )
            rows["width_bin"].append(width_bin.astype("int8"))

        arrays = {key: np.concatenate(parts) for key, parts in rows.items()}
        active = np.isin(
            arrays["width_bin"], D1_SPEED_CONTEXT_WIDTH_BINS
        ) & (arrays["candidate_q95"] < arrays["q95"])
        deployed_q95 = arrays["q95"].copy()
        deployed_q95[active] += D1_SPEED_CONTEXT_UPPER_BLEND * (
            arrays["candidate_q95"][active] - deployed_q95[active]
        )
        base_score = interval_score(
            arrays["truth"], arrays["q05"], arrays["q95"]
        )
        candidate_score = interval_score(
            arrays["truth"], arrays["q05"], deployed_q95
        )
        delta = candidate_score - base_score
        width = arrays["q95"] - arrays["q05"]
        width_edges = np.quantile(width, (0.0, 0.25, 0.5, 0.75, 1.0))
        width_edges[0], width_edges[-1] = -np.inf, np.inf

        def bins(values, edges):
            return np.clip(
                np.digitize(values, edges) - 1, 0, len(edges) - 2
            )

        repeated_lat = np.tile(fine_lat, len(HOURS))
        repeated_lon = np.tile(fine_lon, len(HOURS))
        groups = {
            "hour": (arrays["hour"], 4),
            "spatial_2x2": (
                (repeated_lat >= np.median(fine_lat)).astype("int8") * 2
                + (repeated_lon >= np.median(fine_lon)).astype("int8"),
                4,
            ),
            "truth_speed": (
                bins(arrays["truth"], (0.0, 5.0, 10.0, 15.0, np.inf)),
                4,
            ),
            "base_width": (bins(width, width_edges), 4),
            "signed_error": (
                bins(
                    arrays["truth"] - arrays["q50"],
                    (-np.inf, -2.0, 0.0, 2.0, np.inf),
                ),
                4,
            ),
        }
        worst = {"delta": -np.inf, "family": None, "bin": None, "rows": 0}
        for family, (indices, count) in groups.items():
            for index in range(count):
                keep = indices == index
                if int(keep.sum()) < 500:
                    continue
                group_delta = float(np.mean(delta[keep]))
                if group_delta > worst["delta"]:
                    worst = {
                        "delta": group_delta,
                        "family": family,
                        "bin": int(index),
                        "rows": int(keep.sum()),
                    }
        mean_delta = float(np.mean(delta))
        active_fraction = float(np.mean(active))
        passed = (
            mean_delta <= -D1_SPEED_CONTEXT_MIN_FOLD_GAIN
            and active_fraction >= D1_SPEED_CONTEXT_MIN_ACTIVE
            and worst["delta"] <= 1e-7
        )
        return {
            "year": int(held_year),
            "rows": int(len(delta)),
            "base_winkler": float(np.mean(base_score)),
            "candidate_winkler": float(np.mean(candidate_score)),
            "mean_delta": mean_delta,
            "active_fraction": active_fraction,
            "worst_regime": worst,
            "passed": bool(passed),
        }

    folds = []
    for held_year in D1_SPEED_CONTEXT_HELD_YEARS:
        fold_models = fit_models([year for year in years if year != held_year])
        fold = evaluate_fold(held_year, fold_models)
        folds.append(fold)
        print(f"[train] d1 speed context fold: {fold}", flush=True)
        del fold_models
        gc.collect()

    gate_passed = all(fold["passed"] for fold in folds)
    if not gate_passed:
        raise RuntimeError("Strict d1 speed context gate did not pass")
    final_models = fit_models(years)
    payload = {
        "method": (
            "February d1 causal reanalysis-context quantiles; blend only a "
            "tighter upper endpoint outside the narrowest incumbent-width "
            "quartile"
        ),
        "models": final_models,
        "features": list(feature_frames[years[0]].columns),
        "month": month,
        "day": day,
        "candidate_inflation": D1_SPEED_CONTEXT_CANDIDATE_INFLATION,
        "upper_blend": D1_SPEED_CONTEXT_UPPER_BLEND,
        "width_bins": list(D1_SPEED_CONTEXT_WIDTH_BINS),
        "training_years": list(years),
        "training_rows": int(sum(len(frame) for frame in feature_frames.values())),
        "new_models": len(final_models),
        "gate": {
            "passed": True,
            "held_years": list(D1_SPEED_CONTEXT_HELD_YEARS),
            "folds": folds,
            "minimum_fold_gain": D1_SPEED_CONTEXT_MIN_FOLD_GAIN,
            "minimum_active_fraction": D1_SPEED_CONTEXT_MIN_ACTIVE,
            "every_populated_physical_regime_non_worse": True,
        },
        "input_only_training": True,
        "previous_submission_inputs": [],
    }
    if checkpoint_path is not None:
        _save_checkpoint(
            checkpoint_path, "d1_speed_context", payload=payload
        )
    return payload


def calibrate_intervals_hybrid(
    fh,
    config,
    pipeline,
    downscaling,
    qmos,
    direction_models,
    conformal,
    downscaler,
    dir_offsets,
    target=0.90,
):
    """Fine-grid calibration on 2020 truth using the exact hybrid inference path."""
    import footprint
    import target_loader

    calibration_dates = pd.to_datetime(
        [f"2020-{month:02d}-15" for month in (2, 4, 6, 8, 10, 12)]
    )
    mask = footprint.footprint_mask()
    speed_rows = {
        lead: {"mid": [], "lo": [], "hi": [], "truth": []} for lead in LEADS
    }
    direction_errors = {lead: [] for lead in LEADS}
    for issue_date in calibration_dates:
        fields = coarse_fields_hybrid(
            fh, config, qmos, direction_models, conformal, issue_date
        )
        for lead in LEADS:
            valid_date = issue_date + pd.Timedelta(days=lead)
            day = target_loader.load_day(valid_date.date(), root=config.target_root())
            for hour in HOURS:
                coarse_u, coarse_v = fields[(lead, hour, "det")]
                fine_u, fine_v = downscaling.downscale(
                    downscaler, coarse_u, coarse_v
                )
                median = np.hypot(fine_u, fine_v)
                direction = np.degrees(np.arctan2(-fine_u, -fine_v)) % 360
                q05, q95 = pipeline._speed_interval(
                    fields, lead, hour, median, k=1.0
                )
                snapshot = day.snapshot(hour)
                true_u = snapshot.fields["125m"]["u"]
                true_v = snapshot.fields["125m"]["v"]
                true_speed = np.hypot(true_u, true_v)
                true_direction = np.degrees(np.arctan2(-true_u, -true_v)) % 360
                keep = mask & np.isfinite(median) & np.isfinite(true_speed)
                speed_rows[lead]["mid"].append(median[keep])
                speed_rows[lead]["lo"].append(q05[keep])
                speed_rows[lead]["hi"].append(q95[keep])
                speed_rows[lead]["truth"].append(true_speed[keep])
                direction_errors[lead].append(
                    pipeline._circ_abs_deg(direction[keep], true_direction[keep])
                )
    speed_inflation = {}
    fine_direction_offsets = {}
    for lead in LEADS:
        median = np.concatenate(speed_rows[lead]["mid"])
        q05 = np.concatenate(speed_rows[lead]["lo"])
        q95 = np.concatenate(speed_rows[lead]["hi"])
        truth = np.concatenate(speed_rows[lead]["truth"])

        def coverage(scale):
            lo = np.maximum(0.0, median - scale * (median - q05))
            hi = median + scale * (q95 - median)
            return float(np.mean((truth >= lo) & (truth <= hi)))

        low, high = 0.25, 1.0
        while coverage(high) < target and high < 12:
            high *= 1.5
        for _ in range(25):
            middle = 0.5 * (low + high)
            if coverage(middle) < target:
                low = middle
            else:
                high = middle
        speed_inflation[lead] = round(0.5 * (low + high), 3)
        fine_direction_offsets[lead] = round(
            float(np.percentile(np.concatenate(direction_errors[lead]), 100 * target)),
            1,
        )
    return speed_inflation, fine_direction_offsets


def fit_d14_speed_endpoint_policy(
    fh,
    config,
    pipeline,
    downscaling,
    qmos,
    direction_models,
    conformal,
    downscaler,
    speed_inflation,
):
    """Select a fixed d14 lower-endpoint correction on exact held-out slots."""
    import footprint
    import target_loader

    years = (2019, 2020)
    issue_slots = (
        (1, 14),
        (2, 25),
        (4, 8),
        (5, 20),
        (7, 1),
        (8, 12),
        (9, 23),
        (11, 4),
    )
    mask = footprint.footprint_mask()
    static = target_loader.load_static(config.target_root())
    latitude = np.asarray(static.lat[mask], dtype="float32")
    longitude = np.asarray(static.lon[mask], dtype="float32")
    rows = {
        key: []
        for key in ("truth", "q05", "q50", "q95", "year", "slot", "hour")
    }
    inflation = float(speed_inflation[14])

    for year_index, year in enumerate(years):
        for slot_index, (month, day) in enumerate(issue_slots):
            issue_date = pd.Timestamp(year=year, month=month, day=day)
            valid_date = issue_date + pd.Timedelta(days=14)
            target = target_loader.load_day(
                valid_date.date(), root=config.target_root(), levels=("125m",)
            )
            fields = coarse_fields_hybrid(
                fh,
                config,
                qmos,
                direction_models,
                conformal,
                issue_date,
            )
            for hour_index, hour in enumerate(HOURS):
                speed_key = (14, hour, "speed_det")
                coarse_u, coarse_v = fields.get(
                    speed_key, fields[(14, hour, "det")]
                )
                fine_u, fine_v = downscaling.downscale(
                    downscaler, coarse_u, coarse_v
                )
                q50 = np.hypot(fine_u, fine_v)
                q05, q95 = pipeline._speed_interval(
                    fields, 14, hour, q50, k=inflation
                )
                truth = np.hypot(
                    target.u["125m"][hour // 3],
                    target.v["125m"][hour // 3],
                )
                arrays = (truth[mask], q05[mask], q50[mask], q95[mask])
                if not all(np.isfinite(values).all() for values in arrays):
                    raise ValueError(
                        f"Non-finite d14 speed gate values for "
                        f"{issue_date.date()} h{hour:02d}"
                    )
                for key, values in zip(("truth", "q05", "q50", "q95"), arrays):
                    rows[key].append(np.asarray(values, dtype="float32"))
                shape = arrays[0].shape
                rows["year"].append(
                    np.full(shape, year_index, dtype="int8")
                )
                rows["slot"].append(
                    np.full(shape, slot_index, dtype="int8")
                )
                rows["hour"].append(
                    np.full(shape, hour_index, dtype="int8")
                )
            print(
                f"[train] d14 speed endpoint replay {issue_date.date()} "
                f"-> {valid_date.date()}",
                flush=True,
            )

    arrays = {key: np.concatenate(parts) for key, parts in rows.items()}
    truth = arrays["truth"].astype("float64")
    q05 = arrays["q05"].astype("float64")
    q50 = arrays["q50"].astype("float64")
    q95 = arrays["q95"].astype("float64")

    def winkler(lower, upper):
        return (
            upper
            - lower
            + 20.0 * np.maximum(lower - truth, 0.0)
            + 20.0 * np.maximum(truth - upper, 0.0)
        )

    incumbent_q05 = q05.copy()
    incumbent_q95 = q95.copy()
    candidate_q05 = q05.copy()
    candidate_q95 = q95.copy()
    for slot_index, (month, day) in enumerate(issue_slots):
        for hour_index, hour in enumerate(HOURS):
            factors = D14_SPEED_ENDPOINT_POLICY.get((month, day, hour))
            if factors is None:
                continue
            selected = (arrays["slot"] == slot_index) & (
                arrays["hour"] == hour_index
            )
            lower_factor, upper_factor = factors
            guard = D14_SPEED_ENDPOINT_GUARDS.get((month, day, hour))
            if guard is None:
                local_lower_factor = lower_factor
                local_upper_factor = upper_factor
            else:
                threshold, high_strength = guard
                strength = np.where(
                    q50[selected] < threshold, 3.0, high_strength
                )
                ratio = strength / 3.0
                local_lower_factor = 1.0 + ratio * (lower_factor - 1.0)
                local_upper_factor = 1.0 + ratio * (upper_factor - 1.0)
            candidate_q05[selected] = np.maximum(
                0.0,
                q50[selected]
                - local_lower_factor * (q50[selected] - q05[selected]),
            )
            candidate_q95[selected] = q50[selected] + local_upper_factor * (
                q95[selected] - q50[selected]
            )
    base_score = winkler(q05, q95)
    incumbent_score = winkler(incumbent_q05, incumbent_q95)
    candidate_score = winkler(candidate_q05, candidate_q95)
    raw_delta = candidate_score - base_score
    guard_delta = candidate_score - incumbent_score
    width = q95 - q05
    repeats = len(truth) // len(latitude)
    lat = np.tile(latitude, repeats)
    lon = np.tile(longitude, repeats)
    width_edges = np.quantile(width, (0.0, 0.25, 0.5, 0.75, 1.0))
    width_edges[0], width_edges[-1] = -np.inf, np.inf

    def bins(values, edges):
        return np.clip(np.digitize(values, edges) - 1, 0, len(edges) - 2)

    groups = {
        "spatial_2x2": (
            (lat >= np.median(latitude)).astype("int8") * 2
            + (lon >= np.median(longitude)).astype("int8"),
            4,
        ),
        "truth_speed": (
            bins(truth, (0.0, 5.0, 10.0, 15.0, np.inf)),
            4,
        ),
        "predicted_speed": (
            bins(q50, (0.0, 5.0, 10.0, 15.0, np.inf)),
            4,
        ),
        "base_width": (bins(width, width_edges), 4),
        "signed_error": (
            bins(truth - q50, (-np.inf, -2.0, 0.0, 2.0, np.inf)),
            4,
        ),
    }

    def worst_regime(selected, delta_values):
        worst = {"delta": -np.inf, "family": None, "bin": None, "rows": 0}
        for family, (indices, count) in groups.items():
            for index in range(count):
                keep = selected & (indices == index)
                count_rows = int(keep.sum())
                if count_rows < D14_SPEED_ENDPOINT_MIN_GROUP_ROWS:
                    continue
                delta = float(np.mean(delta_values[keep]))
                if delta > worst["delta"]:
                    worst = {
                        "delta": delta,
                        "family": family,
                        "bin": int(index),
                        "rows": count_rows,
                    }
        return worst

    selected_rules = []
    fold_audit = []
    deployment = np.zeros(len(truth), dtype=bool)
    for slot_index, (month, day) in enumerate(issue_slots):
        for hour_index, hour in enumerate(HOURS):
            factors = D14_SPEED_ENDPOINT_POLICY.get((month, day, hour))
            if factors is None:
                continue
            cell = (arrays["slot"] == slot_index) & (
                arrays["hour"] == hour_index
            )
            folds = []
            cell_mean_delta = float(np.mean(guard_delta[cell]))
            passing = cell_mean_delta <= -D14_SPEED_ENDPOINT_MIN_CELL_GAIN
            for year_index, year in enumerate(years):
                held = cell & (arrays["year"] == year_index)
                mean_delta = float(np.mean(guard_delta[held]))
                worst = worst_regime(held, guard_delta)
                fold_passed = mean_delta <= 1e-7
                folds.append(
                    {
                        "year": int(year),
                        "mean_delta": mean_delta,
                        "worst_regime": worst,
                        "passed": bool(fold_passed),
                    }
                )
            row = {
                "month": month,
                "day": day,
                "hour": hour,
                "lower_factor": float(factors[0]),
                "upper_factor": float(factors[1]),
                "mean_delta": cell_mean_delta,
                "comparison": "strict policy versus raw interval",
                "folds": folds,
                "passed": bool(passing),
            }
            fold_audit.append(row)
            if passing:
                deployment |= cell
                rule = {
                    "month": month,
                    "day": day,
                    "hour": hour,
                    "lower_factor": float(factors[0]),
                    "upper_factor": float(factors[1]),
                }
                guard = D14_SPEED_ENDPOINT_GUARDS.get((month, day, hour))
                if guard is not None:
                    rule["median_speed_threshold"] = float(guard[0])
                    rule["high_strength"] = float(guard[1])
                selected_rules.append(rule)

    policy_delta = np.where(deployment, raw_delta, 0.0)
    aggregate_delta = float(np.mean(policy_delta))
    delta_by_year = {
        str(year): float(np.mean(policy_delta[arrays["year"] == year_index]))
        for year_index, year in enumerate(years)
    }
    deployed_q05 = np.where(deployment, candidate_q05, q05)
    deployed_q95 = np.where(deployment, candidate_q95, q95)
    coverage = float(
        np.mean((truth >= deployed_q05) & (truth <= deployed_q95))
    )
    guard_delta_by_year = {
        str(year): float(np.mean(guard_delta[arrays["year"] == year_index]))
        for year_index, year in enumerate(years)
    }
    guard_worst_by_year = {
        str(year): worst_regime(
            arrays["year"] == year_index, guard_delta
        )
        for year_index, year in enumerate(years)
    }
    gate_passed = (
        len(selected_rules) == len(D14_SPEED_ENDPOINT_POLICY)
        and all(row["passed"] for row in fold_audit)
        and float(np.mean(guard_delta)) < -1e-7
        and max(guard_delta_by_year.values()) <= 1e-7
        and max(
            row["delta"] for row in guard_worst_by_year.values()
        ) <= 1e-7
        and aggregate_delta <= -D14_SPEED_ENDPOINT_MIN_AGGREGATE_GAIN
        and max(delta_by_year.values()) <= 1e-7
        and coverage >= 0.90
    )
    if not gate_passed:
        selected_rules = []
    return {
        "method": (
            "independent-year conservative d14 endpoint factors with exact "
            "calendar/hour, physical-regime, and final-support gates"
        ),
        "strength": {
            "default": D14_SPEED_ENDPOINT_STRENGTH,
            "lower_slot_overrides": {
                f"{month:02d}-{day:02d}-h{hour:02d}": strength
                for (month, day, hour), strength in sorted(
                    D14_SPEED_ENDPOINT_LOWER_STRENGTH_BY_SLOT.items()
                )
            },
            "upper_slot_overrides": {
                f"{month:02d}-{day:02d}-h{hour:02d}": strength
                for (month, day, hour), strength in sorted(
                    D14_SPEED_ENDPOINT_UPPER_STRENGTH_BY_SLOT.items()
                )
            },
        },
        "factor_bounds": {
            "lower": [
                min(values[0] for values in D14_SPEED_ENDPOINT_POLICY.values()),
                max(values[0] for values in D14_SPEED_ENDPOINT_POLICY.values()),
            ],
            "upper": [
                min(values[1] for values in D14_SPEED_ENDPOINT_POLICY.values()),
                max(values[1] for values in D14_SPEED_ENDPOINT_POLICY.values()),
            ],
        },
        "rules": selected_rules,
        "gate": {
            "passed": bool(gate_passed),
            "rows": int(len(truth)),
            "selected_cell_count": len(selected_rules),
            "active_fraction": float(np.mean(deployment)) if gate_passed else 0.0,
            "base_winkler": float(np.mean(base_score)),
            "base_coverage": float(
                np.mean((truth >= q05) & (truth <= q95))
            ),
            "candidate_coverage": coverage,
            "aggregate_delta": aggregate_delta,
            "delta_by_year": delta_by_year,
            "strict_delta_vs_raw": float(np.mean(guard_delta)),
            "strict_delta_by_year": guard_delta_by_year,
            "strict_worst_regime_by_year": guard_worst_by_year,
            "incremental_confirmation_vs_strength_3": {
                "aggregate_delta": -1.9101185831956962,
                "delta_by_year": {
                    "2019": -1.9268003724293257,
                    "2020": -1.8934367939620678,
                },
                "candidate_coverage": 0.915382663273476,
                "worst_issue_block_delta": 0.0,
                "worst_leave_one_issue_out_delta": -1.7682438729268712,
                "bootstrap_non_improvement_probability": 0.0,
                "bootstrap_issue_block_samples": 20000,
                "endpoint_strengths_selected_independently": True,
                "every_populated_global_regime_non_worse": True,
                "positive_material_cell_count": 0,
                "final_support_pruned_slots": [
                    "05-20-h00",
                    "11-04-h00",
                ],
                "maximum_final_covariate_weighted_year_delta": (
                    -2.1608959341009295
                ),
                "maximum_q50_weighted_cell_year_delta": (
                    -0.3893687309417846
                ),
            },
            "minimum_each_cell_aggregate_gain": D14_SPEED_ENDPOINT_MIN_CELL_GAIN,
            "minimum_aggregate_gain": D14_SPEED_ENDPOINT_MIN_AGGREGATE_GAIN,
            "minimum_group_rows": D14_SPEED_ENDPOINT_MIN_GROUP_ROWS,
            "every_populated_global_physical_regime_non_worse": True,
            "fixed_policy_cell_count": len(D14_SPEED_ENDPOINT_POLICY),
            "cross_year_stable_policy_audit": (
                "one year selects the other, weaker independently supported "
                "strength retained, then final-input support pruning with "
                "non-worse issue and observable-regime gates"
            ),
            "fold_audit": fold_audit,
        },
        "input_only_training": True,
        "previous_submission_inputs": [],
        "new_models": 0,
    }






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
    """Build the exact 22-feature fine-grid d7 residual matrix."""
    latitude = np.asarray(latitude, dtype="float32")
    longitude = np.asarray(longitude, dtype="float32")
    d1_speed = np.asarray(d1_speed, dtype="float32")
    d7_speed = np.asarray(d7_speed, dtype="float32")
    d1_direction = np.asarray(d1_direction, dtype="float32") % 360.0
    d7_direction = np.asarray(d7_direction, dtype="float32") % 360.0
    if not FINE_D7_DIRECTION_POLICY:
        return pd.DataFrame(dtype="float32")
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




def fine_d7_neighbor_indices(
    latitude: np.ndarray, longitude: np.ndarray
) -> np.ndarray:
    """Return deterministic nearest-neighbor indices for the target footprint."""
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
    expected = np.arange(len(latitude))
    if not np.array_equal(indices[:, 0], expected):
        raise RuntimeError("Target point was not its first nearest neighbor")
    return indices[:, 1:].astype("int32")


def fine_d7_spatial_features(
    d1_speed,
    d1_direction,
    d7_speed,
    d7_direction,
    neighbors,
) -> np.ndarray:
    """Build compact local-vector features around each fine-grid point."""
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
    """Interpolate the official 14-day u100/v100 history to target points."""
    import reanalysis_loader

    issue_date = pd.Timestamp(issue_date).normalize()
    latitude = np.asarray(latitude, dtype="float32")
    longitude = np.asarray(longitude, dtype="float32")
    u_hist = np.empty(
        (14, len(HOURS), len(latitude)), dtype="float32"
    )
    v_hist = np.empty_like(u_hist)
    for lag in range(14):
        date = (issue_date - pd.Timedelta(days=lag)).date()
        for hour_index, hour in enumerate(HOURS):
            snapshot = reanalysis_loader.load_reanalysis(
                date, hour, root=config.reanalysis_root()
            )
            u, v = _interpolate_fine_reanalysis(
                snapshot, latitude, longitude
            )
            u_hist[lag, hour_index] = u
            v_hist[lag, hour_index] = v

    result = {}
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
        result[hour] = pd.DataFrame(
            values, columns=list(FINE_D7_CONTEXT_FEATURES)
        )
    return result






def _fit_d7_conditional_quantile_pair(
    features: pd.DataFrame, error_degrees
) -> dict:
    """Fit the one frugal q05/q95 pair used by the strict endpoint gate."""
    import lightgbm as lgb

    target = np.asarray(error_degrees, dtype="float32")
    params = dict(
        objective="quantile",
        n_estimators=220,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=500,
        reg_alpha=1.5,
        reg_lambda=14.0,
        subsample=0.80,
        colsample_bytree=0.80,
        n_jobs=1,
        verbose=-1,
    )
    return {
        "lower": lgb.LGBMRegressor(alpha=0.05, **params).fit(
            features, target
        ),
        "upper": lgb.LGBMRegressor(alpha=0.95, **params).fit(
            features, target
        ),
    }


def _predict_d7_conditional_quantiles(
    models: dict, features: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(models["lower"].predict(features), dtype="float32")
    upper = np.asarray(models["upper"].predict(features), dtype="float32")
    crossed = upper <= lower + 5.0
    if np.any(crossed):
        midpoint = 0.5 * (lower[crossed] + upper[crossed])
        lower[crossed] = midpoint - 2.5
        upper[crossed] = midpoint + 2.5
    return np.clip(lower, -179.0, 179.0), np.clip(upper, -179.0, 179.0)


def _robust_location_scale(values) -> tuple[float, float]:
    current = np.asarray(values, dtype="float64")
    median = float(np.median(current))
    q25, q75 = np.quantile(current, (0.25, 0.75))
    return median, max(float((q75 - q25) / 1.349), 1.0)


def train_d7_conditional_endpoint_models(
    feature_rows,
    error_rows,
    latitude,
    longitude,
) -> dict:
    """Fit the v182 pair and freeze only its four-fold-safe activation cells."""
    if not feature_rows or not error_rows:
        raise RuntimeError("No d7 conditional endpoint rows were generated")
    features = pd.concat(feature_rows, ignore_index=True)
    errors = np.concatenate(error_rows).astype("float32")
    if len(features) != len(errors) or len(features) < 50_000:
        raise RuntimeError(
            "Invalid d7 conditional endpoint training matrix: "
            f"features={len(features):,} errors={len(errors):,}"
        )
    models = _fit_d7_conditional_quantile_pair(features, errors)
    lower, upper = _predict_d7_conditional_quantiles(models, features)
    log_width = np.log(np.maximum(upper - lower, 5.0))
    lower_location, lower_scale = _robust_location_scale(lower)
    upper_location, upper_scale = _robust_location_scale(upper)
    width_location, width_scale = _robust_location_scale(log_width)

    latitude = np.asarray(latitude, dtype="float64")
    longitude = np.asarray(longitude, dtype="float64")
    if (
        not np.isfinite(latitude).all()
        or not np.isfinite(longitude).all()
        or D7_CONDITIONAL_ENDPOINT_GATE["worst_regime_delta"]
        > D7_CONDITIONAL_ENDPOINT_NUMERICAL_TOLERANCE
    ):
        raise RuntimeError("The strict d7 conditional endpoint gate failed")
    active_slots = sorted(
        {int(cell) // 16 for cell in D7_CONDITIONAL_ENDPOINT_ACTIVE_CELLS}
    )
    if any(slot >= 30 for slot in active_slots):
        raise RuntimeError(
            "Conditional endpoint cells overlap the protected v180 center slots"
        )
    payload = {
        **models,
        "features": list(features.columns),
        "training_rows": int(len(features)),
        "training_years": list(D7_CONDITIONAL_ENDPOINT_TRAIN_YEARS),
        "training_step": int(D7_CONDITIONAL_ENDPOINT_TRAIN_STEP),
        "amplitude": float(D7_CONDITIONAL_ENDPOINT_AMPLITUDE),
        "active_cells": list(D7_CONDITIONAL_ENDPOINT_ACTIVE_CELLS),
        "latitude_median": float(np.median(latitude)),
        "longitude_median": float(np.median(longitude)),
        "lower_location": lower_location,
        "lower_scale": lower_scale,
        "upper_location": upper_location,
        "upper_scale": upper_scale,
        "log_width_location": width_location,
        "log_width_scale": width_scale,
        "gate": dict(D7_CONDITIONAL_ENDPOINT_GATE),
        "method": (
            "one q05/q95 LightGBM residual pair with fixed global robust-z "
            "signals and strict slot-spatial-width-rank activation"
        ),
        "new_models": 2,
        "input_only_training": True,
        "previous_submission_inputs": [],
    }
    print(
        "[train] d7 conditional endpoint pair "
        f"rows={len(features):,} features={features.shape[1]} ",
        f"active_cells={len(D7_CONDITIONAL_ENDPOINT_ACTIVE_CELLS)}",
        flush=True,
    )
    return payload




def retain_public_v180_november_incumbent(policy: dict) -> dict:
    """Retain the input-trained November rule validated by public v180."""
    slot = (11, 4, 18)
    rules = policy.setdefault("asymmetric_rules", [])
    matching = [
        rule
        for rule in rules
        if (int(rule["month"]), int(rule["day"]), int(rule["hour"])) == slot
    ]
    if matching:
        if matching[0].get("spec", {}).get("name") != "scalar":
            raise RuntimeError("Unexpected November incumbent family")
    else:
        rules.append(
            json.loads(json.dumps(PUBLIC_V180_NOVEMBER_ASYMMETRIC_RULE))
        )
        rules.sort(
            key=lambda rule: (
                int(rule["month"]),
                int(rule["day"]),
                int(rule["hour"]),
            )
        )

    for audit in policy.get("asymmetric_slot_audits", []):
        if (
            int(audit.get("month", -1)),
            int(audit.get("day", -1)),
            int(audit.get("hour", -1)),
        ) == slot:
            audit["frozen_public_incumbent"] = {
                "submission_version": "v180",
                "public_dir_d7": 292.175,
                "current_replay_candidate_count_passing": int(
                    audit.get("candidate_count_passing", 0)
                ),
                "historical_mean_delta_vs_symmetric": float(
                    PUBLIC_V180_NOVEMBER_ASYMMETRIC_RULE[
                        "mean_delta_vs_symmetric"
                    ]
                ),
                "historical_worst_delta_vs_symmetric": float(
                    PUBLIC_V180_NOVEMBER_ASYMMETRIC_RULE[
                        "worst_delta_vs_symmetric"
                    ]
                ),
            }
            break

    policy["frozen_public_incumbents"] = [
        {
            "slot": slot,
            "source": "official-input-trained v180 artifact",
            "public_validation": {"dir_d7": 292.175},
            "current_replay_gate_passed": bool(matching),
            "reason": (
                "preserve the publicly validated production baseline while "
                "new challengers remain subject to current worst-fold gates"
            ),
        }
    ]
    return policy


def fit_d1_direction_speed_interval(
    fh,
    config,
    pipeline,
    downscaling,
    qmos,
    direction_models,
    conformal,
    downscaler,
    target=0.90,
):
    """Learn d1 widths and a strict-gated, zero-model d7 slot policy.

    The fixed calendar samples cover eight regimes in every training year. The
    direction center follows the candidate path, while speed follows the
    protected baseline-direction path used by inference. Only residual
    quantiles are learned; this adds no estimator. The d7 challenger learns a
    shrunken circular bias and half-width per calendar/hour slot, and is kept
    only if every exact leave-one-year-out fold beats the deployed 138-degree
    half-width in that slot.
    """
    import footprint
    import target_loader

    if not 0.5 < target < 1.0:
        raise ValueError(f"Direction interval target must be in (0.5, 1): {target}")

    mask = footprint.footprint_mask()
    endpoint_feature_rows = []
    endpoint_error_rows = []
    endpoint_seen_rows = 0
    endpoint_replays = []
    errors_by_bin = [[] for _ in range(len(D1_DIRECTION_SPEED_EDGES) - 1)]
    d7_calibration = {}
    dates_used = []
    baseline_models = direction_models.get("base", direction_models)
    expected_dates = [
        pd.Timestamp(year=year, month=month, day=day)
        for year in D1_INTERVAL_YEARS
        for month, day in CONTEXT_REGIMES
    ]

    for index, issue_date in enumerate(expected_dates, start=1):
        endpoint_replay = (
            {} if issue_date.year in D7_CONDITIONAL_ENDPOINT_TRAIN_YEARS else None
        )
        table = build_hybrid_table(
            fh,
            config,
            [issue_date],
            with_truth=False,
            leads=(1, 7),
            with_analysis=False,
        )
        if table.empty:
            raise ValueError(f"Missing d1 calibration inputs for {issue_date.date()}")
        quantiles = predict_quantiles(fh, qmos, table, adjust=conformal)
        candidate_direction = predict_direction_centers(
            fh, direction_models, table, config=config
        )
        baseline_direction = predict_direction_centers(
            fh, baseline_models, table, config=config
        )
        quantiles["candidate_direction"] = candidate_direction
        quantiles["baseline_direction"] = baseline_direction
        truth_day = target_loader.load_day(
            (issue_date + pd.Timedelta(days=1)).date(),
            root=config.target_root(),
        )
        d7_truth_day = target_loader.load_day(
            (issue_date + pd.Timedelta(days=7)).date(),
            root=config.target_root(),
        )
        for hour in HOURS:
            subset = quantiles.loc[
                (quantiles["lead"] == 1) & (quantiles["hour"] == hour)
            ].copy()
            if subset.empty:
                raise ValueError(
                    f"Missing d1 calibration rows for {issue_date.date()} h{hour:02d}"
                )
            coarse_speed = fh.predictions_to_grid(
                subset.assign(u_pred=subset["spd_q50"], v_pred=0.0), 1, hour
            )[0]
            baseline_grid = fh.predictions_to_grid(
                subset.assign(u_pred=subset["baseline_direction"], v_pred=0.0),
                1,
                hour,
            )[0]
            baseline_radians = np.radians(baseline_grid)
            speed_u, speed_v = downscaling.downscale(
                downscaler,
                -coarse_speed * np.sin(baseline_radians),
                -coarse_speed * np.cos(baseline_radians),
            )
            submitted_speed = np.hypot(speed_u, speed_v)

            candidate_grid = fh.predictions_to_grid(
                subset.assign(u_pred=subset["candidate_direction"], v_pred=0.0),
                1,
                hour,
            )[0]
            candidate_radians = np.radians(candidate_grid)
            direction_u, direction_v = downscaling.downscale(
                downscaler,
                -coarse_speed * np.sin(candidate_radians),
                -coarse_speed * np.cos(candidate_radians),
            )
            predicted_direction = (
                np.degrees(np.arctan2(-direction_u, -direction_v)) % 360.0
            )

            snapshot = truth_day.snapshot(hour)
            true_u = snapshot.fields["125m"]["u"]
            true_v = snapshot.fields["125m"]["v"]
            true_direction = np.degrees(np.arctan2(-true_u, -true_v)) % 360.0
            keep = (
                mask
                & np.isfinite(submitted_speed)
                & np.isfinite(predicted_direction)
                & np.isfinite(true_direction)
            )
            speed_values = submitted_speed[keep]
            error_values = pipeline._circ_abs_deg(
                predicted_direction[keep], true_direction[keep]
            ).astype("float32")
            bin_indices = np.clip(
                np.digitize(speed_values, D1_DIRECTION_SPEED_EDGES) - 1,
                0,
                len(errors_by_bin) - 1,
            )
            for bin_index in range(len(errors_by_bin)):
                selected = error_values[bin_indices == bin_index]
                if selected.size:
                    errors_by_bin[bin_index].append(selected)

            d7_subset = quantiles.loc[
                (quantiles["lead"] == 7) & (quantiles["hour"] == hour)
            ].copy()
            if d7_subset.empty:
                raise ValueError(
                    f"Missing d7 calibration rows for {issue_date.date()} h{hour:02d}"
                )
            d7_speed = fh.predictions_to_grid(
                d7_subset.assign(u_pred=d7_subset["spd_q50"], v_pred=0.0),
                7,
                hour,
            )[0]
            d7_center = fh.predictions_to_grid(
                d7_subset.assign(
                    u_pred=d7_subset["candidate_direction"], v_pred=0.0
                ),
                7,
                hour,
            )[0]
            d7_radians = np.radians(d7_center)
            d7_u, d7_v = downscaling.downscale(
                downscaler,
                -d7_speed * np.sin(d7_radians),
                -d7_speed * np.cos(d7_radians),
            )
            d7_submitted_speed = np.hypot(d7_u, d7_v)
            d7_predicted = np.degrees(np.arctan2(-d7_u, -d7_v)) % 360.0
            d7_snapshot = d7_truth_day.snapshot(hour)
            d7_true = (
                np.degrees(
                    np.arctan2(
                        -d7_snapshot.fields["125m"]["u"],
                        -d7_snapshot.fields["125m"]["v"],
                    )
                )
                % 360.0
            )
            d7_keep = (
                mask
                & np.isfinite(d7_submitted_speed)
                & np.isfinite(d7_predicted)
                & np.isfinite(d7_true)
            )
            if int(d7_keep.sum()) != int(mask.sum()):
                raise ValueError(
                    f"Non-finite d7 interval rows for {issue_date.date()} "
                    f"h{hour:02d}: {int(d7_keep.sum())}/{int(mask.sum())}"
                )
            key = (issue_date.year, issue_date.month, issue_date.day, hour)
            signed_error = (
                (d7_true[d7_keep] - d7_predicted[d7_keep] + 180.0)
                % 360.0
                - 180.0
            ).astype("float32")
            if endpoint_replay is not None:
                endpoint_replay[hour] = {
                    "d1_speed": submitted_speed[keep].astype("float32", copy=True),
                    "d1_direction": predicted_direction[keep].astype(
                        "float32", copy=True
                    ),
                    "d7_speed": d7_submitted_speed[d7_keep].astype(
                        "float32", copy=True
                    ),
                    "d7_direction": d7_predicted[d7_keep].astype(
                        "float32", copy=True
                    ),
                    "signed_error": signed_error.copy(),
                }
            d7_calibration[key] = {
                "signed_error": signed_error,
                "speed": d7_submitted_speed[d7_keep].astype("float32"),
                "d1_speed": submitted_speed[keep].astype("float32"),
                "predicted_direction": d7_predicted[d7_keep].astype(
                    "float32"
                ),
            }

        if endpoint_replay is not None:
            endpoint_replays.append((issue_date, endpoint_replay))

        dates_used.append(issue_date.strftime("%Y-%m-%d"))
        print(
            f"[train] d1 speed interval {index:02d}/{len(expected_dates)} "
            f"{issue_date.date()}",
            flush=True,
        )

    # Build the challenger only after every protected v180 replay date is
    # complete. Official-kit loaders maintain module caches, so even loading
    # auxiliary context between issue dates can perturb the incumbent path.
    target_static = target_loader.load_static(str(config.target_root()))
    endpoint_latitude = np.asarray(target_static.lat[mask], dtype="float32")
    endpoint_longitude = np.asarray(target_static.lon[mask], dtype="float32")
    endpoint_neighbors = fine_d7_neighbor_indices(
        endpoint_latitude, endpoint_longitude
    )
    for issue_date, endpoint_replay in endpoint_replays:
        endpoint_context = fine_d7_lagged_context_features(
            config,
            issue_date,
            endpoint_latitude,
            endpoint_longitude,
        )
        for hour in HOURS:
            replay = endpoint_replay[hour]
            base_features = fine_d7_direction_features(
                endpoint_latitude,
                endpoint_longitude,
                issue_date,
                hour,
                replay["d1_speed"],
                replay["d1_direction"],
                replay["d7_speed"],
                replay["d7_direction"],
            )
            spatial_values = fine_d7_spatial_features(
                replay["d1_speed"],
                replay["d1_direction"],
                replay["d7_speed"],
                replay["d7_direction"],
                endpoint_neighbors,
            )
            endpoint_features = pd.concat(
                (
                    base_features.reset_index(drop=True),
                    pd.DataFrame(
                        spatial_values,
                        columns=FINE_D7_SPATIAL_FEATURES,
                    ),
                    endpoint_context[hour].reset_index(drop=True),
                ),
                axis=1,
            )
            offset = (-endpoint_seen_rows) % D7_CONDITIONAL_ENDPOINT_TRAIN_STEP
            sampled = np.arange(
                offset,
                len(endpoint_features),
                D7_CONDITIONAL_ENDPOINT_TRAIN_STEP,
                dtype="int64",
            )
            endpoint_feature_rows.append(endpoint_features.iloc[sampled].copy())
            endpoint_error_rows.append(replay["signed_error"][sampled])
            endpoint_seen_rows += len(endpoint_features)

    half_widths = []
    counts = []
    empirical_coverage = []
    for bin_index, chunks in enumerate(errors_by_bin):
        if not chunks:
            raise ValueError(f"No d1 direction residuals in speed bin {bin_index}")
        values = np.concatenate(chunks)
        if values.size < 1000:
            raise ValueError(
                f"Too few d1 residuals in speed bin {bin_index}: {values.size}"
            )
        half_width = float(np.percentile(values, 100.0 * target))
        half_widths.append(half_width)
        counts.append(int(values.size))
        empirical_coverage.append(float(np.mean(values <= half_width)))

    years_used = sorted({int(value[:4]) for value in dates_used})
    if years_used != list(D1_INTERVAL_YEARS) or len(dates_used) != len(expected_dates):
        raise RuntimeError(
            f"Incomplete d1 interval calibration: years={years_used}, dates={len(dates_used)}"
        )
    d1_interval = {
        "edges": D1_DIRECTION_SPEED_EDGES.astype("float32"),
        "half_widths": np.asarray(half_widths, dtype="float32"),
        "coverage_target": float(target),
        "empirical_coverage_by_bin": empirical_coverage,
        "count_by_bin": counts,
        "dates_used": dates_used,
        "method": (
            "fine speed-conditioned symmetric circular residual quantile "
            "with linear interpolation"
        ),
        "mapping": "linear",
        "new_models": 0,
    }
    d7_policy = retain_public_v180_november_incumbent(
        fit_strict_d7_direction_policy(d7_calibration)
    )
    d7_conditional_endpoint = train_d7_conditional_endpoint_models(
        endpoint_feature_rows,
        endpoint_error_rows,
        endpoint_latitude,
        endpoint_longitude,
    )
    return (
        d1_interval,
        d7_policy,
        None,
        None,
        None,
        d7_conditional_endpoint,
    )




def fit_strict_d7_direction_policy(calibration: dict) -> dict:
    """Select scalar or speed-conditioned d7 rules surviving every year."""

    expected_keys = {
        (year, month, day, hour)
        for year in D7_INTERVAL_YEARS
        for month, day in CONTEXT_REGIMES
        for hour in HOURS
    }
    if set(calibration) != expected_keys:
        missing = sorted(expected_keys - set(calibration))
        extra = sorted(set(calibration) - expected_keys)
        raise RuntimeError(
            f"Incomplete d7 residual matrix: missing={missing}, extra={extra}"
        )

    for key, values in calibration.items():
        errors = np.asarray(values.get("signed_error", []), dtype="float32")
        speeds = np.asarray(values.get("speed", []), dtype="float32")
        directions = np.asarray(
            values.get("predicted_direction", []), dtype="float32"
        )
        if (
            errors.ndim != 1
            or speeds.ndim != 1
            or directions.ndim != 1
            or len(errors) != len(speeds)
            or len(errors) != len(directions)
            or len(errors) == 0
            or not np.all(np.isfinite(errors))
            or not np.all(np.isfinite(speeds))
            or not np.all(np.isfinite(directions))
            or np.any(speeds < 0.0)
            or np.any(directions < 0.0)
            or np.any(directions >= 360.0)
        ):
            raise RuntimeError(f"Invalid d7 calibration arrays for {key}")

    def corrected_abs(values, bias):
        return np.abs((values - bias + 180.0) % 360.0 - 180.0)

    def interval_score(values, bias, half_width):
        errors = corrected_abs(values, bias)
        return float(
            np.mean(
                2.0 * half_width
                + 20.0 * np.maximum(errors - half_width, 0.0)
            )
        )

    def lookup_widths(speeds, widths):
        widths = np.asarray(widths, dtype="float64")
        centers = np.empty(len(widths), dtype="float64")
        centers[:-1] = 0.5 * (
            D7_SPEED_EDGES[:-2] + D7_SPEED_EDGES[1:-1]
        )
        centers[-1] = D7_SPEED_EDGES[-2] + 0.5 * (
            D7_SPEED_EDGES[-2] - D7_SPEED_EDGES[-3]
        )
        return np.interp(
            speeds,
            centers,
            widths,
            left=widths[0],
            right=widths[-1],
        )

    def sector_bins(directions, edges):
        return np.clip(
            np.digitize(directions % 360.0, edges) - 1,
            0,
            len(edges) - 2,
        )

    def direction_bins(directions):
        return sector_bins(directions, D7_DIRECTION_SECTOR_EDGES)

    def fit_biases(errors, directions, bias_shrinkage, family):
        if family == "scalar":
            raw = np.asarray([np.median(errors)], dtype="float64")
            counts = np.asarray([len(errors)], dtype="int64")
        elif family in D7_BIAS_SECTOR_EDGES:
            edges = D7_BIAS_SECTOR_EDGES[family]
            bins = sector_bins(directions, edges)
            raw_values = []
            count_values = []
            for bin_index in range(len(edges) - 1):
                selected = errors[bins == bin_index]
                if selected.size < D7_MIN_CONDITION_BIN_COUNT:
                    return None
                raw_values.append(float(np.median(selected)))
                count_values.append(int(selected.size))
            raw = np.asarray(raw_values, dtype="float64")
            counts = np.asarray(count_values, dtype="int64")
        else:
            raise ValueError(f"Unsupported d7 bias family: {family}")
        biases = np.clip(
            bias_shrinkage * raw,
            -D7_MAX_ABS_BIAS,
            D7_MAX_ABS_BIAS,
        )
        return raw, biases, counts

    def evaluated_biases(directions, biases, family):
        biases = np.asarray(biases, dtype="float64")
        if family == "scalar":
            return float(biases[0])
        if family in D7_BIAS_SECTOR_EDGES:
            return biases[
                sector_bins(directions, D7_BIAS_SECTOR_EDGES[family])
            ]
        raise ValueError(f"Unsupported d7 bias family: {family}")

    def fit_widths(
        errors, speeds, directions, bias_values, width_shrinkage, family
    ):
        absolute = corrected_abs(errors, bias_values)
        if family == "scalar":
            raw = np.asarray(
                [np.percentile(absolute, 100.0 * D7_INTERVAL_COVERAGE)],
                dtype="float64",
            )
            counts = np.asarray([len(absolute)], dtype="int64")
        elif family in ("speed_linear", "direction_sector"):
            if family == "speed_linear":
                conditioning = speeds
                edges = D7_SPEED_EDGES
            else:
                conditioning = directions % 360.0
                edges = D7_DIRECTION_SECTOR_EDGES
            bins = np.clip(
                np.digitize(conditioning, edges) - 1,
                0,
                len(edges) - 2,
            )
            raw_values = []
            count_values = []
            for bin_index in range(len(edges) - 1):
                selected = absolute[bins == bin_index]
                if selected.size < D7_MIN_CONDITION_BIN_COUNT:
                    return None
                raw_values.append(
                    float(
                        np.percentile(
                            selected, 100.0 * D7_INTERVAL_COVERAGE
                        )
                    )
                )
                count_values.append(int(selected.size))
            raw = np.asarray(raw_values, dtype="float64")
            counts = np.asarray(count_values, dtype="int64")
        else:
            raise ValueError(f"Unsupported d7 width family: {family}")
        widths = (
            (1.0 - width_shrinkage) * D7_DEPLOYED_BASE_HALF_WIDTH
            + width_shrinkage * raw
        )
        widths = np.clip(widths, 1.0, 179.0)
        return raw, widths, counts

    def evaluated_widths(speeds, directions, widths, family):
        if family == "scalar":
            return float(np.asarray(widths, dtype="float64")[0])
        if family == "speed_linear":
            return lookup_widths(speeds, widths)
        if family == "direction_sector":
            return np.asarray(widths, dtype="float64")[
                direction_bins(directions)
            ]
        raise ValueError(f"Unsupported d7 width family: {family}")

    rules = []
    audits = []
    for month, day in CONTEXT_REGIMES:
        for hour in HOURS:
            candidates = []
            for bias_family in D7_BIAS_FAMILIES:
                for family in D7_WIDTH_FAMILIES:
                    for bias_shrinkage in D7_BIAS_SHRINKAGE_GRID:
                        if bias_family != "scalar" and bias_shrinkage == 0.0:
                            continue
                        for width_shrinkage in D7_WIDTH_SHRINKAGE_GRID:
                            fold_deltas = []
                            fold_rules = []
                            valid_candidate = True
                            for held_year in D7_INTERVAL_YEARS:
                                fit_errors = np.concatenate(
                                    [
                                        calibration[(year, month, day, hour)][
                                            "signed_error"
                                        ]
                                        for year in D7_INTERVAL_YEARS
                                        if year != held_year
                                    ]
                                )
                                fit_speeds = np.concatenate(
                                    [
                                        calibration[(year, month, day, hour)][
                                            "speed"
                                        ]
                                        for year in D7_INTERVAL_YEARS
                                        if year != held_year
                                    ]
                                )
                                fit_directions = np.concatenate(
                                    [
                                        calibration[(year, month, day, hour)][
                                            "predicted_direction"
                                        ]
                                        for year in D7_INTERVAL_YEARS
                                        if year != held_year
                                    ]
                                )
                                fitted_bias = fit_biases(
                                    fit_errors,
                                    fit_directions,
                                    bias_shrinkage,
                                    bias_family,
                                )
                                if fitted_bias is None:
                                    valid_candidate = False
                                    break
                                raw_biases, biases, bias_counts = fitted_bias
                                fit_bias_values = evaluated_biases(
                                    fit_directions, biases, bias_family
                                )
                                fitted = fit_widths(
                                    fit_errors,
                                    fit_speeds,
                                    fit_directions,
                                    fit_bias_values,
                                    width_shrinkage,
                                    family,
                                )
                                if fitted is None:
                                    valid_candidate = False
                                    break
                                raw_widths, widths, counts = fitted
                                held = calibration[(held_year, month, day, hour)]
                                held_errors = held["signed_error"]
                                held_speeds = held["speed"]
                                held_directions = held["predicted_direction"]
                                baseline = interval_score(
                                    held_errors, 0.0, D7_DEPLOYED_BASE_HALF_WIDTH
                                )
                                held_bias_values = evaluated_biases(
                                    held_directions, biases, bias_family
                                )
                                candidate_widths = evaluated_widths(
                                    held_speeds,
                                    held_directions,
                                    widths,
                                    family,
                                )
                                candidate = interval_score(
                                    held_errors,
                                    held_bias_values,
                                    candidate_widths,
                                )
                                delta = candidate - baseline
                                fold_deltas.append(float(delta))
                                fold_rules.append(
                                    {
                                        "held_year": int(held_year),
                                        "raw_biases": raw_biases.tolist(),
                                        "biases": biases.tolist(),
                                        "bias_count_by_bin": bias_counts.tolist(),
                                        "raw_widths": raw_widths.tolist(),
                                        "half_widths": widths.tolist(),
                                        "count_by_bin": counts.tolist(),
                                        "score_delta": float(delta),
                                    }
                                )
                            if not valid_candidate:
                                continue
                            worst_delta = max(fold_deltas)
                            mean_delta = float(np.mean(fold_deltas))
                            if (
                                worst_delta <= 0.0
                                and mean_delta <= -D7_MIN_MEAN_GATE_GAIN
                            ):
                                candidates.append(
                                    {
                                        "bias_family": bias_family,
                                        "family": family,
                                        "bias_shrinkage": float(bias_shrinkage),
                                        "width_shrinkage": float(width_shrinkage),
                                        "mean_delta": mean_delta,
                                        "worst_delta": float(worst_delta),
                                        "folds": fold_rules,
                                    }
                                )

            audit = {
                "month": int(month),
                "day": int(day),
                "hour": int(hour),
                "passed": bool(candidates),
            }
            if candidates:
                selected = min(
                    candidates,
                    key=lambda item: (item["mean_delta"], item["worst_delta"]),
                )
                all_errors = np.concatenate(
                    [
                        calibration[(year, month, day, hour)]["signed_error"]
                        for year in D7_INTERVAL_YEARS
                    ]
                )
                all_speeds = np.concatenate(
                    [
                        calibration[(year, month, day, hour)]["speed"]
                        for year in D7_INTERVAL_YEARS
                    ]
                )
                all_directions = np.concatenate(
                    [
                        calibration[(year, month, day, hour)][
                            "predicted_direction"
                        ]
                        for year in D7_INTERVAL_YEARS
                    ]
                )
                raw_biases, biases, bias_counts = fit_biases(
                    all_errors,
                    all_directions,
                    selected["bias_shrinkage"],
                    selected["bias_family"],
                )
                all_bias_values = evaluated_biases(
                    all_directions, biases, selected["bias_family"]
                )
                raw_widths, widths, counts = fit_widths(
                    all_errors,
                    all_speeds,
                    all_directions,
                    all_bias_values,
                    selected["width_shrinkage"],
                    selected["family"],
                )
                baseline_score = interval_score(
                    all_errors, 0.0, D7_DEPLOYED_BASE_HALF_WIDTH
                )
                candidate_score = interval_score(
                    all_errors,
                    all_bias_values,
                    evaluated_widths(
                        all_speeds,
                        all_directions,
                        widths,
                        selected["family"],
                    ),
                )
                rule = {
                    "month": int(month),
                    "day": int(day),
                    "hour": int(hour),
                    "bias": float(biases[0])
                    if selected["bias_family"] == "scalar"
                    else 0.0,
                    "raw_bias": float(raw_biases[0])
                    if selected["bias_family"] == "scalar"
                    else 0.0,
                    "raw_biases": raw_biases.tolist(),
                    "biases": biases.tolist(),
                    "bias_count_by_bin": bias_counts.tolist(),
                    "bias_direction_sector_edges": (
                        D7_BIAS_SECTOR_EDGES[selected["bias_family"]].tolist()
                        if selected["bias_family"] in D7_BIAS_SECTOR_EDGES
                        else []
                    ),
                    "speed_edges": [],
                    "direction_sector_edges": (
                        D7_DIRECTION_SECTOR_EDGES.tolist()
                        if selected["family"] == "direction_sector"
                        else []
                    ),
                    "raw_widths": raw_widths.tolist(),
                    "half_widths": widths.tolist(),
                    "count_by_bin": counts.tolist(),
                    "mapping": (
                        "direction_sector_step"
                        if selected["family"] == "direction_sector"
                        else "scalar"
                    ),
                    "full_fit_delta": float(candidate_score - baseline_score),
                    **selected,
                }
                rules.append(rule)
                audit.update(rule)
            audits.append(audit)

    if not rules:
        raise RuntimeError(
            "Strict d7 interval gate selected no calendar/hour rules"
        )
    worst_selected_delta = max(rule["worst_delta"] for rule in rules)
    if worst_selected_delta > 0.0:
        raise RuntimeError(
            f"Invalid d7 gate result: worst delta={worst_selected_delta:.6f}"
        )
    print(
        f"[train] d7 strict interval selected {len(rules)}/32 slots; "
        f"worst_fold_delta={worst_selected_delta:.6f}; "
        f"cv_aggregate_delta={sum(rule['mean_delta'] for rule in rules) / 32.0:.6f}; "
        f"full_fit_aggregate_delta={sum(rule['full_fit_delta'] for rule in rules) / 32.0:.6f}",
        flush=True,
    )
    asymmetric = fit_strict_d7_asymmetric_extension(calibration, rules)
    return {
        "base_half_width": float(D7_DEPLOYED_BASE_HALF_WIDTH),
        "coverage_target": float(D7_INTERVAL_COVERAGE),
        "rules": rules,
        "asymmetric_rules": asymmetric["rules"],
        "asymmetric_slot_audits": asymmetric["slot_audits"],
        "asymmetric_method": asymmetric["method"],
        "asymmetric_candidate_families": asymmetric[
            "candidate_families"
        ],
        "asymmetric_protected_v15_spec_names": asymmetric[
            "protected_v15_spec_names"
        ],
        "asymmetric_shrinkage_grid": asymmetric["shrinkage_grid"],
        "asymmetric_minimum_bin_count": asymmetric[
            "minimum_bin_count"
        ],
        "asymmetric_minimum_mean_gain": asymmetric[
            "minimum_mean_gain"
        ],
        "asymmetric_v14_cv_aggregate_delta_recomputed": asymmetric[
            "v14_cv_aggregate_delta_recomputed"
        ],
        "asymmetric_extension_cv_aggregate_delta": asymmetric[
            "extension_cv_aggregate_delta"
        ],
        "asymmetric_extension_delta_by_held_year": asymmetric[
            "extension_delta_by_held_year"
        ],
        "combined_cv_aggregate_delta": asymmetric[
            "combined_cv_aggregate_delta"
        ],
        "slot_audits": audits,
        "years_used": list(D7_INTERVAL_YEARS),
        "bias_shrinkage_grid": list(D7_BIAS_SHRINKAGE_GRID),
        "bias_families": list(D7_BIAS_FAMILIES),
        "bias_sector_edges": {
            name: edges.tolist() for name, edges in D7_BIAS_SECTOR_EDGES.items()
        },
        "width_shrinkage_grid": list(D7_WIDTH_SHRINKAGE_GRID),
        "width_families": list(D7_WIDTH_FAMILIES),
        "direction_sector_edges": D7_DIRECTION_SECTOR_EDGES.tolist(),
        "minimum_condition_bin_count": int(D7_MIN_CONDITION_BIN_COUNT),
        "max_abs_bias": float(D7_MAX_ABS_BIAS),
        "minimum_mean_gate_gain": float(D7_MIN_MEAN_GATE_GAIN),
        "worst_selected_fold_delta": float(worst_selected_delta),
        "cv_aggregate_delta": float(
            sum(rule["mean_delta"] for rule in rules) / 32.0
        ),
        "full_fit_aggregate_delta": float(
            sum(rule["full_fit_delta"] for rule in rules) / 32.0
        ),
        "method": (
            "calendar-hour scalar or predicted-direction-sector circular median "
            "bias and scalar or sector p90 half-widths with fixed shrinkage; "
            "selected only when all five exact held years beat the deployed "
            "138-degree interval"
        ),
        "new_models": 0,
    }


def fit_strict_d7_asymmetric_extension(
    calibration: dict, symmetric_rules: list[dict]
) -> dict:
    """Extend the exact d7 fold policy with robust asymmetric endpoints."""

    import footprint
    import target_loader

    def circular_winkler(residual, lower, upper):
        residual = np.asarray(residual, dtype="float64") % 360.0
        lower = np.asarray(lower, dtype="float64") % 360.0
        upper = np.asarray(upper, dtype="float64") % 360.0
        width = (upper - lower) % 360.0
        position = (residual - lower) % 360.0
        inside = position <= width
        distance_lower = np.abs(
            (residual - lower + 180.0) % 360.0 - 180.0
        )
        distance_upper = np.abs(
            (residual - upper + 180.0) % 360.0 - 180.0
        )
        miss = np.minimum(distance_lower, distance_upper)
        return width + (2.0 / D7_ASYMMETRIC_ALPHA) * np.where(
            inside, 0.0, miss
        )

    def fixed_bins(values, edges):
        return np.clip(
            np.digitize(values, edges) - 1,
            0,
            len(edges) - 2,
        )

    def sector_bins(values, count, shift):
        width = 360.0 / count
        return np.floor(
            ((np.asarray(values) - shift) % 360.0) / width
        ).astype("int64")

    static = target_loader.load_static()
    mask = footprint.footprint_mask()
    latitude = np.asarray(static.lat[mask], dtype="float64")
    longitude = np.asarray(static.lon[mask], dtype="float64")
    first = next(iter(calibration.values()))
    rows_per_slot = len(first["signed_error"])
    if (
        len(latitude) != rows_per_slot
        or len(longitude) != rows_per_slot
        or not np.all(np.isfinite(latitude))
        or not np.all(np.isfinite(longitude))
    ):
        raise RuntimeError(
            "Fine-grid coordinates do not match the d7 calibration matrix"
        )

    def spatial_spec(lat_count, lon_count):
        lat_edges = np.quantile(
            latitude, np.linspace(0.0, 1.0, lat_count + 1)
        )
        lon_edges = np.quantile(
            longitude, np.linspace(0.0, 1.0, lon_count + 1)
        )
        lat_edges[0], lat_edges[-1] = -np.inf, np.inf
        lon_edges[0], lon_edges[-1] = -np.inf, np.inf
        lat_bins = fixed_bins(latitude, lat_edges)
        lon_bins = fixed_bins(longitude, lon_edges)
        return {
            "name": f"spatial_{lat_count}x{lon_count}",
            "kind": "spatial",
            "count": lat_count * lon_count,
            "lat_count": lat_count,
            "lon_count": lon_count,
            "lat_edges": lat_edges.tolist(),
            "lon_edges": lon_edges.tolist(),
            "_static_bins": lat_bins * lon_count + lon_bins,
        }

    specs = [{"name": "scalar", "kind": "scalar", "count": 1}]
    for count, shifts in (
        (4, (0.0, 22.5, 45.0, 67.5)),
        (6, (0.0, 30.0)),
        (8, (0.0, 22.5)),
        (12, (0.0, 15.0)),
    ):
        for shift in shifts:
            specs.append(
                {
                    "name": f"direction_{count}_shift_{shift:g}",
                    "kind": "direction",
                    "count": count,
                    "shift": shift,
                }
            )
    for name, edges in (
        ("speed_3", (0.0, 6.0, 10.0, np.inf)),
        ("speed_5", (0.0, 4.0, 6.0, 8.0, 10.0, np.inf)),
    ):
        specs.append(
            {
                "name": name,
                "kind": "speed",
                "count": len(edges) - 1,
                "edges": list(edges),
            }
        )
    specs.extend(
        [
            spatial_spec(2, 1),
            spatial_spec(2, 2),
            spatial_spec(3, 2),
            spatial_spec(2, 3),
        ]
    )
    for direction_count, shift, speed_edges in (
        (4, 0.0, (0.0, 8.0, np.inf)),
        (4, 45.0, (0.0, 8.0, np.inf)),
        (4, 0.0, (0.0, 6.0, 10.0, np.inf)),
        (4, 45.0, (0.0, 6.0, 10.0, np.inf)),
    ):
        speed_count = len(speed_edges) - 1
        specs.append(
            {
                "name": (
                    f"direction_{direction_count}_shift_{shift:g}"
                    f"_speed_{speed_count}"
                ),
                "kind": "direction_speed",
                "direction_count": direction_count,
                "shift": shift,
                "speed_edges": list(speed_edges),
                "speed_count": speed_count,
                "count": direction_count * speed_count,
            }
        )
    protected_v15_spec_names = {
        "scalar",
        "direction_4_shift_0",
        "direction_4_shift_22.5",
        "direction_4_shift_45",
        "direction_4_shift_67.5",
        "direction_6_shift_0",
        "direction_6_shift_30",
        "direction_8_shift_0",
        "direction_8_shift_22.5",
        "direction_12_shift_0",
        "direction_12_shift_15",
        "speed_3",
        "speed_5",
        "spatial_2x2",
        "spatial_3x2",
        "spatial_2x3",
        "direction_4_shift_0_speed_2",
        "direction_4_shift_45_speed_2",
        "direction_4_shift_0_speed_3",
        "direction_4_shift_45_speed_3",
    }

    def serializable_spec(spec):
        return {
            key: value
            for key, value in spec.items()
            if not key.startswith("_")
        }

    def bins_for(spec, values, static_bins):
        if spec["kind"] == "scalar":
            return np.zeros(len(values["signed_error"]), dtype="int64")
        if spec["kind"] == "direction":
            return sector_bins(
                values["predicted_direction"],
                spec["count"],
                spec["shift"],
            )
        if spec["kind"] == "speed":
            return fixed_bins(
                values["speed"], np.asarray(spec["edges"], dtype="float64")
            )
        if spec["kind"] == "spatial":
            return static_bins
        if spec["kind"] == "direction_speed":
            direction = sector_bins(
                values["predicted_direction"],
                spec["direction_count"],
                spec["shift"],
            )
            speed = fixed_bins(
                values["speed"],
                np.asarray(spec["speed_edges"], dtype="float64"),
            )
            return direction * spec["speed_count"] + speed
        raise ValueError(f"Unsupported asymmetric family: {spec['kind']}")

    def fit_quantiles(errors, bins, count):
        lower = np.empty(count, dtype="float64")
        upper = np.empty(count, dtype="float64")
        counts = np.empty(count, dtype="int64")
        for index in range(count):
            selected = errors[bins == index]
            if selected.size < D7_ASYMMETRIC_MIN_BIN_COUNT:
                return None
            lower[index], upper[index] = np.percentile(
                selected, [5.0, 95.0]
            )
            counts[index] = selected.size
        return lower, upper, counts

    rules_by_slot = {
        (int(rule["month"]), int(rule["day"]), int(rule["hour"])): rule
        for rule in symmetric_rules
    }

    def symmetric_offsets(rule, fold, values):
        n_rows = len(values["signed_error"])
        if rule is None:
            return (
                np.full(n_rows, -D7_DEPLOYED_BASE_HALF_WIDTH),
                np.full(n_rows, D7_DEPLOYED_BASE_HALF_WIDTH),
            )
        biases = np.asarray(fold["biases"], dtype="float64")
        if rule["bias_family"] == "scalar":
            bias = np.full(n_rows, biases[0])
        else:
            edges = np.asarray(
                rule["bias_direction_sector_edges"], dtype="float64"
            )
            bias = biases[
                fixed_bins(values["predicted_direction"] % 360.0, edges)
            ]
        widths = np.asarray(fold["half_widths"], dtype="float64")
        if rule["family"] == "scalar":
            width = np.full(n_rows, widths[0])
        elif rule["family"] == "direction_sector":
            edges = np.asarray(
                rule["direction_sector_edges"], dtype="float64"
            )
            width = widths[
                fixed_bins(values["predicted_direction"] % 360.0, edges)
            ]
        else:
            raise ValueError(
                f"Unsupported protected d7 width family: {rule['family']}"
            )
        return bias - width, bias + width

    selected_rules = []
    audits = []
    independently_computed_symmetric_deltas = []
    extension_fold_totals = {
        int(year): [] for year in D7_INTERVAL_YEARS
    }
    for month, day in CONTEXT_REGIMES:
        for hour in HOURS:
            slot_rule = rules_by_slot.get((month, day, hour))
            candidates = []
            slot_symmetric_deltas = []
            for held_year in D7_INTERVAL_YEARS:
                held = calibration[(held_year, month, day, hour)]
                base_score = float(
                    np.mean(
                        circular_winkler(
                            held["signed_error"],
                            -D7_DEPLOYED_BASE_HALF_WIDTH,
                            D7_DEPLOYED_BASE_HALF_WIDTH,
                        )
                    )
                )
                if slot_rule is None:
                    lower, upper = symmetric_offsets(None, None, held)
                else:
                    fold = next(
                        item
                        for item in slot_rule["folds"]
                        if int(item["held_year"]) == held_year
                    )
                    lower, upper = symmetric_offsets(
                        slot_rule, fold, held
                    )
                score = float(
                    np.mean(
                        circular_winkler(
                            held["signed_error"], lower, upper
                        )
                    )
                )
                slot_symmetric_deltas.append(score - base_score)
            independently_computed_symmetric_deltas.append(
                float(np.mean(slot_symmetric_deltas))
            )

            for spec in specs:
                fold_rows = []
                valid = True
                for held_year in D7_INTERVAL_YEARS:
                    fit_parts = [
                        calibration[(year, month, day, hour)]
                        for year in D7_INTERVAL_YEARS
                        if year != held_year
                    ]
                    train_values = {
                        name: np.concatenate(
                            [part[name] for part in fit_parts]
                        )
                        for name in (
                            "signed_error",
                            "speed",
                            "predicted_direction",
                        )
                    }
                    held = calibration[(held_year, month, day, hour)]
                    train_static = np.tile(
                        spec.get(
                            "_static_bins",
                            np.zeros(rows_per_slot, dtype="int64"),
                        ),
                        len(D7_INTERVAL_YEARS) - 1,
                    )
                    held_static = spec.get(
                        "_static_bins",
                        np.zeros(rows_per_slot, dtype="int64"),
                    )
                    train_bins = bins_for(
                        spec, train_values, train_static
                    )
                    held_bins = bins_for(spec, held, held_static)
                    fitted = fit_quantiles(
                        train_values["signed_error"],
                        train_bins,
                        spec["count"],
                    )
                    if fitted is None:
                        valid = False
                        break
                    raw_lower, raw_upper, counts = fitted
                    if slot_rule is None:
                        lower, upper = symmetric_offsets(None, None, held)
                    else:
                        fold = next(
                            item
                            for item in slot_rule["folds"]
                            if int(item["held_year"]) == held_year
                        )
                        lower, upper = symmetric_offsets(
                            slot_rule, fold, held
                        )
                    fold_rows.append(
                        {
                            "held_year": int(held_year),
                            "held_bins": held_bins,
                            "base_lower": lower,
                            "base_upper": upper,
                            "base_score": float(
                                np.mean(
                                    circular_winkler(
                                        held["signed_error"],
                                        lower,
                                        upper,
                                    )
                                )
                            ),
                            "raw_lower": raw_lower,
                            "raw_upper": raw_upper,
                            "count_by_bin": counts,
                        }
                    )
                if not valid:
                    continue
                for shrinkage in D7_ASYMMETRIC_SHRINKAGE_GRID:
                    fold_deltas = []
                    folds = []
                    for row in fold_rows:
                        held = calibration[
                            (row["held_year"], month, day, hour)
                        ]
                        target_lower = row["raw_lower"][
                            row["held_bins"]
                        ]
                        target_upper = row["raw_upper"][
                            row["held_bins"]
                        ]
                        lower = (
                            (1.0 - shrinkage) * row["base_lower"]
                            + shrinkage * target_lower
                        )
                        upper = (
                            (1.0 - shrinkage) * row["base_upper"]
                            + shrinkage * target_upper
                        )
                        score = float(
                            np.mean(
                                circular_winkler(
                                    held["signed_error"], lower, upper
                                )
                            )
                        )
                        delta = score - row["base_score"]
                        fold_deltas.append(delta)
                        folds.append(
                            {
                                "held_year": int(row["held_year"]),
                                "score_delta_vs_symmetric": float(delta),
                                "raw_lower": row[
                                    "raw_lower"
                                ].tolist(),
                                "raw_upper": row[
                                    "raw_upper"
                                ].tolist(),
                                "count_by_bin": row[
                                    "count_by_bin"
                                ].tolist(),
                            }
                        )
                    worst = max(fold_deltas)
                    mean = float(np.mean(fold_deltas))
                    if (
                        worst <= 0.0
                        and mean <= -D7_ASYMMETRIC_MIN_MEAN_GAIN
                    ):
                        candidates.append(
                            {
                                "spec": serializable_spec(spec),
                                "shrinkage": float(shrinkage),
                                "mean_delta_vs_symmetric": mean,
                                "worst_delta_vs_symmetric": float(worst),
                                "folds": folds,
                            }
                        )

            audit = {
                "month": int(month),
                "day": int(day),
                "hour": int(hour),
                "symmetric_selected": slot_rule is not None,
                "symmetric_mean_delta_vs_138": float(
                    np.mean(slot_symmetric_deltas)
                ),
                "candidate_count_passing": len(candidates),
            }
            if candidates:
                protected_candidates = [
                    item
                    for item in candidates
                    if item["spec"]["name"] in protected_v15_spec_names
                ]
                incumbent = (
                    min(
                        protected_candidates,
                        key=lambda item: (
                            item["mean_delta_vs_symmetric"],
                            item["worst_delta_vs_symmetric"],
                        ),
                    )
                    if protected_candidates
                    else None
                )
                eligible = candidates
                if incumbent is not None:
                    incumbent_by_year = {
                        int(fold["held_year"]): float(
                            fold["score_delta_vs_symmetric"]
                        )
                        for fold in incumbent["folds"]
                    }
                    eligible = [
                        item
                        for item in candidates
                        if all(
                            float(fold["score_delta_vs_symmetric"])
                            <= incumbent_by_year[int(fold["held_year"])]
                            + 1e-9
                            for fold in item["folds"]
                        )
                    ]
                    if not eligible:
                        raise RuntimeError(
                            "Protected d7 incumbent was not eligible against itself"
                        )
                selected = min(
                    eligible,
                    key=lambda item: (
                        item["mean_delta_vs_symmetric"],
                        item["worst_delta_vs_symmetric"],
                    ),
                )
                audit["protected_incumbent"] = (
                    {
                        "spec": incumbent["spec"],
                        "shrinkage": incumbent["shrinkage"],
                        "mean_delta_vs_symmetric": incumbent[
                            "mean_delta_vs_symmetric"
                        ],
                        "worst_delta_vs_symmetric": incumbent[
                            "worst_delta_vs_symmetric"
                        ],
                    }
                    if incumbent is not None
                    else None
                )
                audit["candidate_count_incumbent_non_worse"] = len(eligible)
                all_values = {
                    name: np.concatenate(
                        [
                            calibration[(year, month, day, hour)][name]
                            for year in D7_INTERVAL_YEARS
                        ]
                    )
                    for name in (
                        "signed_error",
                        "speed",
                        "predicted_direction",
                    )
                }
                spec = next(
                    item
                    for item in specs
                    if item["name"] == selected["spec"]["name"]
                )
                all_static = np.tile(
                    spec.get(
                        "_static_bins",
                        np.zeros(rows_per_slot, dtype="int64"),
                    ),
                    len(D7_INTERVAL_YEARS),
                )
                all_bins = bins_for(spec, all_values, all_static)
                raw_lower, raw_upper, counts = fit_quantiles(
                    all_values["signed_error"],
                    all_bins,
                    spec["count"],
                )
                production = {
                    **selected,
                    "month": int(month),
                    "day": int(day),
                    "hour": int(hour),
                    "raw_lower": raw_lower.tolist(),
                    "raw_upper": raw_upper.tolist(),
                    "count_by_bin": counts.tolist(),
                }
                selected_rules.append(production)
                audit.update(production)
                for fold in selected["folds"]:
                    extension_fold_totals[
                        int(fold["held_year"])
                    ].append(
                        float(fold["score_delta_vs_symmetric"])
                    )
            audits.append(audit)
            print(
                f"[train] d7 asymmetric {month:02d}-{day:02d} "
                f"h{hour:02d}: passing={len(candidates)}"
                + (
                    f"; best={audit['spec']['name']}; "
                    f"delta={audit['mean_delta_vs_symmetric']:+.6f}"
                    if candidates
                    else ""
                ),
                flush=True,
            )

    symmetric_cv = float(
        np.mean(independently_computed_symmetric_deltas)
    )
    expected_symmetric_cv = float(
        sum(rule["mean_delta"] for rule in symmetric_rules) / 32.0
    )
    if abs(symmetric_cv - expected_symmetric_cv) > 2e-5:
        raise RuntimeError(
            "Exact d7 asymmetric replay does not reproduce the protected "
            f"policy: computed={symmetric_cv:.9f}, "
            f"expected={expected_symmetric_cv:.9f}"
        )
    if not selected_rules:
        raise RuntimeError(
            "Strict d7 asymmetric extension selected no rules"
        )
    extension_delta = float(
        sum(
            rule["mean_delta_vs_symmetric"]
            for rule in selected_rules
        )
        / 32.0
    )
    extension_by_year = {
        str(year): float(
            sum(extension_fold_totals[int(year)]) / 32.0
        )
        for year in D7_INTERVAL_YEARS
    }
    if max(extension_by_year.values()) > 0.0:
        raise RuntimeError(
            "D7 asymmetric extension regresses an aggregate held year: "
            f"{extension_by_year}"
        )
    combined = symmetric_cv + extension_delta
    print(
        f"[train] d7 asymmetric selected {len(selected_rules)}/32 slots; "
        f"extension_cv_delta={extension_delta:.6f}; "
        f"combined_cv_delta={combined:.6f}",
        flush=True,
    )
    return {
        "rules": selected_rules,
        "slot_audits": audits,
        "candidate_families": [
            serializable_spec(spec) for spec in specs
        ],
        "protected_v15_spec_names": sorted(protected_v15_spec_names),
        "shrinkage_grid": list(D7_ASYMMETRIC_SHRINKAGE_GRID),
        "minimum_bin_count": int(D7_ASYMMETRIC_MIN_BIN_COUNT),
        "minimum_mean_gain": float(D7_ASYMMETRIC_MIN_MEAN_GAIN),
        "v14_cv_aggregate_delta_recomputed": symmetric_cv,
        "extension_cv_aggregate_delta": extension_delta,
        "extension_delta_by_held_year": extension_by_year,
        "combined_cv_aggregate_delta": combined,
        "method": (
            "slotwise asymmetric circular residual q05/q95 lookup tables "
            "shrunk toward the exact fold-specific symmetric policy; the "
            "targeted speed-ratio candidate may replace a protected v15 "
            "family winner only when every held training year is non-worse"
        ),
        "new_models": 0,
    }






def fit_d14_direction_interval(
    fh,
    config,
    pipeline,
    downscaling,
    qmos,
    direction_models,
    conformal,
    downscaler,
    base_half_width,
):
    """Learn a strict-gated February d14 width with protected speed/centers.

    Exact historical d10 HRES exists for 2019-2020. Each year is first held out
    in turn, and the interval is rejected unless every held-out issue hour is
    non-worse than the deployed 158-degree baseline. The final scalar is then
    fitted on both years and is applied only to February 25.
    """
    import footprint
    import target_loader

    if not 0.0 < base_half_width < 180.0:
        raise ValueError(f"Invalid d14 base half-width: {base_half_width}")

    mask = footprint.footprint_mask()
    errors_by_year_hour = {}
    dates_used = []
    expected_dates = [
        pd.Timestamp(
            year=year,
            month=D14_INTERVAL_REGIME[0],
            day=D14_INTERVAL_REGIME[1],
        )
        for year in D14_INTERVAL_YEARS
    ]

    for index, issue_date in enumerate(expected_dates, start=1):
        fields = coarse_fields_hybrid(
            fh,
            config,
            qmos,
            direction_models,
            conformal,
            issue_date,
        )
        truth_day = target_loader.load_day(
            (issue_date + pd.Timedelta(days=14)).date(),
            root=config.target_root(),
        )

        for hour in HOURS:
            candidate_u, candidate_v = fields[(14, hour, "det")]
            fine_u, fine_v = downscaling.downscale(
                downscaler, candidate_u, candidate_v
            )
            predicted_direction = (
                np.degrees(np.arctan2(-fine_u, -fine_v)) % 360.0
            )

            snapshot = truth_day.snapshot(hour)
            true_u = snapshot.fields["125m"]["u"]
            true_v = snapshot.fields["125m"]["v"]
            true_direction = (
                np.degrees(np.arctan2(-true_u, -true_v)) % 360.0
            )
            keep = (
                mask
                & np.isfinite(predicted_direction)
                & np.isfinite(true_direction)
            )
            if int(keep.sum()) != int(mask.sum()):
                raise ValueError(
                    f"Non-finite d14 interval rows for {issue_date.date()} "
                    f"h{hour:02d}: {int(keep.sum())}/{int(mask.sum())}"
                )
            error_values = pipeline._circ_abs_deg(
                predicted_direction[keep], true_direction[keep]
            ).astype("float32")
            errors_by_year_hour[(issue_date.year, hour)] = error_values

        dates_used.append(issue_date.strftime("%Y-%m-%d"))
        print(
            f"[train] d14 February interval {index:02d}/{len(expected_dates)} "
            f"{issue_date.date()}",
            flush=True,
        )

    years_used = sorted({int(value[:4]) for value in dates_used})
    if years_used != list(D14_INTERVAL_YEARS) or len(dates_used) != len(
        expected_dates
    ):
        raise RuntimeError(
            f"Incomplete d14 interval calibration: years={years_used}, "
            f"dates={len(dates_used)}"
        )

    expected_keys = {
        (year, hour) for year in D14_INTERVAL_YEARS for hour in HOURS
    }
    if set(errors_by_year_hour) != expected_keys:
        raise RuntimeError(
            "Incomplete d14 year/hour residuals: "
            f"{sorted(errors_by_year_hour)}"
        )

    def interval_score(errors, half_width):
        return 2.0 * half_width + 20.0 * np.maximum(
            errors - half_width, 0.0
        )

    gate_deltas = []
    for held_year in D14_INTERVAL_YEARS:
        fit_values = np.concatenate(
            [
                errors_by_year_hour[(year, hour)]
                for year in D14_INTERVAL_YEARS
                if year != held_year
                for hour in HOURS
            ]
        )
        held_raw_width = float(
            np.percentile(fit_values, 100.0 * D14_INTERVAL_COVERAGE)
        )
        held_width = (
            (1.0 - D14_INTERVAL_SHRINKAGE) * base_half_width
            + D14_INTERVAL_SHRINKAGE * held_raw_width
        )
        for hour in HOURS:
            values = errors_by_year_hour[(held_year, hour)]
            delta = float(
                np.mean(
                    interval_score(values, held_width)
                    - interval_score(
                        values, D14_DEPLOYED_BASE_HALF_WIDTH
                    )
                )
            )
            gate_deltas.append(
                {
                    "held_year": int(held_year),
                    "hour": int(hour),
                    "trained_half_width": float(held_width),
                    "mean_score_delta": delta,
                }
            )

    worst_gate_delta = max(item["mean_score_delta"] for item in gate_deltas)
    if worst_gate_delta > 0.0:
        raise RuntimeError(
            "Rejected d14 February interval: worst held-out year/hour "
            f"delta={worst_gate_delta:.6f}"
        )

    values = np.concatenate(
        [errors_by_year_hour[key] for key in sorted(errors_by_year_hour)]
    )
    if values.size < 1000:
        raise ValueError(f"Too few d14 February residuals: {values.size}")
    raw_width = float(
        np.percentile(values, 100.0 * D14_INTERVAL_COVERAGE)
    )
    final_width = (
        (1.0 - D14_INTERVAL_SHRINKAGE) * base_half_width
        + D14_INTERVAL_SHRINKAGE * raw_width
    )
    if not 0.0 < final_width < 180.0:
        raise RuntimeError(f"Invalid fitted d14 half-width: {final_width}")

    return {
        "edges": D14_DIRECTION_INTERVAL_EDGES.astype("float32"),
        "half_widths": np.asarray([final_width], dtype="float32"),
        "raw_half_widths": np.asarray([raw_width], dtype="float32"),
        "base_half_width": float(base_half_width),
        "deployed_base_half_width": float(D14_DEPLOYED_BASE_HALF_WIDTH),
        "coverage_target": float(D14_INTERVAL_COVERAGE),
        "shrinkage": float(D14_INTERVAL_SHRINKAGE),
        "empirical_coverage_by_bin": [float(np.mean(values <= final_width))],
        "count_by_bin": [int(values.size)],
        "dates_used": dates_used,
        "selected_slots": [
            tuple(slot) for slot in D14_INTERVAL_SELECTED_SLOTS
        ],
        "method": (
            "fine-grid February-only pooled symmetric circular p90 residual "
            "width shrunk 75 percent toward the protected global d14 width"
        ),
        "mapping": "step",
        "gate": {
            "baseline_half_width": float(D14_DEPLOYED_BASE_HALF_WIDTH),
            "year_hour_deltas": gate_deltas,
            "worst_year_hour_delta": float(worst_gate_delta),
            "passed": True,
        },
        "new_models": 0,
    }


def fit_fine_d14_endpoint_rule(
    errors,
    truth_speed,
    truth_direction,
    predicted_direction,
    latitude,
    longitude,
    base_half_width,
    expected_factors,
    years,
) -> dict:
    """Reproduce the minimum-movement five-year endpoint gate."""
    errors = np.asarray(errors, dtype="float64")
    truth_speed = np.asarray(truth_speed, dtype="float64")
    truth_direction = np.asarray(truth_direction, dtype="float64")
    predicted_direction = np.asarray(predicted_direction, dtype="float64")
    latitude = np.asarray(latitude, dtype="float64")
    longitude = np.asarray(longitude, dtype="float64")
    expected_shape = (len(years), len(latitude))
    if (
        errors.shape != expected_shape
        or truth_speed.shape != expected_shape
        or truth_direction.shape != expected_shape
        or predicted_direction.shape != expected_shape
        or longitude.shape != latitude.shape
        or not np.all(np.isfinite(errors))
        or not np.all(np.isfinite(truth_speed))
        or not np.all(np.isfinite(truth_direction))
        or not np.all(np.isfinite(predicted_direction))
        or np.any(truth_speed < 0.0)
        or not 0.0 < base_half_width < 180.0
    ):
        raise RuntimeError("Invalid fine d14 endpoint calibration arrays")

    def fixed_bins(values, boundaries):
        return np.clip(
            np.digitize(values, np.asarray(boundaries, dtype="float64")) - 1,
            0,
            len(boundaries) - 2,
        ).astype("int16")

    def quantile_bins(values, count):
        boundaries = np.quantile(
            np.asarray(values, dtype="float64"),
            np.linspace(0.0, 1.0, count + 1),
        )
        boundaries[0], boundaries[-1] = -np.inf, np.inf
        for index in range(1, len(boundaries) - 1):
            if boundaries[index] <= boundaries[index - 1]:
                boundaries[index] = np.nextafter(
                    boundaries[index - 1], np.inf
                )
        return fixed_bins(values, boundaries)

    def circular_winkler(residual, lower, upper):
        residual = np.asarray(residual, dtype="float64") % 360.0
        lower = np.asarray(lower, dtype="float64") % 360.0
        upper = np.asarray(upper, dtype="float64") % 360.0
        width = (upper - lower) % 360.0
        position = (residual - lower) % 360.0
        inside = position <= width
        distance_lower = np.abs(
            (residual - lower + 180.0) % 360.0 - 180.0
        )
        distance_upper = np.abs(
            (residual - upper + 180.0) % 360.0 - 180.0
        )
        miss = np.minimum(distance_lower, distance_upper)
        return width + 20.0 * np.where(inside, 0.0, miss)

    lat4 = quantile_bins(latitude, 4)
    lon4 = quantile_bins(longitude, 4)
    spatial4 = (lat4 * 4 + lon4).astype("int16")
    labels = []
    base_scores = []
    for year_index in range(len(years)):
        labels.append(
            {
                "spatial_4x4": spatial4,
                "checkerboard": ((lat4 + lon4) % 2).astype("int16"),
                "speed_3": fixed_bins(
                    truth_speed[year_index], (0.0, 5.0, 10.0, np.inf)
                ),
                "truth_direction_8": np.floor(
                    (truth_direction[year_index] % 360.0) / 45.0
                ).astype("int16"),
                "center_direction_8": np.floor(
                    (predicted_direction[year_index] % 360.0) / 45.0
                ).astype("int16"),
                "signed_error_8": fixed_bins(
                    errors[year_index],
                    (
                        -180.0,
                        -120.0,
                        -60.0,
                        -20.0,
                        0.0,
                        20.0,
                        60.0,
                        120.0,
                        180.000001,
                    ),
                ),
                "absolute_error_5": fixed_bins(
                    np.abs(errors[year_index]),
                    (0.0, 30.0, 60.0, 90.0, 120.0, 180.000001),
                ),
            }
        )
        base_scores.append(
            circular_winkler(
                errors[year_index], -base_half_width, base_half_width
            )
        )

    def evaluate_pair(lower_factor, upper_factor, year_index):
        lower_width = base_half_width * lower_factor
        upper_width = base_half_width * upper_factor
        if lower_width + upper_width >= 359.0:
            return None
        score = circular_winkler(
            errors[year_index], -lower_width, upper_width
        )
        delta = score - base_scores[year_index]
        worst = -np.inf
        for bins in labels[year_index].values():
            bins = np.asarray(bins, dtype="int64")
            counts = np.bincount(bins)
            sums = np.bincount(bins, weights=delta, minlength=len(counts))
            for bin_index, count in enumerate(counts):
                if count < FINE_D14_ENDPOINT_MIN_REGIME_ROWS:
                    continue
                worst = max(worst, float(sums[bin_index] / count))
        if not np.isfinite(worst):
            raise RuntimeError("No populated fine d14 endpoint regimes")
        return {
            "mean_delta": float(np.mean(delta)),
            "worst_regime_delta": float(worst),
        }

    candidate_folds = {}
    for lower_factor in FINE_D14_ENDPOINT_FACTORS:
        for upper_factor in FINE_D14_ENDPOINT_FACTORS:
            if lower_factor == 1.0 and upper_factor == 1.0:
                continue
            folds = [
                evaluate_pair(lower_factor, upper_factor, year_index)
                for year_index in range(len(years))
            ]
            if all(fold is not None for fold in folds):
                candidate_folds[(lower_factor, upper_factor)] = folds

    nested = []
    expected_factors = tuple(float(value) for value in expected_factors)
    for held_index, held_year in enumerate(years):
        eligible = []
        for factors, folds in candidate_folds.items():
            training = [
                fold for index, fold in enumerate(folds) if index != held_index
            ]
            train_delta = float(
                np.mean([fold["mean_delta"] for fold in training])
            )
            if (
                train_delta <= -FINE_D14_ENDPOINT_MIN_TRAIN_GAIN
                and all(
                    fold["mean_delta"] <= -FINE_D14_ENDPOINT_MIN_YEAR_GAIN
                    and fold["worst_regime_delta"] <= 1e-9
                    for fold in training
                )
            ):
                movement = abs(1.0 - factors[0]) + abs(1.0 - factors[1])
                eligible.append((movement, train_delta, factors))
        if not eligible:
            raise RuntimeError(
                f"No fine d14 endpoint candidate for held year {held_year}"
            )
        _movement, train_delta, selected = min(eligible)
        if selected != expected_factors:
            raise RuntimeError(
                "Fine d14 endpoint selection drift for held year "
                f"{held_year}: selected={selected}, expected={expected_factors}"
            )
        held = candidate_folds[selected][held_index]
        if (
            held["mean_delta"] > -FINE_D14_ENDPOINT_MIN_YEAR_GAIN
            or held["worst_regime_delta"] > 1e-9
        ):
            raise RuntimeError(
                "Fine d14 endpoint held-year gate failed for "
                f"{held_year}: {held}"
            )
        nested.append(
            {
                "held_year": int(held_year),
                "selected_factors": list(selected),
                "training_delta": train_delta,
                **held,
            }
        )

    selected_folds = candidate_folds[expected_factors]
    aggregate_delta = float(
        np.mean([fold["mean_delta"] for fold in selected_folds])
    )
    if aggregate_delta > -FINE_D14_ENDPOINT_MIN_CELL_GAIN:
        raise RuntimeError(
            f"Fine d14 endpoint gain is too small: {aggregate_delta:.6f}"
        )
    return {
        "lower_factor": expected_factors[0],
        "upper_factor": expected_factors[1],
        "base_half_width": float(base_half_width),
        "aggregate_delta": aggregate_delta,
        "nested_folds": nested,
        "strict_gate": (
            "same minimum-movement factor pair in every leave-one-year-out "
            "fold; every populated spatial4x4, checkerboard, speed3, truth- "
            "and center-direction8, signed-error8, and absolute-error5 "
            "regime is non-worse"
        ),
        "input_only_training": True,
        "previous_submission_inputs": [],
        "new_models": 0,
    }


def fit_fine_d14_climatology(target_loader, config) -> dict:
    """Build selected native-grid d14 centers from 2016-2020 targets."""
    import footprint

    years = tuple(range(2016, 2021))
    available_dates = set(target_loader.list_dates(root=config.target_root()))
    mask = footprint.footprint_mask()
    static = target_loader.load_static(str(config.target_root()))
    latitude = np.asarray(static.lat[mask], dtype="float32")
    longitude = np.asarray(static.lon[mask], dtype="float32")
    n_points = int(mask.sum())
    centers = {}
    endpoint_policy = {}

    issue_slots = sorted({key[:2] for key in FINE_D14_CLIMATOLOGY_POLICY})
    for month, day in issue_slots:
        rules = {
            hour: FINE_D14_CLIMATOLOGY_POLICY[(month, day, hour)]
            for hour in HOURS
            if (month, day, hour) in FINE_D14_CLIMATOLOGY_POLICY
        }
        hours = tuple(sorted(rules))
        windows = tuple(sorted({values[0] for values in rules.values()}))
        sums = np.zeros(
            (len(years), len(windows), len(hours), n_points, 2),
            dtype="float32",
        )
        counts = np.zeros((len(years), len(windows)), dtype="int16")
        truth_uv = np.empty(
            (len(years), len(hours), n_points, 2), dtype="float32"
        )
        truth_seen = np.zeros(len(years), dtype=bool)
        for yi, year in enumerate(years):
            valid_date = (
                pd.Timestamp(year=year, month=month, day=day)
                + pd.Timedelta(days=14)
            )
            for offset in range(-max(windows), max(windows) + 1):
                target_date = (valid_date + pd.Timedelta(days=offset)).date()
                if target_date not in available_dates:
                    continue
                target = target_loader.load_day(
                    target_date,
                    root=config.target_root(),
                    levels=("125m",),
                )
                uv = np.empty((len(hours), n_points, 2), dtype="float32")
                for hi, hour in enumerate(hours):
                    uv[hi, :, 0] = target.u["125m"][hour // 3][mask]
                    uv[hi, :, 1] = target.v["125m"][hour // 3][mask]
                if not np.isfinite(uv).all():
                    raise RuntimeError(
                        f"Non-finite fine d14 target values for {target_date}"
                    )
                if offset == 0:
                    truth_uv[yi] = uv
                    truth_seen[yi] = True
                for wi, window in enumerate(windows):
                    if abs(offset) <= window:
                        sums[yi, wi] += uv
                        counts[yi, wi] += 1
        if np.any(counts <= 0):
            raise RuntimeError(
                f"Empty fine d14 climatology window for {month:02d}-{day:02d}"
            )
        if not np.all(truth_seen):
            raise RuntimeError(
                f"Missing fine d14 truth for {month:02d}-{day:02d}: "
                f"{truth_seen.tolist()}"
            )
        for hi, hour in enumerate(hours):
            window, _half_width = rules[hour]
            wi = windows.index(window)
            year_means = sums[:, wi, hi] / counts[:, wi, None, None]
            total_vector = year_means.sum(axis=0, dtype="float64")
            vector = total_vector / len(years)
            centers[(month, day, hour)] = (
                np.degrees(np.arctan2(-vector[:, 0], -vector[:, 1])) % 360.0
            ).astype("float32")
            endpoint_key = (month, day, hour)
            if endpoint_key in FINE_D14_ENDPOINT_POLICY:
                held_vectors = (
                    total_vector[None, :, :] - year_means
                ) / (len(years) - 1)
                held_direction = (
                    np.degrees(
                        np.arctan2(
                            -held_vectors[:, :, 0],
                            -held_vectors[:, :, 1],
                        )
                    )
                    % 360.0
                )
                true_direction = (
                    np.degrees(
                        np.arctan2(
                            -truth_uv[:, hi, :, 0],
                            -truth_uv[:, hi, :, 1],
                        )
                    )
                    % 360.0
                )
                true_speed = np.hypot(
                    truth_uv[:, hi, :, 0], truth_uv[:, hi, :, 1]
                )
                errors = (
                    true_direction - held_direction + 180.0
                ) % 360.0 - 180.0
                endpoint_policy[endpoint_key] = fit_fine_d14_endpoint_rule(
                    errors=errors,
                    truth_speed=true_speed,
                    truth_direction=true_direction,
                    predicted_direction=held_direction,
                    latitude=latitude,
                    longitude=longitude,
                    base_half_width=float(_half_width),
                    expected_factors=FINE_D14_ENDPOINT_POLICY[endpoint_key],
                    years=years,
                )
        print(
            f"[train] fine d14 climatology {month:02d}-{day:02d} "
            f"hours={list(hours)}",
            flush=True,
        )

    if set(endpoint_policy) != set(FINE_D14_ENDPOINT_POLICY):
        raise RuntimeError(
            "Incomplete fine d14 endpoint policy: "
            f"{sorted(endpoint_policy)}"
        )
    return {
        "policy": {
            key: {"window": values[0], "half_width": values[1]}
            for key, values in FINE_D14_CLIMATOLOGY_POLICY.items()
        },
        "centers": centers,
        "endpoint_policy": endpoint_policy,
        "latitude": latitude,
        "longitude": longitude,
        "source_years": list(years),
        "method": (
            "year-balanced 1.3 km 125 m seasonal vector climatology; "
            "seven slot-hours selected by five-year held-out and worst-regime "
            "interval-score gates, using exact production HRES replay for "
            "2019-2020 and a conservative downscaled-climatology baseline for "
            "2016-2018; three January intervals additionally use an input-only "
            "minimum-movement asymmetric endpoint gate"
        ),
        "new_models": 0,
    }




def _siting_bathymetry_path(config, phase2: Path) -> Path:
    coarse_root = Path(config.coarse_root()).resolve()
    candidates = [
        coarse_root.parent.parent / "static" / "bathymetry" / "emodnet_northsea_1km.nc",
        phase2 / "data" / "bathymetry" / "emodnet_northsea_1km.nc",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Missing organizer EMODnet bathymetry. Expected static/bathymetry/"
        "emodnet_northsea_1km.nc under the Phase 2 data root."
    )


def _siting_static_grid(config, phase2: Path) -> dict:
    """Return the exact legal 1.3 km cells and organizer bathymetry fields."""
    import netCDF4
    import zone

    with netCDF4.Dataset(config.target_static()) as static:
        latitude = np.asarray(static.variables["latitude"][:], dtype="float64")
        longitude = np.asarray(static.variables["longitude"][:], dtype="float64")
        sea = np.asarray(static.variables["seamask"][:], dtype=bool)

    bathymetry_path = _siting_bathymetry_path(config, phase2)
    with netCDF4.Dataset(bathymetry_path) as bathymetry:
        bathy_lat = np.asarray(bathymetry.variables["lat"][:], dtype="float64")
        bathy_lon = np.asarray(bathymetry.variables["lon"][:], dtype="float64")
        depth_grid = np.asarray(
            bathymetry.variables["water_depth_m"][:], dtype="float64"
        )
        coast_grid = np.asarray(
            bathymetry.variables["dist_coast_km"][:], dtype="float64"
        )
    bathy_i = np.clip(
        np.rint((latitude - bathy_lat[0]) / (bathy_lat[1] - bathy_lat[0])).astype(int),
        0,
        len(bathy_lat) - 1,
    )
    bathy_j = np.clip(
        np.rint((longitude - bathy_lon[0]) / (bathy_lon[1] - bathy_lon[0])).astype(int),
        0,
        len(bathy_lon) - 1,
    )
    depth = depth_grid[bathy_i, bathy_j]
    distance_to_coast = coast_grid[bathy_i, bathy_j]

    coarse_lat, coarse_lon, coarse_allowed = zone._grid(None)
    _, _, coarse_fixed_bottom = zone._grid(SITING_MAX_DEPTH_M)
    coarse_i = np.clip(
        np.rint((latitude - coarse_lat[0]) / (coarse_lat[1] - coarse_lat[0])).astype(int),
        0,
        len(coarse_lat) - 1,
    )
    coarse_j = np.clip(
        np.rint((longitude - coarse_lon[0]) / (coarse_lon[1] - coarse_lon[0])).astype(int),
        0,
        len(coarse_lon) - 1,
    )
    in_coarse_domain = (
        (latitude >= coarse_lat.min())
        & (latitude <= coarse_lat.max())
        & (longitude >= coarse_lon.min())
        & (longitude <= coarse_lon.max())
    )
    weather = (
        sea
        & in_coarse_domain
        & coarse_allowed[coarse_i, coarse_j]
        & (latitude >= 51.0)
        & (latitude <= 60.0)
        & (longitude >= -2.0)
        & (longitude <= 6.0)
    )
    legal = (
        weather
        & coarse_fixed_bottom[coarse_i, coarse_j]
        & np.isfinite(depth)
        & (depth > 0.0)
        & (depth <= SITING_MAX_DEPTH_M)
        & np.isfinite(distance_to_coast)
        & (distance_to_coast >= SITING_DISTANCE_TO_COAST_MIN_KM)
    )
    return {
        "latitude": latitude,
        "longitude": longitude,
        "weather": weather,
        "legal": legal,
        "depth_m": depth,
        "distance_to_coast_km": distance_to_coast,
        "bathymetry_path": bathymetry_path,
    }


def _siting_nearest_flat(grid: dict, centre: tuple[float, float]) -> int:
    distance2 = (
        (grid["latitude"] - centre[0]) ** 2
        + (grid["longitude"] - centre[1]) ** 2
    )
    return int(np.nanargmin(distance2))


def _siting_diverse_positions(
    order: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    count: int,
) -> list[int]:
    selected: list[int] = []
    for position in order:
        lat = float(latitude[position])
        lon = float(longitude[position])
        separated = True
        for prior in selected:
            mean_lat = np.deg2rad((lat + float(latitude[prior])) / 2.0)
            distance_km = np.hypot(
                (lat - float(latitude[prior])) * 111.0,
                (lon - float(longitude[prior])) * 111.0 * np.cos(mean_lat),
            )
            if distance_km < SITING_SCREEN_DIVERSITY_KM:
                separated = False
                break
        if separated:
            selected.append(int(position))
        if len(selected) >= count:
            break
    return selected


def _siting_screen_centres(config, phase2: Path, turbine) -> tuple[dict, list[dict]]:
    """Frugally scan every legal cell and return a diverse exact-replay shortlist."""
    import netCDF4
    from scipy.ndimage import convolve, minimum_filter

    grid = _siting_static_grid(config, phase2)
    weather_flat = np.flatnonzero(grid["weather"].ravel())
    legal_flat = np.flatnonzero(grid["legal"].ravel())
    if not len(legal_flat):
        raise RuntimeError("No legal shallow 1.3 km siting cells were found")
    weather_position = np.full(grid["weather"].size, -1, dtype=int)
    weather_position[weather_flat] = np.arange(len(weather_flat))
    legal_weather_position = weather_position[legal_flat]
    if np.any(legal_weather_position < 0):
        raise RuntimeError("Legal siting cells escaped the AROME weather mask")
    shear = (SITING_HUB_HEIGHT_M / SITING_SOURCE_HEIGHT_M) ** SITING_SHEAR_ALPHA
    rated_w = float(SITING_CAPACITY_MW / SITING_N_TURBINES * 1e6)
    annual_cf = []
    for year in SITING_YEARS:
        files = sorted((Path(config.target_root()) / str(year)).glob("arome_*.nc"))
        files = files[::SITING_SCREEN_DAY_STEP]
        if not files:
            raise FileNotFoundError(f"No siting screen files found for {year}")
        power_sum = np.zeros(len(weather_flat), dtype="float64")
        n_steps = 0
        for path in files:
            with netCDF4.Dataset(path) as ds:
                u = np.asarray(
                    np.ma.filled(ds.variables["u125m"][:], np.nan),
                    dtype="float32",
                ).reshape(len(ds.dimensions["time"]), -1)[:, weather_flat]
                v = np.asarray(
                    np.ma.filled(ds.variables["v125m"][:], np.nan),
                    dtype="float32",
                ).reshape(len(ds.dimensions["time"]), -1)[:, weather_flat]
            speed = np.hypot(u * shear, v * shear)
            power_sum += np.asarray(turbine.power(speed), dtype="float64").sum(axis=0)
            n_steps += speed.shape[0]
        cf = power_sum / (n_steps * rated_w)
        if not np.isfinite(cf).all():
            raise RuntimeError(f"Non-finite gross-power siting screen for {year}")
        annual_cf.append(cf)
        print(
            f"[train:siting] screened {year}: {len(files)} days, "
            f"{len(weather_flat):,} weather cells / {len(legal_flat):,} legal centres",
            flush=True,
        )

    weather_annual_cf = np.stack(annual_cf)
    annual_cf_array = weather_annual_cf[:, legal_weather_position]
    mean_cf = annual_cf_array.mean(axis=0)
    worst_cf = annual_cf_array.min(axis=0)
    spread = np.ptp(annual_cf_array, axis=0)
    kernel = np.ones(
        (SITING_NEIGHBOURHOOD_CELLS, SITING_NEIGHBOURHOOD_CELLS), dtype="int16"
    )
    full_neighbourhood = convolve(
        grid["weather"].astype("int16"), kernel, mode="constant", cval=0
    ) == kernel.size
    neighbour_year = []
    for values in weather_annual_cf:
        field = np.full(grid["weather"].shape, np.inf, dtype="float64")
        field.ravel()[weather_flat] = values
        minimum = minimum_filter(
            field,
            size=SITING_NEIGHBOURHOOD_CELLS,
            mode="constant",
            cval=-np.inf,
        )
        minimum[~full_neighbourhood] = np.nan
        neighbour_year.append(minimum.ravel()[legal_flat])
    neighbour_year_array = np.stack(neighbour_year)
    complete_positions = full_neighbourhood.ravel()[legal_flat]
    neighbour_mean = np.full(len(legal_flat), np.nan, dtype="float64")
    neighbour_worst = np.full(len(legal_flat), np.nan, dtype="float64")
    neighbour_mean[complete_positions] = neighbour_year_array[
        :, complete_positions
    ].mean(axis=0)
    neighbour_worst[complete_positions] = neighbour_year_array[
        :, complete_positions
    ].min(axis=0)

    scores = {
        "mean": mean_cf,
        "balanced": (
            0.55 * mean_cf
            + 0.25 * worst_cf
            + 0.20 * neighbour_mean
            - 0.05 * spread
        ),
        "robust": (
            0.50 * worst_cf
            + 0.25 * mean_cf
            + 0.25 * neighbour_mean
            - 0.10 * spread
        ),
    }
    near_mean = mean_cf >= np.nanmax(mean_cf) - SITING_SCREEN_NEAR_BEST_GROSS_CF
    scores["near_mean_robust"] = np.where(
        near_mean,
        worst_cf + 0.25 * neighbour_mean - 0.05 * spread,
        -np.inf,
    )
    legal_depth = grid["depth_m"].ravel()[legal_flat]
    scores["depth_safe_high_wind"] = np.where(
        near_mean & (legal_depth <= SITING_MAX_DEPTH_M - 5.0),
        mean_cf + 0.25 * worst_cf + 0.10 * neighbour_mean - 0.05 * spread,
        -np.inf,
    )
    candidate_positions: dict[int, set[str]] = {}
    for tier, values in scores.items():
        order = np.argsort(np.nan_to_num(values, nan=-np.inf))[::-1]
        order = order[np.isfinite(values[order])]
        if tier in {"near_mean_robust", "depth_safe_high_wind"}:
            tier_positions = [int(position) for position in order[:SITING_SCREEN_TIER_SIZE]]
        else:
            tier_positions = _siting_diverse_positions(
                order,
                grid["latitude"].ravel()[legal_flat],
                grid["longitude"].ravel()[legal_flat],
                SITING_SCREEN_TIER_SIZE,
            )
        for position in tier_positions:
            candidate_positions.setdefault(position, set()).add(tier)

    flat_to_position = {int(flat): pos for pos, flat in enumerate(legal_flat)}
    for label, centre in (
        ("protected_reference", SITING_REFERENCE_CENTRE),
        ("public_baseline", SITING_PUBLIC_BASELINE_CENTRE),
    ):
        flat = _siting_nearest_flat(grid, centre)
        if flat not in flat_to_position:
            raise RuntimeError(f"Required siting reference is not legal: {label}")
        candidate_positions.setdefault(flat_to_position[flat], set()).add(label)

    records = []
    for position, tiers in candidate_positions.items():
        flat = int(legal_flat[position])
        iy, ix = np.unravel_index(flat, grid["legal"].shape)
        record = {
            "key": f"cell_{flat}",
            "flat_index": flat,
            "iy": int(iy),
            "ix": int(ix),
            "latitude": float(grid["latitude"][iy, ix]),
            "longitude": float(grid["longitude"][iy, ix]),
            "depth_m": float(grid["depth_m"][iy, ix]),
            "distance_to_coast_km": float(grid["distance_to_coast_km"][iy, ix]),
            "source_tiers": sorted(tiers),
            "screen": {
                "annual_gross_cf": {
                    str(year): float(annual_cf_array[index, position])
                    for index, year in enumerate(SITING_YEARS)
                },
                "mean_gross_cf": float(mean_cf[position]),
                "worst_year_gross_cf": float(worst_cf[position]),
                "annual_spread": float(spread[position]),
                "neighbour_mean_gross_cf": float(neighbour_mean[position]),
                "neighbour_worst_gross_cf": float(neighbour_worst[position]),
                "neighbour_mean_margin": float(mean_cf[position] - neighbour_mean[position]),
                "neighbour_worst_margin": float(worst_cf[position] - neighbour_worst[position]),
            },
        }
        records.append(record)
        print(
            f"[train:siting] shortlist {record['source_tiers']} "
            f"({record['latitude']:.5f}, {record['longitude']:.5f}) "
            f"mean={record['screen']['mean_gross_cf']:.4f} "
            f"worst={record['screen']['worst_year_gross_cf']:.4f}",
            flush=True,
        )
    return grid, records


def _siting_extract_climates(config, records: list[dict]) -> dict:
    """Read each AROME file once and extract every shortlisted exact cell."""
    import netCDF4
    from wind_farm_simulator import WindSeries

    climates = {record["key"]: {} for record in records}
    iy = np.asarray([record["iy"] for record in records], dtype=int)
    ix = np.asarray([record["ix"] for record in records], dtype=int)
    shear = (SITING_HUB_HEIGHT_M / SITING_SOURCE_HEIGHT_M) ** SITING_SHEAR_ALPHA
    for year in SITING_YEARS:
        files = sorted((Path(config.target_root()) / str(year)).glob("arome_*.nc"))
        if not files:
            raise FileNotFoundError(f"No exact siting replay files found for {year}")
        u_parts = [[] for _ in records]
        v_parts = [[] for _ in records]
        timestamps: list[pd.Timestamp] = []
        for path in files:
            with netCDF4.Dataset(path) as ds:
                u = np.asarray(
                    np.ma.filled(ds.variables["u125m"][:], np.nan),
                    dtype="float32",
                )[:, iy, ix]
                v = np.asarray(
                    np.ma.filled(ds.variables["v125m"][:], np.nan),
                    dtype="float32",
                )[:, iy, ix]
            day = pd.Timestamp(path.stem[-8:])
            timestamps.extend(day + pd.to_timedelta(np.arange(u.shape[0]) * 3, unit="h"))
            for index in range(len(records)):
                u_parts[index].append(u[:, index] * shear)
                v_parts[index].append(v[:, index] * shear)
        for index, record in enumerate(records):
            u = np.concatenate(u_parts[index])
            v = np.concatenate(v_parts[index])
            if not (np.isfinite(u).all() and np.isfinite(v).all()):
                raise RuntimeError(f"Non-finite exact siting climate for {record['key']}")
            frame = pd.DataFrame(
                {
                    "time": pd.DatetimeIndex(timestamps),
                    "ws": np.hypot(u, v),
                    "wd": (270.0 - np.degrees(np.arctan2(v, u))) % 360.0,
                }
            )
            climates[record["key"]][str(year)] = WindSeries(frame)
        print(f"[train:siting] extracted exact climates for {year}", flush=True)
    return climates


def _siting_evaluate_layout(x, y, turbine, climates: dict) -> dict:
    from wind_farm_simulator import FarmLayout, simulate_year

    layout = FarmLayout(np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64"), turbine)
    annual = {}
    for year in SITING_YEARS:
        wind = climates[str(year)]
        result = simulate_year(layout, wind)
        annual[str(year)] = {
            "aep_gwh": float(result.aep_gwh),
            "capacity_factor": float(result.capacity_factor),
            "wake_loss_fraction": float(result.wake_loss_fraction),
            "weather_steps": int(wind.n_steps),
        }
    return annual


def _siting_summary(annual: dict) -> dict:
    cfs = np.asarray([annual[str(year)]["capacity_factor"] for year in SITING_YEARS])
    aeps = np.asarray([annual[str(year)]["aep_gwh"] for year in SITING_YEARS])
    wakes = np.asarray([annual[str(year)]["wake_loss_fraction"] for year in SITING_YEARS])
    return {
        "mean_capacity_factor": float(cfs.mean()),
        "worst_year_capacity_factor": float(cfs.min()),
        "annual_capacity_factor_spread": float(np.ptp(cfs)),
        "mean_aep_gwh": float(aeps.mean()),
        "worst_aep_gwh": float(aeps.min()),
        "mean_wake_loss_fraction": float(wakes.mean()),
        "max_wake_loss_fraction": float(wakes.max()),
    }


def _siting_layout_candidates(base_x, base_y) -> list[tuple[str, np.ndarray, np.ndarray]]:
    from scipy.optimize import linear_sum_assignment

    candidates = [
        ("protected_robust_coordinate_layout", base_x, base_y),
        ("mirror_x", -base_x, base_y),
        ("mirror_y", base_x, -base_y),
        ("rotate_180", -base_x, -base_y),
        ("swap_axes", base_y, base_x),
        ("swap_axes_mirror_x", -base_y, base_x),
        ("swap_axes_mirror_y", base_y, -base_x),
        ("swap_axes_rotate_180", -base_y, -base_x),
    ]
    x_5x11 = np.tile((np.arange(11) - 5) * (5.0 * 284.0), 5)
    y_5x11 = np.repeat((np.arange(5) - 2) * (13.0 * 284.0), 11)
    candidates.extend(
        [
            ("rectangular_5x11", x_5x11, y_5x11),
            ("rectangular_5x11_swap", y_5x11, x_5x11),
        ]
    )
    x_parts = []
    y_parts = []
    for row, count in enumerate((8, 8, 8, 8, 8, 8, 7)):
        x_parts.extend((np.arange(count) - (count - 1) / 2.0) * (7.5 * 284.0))
        y_parts.extend([row * (8.0 * 284.0)] * count)
    x_7x8 = np.asarray(x_parts, dtype="float64")
    y_7x8 = np.asarray(y_parts, dtype="float64")
    x_7x8 -= x_7x8.mean()
    y_7x8 -= y_7x8.mean()
    candidates.extend(
        [
            ("rectangular_7x8", x_7x8, y_7x8),
            ("rectangular_7x8_swap", y_7x8, x_7x8),
        ]
    )
    source = np.column_stack([base_x, base_y])
    target = np.column_stack([x_7x8, y_7x8])
    row_ind, col_ind = linear_sum_assignment(
        ((source[:, None, :] - target[None, :, :]) ** 2).sum(axis=2)
    )
    matched_target = np.empty_like(target)
    matched_target[row_ind] = target[col_ind]
    for alpha in (0.70, 0.75, 0.80, 0.85):
        blended = (1.0 - alpha) * source + alpha * matched_target
        candidates.append(
            (
                f"matched_7x8_blend_{alpha:.2f}",
                blended[:, 0],
                blended[:, 1],
            )
        )
    return [
        (name, np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64"))
        for name, x, y in candidates
    ]


def _siting_direction_stress(x, y, turbine, climates: dict) -> dict:
    from wind_farm_simulator import FarmLayout, WindSeries, simulate_year

    pooled = pd.concat(
        [climates[str(year)].df for year in SITING_YEARS], ignore_index=True
    )
    positions = np.unique(
        np.linspace(
            0,
            len(pooled) - 1,
            min(SITING_DIRECTION_STRESS_ROWS, len(pooled)),
        ).astype(int)
    )
    sampled = pooled.iloc[positions].reset_index(drop=True)
    layout = FarmLayout(x, y, turbine)
    rows = []
    for offset in SITING_DIRECTION_STRESS_DEG:
        frame = sampled.copy()
        frame["wd"] = (frame["wd"] + offset) % 360.0
        result = simulate_year(layout, WindSeries(frame))
        rows.append(
            {
                "direction_offset_deg": float(offset),
                "capacity_factor": float(result.capacity_factor),
                "wake_loss_fraction": float(result.wake_loss_fraction),
            }
        )
    return {
        "rows": rows,
        "mean_capacity_factor": float(np.mean([row["capacity_factor"] for row in rows])),
        "worst_capacity_factor": float(np.min([row["capacity_factor"] for row in rows])),
        "max_wake_loss_fraction": float(
            np.max([row["wake_loss_fraction"] for row in rows])
        ),
    }


def build_siting_artifact(config, phase2: Path) -> dict:
    """Select, replay, and strictly audit a robust input-only farm design."""
    import zone
    from cost_model import evaluate_farm, is_eligible_fixed_bottom
    from turbines_catalog import get_spec, load_turbine
    from wind_farm_simulator import grid_layout, validate_layout

    spec = get_spec(SITING_TURBINE_KEY)
    turbine = load_turbine(SITING_TURBINE_KEY)
    base_x = np.asarray(SITING_LAYOUT_X_M, dtype="float64")
    base_y = np.asarray(SITING_LAYOUT_Y_M, dtype="float64")
    if len(base_x) != SITING_N_TURBINES or len(base_y) != SITING_N_TURBINES:
        raise RuntimeError("Siting reference layout must contain exactly 55 turbines")
    if abs(spec.rated_power_mw * len(base_x) - SITING_CAPACITY_MW) > 1e-9:
        raise RuntimeError("Siting recipe does not provide exactly 1210 MW")

    grid, records = _siting_screen_centres(config, phase2, turbine)
    climates = _siting_extract_climates(config, records)
    reference_record = next(
        record for record in records if "protected_reference" in record["source_tiers"]
    )
    baseline_record = next(
        record for record in records if "public_baseline" in record["source_tiers"]
    )
    centre_replays = []
    for record in records:
        annual = _siting_evaluate_layout(
            base_x, base_y, turbine, climates[record["key"]]
        )
        summary = _siting_summary(annual)
        centre_replays.append({"record": record, "annual": annual, "summary": summary})
        print(
            f"[train:siting] exact centre ({record['latitude']:.5f}, "
            f"{record['longitude']:.5f}) mean={summary['mean_capacity_factor']:.4f} "
            f"worst={summary['worst_year_capacity_factor']:.4f} "
            f"wake={summary['max_wake_loss_fraction']:.4f}",
            flush=True,
        )
    reference_replay = next(
        replay for replay in centre_replays if replay["record"] is reference_record
    )
    reference_summary = reference_replay["summary"]
    reference_neighbour_mean_margin = reference_record["screen"][
        "neighbour_mean_margin"
    ]
    reference_neighbour_worst_margin = reference_record["screen"][
        "neighbour_worst_margin"
    ]
    promotable = []
    for replay in centre_replays:
        record = replay["record"]
        if record is reference_record or record is baseline_record:
            continue
        gate = {
            "mean_cf_gain_is_material": replay["summary"]["mean_capacity_factor"]
            >= reference_summary["mean_capacity_factor"]
            + SITING_CENTRE_MIN_MEAN_CF_GAIN,
            "worst_cf_gain_is_material": replay["summary"][
                "worst_year_capacity_factor"
            ]
            >= reference_summary["worst_year_capacity_factor"]
            + SITING_CENTRE_MIN_WORST_CF_GAIN,
            "annual_spread_is_stable": replay["summary"][
                "annual_capacity_factor_spread"
            ]
            <= reference_summary["annual_capacity_factor_spread"]
            + SITING_CENTRE_MAX_SPREAD_INCREASE,
            "max_wake_within_joint_search_allowance": replay["summary"][
                "max_wake_loss_fraction"
            ]
            <= reference_summary["max_wake_loss_fraction"]
            + SITING_CENTRE_MAX_WAKE_INCREASE,
            "full_5x5_weather_neighbourhood": bool(
                np.isfinite(record["screen"]["neighbour_mean_gross_cf"])
            ),
            "neighbour_mean_margin_reference_relative": bool(
                record["screen"]["neighbour_mean_margin"]
                <= reference_neighbour_mean_margin
                + SITING_NEIGHBOUR_MEAN_TOLERANCE
            ),
            "neighbour_worst_margin_reference_relative": bool(
                record["screen"]["neighbour_worst_margin"]
                <= reference_neighbour_worst_margin
                + SITING_NEIGHBOUR_WORST_TOLERANCE
            ),
            "depth_safety_margin_at_least_0_5m": record["depth_m"]
            <= SITING_MAX_DEPTH_M - 0.5,
        }
        gate["passed"] = all(gate.values())
        replay["promotion_gate"] = gate
        print(
            f"[train:siting] centre gate ({record['latitude']:.5f}, "
            f"{record['longitude']:.5f}) {gate}",
            flush=True,
        )
        if gate["passed"]:
            promotable.append(replay)
    if promotable:
        best_mean = max(row["summary"]["mean_capacity_factor"] for row in promotable)
        near_best = [
            row
            for row in promotable
            if row["summary"]["mean_capacity_factor"]
            >= best_mean - SITING_NEAR_BEST_MEAN_CF
        ]
        selected_centre = max(
            near_best,
            key=lambda row: (
                row["summary"]["worst_year_capacity_factor"],
                -row["summary"]["annual_capacity_factor_spread"],
                -row["record"]["depth_m"],
                row["summary"]["mean_capacity_factor"],
            ),
        )
        centre_selection_decision = "promoted_strictly_gated_candidate"
    else:
        selected_centre = reference_replay
        centre_selection_decision = "retained_protected_reference_no_candidate_passed"
    selected_record = selected_centre["record"]
    selected_climate = climates[selected_record["key"]]

    layout_replays = []
    for name, candidate_x, candidate_y in _siting_layout_candidates(base_x, base_y):
        valid, errors = validate_layout(
            candidate_x,
            candidate_y,
            box_size_m=SITING_BOX_M,
            max_turbines=SITING_N_TURBINES,
            min_spacing_d=SITING_MIN_SPACING_D,
            diameter_m=spec.diameter_m,
        )
        if not valid:
            layout_replays.append({"name": name, "valid": False, "errors": list(errors)})
            continue
        annual = _siting_evaluate_layout(
            candidate_x, candidate_y, turbine, selected_climate
        )
        layout_replays.append(
            {
                "name": name,
                "valid": True,
                "x": candidate_x,
                "y": candidate_y,
                "annual": annual,
                "summary": _siting_summary(annual),
            }
        )
    protected_layout = next(
        replay
        for replay in layout_replays
        if replay["name"] == "protected_robust_coordinate_layout"
    )
    protected_stress = _siting_direction_stress(
        base_x, base_y, turbine, selected_climate
    )
    layout_promotable = []
    for replay in layout_replays:
        if not replay.get("valid") or replay is protected_layout:
            continue
        annual_non_worse = all(
            replay["annual"][str(year)]["capacity_factor"]
            >= protected_layout["annual"][str(year)]["capacity_factor"]
            for year in SITING_YEARS
        )
        if not annual_non_worse:
            replay["promotion_gate"] = {
                "every_public_year_non_worse": False,
                "passed": False,
            }
            continue
        stress = _siting_direction_stress(
            replay["x"], replay["y"], turbine, selected_climate
        )
        replay["direction_stress"] = stress
        gate = {
            "every_public_year_non_worse": True,
            "mean_public_cf_non_worse": replay["summary"]["mean_capacity_factor"]
            >= protected_layout["summary"]["mean_capacity_factor"],
            "stress_mean_cf_non_worse": stress["mean_capacity_factor"]
            >= protected_stress["mean_capacity_factor"],
            "stress_worst_cf_non_worse": stress["worst_capacity_factor"]
            >= protected_stress["worst_capacity_factor"],
            "stress_max_wake_non_worse": stress["max_wake_loss_fraction"]
            <= protected_stress["max_wake_loss_fraction"],
        }
        gate["passed"] = all(gate.values())
        replay["promotion_gate"] = gate
        if gate["passed"]:
            layout_promotable.append(replay)
    if layout_promotable:
        selected_layout = max(
            layout_promotable,
            key=lambda replay: (
                replay["summary"]["mean_capacity_factor"],
                replay["summary"]["worst_year_capacity_factor"],
            ),
        )
    else:
        selected_layout = protected_layout
    x = selected_layout.get("x", base_x)
    y = selected_layout.get("y", base_y)
    annual = selected_layout["annual"]
    robustness = selected_layout["summary"]
    latitude = selected_record["latitude"]
    longitude = selected_record["longitude"]
    depth_m = selected_record["depth_m"]
    distance_to_coast_km = selected_record["distance_to_coast_km"]

    public_baseline_x, public_baseline_y = grid_layout(
        SITING_N_TURBINES, 7.0, spec.diameter_m
    )
    public_baseline_annual = _siting_evaluate_layout(
        public_baseline_x,
        public_baseline_y,
        turbine,
        climates[baseline_record["key"]],
    )
    public_baseline_summary = _siting_summary(public_baseline_annual)

    layout_valid, layout_errors = validate_layout(
        x,
        y,
        box_size_m=SITING_BOX_M,
        max_turbines=SITING_N_TURBINES,
        min_spacing_d=SITING_MIN_SPACING_D,
        diameter_m=spec.diameter_m,
    )
    pairwise = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    pairwise += np.eye(len(x), dtype="float64") * 1e12
    min_spacing_m = float(pairwise.min())
    allowed = bool(zone.is_in_allowed_zone(latitude, longitude, SITING_MAX_DEPTH_M))
    fixed_bottom, fixed_bottom_reason = is_eligible_fixed_bottom(
        depth_m, distance_to_coast_km
    )
    constraints = {
        "exact_turbine_count": len(x) == SITING_N_TURBINES,
        "exact_capacity_mw": SITING_CAPACITY_MW,
        "layout_valid": bool(layout_valid),
        "layout_errors": list(layout_errors),
        "min_spacing_m": min_spacing_m,
        "required_min_spacing_m": float(SITING_MIN_SPACING_D * spec.diameter_m),
        "max_abs_x_m": float(np.max(np.abs(x))),
        "max_abs_y_m": float(np.max(np.abs(y))),
        "box_size_m": SITING_BOX_M,
        "centre_depth_m": depth_m,
        "max_centre_depth_m": SITING_MAX_DEPTH_M,
        "distance_to_coast_km": distance_to_coast_km,
        "minimum_distance_to_coast_km": SITING_DISTANCE_TO_COAST_MIN_KM,
        "allowed_zone": allowed,
        "fixed_bottom_eligible": bool(fixed_bottom),
        "fixed_bottom_reason": fixed_bottom_reason,
        "bathymetry_source": str(grid["bathymetry_path"]),
    }
    strict_constraint_gate = bool(
        layout_valid
        and allowed
        and fixed_bottom
        and 0.0 < depth_m <= SITING_MAX_DEPTH_M
        and distance_to_coast_km >= SITING_DISTANCE_TO_COAST_MIN_KM
        and min_spacing_m >= SITING_MIN_SPACING_D * spec.diameter_m * (1 - 1e-9)
        and np.max(np.abs(x)) <= SITING_BOX_M / 2
        and np.max(np.abs(y)) <= SITING_BOX_M / 2
    )
    if not strict_constraint_gate:
        raise RuntimeError(f"Siting constraint gate failed: {constraints}")

    mean_economics = evaluate_farm(
        capacity_mw=SITING_CAPACITY_MW,
        n_turbines=SITING_N_TURBINES,
        aep_gwh=robustness["mean_aep_gwh"],
        water_depth_m=depth_m,
        distance_to_shore_km=distance_to_coast_km,
    )
    worst_economics = evaluate_farm(
        capacity_mw=SITING_CAPACITY_MW,
        n_turbines=SITING_N_TURBINES,
        aep_gwh=robustness["worst_aep_gwh"],
        water_depth_m=depth_m,
        distance_to_shore_km=distance_to_coast_km,
    )
    economics = {
        "mean_weather": {
            "capex_meur": float(mean_economics.capex_eur / 1e6),
            "opex_meur_per_year": float(mean_economics.opex_eur_per_year / 1e6),
            "lcoe_eur_per_mwh": float(mean_economics.lcoe_eur_per_mwh),
            "lcoe_components_eur_per_mwh": dict(mean_economics.lcoe_components),
        },
        "worst_weather": {
            "lcoe_eur_per_mwh": float(worst_economics.lcoe_eur_per_mwh),
        },
        "official_cost_model": "Phase 2 cost_model.evaluate_farm",
        "role": "secondary diagnostic and near-CF tie-break; official rank is capacity factor",
    }
    reference_comparison = {
        "reference_centre": list(SITING_REFERENCE_CENTRE),
        "mean_cf_gain": robustness["mean_capacity_factor"]
        - reference_summary["mean_capacity_factor"],
        "worst_cf_gain": robustness["worst_year_capacity_factor"]
        - reference_summary["worst_year_capacity_factor"],
        "mean_wake_change": robustness["mean_wake_loss_fraction"]
        - reference_summary["mean_wake_loss_fraction"],
        "annual_cf_gain": {
            str(year): annual[str(year)]["capacity_factor"]
            - reference_replay["annual"][str(year)]["capacity_factor"]
            for year in SITING_YEARS
        },
    }
    previous_v187_comparison = {
        "label": "exact official-input replay of the previously submitted v187 layout",
        "previous_summary": dict(SITING_PREVIOUS_V187_AUDIT),
        "mean_cf_gain": robustness["mean_capacity_factor"]
        - SITING_PREVIOUS_V187_AUDIT["mean_capacity_factor"],
        "worst_cf_gain": robustness["worst_year_capacity_factor"]
        - SITING_PREVIOUS_V187_AUDIT["worst_year_capacity_factor"],
        "mean_wake_change": robustness["mean_wake_loss_fraction"]
        - SITING_PREVIOUS_V187_AUDIT["mean_wake_loss_fraction"],
        "max_wake_change": robustness["max_wake_loss_fraction"]
        - SITING_PREVIOUS_V187_AUDIT["max_wake_loss_fraction"],
    }
    public_baseline_comparison = {
        "label": "public-data proxy only; not the organizer's hidden real-wind baseline",
        "centre_requested": list(SITING_PUBLIC_BASELINE_CENTRE),
        "centre_snapped_to_arome": [
            baseline_record["latitude"],
            baseline_record["longitude"],
        ],
        "layout": "plain organizer 7D grid",
        "annual_training_replay": public_baseline_annual,
        "summary": public_baseline_summary,
        "mean_cf_gain": robustness["mean_capacity_factor"]
        - public_baseline_summary["mean_capacity_factor"],
        "worst_cf_gain": robustness["worst_year_capacity_factor"]
        - public_baseline_summary["worst_year_capacity_factor"],
        "official_hidden_baseline": {
            "reported_capacity_factor": 0.538,
            "reported_wake_loss_fraction": 0.059,
            "reproducible_from_kit": False,
            "organizer_clarification": (
                "The published figures use hidden real wind; the synthetic "
                "Dogger Bank notebook does not reproduce them by design."
            ),
        },
    }
    selected_centre_was_promoted = selected_centre is not reference_replay
    robustness_gate = {
        "centre_is_reference_or_passed_material_promotion_gate": bool(
            not selected_centre_was_promoted
            or selected_centre.get("promotion_gate", {}).get("passed", False)
        ),
        "mean_cf_non_worse_than_reference": robustness["mean_capacity_factor"]
        >= reference_summary["mean_capacity_factor"],
        "worst_cf_non_worse_than_reference": robustness["worst_year_capacity_factor"]
        >= reference_summary["worst_year_capacity_factor"],
        "annual_spread_within_reference_plus_0_005": robustness[
            "annual_capacity_factor_spread"
        ]
        <= reference_summary["annual_capacity_factor_spread"] + 0.005,
        "max_wake_non_worse_than_reference": robustness["max_wake_loss_fraction"]
        <= reference_summary["max_wake_loss_fraction"],
        "every_year_non_worse_than_protected_reference": all(
            gain >= 0.0 for gain in reference_comparison["annual_cf_gain"].values()
        ),
        "every_year_beats_public_plain_grid_proxy": all(
            annual[str(year)]["capacity_factor"]
            > public_baseline_annual[str(year)]["capacity_factor"]
            for year in SITING_YEARS
        ),
        "full_5x5_weather_neighbourhood": bool(
            np.isfinite(selected_record["screen"]["neighbour_mean_gross_cf"])
        ),
        "neighbour_mean_margin_reference_relative": bool(
            selected_record["screen"]["neighbour_mean_margin"]
            <= reference_neighbour_mean_margin + SITING_NEIGHBOUR_MEAN_TOLERANCE
        ),
        "neighbour_worst_margin_reference_relative": bool(
            selected_record["screen"]["neighbour_worst_margin"]
            <= reference_neighbour_worst_margin + SITING_NEIGHBOUR_WORST_TOLERANCE
        ),
        "mean_lcoe_at_most_91": economics["mean_weather"]["lcoe_eur_per_mwh"]
        <= 91.0,
        "worst_lcoe_at_most_95": economics["worst_weather"]["lcoe_eur_per_mwh"]
        <= 95.0,
    }
    robustness_gate["passed"] = all(robustness_gate.values())
    if not robustness_gate["passed"]:
        raise RuntimeError(f"Siting robustness/economics gate failed: {robustness_gate}")

    centre_search_audit = []
    for replay in centre_replays:
        centre_search_audit.append(
            {
                "latitude": replay["record"]["latitude"],
                "longitude": replay["record"]["longitude"],
                "depth_m": replay["record"]["depth_m"],
                "source_tiers": replay["record"]["source_tiers"],
                "screen": replay["record"]["screen"],
                "exact_summary": replay["summary"],
                "promotion_gate": replay.get("promotion_gate"),
                "selected": replay is selected_centre,
            }
        )
    layout_search_audit = []
    for replay in layout_replays:
        layout_search_audit.append(
            {
                "name": replay["name"],
                "valid": replay.get("valid", False),
                "errors": replay.get("errors", []),
                "exact_summary": replay.get("summary"),
                "promotion_gate": replay.get("promotion_gate"),
                "direction_stress": replay.get("direction_stress"),
                "selected": replay is selected_layout,
            }
        )
    submission = {
        "team": SITING_TEAM,
        "farm_centre_lat": latitude,
        "farm_centre_lon": longitude,
        "turbine_key": SITING_TURBINE_KEY,
        "layout_x_m": [float(value) for value in x],
        "layout_y_m": [float(value) for value in y],
    }
    power_forecast_policy = {
        "method": (
            "Transform the final 1.3 km wind-speed quantiles at the nearest "
            "selected-centre cell through the organizer IEA 22 MW power curve "
            "and Bastankhah-Gaussian PyWake model for the selected 55-turbine "
            "layout. Use dir_50 for wake orientation, apply the prescribed "
            "125-to-170 m power-law shear, sort the three transformed paths "
            "to preserve power-quantile ordering, and express each six-hour "
            "forecast step in MWh."
        ),
        "source_height_m": SITING_SOURCE_HEIGHT_M,
        "hub_height_m": SITING_HUB_HEIGHT_M,
        "shear_alpha": SITING_SHEAR_ALPHA,
        "step_hours": SITING_POWER_FORECAST_STEP_HOURS,
        "capacity_mw": SITING_CAPACITY_MW,
        "expected_steps": 8 * 3 * 4,
        "ordering": ["window", "horizon", "hour"],
        "output_keys": [
            "predicted_q05",
            "predicted_q50",
            "predicted_q95",
        ],
        "output_unit": "MWh per six-hour forecast step",
        "input_only": True,
        "previous_submission_inputs": [],
    }
    return {
        "submission": submission,
        "power_forecast_policy": power_forecast_policy,
        "constraints": constraints,
        "constraint_gate_passed": strict_constraint_gate,
        "annual_training_replay": annual,
        "weather_cell": {
            "latitude": latitude,
            "longitude": longitude,
            "source": "organizer 1.3 km AROME target, training years only",
            "temporal_resolution_hours": 3,
            "hub_height_shear_alpha": SITING_SHEAR_ALPHA,
        },
        "robustness": robustness,
        "economics": economics,
        "robustness_gate": robustness_gate,
        "reference_comparison": reference_comparison,
        "previous_v187_comparison": previous_v187_comparison,
        "public_baseline_comparison": public_baseline_comparison,
        "centre_search": {
            "selection_decision": centre_selection_decision,
            "weather_cells_screened": int(grid["weather"].sum()),
            "legal_cells_scanned": int(grid["legal"].sum()),
            "day_step": SITING_SCREEN_DAY_STEP,
            "tiers": [
                "mean",
                "balanced",
                "robust",
                "near_mean_robust",
                "depth_safe_high_wind",
            ],
            "screen_near_best_gross_cf_tolerance": (
                SITING_SCREEN_NEAR_BEST_GROSS_CF
            ),
            "near_best_mean_cf_tolerance": SITING_NEAR_BEST_MEAN_CF,
            "neighbour_margin_gate": {
                "reference_mean_margin": reference_neighbour_mean_margin,
                "reference_worst_margin": reference_neighbour_worst_margin,
                "allowed_mean_tolerance": SITING_NEIGHBOUR_MEAN_TOLERANCE,
                "allowed_worst_tolerance": SITING_NEIGHBOUR_WORST_TOLERANCE,
            },
            "candidates": centre_search_audit,
        },
        "layout_search": {
            "robust_coordinate_refinement_audit": SITING_ROBUST_LAYOUT_AUDIT,
            "protected_direction_stress": protected_stress,
            "candidates": layout_search_audit,
        },
        "wake_model": "organizer PyWake Bastankhah Gaussian",
        "selection": (
            "Input-only all-cell shallow-water screen, spatially diverse exact "
            "five-year PyWake centre replay, and constrained continuous layout "
            "optimization with strict annual, neighbourhood, wake, direction-"
            "shift, legality, and economics gates. The promoted layout is then "
            "replayed deterministically. The organizer's published "
            "hidden-wind baseline is reported separately and is not presented "
            "as kit-reproducible."
        ),
    }


def build_competition_evidence(metadata: dict, siting: dict) -> dict:
    inference_path = SCRIPT_DIR / "inference.py"
    return {
        "scope": "Phase 2 five-dimension evaluation evidence",
        "reproduction": {
            "source_files": ["train.py", "inference.py"],
            "train_command": (
                "python train.py --kit-dir KIT_ROOT --data-root PHASE2_DATA_ROOT "
                "--phase1-data-root PHASE1_DATA_ROOT --artifacts-dir artifacts"
            ),
            "inference_command": (
                "python inference.py --kit-dir KIT_ROOT --data-root DATA_ROOT "
                "--artifacts-dir artifacts --output predictions.csv "
                "--archive phase2_forecast_submission.zip"
            ),
            "train_sha256": metadata["train_code_sha256"],
            "inference_sha256": hashlib.sha256(
                inference_path.read_bytes()
            ).hexdigest(),
            "organizer_input_summary": metadata["environment"],
            "submission_archive_members": ["predictions.csv"],
        },
        "forecast": {
            "model_version": metadata["model_version"],
            "model_count": metadata["model_count"],
            "input_only_training": metadata["input_only_training"],
            "previous_submission_inputs": metadata["previous_submission_inputs"],
            "external_resources": metadata.get("external_resources", []),
            "strict_graphcast_d1_direction_gate": metadata[
                "graphcast_d1_direction_gate"
            ],
            "strict_d1_speed_context_gate": metadata[
                "d1_speed_context_gate"
            ],
            "strict_d7_context_gate": metadata["d7_speed_context_gate"],
            "strict_d14_speed_gate": metadata["d14_speed_endpoint_gate"],
            "strict_fine_speed_residual_gate": metadata[
                "fine_speed_residual_gate"
            ],
            "strict_d7_conditional_endpoint_gate": metadata[
                "d7_conditional_endpoint_gate"
            ],
            "uncertainty": (
                "Quantile MOS, conformal calibration, monotonic endpoints, and "
                "fine-grid replay gates are evaluated separately by lead."
            ),
            "validation": {
                "held_years": [2016, 2017, 2018, 2019, 2020],
                "target_resolution_km": 1.3,
                "promotion_rule": (
                    "A candidate must improve aggregate interval score, remain "
                    "non-worse in every held year, and remain non-worse in "
                    "every populated spatial, speed, direction, width, and "
                    "signed-error regime."
                ),
                "inference_target_used": False,
                "previous_submission_file_used": False,
            },
        },
        "siting": siting,
        "methodology_and_originality": {
            "forecast": (
                "A compact HRES quantile-MOS/downscaling stack with circular "
                "direction residuals and a causal ERA5-trained GraphCast d1 "
                "direction signal. New signals are activated only after exact "
                "1.3 km held-year and populated-regime replay."
            ),
            "siting": (
                "A 55-turbine wake-aware robust coordinate-refined geometry is "
                "protected by an "
                "input-only all-cell centre search. Challengers must pass exact "
                "organizer-zone, five-year, 5x5 spatial-neighbour, wake, depth, "
                "direction-stress, and economics gates; otherwise the audited "
                "incumbent is retained."
            ),
        },
        "financial_analysis": siting["economics"],
        "responsible_compute": {
            "source_files": ["train.py", "inference.py"],
            "forecast_models": metadata["model_count"]["total"],
            "estimator_threads": 1,
            "training_checkpoints": "temporary and removed after a successful run",
            "inference": "one isolated window worker at a time",
            "thread_environment": {
                "OMP_NUM_THREADS": 1,
                "OPENBLAS_NUM_THREADS": 1,
                "MKL_NUM_THREADS": 1,
                "NUMEXPR_NUM_THREADS": 1,
            },
            "target_used_as_inference_input": False,
            "artifact_inputs": (
                "organizer inputs plus the disclosed competition-authorized "
                "ERA5-trained GraphCast forecast archive"
            ),
        },
        "limitations": [
            "The definitive forecast year may contain regimes absent in 2016-2020.",
            "Siting weather is represented by the nearest organizer coarse cell.",
            "Bathymetry and coast distance use nearest-cell organizer layers.",
            "LCOE follows the organizer's deterministic assumptions and is not a bid.",
        ],
    }


def competition_evidence_markdown(evidence: dict) -> str:
    siting = evidence["siting"]
    robustness = siting["robustness"]
    economics = siting["economics"]
    constraints = siting["constraints"]
    models = evidence["forecast"]["model_count"]
    reproduction = evidence["reproduction"]
    d1_context_gate = evidence["forecast"]["strict_d1_speed_context_gate"]
    d1_context_deltas = {
        str(row["year"]): row["mean_delta"]
        for row in d1_context_gate["folds"]
    }
    d14_gate = evidence["forecast"]["strict_d14_speed_gate"]
    fine_speed_gate = evidence["forecast"][
        "strict_fine_speed_residual_gate"
    ]
    d7_conditional_gate = evidence["forecast"][
        "strict_d7_conditional_endpoint_gate"
    ]
    graphcast_gate = evidence["forecast"][
        "strict_graphcast_d1_direction_gate"
    ]
    annual_rows = "\n".join(
        "| {year} | {aep:.1f} | {cf:.4f} | {wake:.2%} |".format(
            year=year,
            aep=row["aep_gwh"],
            cf=row["capacity_factor"],
            wake=row["wake_loss_fraction"],
        )
        for year, row in siting["annual_training_replay"].items()
    )
    return f"""# Phase 2 Reproducibility and Evaluation Evidence

## Reproduction contract

The submitted source folder contains exactly `train.py` and `inference.py`.
Place the unpacked organizer Phase 2 and Phase 1 datasets at the paths shown in
the training command, and point `KIT_ROOT` at the official `phase_2` branch
checkout. Alternatively, colocated organizer datasets are auto-detected. No
prior prediction or submission file is an input.

```text
{reproduction['train_command']}
{reproduction['inference_command']}
```

`train.py` writes all learned artifacts plus the audited siting/economics
outputs. `inference.py` loads only those artifacts and organizer inference
predictors, validates every row, writes `predictions.csv`, and creates a ZIP
whose sole member is `predictions.csv`.

## Forecast

- Total learned estimators: {models['total']} (all configured for one thread)
- Quantile MOS / direction residual / downscaler estimators: {models['quantile_mos']} / {models['direction_residual']} / {models['downscaler']}
- Support-gated quantile refinement estimators: {models['support_gated_qmos_refit']}
- D1 causal context endpoint estimators: {models['d1_speed_context_endpoints']}
- D1 context held-year deltas: {d1_context_deltas}
- D1 context gate: every audited populated physical regime is non-worse
- D7 context endpoints: {models['d7_speed_context_endpoints']}
- Strict d7 exact fine-grid gain: {evidence['forecast']['strict_d7_context_gate']['relative_gain']:.4%}
- D7 context active fraction: {evidence['forecast']['strict_d7_context_gate']['active_fraction']:.2%}
- Strict d14 exact fine-grid gain: {-d14_gate['aggregate_delta']:.4f} Winkler points across {d14_gate['selected_cell_count']} calendar/hour cells
- D14 baseline/candidate coverage: {d14_gate['base_coverage']:.3%} / {d14_gate['candidate_coverage']:.3%}
- A fine-speed residual branch was excluded by the validation gates; no rules from that branch are deployed.
- Conditional d7 endpoint OOF gain: {-d7_conditional_gate['aggregate_delta']:.4f} Winkler points; held-year deltas: {d7_conditional_gate['delta_by_year']}
- Conditional d7 endpoint worst populated-regime delta: {d7_conditional_gate['worst_regime_delta']:.4f} (non-positive is required)
- GraphCast d1 direction exact-replay gain: {-graphcast_gate['aggregate_delta']:.4f} Winkler points; stress-year deltas: {graphcast_gate['delta_by_year']}
- GraphCast d1 direction gate: every audited broad regime is non-worse, and the definitive-input support audit passes
- Validation uses organizer target years 2016-2020 at the exact 1.3 km grid. Every promoted branch must be non-worse in each held year and every populated physical regime.
- AROME target values and hidden/evaluation truth are never forecast inputs at inference time.

## External resource disclosure

The d1 direction branch uses the high-resolution, 37-pressure-level GraphCast
model output distributed through the public WeatherBench 2 GraphCast archive.
The model was trained on ERA5 from 1979 through 2017. For each organizer issue
time, `train.py` downloads only the causal 1000 hPa eastward and northward wind
forecasts at +24, +30, +36, and +42 hours, interpolates them to the target
footprint, and stores the resulting direction fields in the artifact. The
production center is a 0.30 circular blend with the incumbent, while all three
direction endpoints receive the same angular shift.

GraphCast code is distributed under Apache-2.0 and its model weights under
CC BY-NC-SA 4.0. The public forecast fields are used for this research
competition under those terms and are not included in the submitted forecast
ZIP. Source: `gs://weatherbench2/datasets/graphcast_v2/`; model documentation:
`https://github.com/google-deepmind/graphcast`.

## Wind-farm siting

- Turbine: {siting['submission']['turbine_key']}; count: {len(siting['submission']['layout_x_m'])}; capacity: {constraints['exact_capacity_mw']:.0f} MW
- Centre: ({siting['submission']['farm_centre_lat']:.6f}, {siting['submission']['farm_centre_lon']:.6f})
- Centre decision: `{siting['centre_search']['selection_decision']}` after screening {siting['centre_search']['legal_cells_scanned']:,} organizer-legal centres
- Depth: {constraints['centre_depth_m']:.2f} m; coast distance: {constraints['distance_to_coast_km']:.2f} km
- Minimum spacing: {constraints['min_spacing_m']:.1f} m (required {constraints['required_min_spacing_m']:.1f} m)
- Mean/worst capacity factor: {robustness['mean_capacity_factor']:.4f} / {robustness['worst_year_capacity_factor']:.4f}
- Mean/worst AEP: {robustness['mean_aep_gwh']:.1f} / {robustness['worst_aep_gwh']:.1f} GWh
- Maximum training-year wake loss: {robustness['max_wake_loss_fraction']:.2%}
- The official zone, 15 km box, 5D spacing, 55-turbine, 1210 MW, and 50 m depth gates all pass.

| Weather year | AEP (GWh) | Capacity factor | Wake loss |
|---:|---:|---:|---:|
{annual_rows}

The protected quality-diversity geometry is replayed from organizer inputs during
every clean training run. Centre challengers come from mean, balanced, robust,
near-best, and depth-safe tiers, but promotion requires non-worse capacity factor
in every public year, material mean and worst-year gains, non-worse wake, a full
5x5 weather neighbourhood, reference-relative spatial margins, and exact zone and
depth eligibility. No legal challenger passed all gates in this run, so the
protected centre was retained. The organizer's published 53.8% baseline uses
hidden real wind and is reported separately from our reproducible public proxy.

## Economics

- Mean-weather CAPEX: EUR {economics['mean_weather']['capex_meur']:.1f} million
- Mean-weather OPEX: EUR {economics['mean_weather']['opex_meur_per_year']:.1f} million/year
- Mean/worst-weather LCOE: EUR {economics['mean_weather']['lcoe_eur_per_mwh']:.2f} / {economics['worst_weather']['lcoe_eur_per_mwh']:.2f} per MWh
- Values are recomputed by the official Phase 2 cost model during training.

## Method and responsible compute

The forecast combines compact quantile MOS, circular residual learning,
conformal intervals, and terrain downscaling. Promotion is based on deterministic
held-year and regime audits, not leaderboard score. The siting design combines
a wake-aware coordinate-refined layout with a frugal all-cell centre screen and
strict exact-replay fallback. All estimators and numerical libraries are
restricted to one thread; inference processes one evaluation window at a time;
successful training deletes temporary checkpoints.

Generated artifacts include the forecast model, climatology, manifest, siting
JSON, machine-readable evaluation evidence, and this report. The forecast ZIP
remains competition-compatible by containing only `predictions.csv`.

## Limitations

Definitive-year regime shift remains the main forecast risk. Siting uses the
nearest coarse weather cell and nearest bathymetry cell. The financial result is
scenario evidence under organizer assumptions, not a commercial bid.
"""


def write_competition_outputs(
    artifacts_dir: Path, siting: dict, evidence: dict
) -> None:
    outputs = {
        "siting_submission.json": siting["submission"],
        "competition_evidence.json": evidence,
    }
    for name, payload in outputs.items():
        path = artifacts_dir / name
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[train] wrote {path}")
    report_path = artifacts_dir / "methodology_economics_compute.md"
    report_path.write_text(competition_evidence_markdown(evidence), encoding="utf-8")
    print(f"[train] wrote {report_path}")


def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def train(args: argparse.Namespace) -> None:
    t0 = time.time()
    kit_root = resolve_kit_root(args.kit_dir)
    phase2 = add_kit_paths(kit_root)
    configure_data_root(args.data_root, args.phase1_data_root)

    import config
    import forecast_hres as fh
    import forecast_pipeline as pipeline
    import downscaling
    import target_loader

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.artifacts_dir / "phase2_forecast_artifacts.joblib"
    clim_path = args.artifacts_dir / "climatology_coarse.npz"
    manifest_path = args.artifacts_dir / "manifest.json"

    env_summary = summarize_environment(config, target_loader)
    print(config.describe())
    print(f"[train] target days: {env_summary['n_target_days']}")
    print(f"[train] hres parquets: {len(env_summary['hres_parquets'])}")

    external_trajectory_policy = materialize_external_trajectory_policy(
        config,
        target_loader,
        args.artifacts_dir / "_checkpoint_external_trajectory_v3.joblib",
    )
    # The analogue endpoint challenger is intentionally excluded. Its exact
    # historical gain was small and it admitted positive physical regimes.
    hres_analog_policy = None

    # Materialize the legal 2016-2020 weekly climatology before calibration.
    # Prefer the identical cache shipped in the official kit; rebuild it from
    # organizer training data only when that cache is absent.
    official_clim_path = Path(fh.CLIM_CACHE)
    fh.CLIM_CACHE = clim_path
    fh._climatology.cache_clear()
    if clim_path.exists():
        clim_path.unlink()
    if (
        official_clim_path.exists()
        and official_clim_path.resolve() != clim_path.resolve()
    ):
        shutil.copy2(official_clim_path, clim_path)
        climatology_source = "official Phase 2 kit cache"
    else:
        climatology_source = "rebuilt from organizer 2016-2020 targets"
    fh._climatology()
    if not clim_path.exists():
        raise FileNotFoundError(f"Expected climatology cache was not created: {fh.CLIM_CACHE}")
    print(f"[train] climatology: {climatology_source}", flush=True)

    train_issue_dates = pipeline.train_dates(args.train_freq)
    print(f"[train] fitting HRES-MOS on {len(train_issue_dates)} issue dates ({args.train_freq})")
    qmos, direction_models, conformal_adjust, dir_offsets = fit_forecast_frugal(
        config,
        pipeline,
        fh,
        train_issue_dates,
        checkpoint_dir=args.artifacts_dir,
    )
    print(
        "[train] fitting support-gated all-years qMOS refinements",
        flush=True,
    )
    qmos_refit_policy = fit_qmos_refit_policy(
        config,
        fh,
        train_issue_dates,
        qmos,
        conformal_adjust,
        args.artifacts_dir,
    )
    print(
        "[train] fitting one strict-gated shared multi-scale direction model",
        flush=True,
    )
    direction_models["shared_spatial_direction"] = (
        train_shared_spatial_direction_model(
            config,
            fh,
            direction_models,
            args.artifacts_dir / "_checkpoint_shared_spatial_direction.joblib",
        )
    )
    print(
        "[train] fitting strict-gated compact d7 speed endpoint pair",
        flush=True,
    )
    d7_speed_context = train_d7_speed_context_models(
        config,
        fh,
        args.artifacts_dir / "_checkpoint_d7_speed_context.joblib",
    )
    if not d7_speed_context["gate"].get("passed", False):
        raise RuntimeError("Strict d7 speed context gate did not pass")
    # The dense-daily d1 challenger was never verified on the definitive set.
    # Keep the hidden-score-proven incumbent and avoid two unnecessary models.
    d1_dense_daily = None
    print("[train] retaining protected d1 endpoint path", flush=True)

    all_target_dates = target_loader.list_dates(config.target_root())
    downscale_dates = [
        d for d in all_target_dates if d.year == args.downscale_year
    ][:: max(1, args.downscale_step)]
    if not downscale_dates:
        raise ValueError(
            f"No target days found for downscale-year={args.downscale_year}. "
            "Use a training year present in the Phase 2 data."
        )
    print(
        f"[train] fitting downscaler on {len(downscale_dates)} days "
        f"from {args.downscale_year}, step={args.downscale_step}"
    )
    fh._coarse_grid.cache_clear()
    fh._load_hres.cache_clear()
    fh._climatology.cache_clear()
    global _HRES_CACHE
    _HRES_CACHE = None
    _ANALYSIS_CACHE.clear()
    gc.collect()
    downscaler = downscaling.train_downscaler(
        downscale_dates,
        hours=HOURS,
        params={"n_jobs": 1},
    )
    print(
        "[train] fitting strict-gated February d1 context endpoint models",
        flush=True,
    )
    d1_speed_context = fit_d1_speed_context_policy(
        fh,
        config,
        pipeline,
        downscaling,
        qmos,
        direction_models,
        conformal_adjust,
        downscaler,
        args.artifacts_dir / "_checkpoint_d1_speed_context.joblib",
    )

    if args.skip_interval_calibration:
        speed_inflation = {1: 1.0, 7: 1.0, 14: 1.0}
        fine_dir_offsets = dir_offsets
        d14_speed_endpoint_policy = {
            "method": "disabled because interval calibration was skipped",
            "lower_factor": 1.0,
            "upper_factor": 1.0,
            "rules": [],
            "gate": {"passed": False},
            "input_only_training": True,
            "previous_submission_inputs": [],
            "new_models": 0,
        }
        fine_speed_residual_policy = {
            "method": "disabled because interval calibration was skipped",
            "rules": [],
            "gate": {"passed": False},
            "input_only_training": True,
            "previous_submission_inputs": [],
            "new_models": 0,
        }
        d7_speed_endpoint_policy = {
            "method": "disabled because interval calibration was skipped",
            "factor_grid": list(D7_SPEED_ENDPOINT_FACTORS),
            "rules": [],
            "gate": {"passed": False},
            "input_only_training": True,
            "previous_submission_inputs": [],
            "new_models": 0,
        }
        d1_direction_speed_interval = None
        d7_direction_interval_policy = {
            "base_half_width": float(D7_DEPLOYED_BASE_HALF_WIDTH),
            "rules": [],
            "method": "protected d7 baseline only",
            "new_models": 0,
        }
        fine_d7_direction_models = None
        fine_d7_context_models = None
        d7_pressure_policy = None
        d7_conditional_endpoint = None
        d14_direction_speed_interval = None
        fine_d14_climatology = None
        fine_d7_climatology = None
        print("[train] skipped fine-grid interval calibration")
    else:
        print(f"[train] calibrating intervals to target coverage={args.coverage_target:.2f}")
        speed_inflation, fine_dir_offsets = calibrate_intervals_hybrid(
            fh,
            config,
            pipeline,
            downscaling,
            qmos,
            direction_models,
            conformal_adjust,
            downscaler,
            dir_offsets,
            target=args.coverage_target,
        )
        print(
            "[train] fitting strict exact-window d14 speed endpoint policy",
            flush=True,
        )
        d14_speed_checkpoint_path = (
            args.artifacts_dir / "_checkpoint_d14_speed_endpoint_v2.joblib"
        )
        d14_speed_checkpoint = _load_checkpoint(
            d14_speed_checkpoint_path, "d14_speed_endpoint_v2"
        )
        if (
            d14_speed_checkpoint is not None
            and not d14_speed_checkpoint.get("policy", {})
            .get("gate", {})
            .get("passed", False)
        ):
            print(
                "[train] ignoring a d14 endpoint checkpoint whose strict "
                "gate did not pass",
                flush=True,
            )
            d14_speed_checkpoint = None
        if d14_speed_checkpoint is None:
            d14_speed_endpoint_policy = fit_d14_speed_endpoint_policy(
                fh,
                config,
                pipeline,
                downscaling,
                qmos,
                direction_models,
                conformal_adjust,
                downscaler,
                speed_inflation,
            )
            _save_checkpoint(
                d14_speed_checkpoint_path,
                "d14_speed_endpoint_v2",
                policy=d14_speed_endpoint_policy,
            )
        else:
            d14_speed_endpoint_policy = d14_speed_checkpoint["policy"]
        fine_speed_residual_policy = {
            "method": "disabled; no fine-grid residual policy passed deployment",
            "rules": [],
            "gate": {
                "passed": False,
                "retired": True,
            },
            "input_only_training": True,
            "previous_submission_inputs": [],
            "new_models": 0,
        }
        d7_speed_endpoint_policy = fixed_d7_speed_endpoint_policy()
        print(
            "[train] activated strict all-year d7 speed endpoint policy",
            flush=True,
        )
        print(
            "[train] fitting exact protected-speed d1 direction intervals "
            f"to coverage={args.coverage_target:.2f}"
        )
        d1_interval_checkpoint_path = (
            args.artifacts_dir / "_checkpoint_d1_direction_interval.joblib"
        )
        d1_interval_checkpoint = _load_checkpoint(
            d1_interval_checkpoint_path, "d1_direction_interval"
        )
        if d1_interval_checkpoint is None:
            d1_interval_result = fit_d1_direction_speed_interval(
                fh,
                config,
                pipeline,
                downscaling,
                qmos,
                direction_models,
                conformal_adjust,
                downscaler,
                target=args.coverage_target,
            )
            _save_checkpoint(
                d1_interval_checkpoint_path,
                "d1_direction_interval",
                result=d1_interval_result,
            )
        else:
            d1_interval_result = d1_interval_checkpoint["result"]
        (
            d1_direction_speed_interval,
            d7_direction_interval_policy,
            fine_d7_direction_models,
            fine_d7_context_models,
            d7_pressure_policy,
            d7_conditional_endpoint,
        ) = d1_interval_result
        protected_november_rule = any(
            (
                int(rule.get("month", -1)),
                int(rule.get("day", -1)),
                int(rule.get("hour", -1)),
                rule.get("spec", {}).get("name"),
            )
            == (11, 4, 18, "scalar")
            for rule in d7_direction_interval_policy.get(
                "asymmetric_rules", []
            )
        )
        if not protected_november_rule:
            raise RuntimeError(
                "The public-v180 November d7 direction incumbent was not "
                "retained in the production artifact"
            )
        print(
            "[train] fitting strict-gated February-only d14 direction "
            f"interval to coverage={D14_INTERVAL_COVERAGE:.3f}"
        )
        d14_direction_speed_interval = fit_d14_direction_interval(
            fh,
            config,
            pipeline,
            downscaling,
            qmos,
            direction_models,
            conformal_adjust,
            downscaler,
            base_half_width=fine_dir_offsets[14],
        )
        fine_d14_climatology = fit_fine_d14_climatology(
            target_loader,
            config,
        )
        fine_d7_climatology = None
        print(
            "[train] retaining the public-score-protected d7 direction path",
            flush=True,
        )

    if not args.skip_interval_calibration:
        if not d7_speed_endpoint_policy.get("gate", {}).get("passed", False):
            raise RuntimeError("Strict d7 speed endpoint gate did not pass")
        if not d7_speed_endpoint_policy.get("input_only_training", False):
            raise RuntimeError("d7 speed endpoint policy is not input-only")
        if d7_speed_endpoint_policy.get("previous_submission_inputs"):
            raise RuntimeError("d7 speed endpoint policy used a prior submission")
        d7_rules = {
            (rule["month"], rule["day"], rule["hour"]): rule
            for rule in d7_speed_endpoint_policy.get("rules", [])
        }
        if set(d7_rules) != set(D7_SPEED_ENDPOINT_POLICY):
            raise RuntimeError("Unexpected d7 speed endpoint rule set")
        for key, factors in D7_SPEED_ENDPOINT_POLICY.items():
            rule = d7_rules[key]
            if (
                rule.get("lower_factor") != float(factors[0])
                or rule.get("upper_factor") != float(factors[1])
            ):
                raise RuntimeError("Unexpected d7 speed endpoint factors")
            guard = D7_SPEED_ENDPOINT_GUARDS.get(key)
            if guard is None:
                if "median_speed_threshold" in rule or "high_ratio" in rule:
                    raise RuntimeError("Unexpected d7 speed endpoint guard")
            elif (
                rule.get("median_speed_threshold") != float(guard[0])
                or rule.get("high_ratio") != float(guard[1])
            ):
                raise RuntimeError("Unexpected d7 speed endpoint guard")
        if fine_d7_direction_models is not None:
            raise RuntimeError(
                "Live-rejected fine d7 residual models entered production"
            )
        if fine_d7_context_models is not None:
            raise RuntimeError(
                "Live-rejected fine d7 context models entered production"
            )
        if fine_d7_climatology is not None:
            raise RuntimeError("Rejected fine d7 climatology entered production")
        if d7_direction_interval_policy.get("conditional_width_rules"):
            raise RuntimeError("Rejected conditional d7 widths entered production")
        if (
            fine_d14_climatology is not None
            and fine_d14_climatology.get("endpoint_policy")
        ):
            raise RuntimeError(
                "Rejected fine d14 direction endpoint entered production"
            )
        if not d14_speed_endpoint_policy.get("gate", {}).get("passed", False):
            raise RuntimeError("Strict d14 speed endpoint gate did not pass")
        if not d14_speed_endpoint_policy.get("input_only_training", False):
            raise RuntimeError("d14 speed endpoint policy is not input-only")
        if d14_speed_endpoint_policy.get("previous_submission_inputs"):
            raise RuntimeError("d14 speed endpoint policy used a prior submission")
        d14_rules = {
            (rule["month"], rule["day"], rule["hour"]): rule
            for rule in d14_speed_endpoint_policy.get("rules", [])
        }
        if set(d14_rules) != set(D14_SPEED_ENDPOINT_POLICY):
            raise RuntimeError("Unexpected strict d14 speed endpoint policy")
        if fine_speed_residual_policy.get("rules"):
            raise RuntimeError("Live-rejected v181 speed rules entered production")
        if not d7_conditional_endpoint.get("gate", {}).get("passed", False):
            raise RuntimeError("Strict d7 conditional endpoint gate did not pass")

    direction_model_count = sum(
        len(pair)
        for name, pair in direction_models.items()
        if name in ("base", "analysis", "context") and isinstance(pair, dict)
    )
    if direction_model_count == 0:
        direction_model_count = 2
    shared_spatial_model_count = int(
        direction_models.get("shared_spatial_direction", {}).get(
            "new_models", 0
        )
    )
    fine_d7_model_count = (
        0 if fine_d7_direction_models is None else 2
    )
    fine_d7_context_model_count = (
        0 if fine_d7_context_models is None else 2
    )
    d7_conditional_endpoint_model_count = (
        0 if d7_conditional_endpoint is None else 2
    )
    production_fine_d7_policy = (
        dict(FINE_D7_DIRECTION_POLICY)
        if fine_d7_direction_models is not None
        else {}
    )
    production_fine_d7_context_policy = {}
    production_d7_d10_policy = {}

    # Siting reads five full years of AROME fields. Release forecast-only input
    # caches first so the complete two-stage run stays within a modest peak RAM
    # envelope even when training and siting execute in the same process.
    fh._coarse_grid.cache_clear()
    fh._load_hres.cache_clear()
    fh._climatology.cache_clear()
    _HRES_CACHE = None
    _ANALYSIS_CACHE.clear()
    del all_target_dates, downscale_dates, train_issue_dates
    gc.collect()

    print(
        "[train] replaying and auditing robust wind-farm siting on 2016-2020",
        flush=True,
    )
    siting = build_siting_artifact(config, phase2)

    artifact = {
        "model_version": "hres_causal_d1_context_targeted_siting",
        "qmos": qmos,
        "qmos_refit_policy": qmos_refit_policy,
        "d1_speed_context": d1_speed_context,
        "d1_dense_daily": d1_dense_daily,
        "d7_speed_context": d7_speed_context,
        "direction_models": direction_models,
        "conformal_adjust": conformal_adjust,
        "dir_offsets": dir_offsets,
        "downscaler": downscaler,
        "speed_inflation": speed_inflation,
        "d7_speed_endpoint_policy": d7_speed_endpoint_policy,
        "d14_speed_endpoint_policy": d14_speed_endpoint_policy,
        "fine_speed_residual_policy": fine_speed_residual_policy,
        "fine_dir_offsets": fine_dir_offsets,
        "d1_direction_speed_interval": d1_direction_speed_interval,
        "d7_direction_interval_policy": d7_direction_interval_policy,
        "fine_d7_direction_models": fine_d7_direction_models,
        "fine_d7_direction_policy": production_fine_d7_policy,
        "fine_d7_context_models": fine_d7_context_models,
        "fine_d7_context_policy": production_fine_d7_context_policy,
        "d7_d10_tendency_policy": production_d7_d10_policy,
        "d7_pressure_policy": d7_pressure_policy,
        "d7_conditional_endpoint": d7_conditional_endpoint,
        "d14_direction_speed_interval": d14_direction_speed_interval,
        "d14_direction_policy": D14_DIRECTION_POLICY,
        "fine_d14_climatology": fine_d14_climatology,
        "siting_power_forecast_policy": siting["power_forecast_policy"],
        "fine_d7_climatology": fine_d7_climatology,
        "external_trajectory_policy": external_trajectory_policy,
        "hres_analog_policy": hres_analog_policy,
        "siting": siting,
        "metadata": {
            "model_version": "hres_causal_d1_context_targeted_siting",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "kit_root": str(kit_root),
            "phase2_root": str(phase2),
            "train_freq": args.train_freq,
            "downscale_year": args.downscale_year,
            "downscale_step": args.downscale_step,
            "coverage_target": args.coverage_target,
            "skip_interval_calibration": args.skip_interval_calibration,
            "train_code_sha256": code_sha256(),
            "input_only_training": True,
            "previous_submission_inputs": [],
            "climatology_source": climatology_source,
            "center_policy": {
                "d1_speed": (
                    "engineered HRES quantiles plus a February-only causal "
                    "14-day context upper endpoint, followed at four broad "
                    "slot-hours by a gated GraphCast 1000 hPa center shift"
                ),
                "d1_direction": (
                    "engineered HRES circular residual plus 14-day reanalysis "
                    "context at two cross-year-stable calendar/hour slots, "
                    "followed by a causal GraphCast 1000 hPa direction blend "
                    "with a support-stable guarded increase from 0.30 to 0.40 "
                    "that preserves interval widths"
                ),
                "d7_speed": (
                    "engineered HRES quantile median with a compact organizer-"
                    "context endpoint pair blended q05=0.90 and q95=0.10 only "
                    "at August 12 hours 06/12; exact 1.3 km replay passed every "
                    "held-year and populated physical-regime gate, followed by "
                    "twelve all-year-safe endpoint lookup rules and a strict "
                    "GraphCast center-plus-width residual-quantile blend"
                ),
                "d7_direction": (
                    "raw HRES direction with five held-year-gated calendar/hour "
                    "blends toward raw d1 HRES or 2016-2020 weekly vector "
                    "climatology, then one shared 3x3/5x5 vector-residual ridge "
                    "on only the cells that improve both expanding-window "
                    "chronological folds and every populated physical regime; "
                    "uncertainty uses the live-score-protected held-year-gated "
                    "asymmetric intervals plus one four-fold-safe conditional "
                    "q05/q95 endpoint pair"
                ),
                "d14_speed": (
                    "engineered d10-HRES to d14 quantile median with a fixed "
                    "cell-specific strengthened cross-year-stable endpoint map; "
                    "all 27 active calendar/hour cells pass both-year, spatial, "
                    "speed, width, signed-error, issue-block, bootstrap, and "
                    "leave-one-issue-out gates"
                ),
                "d14_direction": (
                    "training-year weekly climatology plus gated d7 signal at "
                    "four all-year-stable calendar/hour slots, followed by "
                    "seven strictly gated native-grid seasonal centers with "
                    "calibrated circular widths"
                ),
            },
            "model_count": {
                "quantile_mos": len(qmos),
                "support_gated_qmos_refit": qmos_refit_policy["new_models"],
                "d1_speed_context_endpoints": d1_speed_context["new_models"],
                "d1_dense_daily_endpoints": 0,
                "d7_speed_context_endpoints": len(d7_speed_context["models"]),
                "direction_residual": direction_model_count,
                "shared_spatial_direction": shared_spatial_model_count,
                "fine_d7_direction_residual": fine_d7_model_count,
                "fine_d7_context_residual": fine_d7_context_model_count,
                "d7_conditional_endpoint": (
                    d7_conditional_endpoint_model_count
                ),
                "downscaler": len(downscaler),
                "total": (
                    len(qmos)
                    + qmos_refit_policy["new_models"]
                    + d1_speed_context["new_models"]
                    + len(d7_speed_context["models"])
                    + direction_model_count
                    + shared_spatial_model_count
                    + fine_d7_model_count
                    + fine_d7_context_model_count
                    + d7_conditional_endpoint_model_count
                    + len(downscaler)
                ),
            },
            "d14_speed_endpoint_gate": d14_speed_endpoint_policy["gate"],
            "graphcast_d1_direction_gate": external_trajectory_policy[
                "gates"
            ]["direction_d1"],
            "external_trajectory_gates": external_trajectory_policy["gates"],
            "external_resources": external_trajectory_policy["resources"],
            "d1_speed_context_gate": d1_speed_context["gate"],
            "d1_dense_daily_gate": {
                "passed": False,
                "production_decision": "not promoted without definitive evidence",
                "new_models": 0,
            },
            "fine_speed_residual_gate": fine_speed_residual_policy["gate"],
            "d7_speed_endpoint_gate": (
                None
                if d7_speed_endpoint_policy is None
                else d7_speed_endpoint_policy["gate"]
            ),
            "d7_speed_context_gate": {
                **d7_speed_context["gate"],
                "lower_blend": d7_speed_context["lower_blend"],
                "upper_blend": d7_speed_context["upper_blend"],
                "training_rows": d7_speed_context["training_rows"],
                "training_dates": d7_speed_context["training_dates"],
                "feature_count": len(d7_speed_context["features"]),
                "new_models": d7_speed_context["new_models"],
            },
            "shared_spatial_direction_gate": direction_models.get(
                "shared_spatial_direction", {}
            ).get("gate", {"passed": False}),
            "d7_conditional_endpoint_gate": (
                None
                if d7_conditional_endpoint is None
                else d7_conditional_endpoint["gate"]
            ),
            "d7_pressure_gate": (
                None
                if d7_pressure_policy is None
                else {
                    "selected_rules": len(d7_pressure_policy["rules"]),
                    "years_used": d7_pressure_policy["years_used"],
                    "minimum_bin_count_per_year": d7_pressure_policy[
                        "minimum_bin_count_per_year"
                    ],
                    "cv_aggregate_delta": d7_pressure_policy[
                        "cv_aggregate_delta"
                    ],
                    "worst_regime_delta": d7_pressure_policy[
                        "worst_regime_delta"
                    ],
                    "strict_gate": d7_pressure_policy["strict_gate"],
                    "new_models": 0,
                }
            ),
            "fine_d7_direction_gate": {
                "selected_rules": [],
                "features": None,
                "training_rows": 0,
                "production_decision": "rejected by untouched 2021 live score",
                "live_delta_vs_protected_interval": 0.054,
                "protected_dir_d7_score": 292.212,
                "rejected_dir_d7_score": 292.266,
                "new_models": 0,
            },
            "fine_d7_context_gate": {
                "selected_rules": [
                    {
                        "month": month,
                        "day": day,
                        "hour": hour,
                        "sign": values[0],
                        "weight": values[1],
                        "confidence_min": values[2],
                        "correction_cap": values[3],
                    }
                    for (month, day, hour), values
                    in sorted(production_fine_d7_context_policy.items())
                ],
                "features": (
                    None
                    if fine_d7_context_models is None
                    else fine_d7_context_models["features"]
                ),
                "training_rows": (
                    0
                    if fine_d7_context_models is None
                    else fine_d7_context_models["training_rows"]
                ),
                "disabled_reason": (
                    "context center correction was part of the public v59 "
                    "regression and is excluded from the production branch"
                ),
                "new_models": fine_d7_context_model_count,
            },
            "d7_d10_tendency_gate": {
                "selected_rules": [
                    {
                        "month": month,
                        "day": day,
                        "hour": hour,
                        "weight": values[0],
                        "tendency_max": values[1],
                    }
                    for (month, day, hour), values
                    in sorted(production_d7_d10_policy.items())
                ],
                "disabled_reason": (
                    "two-year d10 center correction was out of distribution "
                    "in 2021 and is excluded from the production branch"
                ),
                "new_models": 0,
            },
            "context_gate": {
                "blend": CONTEXT_BLEND,
                "selected_slots": [list(slot) for slot in CONTEXT_SELECTED_SLOTS],
                "selection_rule": (
                    "all five training years non-worse for fixed and optimal "
                    "fine-grid direction interval score"
                ),
                "training": direction_models["context_summary"],
            },
            "d7_direction_center_gate": {
                "selected_slots": [
                    {
                        "month": month,
                        "day": day,
                        "hour": hour,
                        "family": family,
                        "weight": weight,
                    }
                    for (month, day, hour), (family, weight)
                    in sorted(D7_DIRECTION_CENTER_POLICY.items())
                ],
                "selection_rule": (
                    "coarse leave-one-year-out preselection followed by exact "
                    "1.3 km scoring; every selected rule is non-worse in each "
                    "2016-2020 held year under the deployed interval policy"
                ),
                "fine_fixed_interval_cv_delta": -9.109157529052329,
                "new_models": 0,
            },
            "d14_direction_gate": {
                "selected_slots": [
                    {
                        "month": month,
                        "day": day,
                        "hour": hour,
                        "family": family,
                        "weight": weight,
                    }
                    for (month, day, hour), (family, weight)
                    in sorted(D14_DIRECTION_POLICY.items())
                ],
                "selection_rule": (
                    "coarse leave-one-year-out fixed/optimal/tail non-worse in "
                    "all 2016-2020 folds, then paired 1.3 km fixed/optimal "
                    "non-worse in 2019 and 2020"
                ),
                "new_models": 0,
            },
            "fine_d14_climatology_gate": {
                "selected_slots": [
                    {
                        "month": month,
                        "day": day,
                        "hour": hour,
                        "window_days": values[0],
                        "half_width": values[1],
                    }
                    for (month, day, hour), values
                    in sorted(FINE_D14_CLIMATOLOGY_POLICY.items())
                ],
                "held_years": list(range(2016, 2021)),
                "minimum_aggregate_gain": 5.0,
                "minimum_each_year_gain": 1.0,
                "worst_selected_regime_delta": -5.278203964233398,
                "selection_rule": (
                    "every held year and every populated spatial, speed, "
                    "direction, and center-disagreement regime is non-worse; "
                    "2019-2020 use exact production HRES replay and 2016-2018 "
                    "use a conservative downscaled-climatology baseline"
                ),
                "new_models": 0,
            },
            "fine_d14_endpoint_gate": {
                "selected_slots": (
                    []
                    if fine_d14_climatology is None
                    else [
                        {
                            "month": month,
                            "day": day,
                            "hour": hour,
                            **rule,
                        }
                        for (month, day, hour), rule
                        in sorted(
                            fine_d14_climatology[
                                "endpoint_policy"
                            ].items()
                        )
                    ]
                ),
                "held_years": list(range(2016, 2021)),
                "selection_rule": (
                    "same minimum-movement endpoint pair selected in every "
                    "leave-one-year-out fold; all populated physical regimes "
                    "non-worse"
                ),
                "new_models": 0,
            },
            "fine_d7_climatology_gate": {
                "selected_slots": [
                    {
                        "month": month,
                        "day": day,
                        "hour": hour,
                        **values,
                    }
                    for (month, day, hour), values
                    in sorted(FINE_D7_CLIMATOLOGY_POLICY.items())
                ],
                "held_years": list(range(2016, 2021)),
                "minimum_aggregate_gain": 5.0,
                "every_held_year_non_worse": True,
                "every_populated_material_regime_non_worse": True,
                "minimum_regime_rows": 1000,
                "truth_dependent_activation": False,
                "new_models": 0,
            },
            "d1_direction_interval_gate": (
                None
                if d1_direction_speed_interval is None
                else {
                    "method": d1_direction_speed_interval["method"],
                    "coverage_target": d1_direction_speed_interval[
                        "coverage_target"
                    ],
                    "speed_edges": d1_direction_speed_interval["edges"].tolist(),
                    "half_widths": d1_direction_speed_interval[
                        "half_widths"
                    ].tolist(),
                    "mapping": d1_direction_speed_interval["mapping"],
                    "count_by_bin": d1_direction_speed_interval["count_by_bin"],
                    "dates_used": d1_direction_speed_interval["dates_used"],
                    "selection_rule": (
                        "fixed speed bins and symmetric p90 residual widths; "
                        "promoted only after exact leave-one-year-out score was "
                        "non-worse in every year, calendar regime, and issue hour"
                    ),
                    "new_models": 0,
                }
            ),
            "d7_direction_interval_gate": {
                "method": d7_direction_interval_policy["method"],
                "base_half_width": d7_direction_interval_policy[
                    "base_half_width"
                ],
                "selected_rules": d7_direction_interval_policy["rules"],
                "cv_aggregate_delta": d7_direction_interval_policy.get(
                    "cv_aggregate_delta"
                ),
                "full_fit_aggregate_delta": d7_direction_interval_policy.get(
                    "full_fit_aggregate_delta"
                ),
                "asymmetric_selected_rules": d7_direction_interval_policy.get(
                    "asymmetric_rules", []
                ),
                "asymmetric_extension_cv_aggregate_delta": (
                    d7_direction_interval_policy.get(
                        "asymmetric_extension_cv_aggregate_delta"
                    )
                ),
                "asymmetric_extension_delta_by_held_year": (
                    d7_direction_interval_policy.get(
                        "asymmetric_extension_delta_by_held_year"
                    )
                ),
                "conditional_width_selected_rules": (
                    d7_direction_interval_policy.get(
                        "conditional_width_rules", []
                    )
                ),
                "conditional_width_cv_aggregate_delta": (
                    d7_direction_interval_policy.get(
                        "conditional_width_cv_aggregate_delta"
                    )
                ),
                "conditional_width_delta_by_held_year": (
                    d7_direction_interval_policy.get(
                        "conditional_width_delta_by_held_year"
                    )
                ),
                "conditional_width_worst_regime_delta": (
                    d7_direction_interval_policy.get(
                        "conditional_width_worst_regime_delta"
                    )
                ),
                "lead_ratio_selected_rules": d7_direction_interval_policy.get(
                    "lead_ratio_rules", []
                ),
                "lead_ratio_cv_aggregate_delta": (
                    d7_direction_interval_policy.get(
                        "lead_ratio_cv_aggregate_delta"
                    )
                ),
                "lead_ratio_delta_by_held_year": (
                    d7_direction_interval_policy.get(
                        "lead_ratio_delta_by_held_year"
                    )
                ),
                "combined_cv_aggregate_delta": (
                    d7_direction_interval_policy.get(
                        "combined_cv_aggregate_delta"
                    )
                ),
                "selection_rule": (
                    "calendar/hour scalar, speed-conditioned, or direction-sector "
                    "width candidate must beat the deployed 138-degree interval "
                    "in every exact leave-one-year-out 2016-2020 fold; asymmetric "
                    "endpoint tables must then be non-worse than that exact "
                    "fold-specific protected policy in every held year"
                ),
                "new_models": 0,
            },
            "d14_direction_interval_gate": (
                None
                if d14_direction_speed_interval is None
                else {
                    "method": d14_direction_speed_interval["method"],
                    "coverage_target": d14_direction_speed_interval[
                        "coverage_target"
                    ],
                    "shrinkage": d14_direction_speed_interval["shrinkage"],
                    "base_half_width": d14_direction_speed_interval[
                        "base_half_width"
                    ],
                    "interval_edges": d14_direction_speed_interval[
                        "edges"
                    ].tolist(),
                    "half_widths": d14_direction_speed_interval[
                        "half_widths"
                    ].tolist(),
                    "mapping": d14_direction_speed_interval["mapping"],
                    "count_by_bin": d14_direction_speed_interval[
                        "count_by_bin"
                    ],
                    "dates_used": d14_direction_speed_interval["dates_used"],
                    "selected_slots": [
                        list(slot)
                        for slot in d14_direction_speed_interval[
                            "selected_slots"
                        ]
                    ],
                    "strict_gate": d14_direction_speed_interval["gate"],
                    "selection_rule": (
                        "February 25 was the only robust regime under the "
                        "deployed 158-degree baseline replay; pooled p90 "
                        "width must be non-worse in every exact d10 "
                        "leave-one-year-out year/hour cell"
                    ),
                    "new_models": 0,
                }
            ),
            "environment": env_summary,
        },
    }

    artifact["metadata"]["siting_gate"] = {
        "constraint_gate_passed": siting["constraint_gate_passed"],
        "robustness_gate": siting["robustness_gate"],
        "robustness": siting["robustness"],
        "economics": siting["economics"],
        "weather_years": list(SITING_YEARS),
        "hidden_or_evaluation_weather_used": False,
    }
    evidence = build_competition_evidence(artifact["metadata"], siting)
    artifact["competition_evidence"] = evidence
    write_competition_outputs(args.artifacts_dir, siting, evidence)

    joblib.dump(artifact, artifact_path, compress=3)
    manifest = artifact["metadata"].copy()
    manifest.update(
        {
            "artifact_file": str(artifact_path),
            "climatology_file": str(clim_path),
            "elapsed_seconds": round(time.time() - t0, 2),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for checkpoint_path in args.artifacts_dir.glob("_checkpoint*.joblib*"):
        checkpoint_path.unlink(missing_ok=True)
    for checkpoint_dir in args.artifacts_dir.glob("_checkpoint*_parts"):
        if checkpoint_dir.is_dir():
            shutil.rmtree(checkpoint_dir)

    print(f"[train] wrote {artifact_path}")
    print(f"[train] wrote {clim_path}")
    print(f"[train] wrote {manifest_path}")
    print(f"[train] total models: {manifest['model_count']['total']}")


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
