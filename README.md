# Sea Wind Phase 2 Code

Reproducible Phase 2 solution for the Sea Winds Predictions competition. The
repository contains exactly two executable Python files:

- `train.py` trains the forecasting system and wind-farm siting policy from the
  permitted input data, then saves all learned artifacts.
- `inference.py` loads those artifacts, generates `predictions.csv`, generates
  the wind-farm `submission.json`, and packages both files in a submission ZIP.

No previous submission, hidden label, or precomputed competition prediction is
read by either program. Generated data, trained artifacts, and competition
datasets are intentionally excluded from Git.

## Method summary

The forecasting system combines organizer-provided HRES forecasts with compact
statistical post-processing, terrain-aware downscaling, conformal interval
calibration, circular wind-direction models, and a strictly validated causal
GraphCast trajectory component. Model selection uses chronological and
worst-regime gates. The siting component searches eligible shallow-water cells
and optimizes a 55-turbine IEA 22 MW layout under the competition's geographic,
depth, footprint, and spacing constraints.

The full scientific description is available in
[`docs/methodology-report.pdf`](docs/methodology-report.pdf).

## Repository contents

```text
Sea-Wind-Phase-2-Code/
|-- train.py
|-- inference.py
|-- requirements.txt
|-- README.md
|-- THIRD_PARTY.md
|-- LICENSE
|-- data/
|   `-- README.md
`-- docs/
    `-- methodology-report.pdf
```

`artifacts/` and `output/` are created during execution and are ignored by Git.

## Required inputs

1. The official Phase 2 kit, branch `phase_2`:
   <https://github.com/DavidMedernach/Hackathon-Sea-Winds-Predictions/tree/phase_2>
2. The official Phase 2 and Phase 1 datasets from Zenodo record `20335351`:
   <https://zenodo.org/records/20335351>
3. The definitive Phase 2 inference windows from Zenodo record `20874645`:
   <https://zenodo.org/records/20874645>
4. Internet access during training for the authorized ERA5-trained GraphCast
   output hosted by WeatherBench 2. The materialized causal fields are stored in
   the trained artifact bundle; inference does not train or download a model.

The datasets are too large to redistribute in this repository. See
[`data/README.md`](data/README.md) for the exact expected layout.

## Environment setup

Python 3.11 is recommended. Create an isolated environment and install the
runtime dependencies:

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Clone the official kit beside the two scripts:

```bash
git clone --branch phase_2 --single-branch \
  https://github.com/DavidMedernach/Hackathon-Sea-Winds-Predictions.git \
  official-kit
```

## Train

From the repository root, run:

```bash
python train.py \
  --kit-dir official-kit \
  --data-root data/phase2/phase2_dataset_ship \
  --phase1-data-root data/phase1/phase1_dataset \
  --artifacts-dir artifacts
```

The default training frequency is `6D`, chosen as the frugal configuration.
`--train-freq 3D` is supported for a slower, denser fit. The default run saves a
single serialized forecasting bundle and the siting evidence under
`artifacts/`, including:

```text
artifacts/phase2_forecast_artifacts.joblib
artifacts/climatology_coarse.npz
artifacts/siting_submission.json
artifacts/competition_evidence.json
artifacts/methodology_economics_compute.md
artifacts/manifest.json
```

## Infer and package

After training, run:

```bash
python inference.py \
  --kit-dir official-kit \
  --data-root data/phase2/phase2_dataset_ship \
  --artifacts-dir artifacts \
  --output output/predictions.csv \
  --archive output/submission.zip
```

The resulting `output/submission.zip` contains the two competition products at
the archive root:

```text
predictions.csv
submission.json
```

The forecast table is validated before packaging for row count, key coverage,
finite values, quantile ordering, and direction normalization. The siting JSON
is generated from the training artifact and includes exactly 55 relative
turbine coordinates plus optional farm-power quantiles derived from the same
forecast.

## Reproducibility notes

- Training targets are restricted to the organizer-provided 2016-2020 period.
- Definitive inference inputs are used only as unlabeled causal predictors.
- Random seeds, single-threaded model fitting, feature definitions, gates, and
  post-processing policies are encoded in `train.py` and serialized.
- The organizer kit remains an explicit dependency because it defines the
  official data readers, footprint, turbine model, and PyWake simulation path.
- External resources and their roles are disclosed in `THIRD_PARTY.md`.

## Author

Michael Ibrahim
