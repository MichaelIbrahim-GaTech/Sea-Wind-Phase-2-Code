# Sea Wind Phase 2 Code

Repository: <https://github.com/MichaelIbrahim-GaTech/Sea-Wind-Phase-2-Code>

Reproducible Phase 2 solution for the Sea Winds Predictions competition. The
repository contains exactly two executable Python files:

- `train.py` trains the forecasting system and wind-farm siting policy from the
  permitted input data, then saves all learned artifacts.
- `inference.py` loads those artifacts, generates `predictions.csv`, generates
  the wind-farm `submission.json`, and packages both files in a submission ZIP.

No previous submission, hidden label, or precomputed competition prediction is
read by either program. Competition datasets and trained artifacts are excluded
from Git; the exact final submission archive is retained through Git LFS for
verification and direct inspection.

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
|-- docs/
|   `-- methodology-report.pdf
`-- submission/
    `-- final_submission.zip
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
  --artifacts-dir artifacts \
  --train-freq 6D \
  --downscale-year 2020 \
  --downscale-step 20 \
  --coverage-target 0.90
```

These are the settings used to generate the final submission. Interval
calibration was enabled. The fit retained 29 models after the chronological and
worst-regime gates: 15 speed-quantile models, five speed-context endpoint
models, four circular direction-residual models, one shared spatial direction
model, two conditional direction-endpoint models, and two terrain-downscaling
models. The run saved one serialized forecasting bundle and the audited siting
and economics evidence under `artifacts/`, including:

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
  --archive output/submission.zip \
  --eval-year 2022 \
  --window-base 0 \
  --speed-width-scale 1:0.75 \
  --d7-center-policy-max-weight 0.4 \
  --d1-context-blend-scale 1.25 \
  --dir-halfwidth-cap-deg 14:145
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

## Included final submission

[`submission/final_submission.zip`](submission/final_submission.zip) is the
exact final archive generated by the commands above. It contains the scored
`predictions.csv` and the economy-aware `submission.json`. Because the archive
is larger than GitHub's regular-file limit, it is stored with Git LFS; run
`git lfs install` before cloning or pulling the repository.

The final forecast produced 4,196,640 matched rows. Its definitive evaluation
metrics were:

| Component | Score |
|---|---:|
| Wind speed +1 day | 9.511 |
| Wind speed +7 days | 17.124 |
| Wind speed +14 days | 16.835 |
| Wind direction +1 day | 76.400 |
| Wind direction +7 days | 335.571 |
| Wind direction +14 days | 315.183 |
| Primary score | 1.497276 |

## Final wind-farm placement

The included `submission.json` uses the final economics-gated placement, not an
earlier yield-only candidate:

- Farm centre: 54.109676 degrees north, 0.952316 degrees east
- Turbines: 55 IEA 22 MW units, 1.21 GW total capacity
- Centre depth: 44.14 m
- Minimum turbine spacing: 1,554.64 m
- Mean and worst-year training capacity factor: 0.5036 and 0.4818
- Mean and worst-year AEP: 5,337.7 and 5,106.9 GWh
- Mean and maximum wake loss: 7.10% and 7.31%
- Mean-weather CAPEX: EUR 4,478.4 million
- Mean and worst-weather LCOE: EUR 88.69 and EUR 92.70 per MWh

Candidate promotion required all geographic and engineering constraints to
pass, non-worse capacity factor in every training year, robust neighboring-cell
performance, controlled wake loss, and explicit mean- and worst-weather LCOE
gates using the organizer's cost model. The final archive also contains 96
ordered farm-power quantiles for the optional bidding component.

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
