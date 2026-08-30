# External resources

This repository does not redistribute competition datasets, organizer code,
external model weights, or external forecast archives.

## Sea Winds Phase 2 kit and datasets

- Source code: <https://github.com/DavidMedernach/Hackathon-Sea-Winds-Predictions/tree/phase_2>
- Training data: <https://zenodo.org/records/20335351>
- Definitive inference data: <https://zenodo.org/records/20874645>
- Use: official data loading, forecast assembly, target-grid definition,
  bathymetry, turbine specification, wake simulation, and competition inputs.

Users must obtain these resources directly from the organizers and comply with
their published terms.

## GraphCast and WeatherBench 2

- GraphCast source: <https://github.com/google-deepmind/graphcast>
- WeatherBench 2: <https://weatherbench2.readthedocs.io/>
- Use: causal ERA5-trained trajectory output for a strictly gated
  wind-direction component. Only information initialized at the relevant issue
  time is materialized. The external output is not redistributed by this
  repository.

The competition organizers explicitly permitted outputs from ERA5-trained
public weather models under causal initialization conditions. This resource is
disclosed here and in the accompanying methodology report.

## Python libraries

Runtime libraries are listed in `requirements.txt`. They remain subject to
their respective licenses; no library source is vendored into this repository.
