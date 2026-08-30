# Data placement

This directory is a mount point for official competition data. Dataset contents
are ignored by Git and must not be committed.

## Expected layout

After downloading and extracting the official archives, arrange the files as:

```text
data/
|-- phase2/
|   `-- phase2_dataset_ship/
|       |-- train/
|       |-- static/
|       `-- inference/
|           |-- window_1/
|           |-- window_2/
|           |-- ...
|           `-- window_8/
`-- phase1/
    `-- phase1_dataset/
        |-- train/
        |-- inference/
        `-- scoring/
```

Use the `train/` and `static/` directories from Zenodo record `20335351`. For
the definitive run, use the eight inference windows from Zenodo record
`20874645` as `phase2_dataset_ship/inference/`. The resulting Phase 2 root must
therefore expose `train`, `static`, and the definitive `inference` directory at
the same level.

The Phase 1 root is supplied separately to `train.py` through
`--phase1-data-root`; it provides the official HRES driver inputs required by
the Phase 2 kit.

Example arguments from the repository root:

```text
--data-root data/phase2/phase2_dataset_ship
--phase1-data-root data/phase1/phase1_dataset
```

The scripts set `PHASE2_DATA_ROOT` internally and validate the discovered files
before model fitting or inference.
