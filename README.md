# Identifying flow features with learned color & opacity transfer functions

## Install

```sh
uv sync
source .venv/bin/activate
```

## Build train data

```sh
cd ahmedML

# Prepares training datasets for all the runs listed in the CSV
python batch_build_cases.py --runs data/geo_parameters_all.csv --workers 3 --no-vti

# See usage
python batch_build_cases.py --help

usage: batch_build_cases.py [-h] --runs RUNS [--data DATA] [--out OUT] [--n-per-H N_PER_H] [--keep-vtu] [--workers WORKERS]
                            [--save-vti]
                            [extra ...]

positional arguments:
  extra              extra args passed to build_training_case.py (after --

options:
  -h, --help         show this help message and exit
  --runs RUNS        comma-separated ids, or path to a csv with a 'run' column.
  --data DATA        Where the downloaded datasets go
  --out OUT          Where the {tag}_labels.npz go
  --n-per-H N_PER_H
  --keep-vtu
  --workers WORKERS  number of builds to run at the same time
  --save-vti         pass --save-vti to build_training_case.py
```
