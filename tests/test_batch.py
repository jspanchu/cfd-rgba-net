"""Verification: the run specification, and which runs the batch driver builds, keeps and reports."""

import pytest

from cfd_rgba_net import batch
from cfd_rgba_net.batch import parse_runs

DATASET_YAML = """dataset: T
files:
  volume: run_{run}/v.vtu
  body: run_{run}/b.stl
fields: {velocity: [U], pressure: [p]}
field_source: point
flow:
  U_inf: 1.0
  nu: 1.0e-5
"""


def test_parse_runs_lists_and_ranges():
    assert parse_runs("1,5,7") == [1, 5, 7]
    assert parse_runs("0-3,10") == [0, 1, 2, 3, 10]


def test_parse_runs_reads_a_csv_column_and_drops_duplicates(tmp_path):
    f = tmp_path / "runs.csv"
    f.write_text("run,length\n3,1.0\n1,2.0\n3,3.0\n")
    assert parse_runs(str(f)) == [3, 1]
    assert parse_runs("1,2-3,3") == [1, 2, 3]


def test_parse_runs_rejects_a_backwards_range():
    with pytest.raises(ValueError, match="backwards"):
        parse_runs("5-3")


@pytest.mark.parametrize("keep,kept", [(False, {3}), (True, {2, 3, 4})])
def test_main_skips_built_runs_reports_failures_and_cleans_up(tmp_path, monkeypatch, keep, kept):
    """Run 1 is already built, run 3 fails: only 2-4 are built, the exit code is 1, and a failed run's files stay."""
    dataset, out, work = tmp_path / "t.yaml", tmp_path / "out", tmp_path / "work"
    dataset.write_text(DATASET_YAML)
    (out / "t_run1.zarr").mkdir(parents=True)
    built = []

    def fake_download(cfg, run, work):
        (work / f"run_{run}").mkdir(parents=True, exist_ok=True)

    def fake_build(cfg, run, args):
        built.append(run)
        return run != 3

    monkeypatch.setattr(batch, "download", fake_download)
    monkeypatch.setattr(batch, "build", fake_build)
    argv = ["--dataset", str(dataset), "--runs", "1-4", "--out", str(out), "--work", str(work)]
    code = batch.main(argv + (["--keep"] if keep else []))

    assert sorted(built) == [2, 3, 4]
    assert code == 1
    assert {r for r in range(1, 5) if (work / f"run_{r}").exists()} == kept
