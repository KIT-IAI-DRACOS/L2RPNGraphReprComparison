"""Tests for the multi-seed survival helpers of analysis.episodes."""
from pathlib import Path

import numpy as np
import pytest

from analysis.episodes import make_episodes, save_episodes, survival_over_seeds, survival_per_seed

MAX_STEP = 100


def write_task(folder: Path, tag: str, task: int, steps: list[int]) -> None:
    """Write one array task's episode table with one chronic per entry of `steps`."""
    rows = [{"agent": "a", "chronic": f"c{task}_{i}", "seed": i, "steps_survived": s, "max_step": MAX_STEP}
            for i, s in enumerate(steps)]
    save_episodes(make_episodes(rows), folder / f"episodes_{tag}_{task:03d}.csv")


@pytest.fixture
def folders(tmp_path: Path) -> dict[int, Path]:
    """Seed 0: survives 2 of 4 chronics; seed 1: survives 1 of 4; seed 2: no files."""
    seed0, seed1, seed2 = tmp_path / "s0", tmp_path / "s1", tmp_path / "s2"
    write_task(seed0, "valenv", 0, [100, 50])
    write_task(seed0, "valenv", 1, [100, 50])
    write_task(seed1, "valenv", 0, [100, 0])
    write_task(seed1, "valenv", 1, [20, 20])
    write_task(seed1, "testenv", 0, [100])  # other split, must not be mixed in
    seed2.mkdir()
    return {0: seed0, 1: seed1, 2: seed2}


def test_survival_per_seed_concatenates_tasks_and_skips_seeds_without_files(folders: dict[int, Path]) -> None:
    per_seed = survival_per_seed(folders, "valenv")
    assert list(per_seed.index) == [0, 1]
    assert per_seed.loc[0, "chronics"] == 4
    assert per_seed.loc[0, "full_survival_pct"] == pytest.approx(50.0)
    assert per_seed.loc[0, "mean_survival_pct"] == pytest.approx(75.0)
    assert per_seed.loc[1, "full_survival_pct"] == pytest.approx(25.0)
    assert per_seed.loc[1, "mean_survival_pct"] == pytest.approx(35.0)


def test_survival_per_seed_without_files_is_empty_with_columns(tmp_path: Path) -> None:
    per_seed = survival_per_seed({0: tmp_path}, "valenv")
    assert per_seed.empty
    assert {"chronics", "mean_survival_pct", "full_survival_pct"} <= set(per_seed.columns)
    assert survival_over_seeds(per_seed).seeds == 0


def test_survival_over_seeds_uses_sample_sd(folders: dict[int, Path]) -> None:
    summary = survival_over_seeds(survival_per_seed(folders, "valenv"))
    assert summary.seeds == 2
    assert summary.steps_pct == pytest.approx(55.0)
    assert summary.steps_sd == pytest.approx(np.std([75.0, 35.0], ddof=1))
    assert summary.episodes_pct == pytest.approx(37.5)
    assert summary.episodes_sd == pytest.approx(np.std([50.0, 25.0], ddof=1))


def test_survival_over_seeds_single_seed_has_nan_sd(folders: dict[int, Path]) -> None:
    summary = survival_over_seeds(survival_per_seed({0: folders[0]}, "valenv"))
    assert summary.seeds == 1
    assert np.isnan(summary.steps_sd) and np.isnan(summary.episodes_sd)
