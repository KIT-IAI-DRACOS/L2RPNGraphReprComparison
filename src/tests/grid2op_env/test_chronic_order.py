"""
Tests for the per-worker seeds and chronic orders of grid2op_env.utils (shuffled training, sharded evaluation).

Needs the local l2rpn_case14_sandbox_val env (python setup_envs.py).
"""
from __future__ import annotations

import unittest

import grid2op
from lightsim2grid import LightSimBackend

from grid2op_env.utils import chronic_seed, chronic_shard, shuffle_chronics, worker_seed

ENV_NAME = "l2rpn_case14_sandbox_val"


def make_env() -> grid2op.Environment.Environment:
    """A fresh case14 val env (chronics in grid2op's default sorted order)."""
    return grid2op.make(ENV_NAME, backend=LightSimBackend())


def reset_names(env, n: int) -> list[str]:
    """Names of the chronics played by the next `n` resets."""
    names = []
    for _ in range(n):
        env.reset()
        names.append(env.chronics_handler.get_name())
    return names


class TestWorkerSeed(unittest.TestCase):
    def test_differs_per_worker_and_env_and_is_reproducible(self):
        seeds = {worker_seed(42, w, v) for w in range(25) for v in range(2)}
        self.assertEqual(len(seeds), 50)
        self.assertEqual(worker_seed(42, 3), worker_seed(42, 3))
        self.assertNotEqual(worker_seed(42, 3), worker_seed(43, 3))
        self.assertTrue(all(0 <= s < 2**31 for s in seeds))


class TestShuffleChronics(unittest.TestCase):
    def setUp(self):
        self.env = make_env()
        self.n = len(self.env.chronics_handler.subpaths)

    def tearDown(self):
        self.env.close()

    def test_resets_follow_the_returned_order_without_repeats(self):
        order = shuffle_chronics(self.env, seed=1)
        played = reset_names(self.env, self.n)
        self.assertEqual(sorted(played), sorted(str(p).split("/")[-1] for p in self.env.chronics_handler.subpaths))
        self.assertEqual(len(set(played)), self.n)
        # grid2op's next reset plays the chronic after the current one, so the first of the order comes last.
        names = [p.split("/")[-1] for p in order]
        self.assertEqual(played, names[1:] + names[:1])

    def test_same_seed_same_order_other_seed_other_order(self):
        first = shuffle_chronics(self.env, seed=1)
        other = make_env()
        try:
            self.assertEqual(shuffle_chronics(other, seed=1), first)
        finally:
            other.close()
        third = make_env()
        try:
            self.assertNotEqual(shuffle_chronics(third, seed=2), first)
        finally:
            third.close()

    def test_order_is_not_sorted(self):
        order = shuffle_chronics(self.env, seed=1)
        self.assertNotEqual(order, sorted(order))


class TestChronicShard(unittest.TestCase):
    def setUp(self):
        self.env = make_env()
        self.all = sorted(str(p) for p in self.env.chronics_handler.subpaths)

    def tearDown(self):
        self.env.close()

    def test_shards_cover_every_chronic_once(self):
        shards = [chronic_shard(self.env, i, 4) for i in range(4)]
        self.assertEqual(sorted(c for shard in shards for c in shard), self.all)
        self.assertEqual(shards[1], self.all[1::4])

    def test_set_id_with_seed_plays_the_shard_reproducibly(self):
        """What CustomizedGrid2OpEnvironment does per evaluation episode: same chronic + seed -> same maintenance."""
        shard = chronic_shard(self.env, 0, 25)
        played, maintenance = [], []
        for chronic in shard * 2:
            self.env.seed(chronic_seed(42, chronic))
            self.env.set_id(chronic)
            obs = self.env.reset()
            played.append(self.env.chronics_handler.get_name())
            maintenance.append(obs.time_next_maintenance.tolist())
        self.assertEqual(played, [c.split("/")[-1] for c in shard * 2])
        self.assertEqual(maintenance[: len(shard)], maintenance[len(shard):])


if __name__ == "__main__":
    unittest.main()
