"""Offline tests for lab/ytlab.py (no GPU, no network): python -m unittest lab.test_ytlab"""
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import synth  # noqa: E402
import ytlab  # noqa: E402


class LabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        synth.make(cls.root, n_channels=60, per_channel=60)
        runs = ytlab.list_runs(cls.root)
        cls.full_run = ytlab.pick_full_run(runs)
        cls.videos, cls.channels = ytlab.load_run(cls.full_run)

    def test_load(self):
        self.assertEqual(len(self.videos), 3600)
        self.assertEqual(len(self.channels), 60)

    def test_prepare_filters(self):
        df = ytlab.prepare(self.videos)
        self.assertTrue((df["duration_s"] > ytlab.LONGFORM_S).all())
        self.assertTrue((df["age_days"] >= 30).all())
        self.assertFalse(df["was_live"].any())
        self.assertTrue((df["published_at"] < ytlab.VIEW_RULE_CHANGE).all())
        self.assertTrue((df.groupby("channel_id").size() >= 20).all())

    def test_slope_and_target(self):
        df = ytlab.assign_splits(ytlab.prepare(self.videos))
        slope = ytlab.fit_age_slope(df, df["split"] == "train")
        self.assertAlmostEqual(slope, 0.4, delta=0.08)
        df = ytlab.add_target(df, slope)
        med = df.groupby("channel_id")["y"].median().abs().max()
        self.assertLess(med, 1e-9)
        self.assertTrue(df["y"].between(-5, 5).all())

    def test_splits_disjoint_and_stable(self):
        df = ytlab.prepare(self.videos)
        a = ytlab.assign_splits(df, seed=0)
        b = ytlab.assign_splits(df, seed=0)
        pd.testing.assert_series_equal(a["split"], b["split"])
        test_ch = set(a.loc[a.split == "test_channel", "channel_id"])
        other_ch = set(a.loc[a.split != "test_channel", "channel_id"])
        self.assertFalse(test_ch & other_ch)
        val_ch = set(a.loc[a.split == "val", "channel_id"])
        self.assertFalse(val_ch & set(a.loc[a.split.isin(["train", "test_time"]), "channel_id"]))
        self.assertGreater((a.split == "test_time").sum(), 0)

    def test_spearman(self):
        df = pd.DataFrame({"channel_id": ["a"] * 12 + ["b"] * 12, "y": list(range(12)) * 2})
        df["pred"] = df["y"] * 2.0
        r = ytlab.within_channel_spearman(df, "pred", n_boot=50)
        self.assertAlmostEqual(r["rho"], 1.0)
        self.assertEqual(r["channels"], 2)
        df["pred"] = -df["y"]
        self.assertAlmostEqual(ytlab.within_channel_spearman(df, "pred", n_boot=50)["rho"], -1.0)
        small = df.groupby("channel_id").head(5)
        self.assertEqual(ytlab.within_channel_spearman(small, "pred")["channels"], 0)

    def test_pick_prefers_complete_and_skips_smoke(self):
        runs = [{"run_id": "1", "mode": "full", "status": "complete", "params": {"limit": 3}, "completed": [1, 2, 3]},
                {"run_id": "2", "mode": "full", "status": "paused_quota", "params": {}, "completed": list(range(800))},
                {"run_id": "3", "mode": "refresh", "status": "complete", "params": {}, "completed": []}]
        self.assertEqual(ytlab.pick_full_run(runs)["run_id"], "2")
        runs.append({"run_id": "4", "mode": "full", "status": "complete", "params": {}, "completed": [1]})
        self.assertEqual(ytlab.pick_full_run(runs)["run_id"], "4")

    def test_title_features(self):
        df = pd.DataFrame({"title": ["GPU vs CPU: Which Wins?", "my $500 NAS build!!", None],
                           "duration_s": [600, 1200, None],
                           "published_at": pd.to_datetime(["2026-01-01T10:00Z"] * 3, utc=True)})
        f = ytlab.title_features(df)
        self.assertEqual(f["t_has_vs"].tolist(), [1, 0, 0])
        self.assertEqual(f["t_has_dollar"].tolist(), [0, 1, 0])
        self.assertEqual(f["t_has_question"].tolist(), [1, 0, 0])
        self.assertEqual(f["t_chars"].tolist()[2], 0)

    def test_image_stats_missing(self):
        s = ytlab.image_stats([None, "/nonexistent.jpg"])
        self.assertTrue(s.isna().all().all())


if __name__ == "__main__":
    unittest.main()
