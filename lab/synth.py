"""Synthetic collector run for testing the lab pipeline without API data.

python lab/synth.py /tmp/ytsynth   ->  /tmp/ytsynth/raw/<run>/{run.json,videos.jsonl.gz,channels.jsonl.gz}

Views follow channel_base * age^0.4 * exp(packaging effect + noise), with a known
title effect ("vs" and digits help), so a working pipeline must recover slope ~0.4
and beat the naive baseline.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORDS = "server homelab build gpu cheap nas proxmox linux mini pc upgrade test review fast quiet power".split()


def make(out: str | Path, n_channels: int = 60, per_channel: int = 60, seed: int = 0,
         thumbnail_urls: list[str] | None = None) -> Path:
    rng = np.random.default_rng(seed)
    captured = pd.Timestamp("2026-10-16T12:00:00Z")
    run_id = "20261016T000000Z"
    d = Path(out) / "raw" / run_id
    d.mkdir(parents=True, exist_ok=True)
    vids, chans = [], []
    for c in range(n_channels):
        cid = "UC" + f"{c:022d}"
        base = rng.lognormal(10, 1.2)
        chans.append({"run_id": run_id, "captured_at": captured.isoformat().replace("+00:00", "Z"),
                      "channel_id": cid, "title": f"Chan {c}", "bucket": ["niche", "adjacent", "far"][c % 3],
                      "subscriber_count": int(base * 3)})
        for i in range(per_channel):
            age = rng.uniform(5, 720)
            pub = captured - pd.Timedelta(days=float(age))
            words = list(rng.choice(WORDS, size=rng.integers(3, 9)))
            vs = rng.random() < 0.25
            digit = rng.random() < 0.3
            if vs:
                words.insert(1, "vs")
            if digit:
                words.append(str(int(rng.integers(2, 99))))
            title = " ".join(words).title()
            effect = 0.9 * vs + 0.5 * digit
            views = base * age ** 0.4 * np.exp(effect + rng.normal(0, 0.6))
            dur = float(rng.choice([45, rng.uniform(240, 2400)], p=[0.15, 0.85]))
            vid = f"v{c:03d}x{i:04d}"
            vids.append({
                "schema_version": 1, "run_id": run_id,
                "captured_at": captured.isoformat().replace("+00:00", "Z"),
                "video_id": vid, "channel_id": cid, "title": title,
                "published_at": pub.isoformat().replace("+00:00", "Z"),
                "duration_s": int(dur), "live_broadcast_content": "none",
                "was_live": bool(rng.random() < 0.03), "privacy_status": "public",
                "view_count": int(views), "like_count": int(views / 40), "comment_count": 3,
                "thumbnail_url": thumbnail_urls[i % len(thumbnail_urls)] if thumbnail_urls else None,
            })
    for name, rows in (("videos", vids), ("channels", chans)):
        with gzip.open(d / f"{name}.jsonl.gz", "wt") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    (d / "run.json").write_text(json.dumps({
        "run_id": run_id, "mode": "full", "status": "complete", "params": {"limit": None},
        "completed": [c["channel_id"] for c in chans], "n_channels": n_channels}))
    return d


if __name__ == "__main__":
    print(make(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ytsynth"))
