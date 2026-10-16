"""Helpers for the lab notebooks: load collector runs, build within-channel targets,
featurize titles and thumbnails, and evaluate within-channel ranking.

Kept out of the notebooks so they stay readable and the logic stays testable
(python -m unittest lab.test_ytlab).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

LONGFORM_S = 180
# YouTube began counting views from the first frame on this date; later uploads
# run at a different views-per-like ratio, so v0 leaves them out of the targets.
VIEW_RULE_CHANGE = pd.Timestamp("2026-08-24", tz="UTC")


# ---------------------------------------------------------------- loading

def list_runs(data_root: str | Path) -> list[dict]:
    root = Path(data_root) / "raw"
    runs = []
    for p in sorted(root.glob("*/run.json")):
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        s["_dir"] = p.parent
        s.setdefault("mode", "full")
        runs.append(s)
    return runs


def pick_full_run(runs: list[dict]) -> dict:
    """Latest complete full run; else the full run with the most collected channels
    (a paused first run is still usable, just with fewer channels)."""
    full = [r for r in runs if r["mode"] == "full" and not (r.get("params") or {}).get("limit")]
    if not full:
        raise FileNotFoundError("no full collector run found")
    done = [r for r in full if r.get("status") == "complete"]
    if done:
        return done[-1]
    return max(full, key=lambda r: len(r.get("completed", [])))


def read_jsonl_gz(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(rows)


def load_run(run: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    videos = read_jsonl_gz(run["_dir"] / "videos.jsonl.gz")
    if not videos.empty:
        videos = videos.drop_duplicates(["run_id", "video_id"], keep="last")
    channels = read_jsonl_gz(run["_dir"] / "channels.jsonl.gz")
    if not channels.empty:
        channels = channels.drop_duplicates(["run_id", "channel_id"], keep="last")
    return videos, channels


# ---------------------------------------------------------------- targets

def prepare(videos: pd.DataFrame, min_age_days: float = 30, min_videos: int = 20,
            exclude_after_rule_change: bool = True) -> pd.DataFrame:
    """Long-form, non-live, public, at least min_age_days old, positive views; channels with
    at least min_videos qualifying uploads. Adds age_days, log_views, log_age."""
    df = videos.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True)
    df["age_days"] = (df["captured_at"] - df["published_at"]).dt.total_seconds() / 86400
    keep = (
        (df["duration_s"].fillna(0) > LONGFORM_S)
        & (df["live_broadcast_content"].fillna("none") == "none")
        & (~df["was_live"].fillna(False).astype(bool))
        & (df["privacy_status"].fillna("public") == "public")
        & (df["view_count"].fillna(0) > 0)
        & (df["age_days"] >= min_age_days)
    )
    if exclude_after_rule_change:
        keep &= df["published_at"] < VIEW_RULE_CHANGE
    df = df[keep].copy()
    counts = df.groupby("channel_id")["video_id"].transform("size")
    df = df[counts >= min_videos].copy()
    df["log_views"] = np.log(df["view_count"].astype(float))
    df["log_age"] = np.log(df["age_days"].astype(float))
    return df.reset_index(drop=True)


def fit_age_slope(df: pd.DataFrame, mask: pd.Series | None = None) -> float:
    """Pooled within-channel slope of log views on log age (channel fixed effects)."""
    d = df if mask is None else df[mask]
    lv = d["log_views"] - d.groupby("channel_id")["log_views"].transform("mean")
    la = d["log_age"] - d.groupby("channel_id")["log_age"].transform("mean")
    den = float((la * la).sum())
    return float((lv * la).sum() / den) if den > 0 else 0.0


def add_target(df: pd.DataFrame, slope: float, clip: float = 5.0) -> pd.DataFrame:
    """y = robust z (median / 1.4826*MAD) of age-adjusted log views within each channel."""
    df = df.copy()
    df["resid"] = df["log_views"] - slope * df["log_age"]
    g = df.groupby("channel_id")["resid"]
    med = g.transform("median")
    mad = (df["resid"] - med).abs().groupby(df["channel_id"]).transform("median")
    scale = (1.4826 * mad).replace(0, np.nan)
    df["y"] = ((df["resid"] - med) / scale).clip(-clip, clip)
    df["multiple"] = np.exp(df["resid"] - med)  # age-adjusted views / channel median
    return df[df["y"].notna()].reset_index(drop=True)


def stable_hash(s: str, seed: int = 0) -> float:
    h = hashlib.sha256(f"{seed}:{s}".encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def assign_splits(df: pd.DataFrame, test_channel_frac: float = 0.2, val_channel_frac: float = 0.1,
                  time_holdout_days: int = 120, seed: int = 0) -> pd.DataFrame:
    """split column:
    test_channel  every video of a held-out channel (generalization to unseen channels)
    test_time     the newest uploads of the remaining channels (generalization forward in time)
    val           early-stopping channels
    train         everything else"""
    df = df.copy()
    h = df["channel_id"].map(lambda c: stable_hash(c, seed))
    is_test_ch = h < test_channel_frac
    is_val_ch = (h >= test_channel_frac) & (h < test_channel_frac + val_channel_frac)
    cutoff = df["published_at"].max() - pd.Timedelta(days=time_holdout_days)
    is_recent = df["published_at"] > cutoff
    df["split"] = "train"
    df.loc[is_val_ch, "split"] = "val"
    df.loc[~is_test_ch & ~is_val_ch & is_recent, "split"] = "test_time"
    df.loc[is_test_ch, "split"] = "test_channel"
    return df


# ---------------------------------------------------------------- features

_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


def title_features(df: pd.DataFrame) -> pd.DataFrame:
    t = df["title"].fillna("")
    letters = t.str.count(r"[A-Za-z]").replace(0, np.nan)
    out = pd.DataFrame({
        "t_chars": t.str.len(),
        "t_words": t.str.split().str.len(),
        "t_has_digit": t.str.contains(r"\d").astype(int),
        "t_has_question": t.str.contains(r"\?").astype(int),
        "t_has_exclaim": t.str.contains("!").astype(int),
        "t_has_dollar": t.str.contains(r"\$").astype(int),
        "t_has_vs": t.str.contains(r"\bvs\.?\b", case=False).astype(int),
        "t_has_paren": t.str.contains(r"[\(\[]").astype(int),
        "t_caps_ratio": (t.str.count(r"[A-Z]") / letters).fillna(0),
        "t_emoji": t.map(lambda s: len(_EMOJI.findall(s))),
        "log_duration": np.log1p(df["duration_s"].fillna(0)),
        "pub_hour_utc": df["published_at"].dt.hour,
        "pub_dow": df["published_at"].dt.dayofweek,
    }, index=df.index)
    return out


def _cache_key(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def embed_titles(df: pd.DataFrame, model_name: str, device: str, cache_dir: Path,
                 batch_size: int = 256) -> np.ndarray:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir) / f"title_{_cache_key(model_name)}.npz"
    have = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        have = dict(zip(z["ids"].tolist(), z["emb"]))
    # Key by id + title: a retitled video (A/B test) gets a fresh embedding.
    keys = (df["video_id"] + "|" + df["title"].fillna("")).tolist()
    missing = [i for i, k in enumerate(keys) if k not in have]
    if missing:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device=device)
        texts = [df["title"].iloc[i] or "" for i in missing]
        emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True)
        for i, e in zip(missing, emb):
            have[keys[i]] = e.astype(np.float32)
        ids = np.array(list(have.keys()))
        np.savez(cache, ids=ids, emb=np.stack(list(have.values())))
    return np.stack([have[k] for k in keys])


def download_thumbnails(df: pd.DataFrame, thumb_dir: Path, workers: int = 16,
                        timeout: float = 20.0) -> pd.Series:
    """Fetch each video's thumbnail_url to thumb_dir/<video_id>.jpg (skips existing).
    Returns a Series of local paths (None where the download failed)."""
    import requests
    thumb_dir = Path(thumb_dir)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    def fetch(row):
        vid, url = row
        path = thumb_dir / f"{vid}.jpg"
        if path.exists() and path.stat().st_size > 0:
            return str(path)
        if not isinstance(url, str) or not url:
            return None
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code != 200 or not r.content:
                return None
            tmp = path.with_suffix(".part")
            tmp.write_bytes(r.content)
            os.replace(tmp, path)
            return str(path)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        paths = list(ex.map(fetch, zip(df["video_id"], df["thumbnail_url"])))
    return pd.Series(paths, index=df.index)


def _open_rgb(path: str | None, small: tuple[int, int] | None = None):
    from PIL import Image
    if not isinstance(path, str) or not path:
        return None
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return None
    if small:
        # Render at phone-thumbnail size, then let the processor upscale: tests legibility.
        im = im.resize(small, Image.BILINEAR)
    return im


def embed_images(paths: list[str | None], model_name: str, device: str, cache_dir: Path,
                 tag: str = "full", small: tuple[int, int] | None = None,
                 batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Image embeddings (L2-normalized) and a has_image mask. Cached per path+tag."""
    import torch
    from transformers import AutoModel, AutoProcessor
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir) / f"img_{_cache_key(model_name)}_{tag}.npz"
    have = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        have = dict(zip(z["ids"].tolist(), z["emb"]))
    paths = [p if isinstance(p, str) and p else None for p in paths]
    keys = [Path(p).stem if p else None for p in paths]
    todo = [i for i, k in enumerate(keys) if k and k not in have]
    if todo:
        model = AutoModel.from_pretrained(model_name).to(device).eval()
        proc = AutoProcessor.from_pretrained(model_name)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = model.to(dtype)
        for s in range(0, len(todo), batch_size):
            idx = todo[s:s + batch_size]
            ims = [(i, _open_rgb(paths[i], small)) for i in idx]
            ims = [(i, im) for i, im in ims if im is not None]
            if not ims:
                continue
            inputs = proc(images=[im for _, im in ims], return_tensors="pt")
            inputs = {k: (v.to(device, dtype) if v.is_floating_point() else v.to(device))
                      for k, v in inputs.items() if hasattr(v, "to")}
            with torch.no_grad():
                out = model.get_image_features(**inputs)
            feats = out.pooler_output if hasattr(out, "pooler_output") else out
            feats = torch.nn.functional.normalize(feats.float(), dim=-1).cpu().numpy()
            for (i, _), e in zip(ims, feats):
                have[keys[i]] = e.astype(np.float32)
        if have:
            np.savez(cache, ids=np.array(list(have.keys())), emb=np.stack(list(have.values())))
    dim = next(iter(have.values())).shape[0] if have else 1
    emb = np.zeros((len(paths), dim), dtype=np.float32)
    mask = np.zeros(len(paths), dtype=bool)
    for i, k in enumerate(keys):
        if k and k in have:
            emb[i] = have[k]
            mask[i] = True
    return emb, mask


def image_stats(paths: list[str | None]) -> pd.DataFrame:
    """Cheap, interpretable thumbnail statistics on a 64x36 render."""
    rows = []
    for p in paths:
        im = _open_rgb(p, (64, 36))
        if im is None:
            rows.append({"img_brightness": np.nan, "img_contrast": np.nan, "img_saturation": np.nan,
                         "img_colorfulness": np.nan})
            continue
        a = np.asarray(im, dtype=np.float32) / 255.0
        hsv = np.asarray(im.convert("HSV"), dtype=np.float32) / 255.0
        lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        rg = a[..., 0] - a[..., 1]
        yb = 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
        rows.append({"img_brightness": float(lum.mean()), "img_contrast": float(lum.std()),
                     "img_saturation": float(hsv[..., 1].mean()),
                     "img_colorfulness": float(math.hypot(rg.std(), yb.std())
                                               + 0.3 * math.hypot(rg.mean(), yb.mean()))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- evaluation

def within_channel_spearman(df: pd.DataFrame, pred: str, y: str = "y", min_n: int = 10,
                            n_boot: int = 1000, seed: int = 0) -> dict:
    """Mean per-channel Spearman correlation between prediction and target, with a
    bootstrap 95% CI over channels. Channels with fewer than min_n videos are skipped."""
    from scipy.stats import spearmanr
    rhos = []
    for _, g in df.groupby("channel_id"):
        if len(g) < min_n or g[y].nunique() < 2 or g[pred].nunique() < 2:
            continue
        rhos.append(spearmanr(g[pred], g[y]).statistic)
    rhos = np.array([r for r in rhos if np.isfinite(r)])
    if len(rhos) == 0:
        return {"rho": np.nan, "lo": np.nan, "hi": np.nan, "channels": 0}
    rng = np.random.default_rng(seed)
    boots = rng.choice(rhos, size=(n_boot, len(rhos)), replace=True).mean(axis=1)
    return {"rho": float(rhos.mean()), "lo": float(np.quantile(boots, 0.025)),
            "hi": float(np.quantile(boots, 0.975)), "channels": int(len(rhos))}
