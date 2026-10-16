#!/usr/bin/env python3
"""yt-collector: raw, append-only YouTube Data API v3 snapshots for a list of channels.

No derived scores are stored; compute targets downstream so the method can
change without re-collecting. Standard library only (Python 3.9+).

Usage:
  export YT_API_KEY=...            # or --key-file path
  python3 yt_collect.py collect --limit 3          # smoke test
  python3 yt_collect.py collect                    # full run over channels.csv
  python3 yt_collect.py collect --resume RUN_ID    # continue a paused run
  python3 yt_collect.py daily                      # scheduler entry (full weekly, refresh daily)
  python3 yt_collect.py summary
  python3 yt_collect.py prune --older-than-days 30 # retention (dry run without --yes)
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import os
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Iterable

SCHEMA_VERSION = 1
API_BASE = "https://www.googleapis.com/youtube/v3/"
BATCH = 50  # max ids per channels.list / videos.list call, max playlistItems page size
VIDEO_PARTS = "snippet,contentDetails,statistics,status,liveStreamingDetails,topicDetails"
CHANNEL_PARTS = "snippet,statistics,contentDetails"
RETRYABLE_HTTP = {429, 500, 502, 503, 504}
FATAL_403_REASONS = {"keyInvalid", "accessNotConfigured", "forbidden", "ipRefererBlocked",
                     "keyExpired", "API_KEY_INVALID", "SERVICE_DISABLED"}
QUOTA_403_REASONS = {"quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded",
                     "userRateLimitExceeded"}


# --------------------------------------------------------------------------- errors

class ApiError(Exception):
    def __init__(self, status: int, reason: str, message: str):
        super().__init__(f"HTTP {status} {reason}: {message}")
        self.status, self.reason, self.message = status, reason, message


class QuotaExhausted(Exception):
    """Daily API quota hit (server side) or local --max-units budget reached."""


# --------------------------------------------------------------------------- helpers

def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso_time(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


_DUR = re.compile(r"^P(?:(\d+)W)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?$")


def parse_duration(s: str | None) -> int | None:
    """ISO 8601 duration (PT1H2M3S, P1DT2S, P0D) -> seconds. None if absent or unparseable."""
    if not s:
        return None
    m = _DUR.match(s)
    if not m or s in ("P", "PT"):
        return None
    w, d, h, mi, sec = m.groups()
    total = (int(w or 0) * 7 * 86400 + int(d or 0) * 86400 + int(h or 0) * 3600
             + int(mi or 0) * 60 + float(sec or 0))
    return int(round(total))


def to_int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def best_thumbnail(thumbs: dict | None) -> tuple[str | None, str | None, int | None, int | None]:
    if not thumbs:
        return None, None, None, None
    for key in ("maxres", "standard", "high", "medium", "default"):
        t = thumbs.get(key)
        if t and t.get("url"):
            return t["url"], key, to_int(t.get("width")), to_int(t.get("height"))
    return None, None, None, None


def chunks(seq: list, n: int) -> Iterable[list]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def append_jsonl_gz(path: str, rows: list[dict]) -> None:
    """Append rows as a new gzip member. Multi-member gzip reads back as one stream
    (gzip module, pandas.read_json(..., lines=True, compression='gzip'))."""
    if not rows:
        return
    with gzip.open(path, "at", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_jsonl_gz(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json_atomic(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- transport

Transport = Callable[[str, dict], dict]  # (endpoint, params) -> parsed JSON; raises ApiError


def make_http_transport(api_key: str, timeout: float = 30.0) -> Transport:
    def transport(endpoint: str, params: dict) -> dict:
        q = dict(params)
        q["key"] = api_key
        url = API_BASE + endpoint + "?" + urllib.parse.urlencode(q)
        req = urllib.request.Request(url, headers={"Accept": "application/json",
                                                   "Accept-Encoding": "gzip"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            body = e.read()
            try:
                if e.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                err = json.loads(body).get("error", {})
                errs = err.get("errors") or [{}]
                reason = errs[0].get("reason") or err.get("status") or ""
                message = err.get("message", "")
            except Exception:
                reason, message = "", ""
            # Never let the key leak into logs.
            raise ApiError(e.code, reason, message.replace(api_key, "***")) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise ApiError(0, "network", str(e).replace(api_key, "***")) from None
    return transport


# --------------------------------------------------------------------------- client

class Client:
    """Counts quota units, enforces a local budget, retries transient failures."""

    def __init__(self, transport: Transport, max_units: int, units_used: int = 0,
                 retries: int = 4, sleep: Callable[[float], None] = time.sleep):
        self.transport, self.max_units, self.units = transport, max_units, units_used
        self.retries, self.sleep = retries, sleep

    def call(self, endpoint: str, params: dict) -> dict:
        for attempt in range(self.retries + 1):
            if self.units + 1 > self.max_units:
                raise QuotaExhausted(f"local budget of {self.max_units} units reached")
            self.units += 1  # every request costs at least 1 unit, failed ones included
            try:
                return self.transport(endpoint, params)
            except ApiError as e:
                if e.status == 403 and e.reason in QUOTA_403_REASONS:
                    raise QuotaExhausted(f"API quota: {e.reason}") from None
                retryable = e.status in RETRYABLE_HTTP or e.status == 0
                if not retryable or attempt == self.retries:
                    raise
                self.sleep(min(60.0, 2 ** attempt + random.random()))
        raise AssertionError("unreachable")


# --------------------------------------------------------------------------- row builders

def channel_row(item: dict, run_id: str, captured_at: str, sheet: dict) -> dict:
    sn, st, cd = item.get("snippet", {}), item.get("statistics", {}), item.get("contentDetails", {})
    return {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "captured_at": captured_at,
        "channel_id": item["id"],
        "title": sn.get("title"), "custom_url": sn.get("customUrl"),
        "sheet_handle": sheet.get("handle"), "sheet_name": sheet.get("name"),
        "bucket": sheet.get("bucket"),
        "published_at": sn.get("publishedAt"), "country": sn.get("country"),
        "default_language": sn.get("defaultLanguage"),
        # API rounds subscriberCount down to 3 significant figures.
        "subscriber_count": to_int(st.get("subscriberCount")),
        "hidden_subscriber_count": st.get("hiddenSubscriberCount"),
        "view_count": to_int(st.get("viewCount")), "video_count": to_int(st.get("videoCount")),
        "uploads_playlist_id": (cd.get("relatedPlaylists") or {}).get("uploads"),
    }


def video_row(item: dict, run_id: str, captured_at: str) -> dict:
    sn = item.get("snippet", {})
    cd = item.get("contentDetails", {})
    st = item.get("statistics", {})
    ss = item.get("status", {})
    ls = item.get("liveStreamingDetails") or {}
    td = item.get("topicDetails") or {}
    url, key, w, h = best_thumbnail(sn.get("thumbnails"))
    return {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "captured_at": captured_at,
        "video_id": item["id"], "channel_id": sn.get("channelId"),
        "title": sn.get("title"), "description": sn.get("description"),
        "tags": sn.get("tags") or [], "category_id": sn.get("categoryId"),
        "default_language": sn.get("defaultLanguage"),
        "default_audio_language": sn.get("defaultAudioLanguage"),
        "published_at": sn.get("publishedAt"),
        "duration_iso": cd.get("duration"), "duration_s": parse_duration(cd.get("duration")),
        "definition": cd.get("definition"), "caption": cd.get("caption"),
        "licensed_content": cd.get("licensedContent"), "projection": cd.get("projection"),
        "live_broadcast_content": sn.get("liveBroadcastContent"),
        "was_live": bool(ls.get("actualStartTime")),
        "scheduled_start": ls.get("scheduledStartTime"),
        "actual_start": ls.get("actualStartTime"), "actual_end": ls.get("actualEndTime"),
        "privacy_status": ss.get("privacyStatus"), "upload_status": ss.get("uploadStatus"),
        "made_for_kids": ss.get("madeForKids"), "embeddable": ss.get("embeddable"),
        # None means hidden or disabled, not zero.
        "view_count": to_int(st.get("viewCount")), "like_count": to_int(st.get("likeCount")),
        "comment_count": to_int(st.get("commentCount")),
        "thumbnail_url": url, "thumbnail_key": key, "thumbnail_w": w, "thumbnail_h": h,
        "topic_categories": td.get("topicCategories") or [],
    }


# --------------------------------------------------------------------------- collection

def list_upload_ids(client: Client, playlist_id: str, cutoff: dt.datetime,
                    max_videos: int) -> tuple[list[str], int]:
    """Video ids in the uploads playlist published on/after cutoff.

    The uploads playlist comes back newest first in practice (not a documented
    guarantee). Items within a page are all checked against the window, and
    paging stops when a page ends with an out-of-window item or contains nothing
    in the window. Returns (ids, pages_fetched)."""
    ids: list[str] = []
    seen: set[str] = set()
    token, pages = None, 0
    while True:
        params = {"part": "contentDetails", "playlistId": playlist_id, "maxResults": BATCH}
        if token:
            params["pageToken"] = token
        resp = client.call("playlistItems", params)
        pages += 1
        in_window_on_page = 0
        last_pub = None
        for it in resp.get("items", []):
            c = it.get("contentDetails", {})
            vid = c.get("videoId")
            pub = parse_iso_time(c.get("videoPublishedAt"))
            if not vid or vid in seen:
                continue
            if pub is None:
                continue  # private/deleted entries carry no publish time; videos.list would omit them
            last_pub = pub
            if pub >= cutoff:
                in_window_on_page += 1
                seen.add(vid)
                ids.append(vid)
                if len(ids) >= max_videos:
                    return ids, pages
        token = resp.get("nextPageToken")
        if not token or in_window_on_page == 0 or (last_pub is not None and last_pub < cutoff):
            return ids, pages


class Run:
    def __init__(self, out_dir: str, run_id: str):
        self.dir = os.path.join(out_dir, "raw", run_id)
        self.run_id = run_id
        self.state_path = os.path.join(self.dir, "run.json")
        self.channels_path = os.path.join(self.dir, "channels.jsonl.gz")
        self.videos_path = os.path.join(self.dir, "videos.jsonl.gz")
        self.stats_path = os.path.join(self.dir, "stats.jsonl.gz")
        self.ids_path = os.path.join(self.dir, "refresh_ids.json")
        self.state: dict = {}

    def load(self) -> None:
        with open(self.state_path, encoding="utf-8") as f:
            self.state = json.load(f)

    def save(self) -> None:
        write_json_atomic(self.state_path, self.state)


def load_channels_csv(path: str, buckets: set[str] | None, limit: int | None) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "channel_id" not in rows[0]:
        sys.exit(f"{path}: needs a channel_id column")
    out, seen = [], set()
    for r in rows:
        cid = (r.get("channel_id") or "").strip()
        if not re.fullmatch(r"UC[0-9A-Za-z_-]{22}", cid) or cid in seen:
            continue
        if buckets and (r.get("bucket") or "").strip() not in buckets:
            continue
        seen.add(cid)
        out.append({"channel_id": cid, "handle": (r.get("handle") or "").strip(),
                    "name": (r.get("name") or "").strip(), "bucket": (r.get("bucket") or "").strip()})
    return out[:limit] if limit else out


def collect(run: Run, client: Client, channels: list[dict], months: int, max_videos: int,
            log: Callable[[str], None] = print, now: Callable[[], dt.datetime] = utcnow) -> str:
    """Runs (or resumes) a collection. Returns final status."""
    st = run.state
    st.setdefault("completed", [])
    st.setdefault("errors", {})
    st.setdefault("channel_meta", {})  # channel_id -> uploads playlist id
    done = set(st["completed"]) | set(st["errors"])
    cutoff = parse_iso_time(st["cutoff"])
    sheet_by_id = {c["channel_id"]: c for c in channels}

    def checkpoint(status: str) -> None:
        st["status"], st["units_used"], st["updated_at"] = status, client.units, iso(now())
        run.save()

    try:
        # 1. channels.list for channels without metadata yet
        need_meta = [c["channel_id"] for c in channels
                     if c["channel_id"] not in st["channel_meta"] and c["channel_id"] not in done]
        for batch in chunks(need_meta, BATCH):
            resp = client.call("channels", {"part": CHANNEL_PARTS, "id": ",".join(batch),
                                            "maxResults": BATCH})
            cap = iso(now())
            rows = []
            returned = set()
            for item in resp.get("items", []):
                row = channel_row(item, run.run_id, cap, sheet_by_id.get(item["id"], {}))
                rows.append(row)
                returned.add(item["id"])
                st["channel_meta"][item["id"]] = row["uploads_playlist_id"]
            append_jsonl_gz(run.channels_path, rows)
            for cid in batch:
                if cid not in returned:
                    st["errors"][cid] = "channel not returned by channels.list (deleted or terminated?)"
            checkpoint("running")

        # 2. uploads + videos per channel
        todo = [c for c in channels if c["channel_id"] not in done
                and c["channel_id"] not in st["errors"]]
        for i, c in enumerate(todo, 1):
            cid = c["channel_id"]
            playlist = st["channel_meta"].get(cid)
            if not playlist:
                st["errors"][cid] = "no uploads playlist"
                checkpoint("running")
                continue
            try:
                ids, pages = list_upload_ids(client, playlist, cutoff, max_videos)
            except ApiError as e:
                if e.status == 404:
                    st["errors"][cid] = f"uploads playlist not found ({e.reason})"
                    checkpoint("running")
                    continue
                raise
            rows = []
            for batch in chunks(ids, BATCH):
                resp = client.call("videos", {"part": VIDEO_PARTS, "id": ",".join(batch),
                                              "maxResults": BATCH})
                cap = iso(now())
                rows.extend(video_row(it, run.run_id, cap) for it in resp.get("items", []))
            append_jsonl_gz(run.videos_path, rows)
            st["completed"].append(cid)
            st.setdefault("per_channel", {})[cid] = {"ids_listed": len(ids), "rows": len(rows),
                                                     "playlist_pages": pages}
            checkpoint("running")
            if i % 25 == 0 or i == len(todo):
                log(f"[{i}/{len(todo)}] {c.get('name') or cid}: {len(rows)} videos, "
                    f"{client.units} units used")
    except QuotaExhausted as e:
        log(f"Stopped: {e}. Resume with: collect --resume {run.run_id}")
        checkpoint("paused_quota")
        return "paused_quota"
    except ApiError as e:
        log(f"Stopped on API error: {e}. Fix the cause, then: collect --resume {run.run_id}")
        st["last_error"] = str(e)
        checkpoint("failed")
        return "failed"

    checkpoint("complete")
    log(f"Complete: {len(st['completed'])} channels, {len(st['errors'])} errors, "
        f"{client.units} units. Output: {run.dir}")
    return "complete"


# --------------------------------------------------------------------------- refresh mode

STATS_DROP = ("description", "tags", "topic_categories")
STATS_PARTS = "snippet,contentDetails,statistics,liveStreamingDetails"  # same 1-unit cost


def stats_row(item: dict, run_id: str, captured_at: str) -> dict:
    r = video_row(item, run_id, captured_at)
    for k in STATS_DROP:
        r.pop(k, None)
    return r


def run_mode(state: dict) -> str:
    return state.get("mode", "full")


def known_videos(out: str, cutoff: dt.datetime, before_run: str | None = None) -> dict[str, dict[str, str]]:
    """{channel_id: {video_id: published_at}} from the latest complete full run plus every
    refresh run after it (up to before_run, exclusive). Filtered to published_at >= cutoff."""
    root = os.path.join(out, "raw")
    runs = []
    for rid in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        p = os.path.join(root, rid, "run.json")
        if before_run and rid >= before_run or not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            runs.append((rid, json.load(f)))
    full_idx = max((i for i, (_, s) in enumerate(runs)
                    if run_mode(s) == "full" and s.get("status") == "complete"
                    and not s.get("params", {}).get("limit")), default=None)
    if full_idx is None:
        return {}
    known: dict[str, dict[str, str]] = {}
    for rid, s in runs[full_idx:]:
        if run_mode(s) == "full" and rid != runs[full_idx][0]:
            continue
        files = ["videos.jsonl.gz", "stats.jsonl.gz"]
        for fn in files:
            for r in read_jsonl_gz(os.path.join(root, rid, fn)):
                pub = parse_iso_time(r.get("published_at"))
                if r.get("channel_id") and pub and pub >= cutoff:
                    known.setdefault(r["channel_id"], {})[r["video_id"]] = r["published_at"]
    return known


def refresh(run: Run, client: Client, channels: list[dict], months: int, max_videos: int,
            known: dict[str, dict[str, str]], log: Callable[[str], None] = print,
            now: Callable[[], dt.datetime] = utcnow) -> str:
    """Daily snapshot: channel stats, new uploads (first playlist page), and current stats for
    every known in-window video. Channels not seen before get a full pull into videos.jsonl.gz."""
    st = run.state
    cutoff = parse_iso_time(st["cutoff"])
    sheet_by_id = {c["channel_id"]: c for c in channels}
    st.setdefault("phase", "new_channels")
    st.setdefault("errors", {})
    st.setdefault("channel_meta", {})
    if "new_channels" not in st:
        st["new_channels"] = [c["channel_id"] for c in channels if c["channel_id"] not in known]

    def checkpoint(status: str) -> None:
        st["status"], st["units_used"], st["updated_at"] = status, client.units, iso(now())
        run.save()

    # Phase A: full pull for channels with no history (reuses the full-run code path)
    if st["phase"] == "new_channels":
        new = [c for c in channels if c["channel_id"] in set(st["new_channels"])]
        if new:
            log(f"{len(new)} new channels get a full pull")
            status = collect(run, client, new, months, max_videos, log=log, now=now)
            if status != "complete":
                return status
        st["phase"] = "meta"
        checkpoint("running")

    old = [c for c in channels if c["channel_id"] not in set(st["new_channels"])]
    try:
        # Phase B: channel snapshot rows + uploads playlist ids
        if st["phase"] == "meta":
            for batch in chunks([c["channel_id"] for c in old], BATCH):
                resp = client.call("channels", {"part": CHANNEL_PARTS, "id": ",".join(batch),
                                                "maxResults": BATCH})
                cap = iso(now())
                rows, returned = [], set()
                for item in resp.get("items", []):
                    row = channel_row(item, run.run_id, cap, sheet_by_id.get(item["id"], {}))
                    rows.append(row)
                    returned.add(item["id"])
                    st["channel_meta"][item["id"]] = row["uploads_playlist_id"]
                append_jsonl_gz(run.channels_path, rows)
                for cid in batch:
                    if cid not in returned:
                        st["errors"][cid] = "channel not returned by channels.list (deleted or terminated?)"
            st["phase"] = "new_uploads"
            st["scanned"] = []
            checkpoint("running")

        # Phase C: newest playlist page per channel to catch new uploads
        if st["phase"] == "new_uploads":
            scanned = set(st["scanned"])
            added = st.setdefault("added_ids", {})
            for c in old:
                cid = c["channel_id"]
                if cid in scanned or cid in st["errors"] or not st["channel_meta"].get(cid):
                    continue
                try:
                    resp = client.call("playlistItems", {"part": "contentDetails", "maxResults": BATCH,
                                                         "playlistId": st["channel_meta"][cid]})
                except ApiError as e:
                    if e.status != 404:
                        raise
                    st["errors"][cid] = f"uploads playlist not found ({e.reason})"
                    resp = {"items": []}
                have = known.get(cid, {})
                for it in resp.get("items", []):
                    cd_ = it.get("contentDetails", {})
                    vid, pub = cd_.get("videoId"), parse_iso_time(cd_.get("videoPublishedAt"))
                    if vid and pub and pub >= cutoff and vid not in have:
                        added.setdefault(cid, []).append(vid)
                st["scanned"].append(cid)
                if len(st["scanned"]) % 100 == 0:
                    checkpoint("running")
            ids = sorted({v for cid, vids in known.items() if cid in {c["channel_id"] for c in old}
                          for v in vids} | {v for vids in added.values() for v in vids})
            write_json_atomic(run.ids_path, ids)
            st["phase"], st["batches_done"], st["n_ids"] = "stats", 0, len(ids)
            st["new_uploads"] = sum(len(v) for v in added.values())
            checkpoint("running")

        # Phase D: stats for all ids, 50 per call regardless of channel
        if st["phase"] == "stats":
            with open(run.ids_path, encoding="utf-8") as f:
                ids = json.load(f)
            batches = list(chunks(ids, BATCH))
            missing = st.setdefault("missing", 0)
            for bi in range(st["batches_done"], len(batches)):
                batch = batches[bi]
                resp = client.call("videos", {"part": STATS_PARTS, "id": ",".join(batch),
                                              "maxResults": BATCH})
                cap = iso(now())
                items = resp.get("items", [])
                missing += len(batch) - len(items)
                append_jsonl_gz(run.stats_path, [stats_row(it, run.run_id, cap) for it in items])
                st["batches_done"], st["missing"] = bi + 1, missing
                if (bi + 1) % 50 == 0:
                    checkpoint("running")
                    log(f"stats {bi + 1}/{len(batches)} batches, {client.units} units used")
            st["phase"] = "done"
    except QuotaExhausted as e:
        log(f"Stopped: {e}. The next daily run resumes {run.run_id}")
        checkpoint("paused_quota")
        return "paused_quota"
    except ApiError as e:
        log(f"Stopped on API error: {e}")
        st["last_error"] = str(e)
        checkpoint("failed")
        return "failed"

    checkpoint("complete")
    log(f"Refresh complete: {st.get('n_ids', 0)} videos, {st.get('new_uploads', 0)} new uploads, "
        f"{len(st['new_channels'])} new channels, {st.get('missing', 0)} gone, {client.units} units.")
    return "complete"


def estimate_refresh_units(n_channels: int, n_videos: int) -> int:
    return -(-n_channels // BATCH) + n_channels + -(-n_videos // BATCH)


# --------------------------------------------------------------------------- CLI

def load_key(args) -> str:
    if args.key_file:
        with open(os.path.expanduser(args.key_file), encoding="utf-8") as f:
            key = f.read().strip()
    else:
        key = os.environ.get("YT_API_KEY", "").strip()
    if not key:
        sys.exit("No API key: set YT_API_KEY or pass --key-file")
    return key


def cmd_collect(args, mode: str = "full", transport: Transport | None = None) -> int:
    key = None if transport else load_key(args)
    transport = transport or make_http_transport(key)
    if args.resume:
        run = Run(args.out, args.resume)
        if not os.path.exists(run.state_path):
            sys.exit(f"No run at {run.dir}")
        run.load()
        p = run.state["params"]
        channels = load_channels_csv(p["channels_csv"], set(p["buckets"]) if p["buckets"] else None,
                                     p["limit"])
        # A resumed run gets a fresh daily budget; units_used stays the cumulative total.
        client = Client(transport, max_units=run.state.get("units_used", 0) + args.max_units,
                        units_used=run.state.get("units_used", 0))
        months, max_videos = p["months"], p["max_videos"]
        mode = run_mode(run.state)
    else:
        started = utcnow()
        run_id = started.strftime("%Y%m%dT%H%M%SZ")
        run = Run(args.out, run_id)
        os.makedirs(run.dir, exist_ok=False)
        buckets = sorted(set(args.bucket)) if args.bucket else None
        channels = load_channels_csv(args.channels, set(buckets) if buckets else None, args.limit)
        months, max_videos = args.months, args.max_videos
        cutoff = started - dt.timedelta(days=round(months * 30.4375))
        run.state = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "started_at": iso(started),
                     "cutoff": iso(cutoff), "mode": mode,
                     "params": {"channels_csv": os.path.abspath(args.channels), "buckets": buckets,
                                "limit": args.limit, "months": months, "max_videos": max_videos},
                     "n_channels": len(channels)}
        run.save()
        client = Client(transport, max_units=args.max_units)
        print(f"Run {run_id} ({mode}): {len(channels)} channels, window from {run.state['cutoff']}, "
              f"budget {args.max_units} units")
    if mode == "refresh":
        known = known_videos(args.out, parse_iso_time(run.state["cutoff"]), before_run=run.run_id)
        status = refresh(run, client, channels, months, max_videos, known)
    else:
        status = collect(run, client, channels, months, max_videos)
    return {"complete": 0, "paused_quota": 3}.get(status, 2)


def last_full_complete(out: str) -> dt.datetime | None:
    root = os.path.join(out, "raw")
    best = None
    for rid in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        p = os.path.join(root, rid, "run.json")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        if run_mode(s) == "full" and s.get("status") == "complete" and not s.get("params", {}).get("limit"):
            best = parse_iso_time(s.get("started_at")) or best
    return best


RESUMABLE = {"paused_quota", "failed", "running"}


def latest_run_id(out: str) -> str | None:
    root = os.path.join(out, "raw")
    if not os.path.isdir(root):
        return None
    ids = sorted(r for r in os.listdir(root) if os.path.exists(os.path.join(root, r, "run.json")))
    return ids[-1] if ids else None


def cmd_daily(args) -> int:
    """Scheduler entry point: resume the latest run if it did not finish, else start a new one.
    Holds an exclusive lock so overlapping invocations exit instead of double-spending quota."""
    import fcntl
    os.makedirs(args.out, exist_ok=True)
    lock = open(os.path.join(args.out, ".lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        print("Another collection is running; exiting.")
        return 0
    try:
        return _daily_locked(args)
    finally:
        lock.close()  # releases the flock


def _daily_locked(args) -> int:
    last = latest_run_id(args.out)
    args.resume = None
    if last:
        with open(os.path.join(args.out, "raw", last, "run.json"), encoding="utf-8") as f:
            status = json.load(f).get("status")
        if status in RESUMABLE:
            print(f"{utcnow():%Y-%m-%d %H:%M}Z resuming {last} (status {status})")
            args.resume = last
    mode = "full"
    if not args.resume:
        last_full = last_full_complete(args.out)
        if last_full and utcnow() - last_full < dt.timedelta(days=args.full_every_days):
            mode = "refresh"
        print(f"{utcnow():%Y-%m-%d %H:%M}Z starting a new {mode} run")
    rc = cmd_collect(args, mode=mode, transport=getattr(args, "_transport", None))
    return 0 if rc in (0, 3) else rc  # a quota pause is expected, not a failure


def cmd_lookup(args, transport: Transport | None = None) -> int:
    """Curation helper. Prints one JSON object per channel; costs are noted per option."""
    transport = transport or make_http_transport(load_key(args))
    client = Client(transport, max_units=2000)
    sources: dict[str, list[str]] = {}

    def add(cid: str, src: str):
        sources.setdefault(cid, [])
        if src not in sources[cid]:
            sources[cid].append(src)

    for cid in args.id:
        add(cid, "id")
    for h in args.handle:
        h = h if h.startswith("@") else "@" + h
        for it in client.call("channels", {"part": "id", "forHandle": h}).get("items", []):
            add(it["id"], f"handle:{h}")
    for q in args.search:
        resp = client.call("search", {"part": "snippet", "type": "channel", "q": q,
                                      "maxResults": min(50, args.max_results)})
        for it in resp.get("items", []):
            cid = (it.get("id") or {}).get("channelId") or (it.get("snippet") or {}).get("channelId")
            if cid:
                add(cid, f"search:{q}")
    for src_cid in args.featured:
        resp = client.call("channelSections", {"part": "contentDetails", "channelId": src_cid})
        for sec in resp.get("items", []):
            for cid in (sec.get("contentDetails") or {}).get("channels", []) or []:
                add(cid, f"featured_by:{src_cid}")
    ids = list(sources)
    cap = iso(utcnow())
    for batch in chunks(ids, BATCH):
        resp = client.call("channels", {"part": CHANNEL_PARTS, "id": ",".join(batch), "maxResults": BATCH})
        for item in resp.get("items", []):
            row = channel_row(item, "lookup", cap, {})
            latest = None
            if row["uploads_playlist_id"]:
                try:
                    pl = client.call("playlistItems", {"part": "contentDetails", "maxResults": 1,
                                                       "playlistId": row["uploads_playlist_id"]})
                    its = pl.get("items", [])
                    latest = its[0]["contentDetails"].get("videoPublishedAt") if its else None
                except ApiError:
                    latest = None
            out = {k: row[k] for k in ("channel_id", "title", "custom_url", "subscriber_count",
                                       "video_count", "view_count", "country", "default_language",
                                       "published_at")}
            out["latest_upload_at"] = latest
            out["sources"] = sources[item["id"]]
            print(json.dumps(out, ensure_ascii=False))
    print(json.dumps({"_units_used": client.units}), file=sys.stderr)
    return 0


def cmd_estimate(args) -> int:
    channels = load_channels_csv(args.channels, None, None)
    cutoff = utcnow() - dt.timedelta(days=round(24 * 30.4375))
    known = known_videos(args.out, cutoff)
    ids_known = sum(len(known.get(c["channel_id"], {})) for c in channels)
    with_hist = [c for c in channels if c["channel_id"] in known]
    avg = ids_known / len(with_hist) if with_hist else 0
    n_new = len(channels) - len(with_hist)
    projected = estimate_refresh_units(len(channels), int(ids_known + avg * n_new))
    print(json.dumps({"channels": len(channels), "channels_with_history": len(with_hist),
                      "new_channels": n_new, "known_videos": ids_known,
                      "avg_videos_per_channel": round(avg, 1),
                      "projected_refresh_units": projected,
                      "projected_full_run_units_per_new_channel": round(2 + 2 * avg / BATCH, 1)}))
    return 0


def cmd_summary(args) -> int:
    root = os.path.join(args.out, "raw")
    if not os.path.isdir(root):
        print("No runs yet.")
        return 0
    for run_id in sorted(os.listdir(root)):
        path = os.path.join(root, run_id, "run.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
        videos = sum(v.get("rows", 0) for v in s.get("per_channel", {}).values())
        extra = ""
        if run_mode(s) == "refresh":
            extra = (f"  phase {s.get('phase')}  refreshed {s.get('n_ids', 0)}"
                     f"  new_uploads {s.get('new_uploads', 0)}  gone {s.get('missing', 0)}")
        print(f"{run_id}  {run_mode(s):<7} {s.get('status'):<13} channels {len(s.get('completed', []))}/"
              f"{s.get('n_channels')}  errors {len(s.get('errors', {}))}  videos {videos}  "
              f"units {s.get('units_used')}{extra}")
    return 0


def cmd_prune(args) -> int:
    root = os.path.join(args.out, "raw")
    if not os.path.isdir(root):
        return 0
    limit = utcnow() - dt.timedelta(days=args.older_than_days)
    victims = []
    for run_id in sorted(os.listdir(root)):
        try:
            t = dt.datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
        if t < limit:
            victims.append(run_id)
    if not victims:
        print("Nothing older than the limit.")
        return 0
    print("Runs to delete:", ", ".join(victims))
    if not args.yes:
        print("Dry run. Re-run with --yes to delete.")
        return 0
    for v in victims:
        shutil.rmtree(os.path.join(root, v))
    print(f"Deleted {len(victims)} runs.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data", help="output root (default: ./data)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="start or resume a collection run")
    def add_collect_args(p, with_resume: bool):
        p.add_argument("--channels", default="channels.csv", help="CSV with a channel_id column")
        p.add_argument("--bucket", action="append",
                       help="only these buckets (repeatable): own, niche, adjacent, far")
        p.add_argument("--limit", type=int, help="first N channels only (smoke tests)")
        p.add_argument("--months", type=int, default=24, help="upload window in months (default 24)")
        p.add_argument("--max-videos", type=int, default=600,
                       help="cap per channel, Shorts included (default 600)")
        p.add_argument("--max-units", type=int, default=9000,
                       help="local quota budget for this invocation (default 9000 of the 10,000 daily)")
        p.add_argument("--key-file", help="file containing the API key (else YT_API_KEY)")
        if with_resume:
            p.add_argument("--resume", metavar="RUN_ID", help="continue a paused or failed run")

    add_collect_args(c, True)
    c.set_defaults(func=cmd_collect)

    dly = sub.add_parser("daily", help="scheduler entry: resume an unfinished run, else start a new one")
    add_collect_args(dly, False)
    dly.add_argument("--full-every-days", type=int, default=7,
                     help="start a full metadata run when the last complete one is this old (default 7)")
    dly.set_defaults(func=cmd_daily)

    lk = sub.add_parser("lookup", help="resolve and describe channels (JSON lines) for curation")
    lk.add_argument("--id", action="append", default=[], help="channel id (repeatable)")
    lk.add_argument("--handle", action="append", default=[], help="@handle (repeatable, 1 unit each)")
    lk.add_argument("--search", action="append", default=[],
                    help="channel search query (repeatable; search.list, own 100-call daily bucket)")
    lk.add_argument("--featured", action="append", default=[],
                    help="list channels featured by this channel id (channelSections, 1 unit)")
    lk.add_argument("--max-results", type=int, default=15, help="results per search (max 50)")
    lk.add_argument("--key-file", help="file containing the API key (else YT_API_KEY)")
    lk.set_defaults(func=cmd_lookup)

    es = sub.add_parser("estimate", help="projected daily refresh cost for a channels CSV")
    es.add_argument("--channels", default="channels.csv")
    es.set_defaults(func=cmd_estimate)

    s = sub.add_parser("summary", help="list runs and their status")
    s.set_defaults(func=cmd_summary)

    p = sub.add_parser("prune", help="delete runs older than N days (dry run unless --yes)")
    p.add_argument("--older-than-days", type=int, required=True)
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_prune)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
