"""Offline tests for yt_collect.py. Run: python3 -m unittest -v test_yt_collect"""
import datetime as dt
import gzip
import json
import os
import tempfile
import unittest

import yt_collect as yc

NOW = dt.datetime(2026, 10, 16, 12, 0, tzinfo=dt.timezone.utc)
CUTOFF = NOW - dt.timedelta(days=730)


def cid(n):  # valid-looking channel id
    return "UC" + f"{n:022d}"


def vid(c, i):
    return f"v{c:03d}_{i:04d}"


class FakeYouTube:
    """Simulates channels.list, playlistItems.list, videos.list.

    channels: {n: {"videos": count, "spacing_days": d, "missing": bool, "no_playlist": bool}}
    """

    def __init__(self, channels, fail_plan=None, hide_likes=()):
        self.channels = channels
        self.calls = []
        self.fail_plan = list(fail_plan or [])  # list of (endpoint, ApiError) consumed in order
        self.hide_likes = set(hide_likes)
        self.videos = {}
        for n, spec in channels.items():
            for i in range(spec.get("videos", 0)):
                pub = NOW - dt.timedelta(days=1 + i * spec.get("spacing_days", 7))
                self.videos[vid(n, i)] = (n, pub)

    def __call__(self, endpoint, params):
        self.calls.append((endpoint, dict(params)))
        if self.fail_plan and self.fail_plan[0][0] == endpoint:
            raise self.fail_plan.pop(0)[1]
        if endpoint == "channels":
            ids = params["id"].split(",")
            assert len(ids) <= 50
            items = []
            for c in ids:
                n = int(c[2:])
                spec = self.channels.get(n)
                if spec is None or spec.get("missing"):
                    continue
                items.append({"id": c,
                              "snippet": {"title": f"Chan {n}", "customUrl": f"@chan{n}",
                                          "publishedAt": "2015-01-01T00:00:00Z"},
                              "statistics": {"subscriberCount": "123000", "viewCount": "1",
                                             "videoCount": str(spec.get("videos", 0)),
                                             "hiddenSubscriberCount": False},
                              "contentDetails": {"relatedPlaylists": {
                                  "uploads": None if spec.get("no_playlist") else f"UU{n}"}}})
            return {"items": items}
        if endpoint == "playlistItems":
            n = int(params["playlistId"][2:])
            if self.channels[n].get("playlist_404"):
                raise yc.ApiError(404, "playlistNotFound", "nope")
            start = int(params.get("pageToken", "0"))
            size = params["maxResults"]
            all_ids = [v for v, (c, _) in self.videos.items() if c == n]
            page = all_ids[start:start + size]
            items = [{"contentDetails": {"videoId": v,
                                         "videoPublishedAt": yc.iso(self.videos[v][1])}} for v in page]
            resp = {"items": items}
            if start + size < len(all_ids):
                resp["nextPageToken"] = str(start + size)
            return resp
        if endpoint == "videos":
            ids = params["id"].split(",")
            assert len(ids) <= 50, len(ids)
            items = []
            for v in ids:
                n, pub = self.videos[v]
                stats = {"viewCount": "93500", "commentCount": "10"}
                if v not in self.hide_likes:
                    stats["likeCount"] = "1200"
                items.append({"id": v,
                              "snippet": {"channelId": cid(n), "title": f"Title {v}",
                                          "publishedAt": yc.iso(pub), "liveBroadcastContent": "none",
                                          "thumbnails": {"high": {"url": f"https://i.ytimg.com/vi/{v}/hqdefault.jpg",
                                                                  "width": 480, "height": 360}}},
                              "contentDetails": {"duration": "PT12M30S", "definition": "hd"},
                              "statistics": stats,
                              "status": {"privacyStatus": "public"}})
            return {"items": items}
        raise AssertionError(endpoint)


def make_run(tmp, channels_spec, n_limit=None):
    run = yc.Run(tmp, "20261016T120000Z")
    os.makedirs(run.dir)
    run.state = {"run_id": run.run_id, "cutoff": yc.iso(CUTOFF), "params": {}}
    chans = [{"channel_id": cid(n), "handle": f"@chan{n}", "name": f"Chan {n}", "bucket": "niche"}
             for n in channels_spec]
    return run, chans[:n_limit] if n_limit else chans


def quiet(_):
    pass


class DurationTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(yc.parse_duration("PT12M30S"), 750)
        self.assertEqual(yc.parse_duration("PT1H2M3S"), 3723)
        self.assertEqual(yc.parse_duration("P1DT2S"), 86402)
        self.assertEqual(yc.parse_duration("P0D"), 0)
        self.assertEqual(yc.parse_duration("PT0S"), 0)
        self.assertEqual(yc.parse_duration("PT59.6S"), 60)
        self.assertIsNone(yc.parse_duration(None))
        self.assertIsNone(yc.parse_duration("garbage"))
        self.assertIsNone(yc.parse_duration("PT"))


class CollectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def run_collect(self, spec, fake=None, max_units=10_000, max_videos=300, run=None, chans=None,
                    units_used=0):
        fake = fake or FakeYouTube(spec)
        if run is None:
            run, chans = make_run(self.tmp, spec)
        client = yc.Client(fake, max_units=max_units, units_used=units_used, sleep=lambda s: None)
        status = yc.collect(run, client, chans, 24, max_videos, log=quiet, now=lambda: NOW)
        return status, run, fake, client

    def test_window_and_pagination(self):
        # 200 weekly uploads: 1 + 7i days old; within 730 days -> i <= 104 -> 105 videos
        status, run, fake, client = self.run_collect({1: {"videos": 200, "spacing_days": 7}})
        self.assertEqual(status, "complete")
        rows = yc.read_jsonl_gz(run.videos_path)
        self.assertEqual(len(rows), 105)
        self.assertTrue(all(yc.parse_iso_time(r["published_at"]) >= CUTOFF for r in rows))
        pl_calls = [c for c in fake.calls if c[0] == "playlistItems"]
        self.assertEqual(len(pl_calls), 3)  # pages 0-49, 50-99, 100-149 (last has 5 in window)
        v_calls = [c for c in fake.calls if c[0] == "videos"]
        self.assertEqual(len(v_calls), 3)  # 50 + 50 + 5
        self.assertEqual(client.units, 1 + 3 + 3)

    def test_max_videos_cap(self):
        status, run, *_ = self.run_collect({1: {"videos": 200, "spacing_days": 1}}, max_videos=60)
        self.assertEqual(len(yc.read_jsonl_gz(run.videos_path)), 60)

    def test_row_fields(self):
        spec = {1: {"videos": 3}}
        fake = FakeYouTube(spec, hide_likes={vid(1, 0)})
        status, run, *_ = self.run_collect(spec, fake=fake)
        rows = {r["video_id"]: r for r in yc.read_jsonl_gz(run.videos_path)}
        r = rows[vid(1, 0)]
        self.assertIsNone(r["like_count"])  # hidden, not zero
        self.assertEqual(rows[vid(1, 1)]["like_count"], 1200)
        self.assertEqual(r["view_count"], 93500)
        self.assertEqual(r["duration_s"], 750)
        self.assertEqual(r["thumbnail_key"], "high")
        self.assertEqual(r["captured_at"], "2026-10-16T12:00:00Z")
        self.assertEqual(r["channel_id"], cid(1))
        ch = yc.read_jsonl_gz(run.channels_path)
        self.assertEqual(ch[0]["subscriber_count"], 123000)
        self.assertEqual(ch[0]["bucket"], "niche")

    def test_channels_batched_and_missing(self):
        spec = {n: {"videos": 1} for n in range(120)}
        spec[5]["missing"] = True
        spec[7]["no_playlist"] = True
        spec[9]["playlist_404"] = True
        status, run, fake, _ = self.run_collect(spec)
        self.assertEqual(status, "complete")
        ch_calls = [c for c in fake.calls if c[0] == "channels"]
        self.assertEqual([len(c[1]["id"].split(",")) for c in ch_calls], [50, 50, 20])
        self.assertIn(cid(5), run.state["errors"])
        self.assertIn(cid(7), run.state["errors"])
        self.assertIn(cid(9), run.state["errors"])
        self.assertEqual(len(run.state["completed"]), 117)

    def test_budget_pause_and_resume_no_duplicates(self):
        spec = {n: {"videos": 10} for n in range(10)}
        # 1 channels call + 2 per channel = 21 units total. Budget 9 -> pause mid-way.
        status, run, fake, client = self.run_collect(spec, max_units=9)
        self.assertEqual(status, "paused_quota")
        with open(run.state_path) as f:
            saved = json.load(f)
        self.assertEqual(saved["status"], "paused_quota")
        done_first = len(saved["completed"])
        self.assertEqual(done_first, 4)  # 1 + 4*2 = 9
        # Resume with fresh budget, reloaded state
        run2 = yc.Run(self.tmp, run.run_id)
        run2.load()
        _, chans = make_run(tempfile.mkdtemp(), spec)
        status2, run2, fake2, client2 = self.run_collect(spec, run=run2, chans=chans,
                                                         max_units=saved["units_used"] + 100,
                                                         units_used=saved["units_used"])
        self.assertEqual(status2, "complete")
        self.assertFalse([c for c in fake2.calls if c[0] == "channels"])  # meta cached
        rows = yc.read_jsonl_gz(run2.videos_path)
        self.assertEqual(len(rows), 100)
        self.assertEqual(len({r["video_id"] for r in rows}), 100)
        self.assertEqual(client2.units, 21)

    def test_server_quota_exceeded(self):
        spec = {1: {"videos": 5}, 2: {"videos": 5}}
        fake = FakeYouTube(spec, fail_plan=[("videos", yc.ApiError(403, "quotaExceeded", "x"))])
        status, run, *_ = self.run_collect(spec, fake=fake)
        self.assertEqual(status, "paused_quota")
        self.assertEqual(run.state["completed"], [])

    def test_retry_then_success(self):
        spec = {1: {"videos": 5}}
        fake = FakeYouTube(spec, fail_plan=[("videos", yc.ApiError(503, "backendError", "x")),
                                            ("videos", yc.ApiError(0, "network", "reset"))])
        status, run, fake, client = self.run_collect(spec, fake=fake)
        self.assertEqual(status, "complete")
        self.assertEqual(len(yc.read_jsonl_gz(run.videos_path)), 5)
        self.assertEqual(client.units, 1 + 1 + 3)  # failed calls are charged

    def test_fatal_error_stops(self):
        spec = {1: {"videos": 5}}
        fake = FakeYouTube(spec, fail_plan=[("channels", yc.ApiError(400, "keyInvalid", "bad key"))])
        status, run, *_ = self.run_collect(spec, fake=fake)
        self.assertEqual(status, "failed")
        self.assertIn("keyInvalid", run.state["last_error"])


class RefreshTests(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.mkdtemp()
        self.spec = {1: {"videos": 120, "spacing_days": 7}, 2: {"videos": 30, "spacing_days": 3}}
        self.fake = FakeYouTube(self.spec)
        run, chans = make_run(self.out, self.spec)
        client = yc.Client(self.fake, max_units=10_000, sleep=lambda s: None)
        self.assertEqual(yc.collect(run, client, chans, 24, 600, log=quiet, now=lambda: NOW), "complete")
        self.full_rows = len(yc.read_jsonl_gz(run.videos_path))

    def _refresh(self, chans, run_id="20261017T072300Z", max_units=10_000, units_used=0, resume=False):
        run = yc.Run(self.out, run_id)
        if resume:
            run.load()
        else:
            os.makedirs(run.dir)
            run.state = {"run_id": run_id, "mode": "refresh", "cutoff": yc.iso(CUTOFF), "params": {}}
        known = yc.known_videos(self.out, CUTOFF, before_run=run_id)
        client = yc.Client(self.fake, max_units=max_units, units_used=units_used, sleep=lambda s: None)
        status = yc.refresh(run, client, chans, 24, 600, known, log=quiet, now=lambda: NOW)
        return status, run, client

    def _chans(self, ns):
        return [{"channel_id": cid(n), "handle": "", "name": "", "bucket": "niche"} for n in ns]

    def test_refresh_snapshot_new_uploads_and_new_channel(self):
        # a new upload on channel 1 that the playlist's first page will show
        self.fake.videos["vNEW_0001"] = (1, NOW - dt.timedelta(hours=3))
        # put it first so the newest-first first page contains it
        self.fake.videos = {"vNEW_0001": self.fake.videos.pop("vNEW_0001"), **self.fake.videos}
        self.spec[3] = {"videos": 4}
        for i in range(4):
            self.fake.videos[vid(3, i)] = (3, NOW - dt.timedelta(days=1 + i))
        self.fake.calls.clear()
        status, run, client = self._refresh(self._chans([1, 2, 3]))
        self.assertEqual(status, "complete")
        stats = yc.read_jsonl_gz(run.stats_path)
        ids = [r["video_id"] for r in stats]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), self.full_rows + 1)  # every known video + the new upload
        self.assertIn("vNEW_0001", ids)
        self.assertNotIn("description", stats[0])
        self.assertEqual(stats[0]["view_count"], 93500)
        new_full = yc.read_jsonl_gz(run.videos_path)
        self.assertEqual({r["channel_id"] for r in new_full}, {cid(3)})
        self.assertEqual(run.state["new_channels"], [cid(3)])
        self.assertEqual(run.state["new_uploads"], 1)
        n = len(ids)
        expected = (1 + 1 + 1) + 1 + 2 + -(-n // 50)  # new chan: meta+list+videos; meta; 2 first pages; stats
        self.assertEqual(client.units, expected)
        # channel snapshot rows written for both old channels and the new one
        self.assertEqual(len(yc.read_jsonl_gz(run.channels_path)), 3)

    def test_refresh_pause_and_resume_in_stats(self):
        chans = self._chans([1, 2])
        status, run, client = self._refresh(chans, max_units=4)  # meta 1 + pages 2 + 1 stats batch
        self.assertEqual(status, "paused_quota")
        self.assertEqual(run.state["phase"], "stats")
        status, run, client = self._refresh(chans, max_units=run.state["units_used"] + 50,
                                            units_used=run.state["units_used"], resume=True)
        self.assertEqual(status, "complete")
        ids = [r["video_id"] for r in yc.read_jsonl_gz(run.stats_path)]
        self.assertEqual(len(ids), self.full_rows)
        self.assertEqual(len(set(ids)), len(ids))

    def test_removed_channel_not_refreshed(self):
        status, run, _ = self._refresh(self._chans([1]))
        chans_in_stats = {r["channel_id"] for r in yc.read_jsonl_gz(run.stats_path)}
        self.assertEqual(chans_in_stats, {cid(1)})

    def test_known_includes_prior_refresh_new_uploads(self):
        self.fake.videos = {"vNEW_0002": (2, NOW - dt.timedelta(hours=1)), **self.fake.videos}
        self._refresh(self._chans([1, 2]))
        known = yc.known_videos(self.out, CUTOFF)
        self.assertIn("vNEW_0002", known[cid(2)])


class LookupTests(unittest.TestCase):
    def test_lookup_sources(self):
        import contextlib
        import io
        fake = FakeYouTube({1: {"videos": 2}, 2: {"videos": 2}})

        def transport(endpoint, params):
            if endpoint == "search":
                return {"items": [{"id": {"kind": "youtube#channel", "channelId": cid(1)}}]}
            if endpoint == "channelSections":
                return {"items": [{"contentDetails": {"channels": [cid(2), cid(1)]}}]}
            if endpoint == "channels" and "forHandle" in params:
                return {"items": [{"id": cid(2)}]}
            return fake(endpoint, params)
        args = yc.main.__globals__["argparse"].Namespace(id=[], handle=["chan2"], search=["homelab"],
                                                         featured=[cid(9)], max_results=5, key_file=None)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            yc.cmd_lookup(args, transport=transport)
        rows = [json.loads(l) for l in buf.getvalue().splitlines()]
        by = {r["channel_id"]: r for r in rows}
        self.assertEqual(set(by), {cid(1), cid(2)})
        self.assertIn("search:homelab", by[cid(1)]["sources"])
        self.assertIn(f"featured_by:{cid(9)}", by[cid(1)]["sources"])
        self.assertIn("handle:@chan2", by[cid(2)]["sources"])
        self.assertIsNotNone(by[cid(1)]["latest_upload_at"])


class DailyTests(unittest.TestCase):
    def _mk(self, out, run_id, status):
        d = os.path.join(out, "raw", run_id)
        os.makedirs(d)
        with open(os.path.join(d, "run.json"), "w") as f:
            json.dump({"status": status}, f)

    def _mode(self, out):
        seen = {}
        orig = yc.cmd_collect
        yc.cmd_collect = lambda args, mode="full", transport=None: seen.update(mode=mode, resume=args.resume) or 0
        try:
            yc.main(["--out", out, "daily"])
        finally:
            yc.cmd_collect = orig
        return seen

    def _mkfull(self, out, run_id, started, status="complete", limit=None):
        d = os.path.join(out, "raw", run_id)
        os.makedirs(d)
        with open(os.path.join(d, "run.json"), "w") as f:
            json.dump({"status": status, "mode": "full", "started_at": yc.iso(started),
                       "params": {"limit": limit}}, f)

    def test_mode_selection(self):
        out = tempfile.mkdtemp()
        self.assertEqual(self._mode(out)["mode"], "full")  # no history
        self._mkfull(out, "20200101T000000Z", yc.utcnow() - dt.timedelta(days=1), limit=3)
        self.assertEqual(self._mode(out)["mode"], "full")  # smoke runs don't count
        self._mkfull(out, "20200102T000000Z", yc.utcnow() - dt.timedelta(days=2))
        self.assertEqual(self._mode(out)["mode"], "refresh")
        out2 = tempfile.mkdtemp()
        self._mkfull(out2, "20200101T000000Z", yc.utcnow() - dt.timedelta(days=8))
        self.assertEqual(self._mode(out2)["mode"], "full")

    def _daily(self, out):
        seen = {}
        orig = yc.cmd_collect
        yc.cmd_collect = lambda args, mode="full", transport=None: seen.setdefault("resume", args.resume) and 0 or 0
        try:
            yc.main(["--out", out, "daily"])
        finally:
            yc.cmd_collect = orig
        return seen.get("resume")

    def test_new_when_complete_or_empty(self):
        out = tempfile.mkdtemp()
        self.assertIsNone(self._daily(out))
        self._mk(out, "20261015T072300Z", "complete")
        self.assertIsNone(self._daily(out))

    def test_resumes_latest_unfinished(self):
        out = tempfile.mkdtemp()
        self._mk(out, "20261014T072300Z", "paused_quota")
        self._mk(out, "20261015T072300Z", "paused_quota")
        self.assertEqual(self._daily(out), "20261015T072300Z")

    def test_lock_blocks_overlap(self):
        import fcntl
        out = tempfile.mkdtemp()
        held = open(os.path.join(out, ".lock"), "w")
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        called = []
        orig = yc.cmd_collect
        yc.cmd_collect = lambda args, mode="full", transport=None: called.append(1) or 0
        try:
            self.assertEqual(yc.main(["--out", out, "daily"]), 0)
        finally:
            yc.cmd_collect = orig
            held.close()
        self.assertEqual(called, [])


class MiscTests(unittest.TestCase):
    def test_gzip_append_members(self):
        p = os.path.join(tempfile.mkdtemp(), "x.jsonl.gz")
        yc.append_jsonl_gz(p, [{"a": 1}])
        yc.append_jsonl_gz(p, [{"a": 2}, {"a": 3}])
        self.assertEqual([r["a"] for r in yc.read_jsonl_gz(p)], [1, 2, 3])

    def test_key_redacted(self):
        import io
        import urllib.error
        import urllib.request
        key = "AIzaSECRETKEY123"
        body = json.dumps({"error": {"message": f"bad key {key}",
                                     "errors": [{"reason": "keyInvalid"}]}}).encode()

        def fake_open(req, timeout=None):
            self.assertIn(key, req.full_url)
            raise urllib.error.HTTPError(req.full_url, 400, "Bad", {}, io.BytesIO(body))
        orig = urllib.request.urlopen
        urllib.request.urlopen = fake_open
        try:
            t = yc.make_http_transport(key)
            with self.assertRaises(yc.ApiError) as cm:
                t("videos", {"part": "id", "id": "x"})
        finally:
            urllib.request.urlopen = orig
        self.assertNotIn(key, str(cm.exception))
        self.assertEqual(cm.exception.reason, "keyInvalid")

    def test_channels_csv(self):
        p = os.path.join(tempfile.mkdtemp(), "c.csv")
        with open(p, "w") as f:
            f.write("channel_id,handle,name,bucket\n")
            f.write(f"{cid(1)},@a,A,niche\n{cid(1)},@a,A,niche\nnotanid,@b,B,far\n{cid(2)},@c,C,far\n")
        self.assertEqual(len(yc.load_channels_csv(p, None, None)), 2)
        self.assertEqual(len(yc.load_channels_csv(p, {"far"}, None)), 1)


if __name__ == "__main__":
    unittest.main()
