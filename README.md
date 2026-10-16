# yt-collector

[![tests](https://github.com/Adithya-Rajendran/yt-collector/actions/workflows/tests.yml/badge.svg)](https://github.com/Adithya-Rajendran/yt-collector/actions/workflows/tests.yml)

This collects raw, timestamped snapshots of YouTube channel and video metadata from the YouTube Data API v3 for a list of channels, as a dataset for studying how videos perform relative to their own channel.

- One Python file, standard library only, Python 3.9+.
- Append-only storage: every run is a new folder of gzipped JSON Lines files, and nothing is overwritten. Repeated snapshots of the same video give view counts over time.
- No derived metrics are stored, only raw values. Compute your targets downstream so the method can change without collecting again.
- It is quota-aware: it counts units, stops at a local budget, saves its place and resumes the next day.

## Quick start

```bash
git clone https://github.com/Adithya-Rajendran/yt-collector && cd yt-collector
printf "channel_id,handle,name,bucket\nUC...,@handle,Name,niche\n" > channels.csv   # your channel list
export YT_API_KEY=...                           # or --key-file ~/.config/yt/key (chmod 600)
python3 yt_collect.py collect --limit 3         # smoke test, about 30 units
python3 yt_collect.py summary
python3 -m unittest -v test_yt_collect          # offline tests, no key needed
```

You get an API key from a Google Cloud project with the YouTube Data API v3 enabled.

## Commands

| Command | What it does |
|---|---|
| `collect` | Start a full run, or `--resume RUN_ID` |
| `daily` | Scheduler entry point. Resumes an unfinished run, otherwise starts a full run if the last complete one is 7 or more days old, otherwise a refresh |
| `summary` | Lists runs with their status, units used and counts |
| `estimate` | Projected daily refresh cost for a channels CSV |
| `lookup` | Describes channels as JSON lines: `--id`, `--handle @x`, `--search "query"`, `--featured UC...` |
| `prune` | Deletes runs older than N days (dry run unless `--yes`) |

The collection commands take `--channels`, `--bucket` (repeatable), `--limit`, `--months` (default 24), `--max-videos` (default 600, Shorts included) and `--max-units` (default 9000).

## Modes and cost

Calls to `channels.list`, `playlistItems.list`, `videos.list` and `channelSections.list` cost 1 unit each, including failed ones, and return up to 50 items. The default daily quota is 10,000 units and resets at midnight Pacific. `search.list` uses a separate bucket of 100 calls a day.

- **full**: Channel metadata, then every upload inside the window with titles, descriptions, tags, durations and stats. It costs about 2 units per channel plus 2 per 50 uploads. Channels that post many Shorts cost more, and about 9 units per channel is typical.
- **refresh**: One channels row per channel, the newest playlist page per channel (to find new uploads), then current stats for every known video in the window, batched 50 per call across channels. It costs about 1 unit per channel plus 1 per 50 videos. Channels new to the CSV get a full pull inside the refresh, and channels removed from it stop being refreshed.

## Output

```
data/raw/<RUN_ID>/
  run.json            mode, params, cutoff, status, phase, units used, per-channel counts, errors
  channels.jsonl.gz   one row per channel per run
  videos.jsonl.gz     full rows (full runs, and new channels in refresh runs)
  stats.jsonl.gz      refresh rows: videos minus description, tags and topics
```

```python
import pandas as pd
df = pd.read_json("data/raw/<RUN_ID>/videos.jsonl.gz", lines=True, compression="gzip")
```

Notes on the data:
- **Dedupe key:** `(run_id, video_id)`, keeping the last row. A crash between writing a channel's rows and checkpointing can repeat that channel when the run resumes.
- **Counts:** `view_count`, `like_count` and `comment_count` are exact. `None` means hidden or disabled, not zero. `subscriber_count` is abbreviated by YouTube to 3 significant figures.
- **Dates:** `published_at` is exact, and age at capture is `captured_at - published_at`.
- **Shorts and lives:** The API has no Shorts flag. Use `duration_s`; this project treats more than 180 seconds as long-form. Filter lives with `was_live` and `live_broadcast_content`.
- **Thumbnails:** These are stored as URLs, with `thumbnail_key` recording the best size available.
- **Upload order:** The uploads playlist comes back newest first in practice, but that isn't documented. Paging stops when a page ends before the cutoff.

## Scheduling (systemd user timer)

```bash
mkdir -p ~/.config/systemd/user
cp systemd/yt-collect.{service,timer} ~/.config/systemd/user/
# edit WorkingDirectory and --key-file in yt-collect.service if your paths differ
sudo loginctl enable-linger "$USER"
systemctl --user daemon-reload
systemctl --user enable --now yt-collect.timer
journalctl --user -u yt-collect -n 50
```

The timer fires at 00:23 Pacific, just after the quota resets, with `Persistent=true`. A lock file stops overlapping runs. A pause for quota exits 0, and a real failure exits non-zero.

## Policy: read before you collect

YouTube's [API Developer Policies](https://developers.google.com/youtube/terms/developer-policies) (section III.E) limit what a client may do with API data:
- It may not store non-authorized statistics such as view counts for more than 30 days.
- It may not create new or derived metrics from API data.
- It may not aggregate API data across channel owners.

A long-lived, cross-channel dataset of view snapshots falls outside the standard policy. Before building one, check whether your use is permitted. The [YouTube Researcher Program](https://research.youtube.com/) offers expanded access with its own terms to researchers at accredited institutions. `prune --older-than-days 30 --yes` enforces a retention window.

This tool uses only the official API. It does not scrape YouTube pages.

## Lab: GPU notebooks (`lab/`)

`lab/` holds a GPU container and notebooks for modeling the collected data. The first notebook, `lab/notebooks/01_v0_baseline.ipynb`, asks how well title and thumbnail alone rank a channel's videos against that channel's own norm.

- **Target:** the within-channel robust z-score of age-adjusted log views.
- **Features:** title and format metadata, frozen `BAAI/bge-base-en-v1.5` title embeddings, and frozen SigLIP 2 NaFlex thumbnail embeddings at full size and at 168x94.
- **Model:** LightGBM regression, plus a `lambdarank` variant.
- **Evaluation:** within-channel Spearman correlation on held-out channels and on the newest uploads, with bootstrap CIs, compared against a title-length-plus-duration baseline.

The container uses the NGC PyTorch image. `26.05-py3` (CUDA 13.2) matches a CUDA 13.2 driver; check `nvidia-smi` before raising the tag. The repo checkout is mounted at `/workspace`, so a `git pull` on the host updates the notebooks the container sees. Collected data is mounted read-only at `/data/yt`, from `/srv/yt-data` by default; override it with `YT_DATA_HOST`.

```bash
git clone https://github.com/Adithya-Rajendran/yt-collector && cd yt-collector
docker compose -f lab/compose.yaml up -d --build
docker exec yt-lab python lab/smoke.py            # GPU check
```

In VS Code, connect with Remote-SSH to the GPU host, open the repo, and choose "Reopen in Container" (`.devcontainer/` points at `lab/compose.yaml`). Pick the container's Python as the notebook kernel.

To run headless: `docker exec yt-lab jupyter nbconvert --to notebook --execute --inplace lab/notebooks/01_v0_baseline.ipynb`. Caches and results go to `lab/.cache/` and `lab/artifacts/`, which git ignores.

Environment knobs: `LAB_SAMPLE_CHANNELS`, `LAB_SKIP_IMAGES=1`, `LAB_TEXT_MODEL` and `LAB_IMAGE_MODEL`. The offline tests (`python -m unittest lab.test_ytlab`) use synthetic data from `lab/synth.py`.

Thumbnails are downloaded from YouTube's image CDN for embedding. They fall under the same policy caveats as the rest of the data.

## License

MIT
