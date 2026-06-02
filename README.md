# Real-Time Speech Translation Engine

A **local-first** engine that listens to live audio, transcribes it with streaming
ASR, translates it into a target language with a context-aware translator, and
renders **low-latency captions** suitable for a floating overlay on top of
Microsoft Teams / Google Meet / Zoom — without modifying those apps.

> Priorities: **speed → meaning → low-latency captions → stability in live meetings.**
> This is not a chatbot; it's a streaming translation pipeline.

## Highlights

- **Streaming, not batch** — voice-activity segmentation emits *partial* hypotheses
  while you speak and *final* ones when you pause, so captions appear in ~1–2s.
- **Offline-first / privacy mode** — ASR (faster-whisper) and translation
  (NLLB-200 / Argos) run locally. Nothing leaves your machine unless you pass
  `--allow-cloud`.
- **Context-aware** — keeps a rolling window of recent utterances to translate by
  *meaning*, not word-by-word.
- **Swappable layers** — audio / ASR / translation / captions are independent
  backends behind small interfaces, so the orchestration logic is fully unit-tested
  without downloading any models.
- **Two caption sinks** — a terminal sink (in-place updating, works over SSH) and a
  PyQt6 always-on-top overlay window for meetings.
- **CPU fallback, GPU optional** — device auto-detected.

## Architecture

```
            ┌──────────┐   frames   ┌───────────────────┐ segments ┌──────────────┐
 mic / ───▶ │  capture │ ─────────▶ │ StreamingSegmenter│ ───────▶ │  ASR backend │
 loopback   │(sounddev)│            │   (VAD + timing)  │ partial/ │(faster-whisper)
            └──────────┘            └───────────────────┘  final   └──────┬───────┘
                                                                          │ text
                                          ┌───────────────────┐           ▼
              captions  ◀──────────────── │     Pipeline      │ ◀── ConversationContext
        (console / PyQt overlay)          │ throttle + dedup  │ ──▶ Translator (NLLB/Argos)
                                          └───────────────────┘
```

| Layer | Module | Default backend |
|-------|--------|-----------------|
| Audio capture | `rtst.audio.capture` | `sounddevice` (mic / loopback) |
| Segmentation/VAD | `rtst.audio.vad` | WebRTC VAD, energy-gate fallback |
| Speech-to-text | `rtst.asr` | `faster-whisper` |
| Translation | `rtst.translate` | NLLB-200 distilled (Argos / identity also available) |
| Captions | `rtst.captions` | console (default) / PyQt6 overlay |
| Orchestration | `rtst.pipeline` | partial throttling, de-dup, context |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"      # everything (audio + ASR + translation + UI)
# or pick extras: .[asr]  .[translate]  .[audio]  .[ui]  .[argos]
# .[translate] pulls transformers + torch for NLLB; .[argos] is lighter (CPU-friendly)
```

The core package is lightweight; the heavy ML/GUI/audio dependencies live in the
extras above. Whisper and NLLB weights download automatically on first run and are
cached locally.

System audio libraries you may need:

- **Audio capture**: PortAudio (`sudo apt-get install libportaudio2` on Debian/Ubuntu).
- **Robust VAD (optional)**: `pip install webrtcvad` (the engine falls back to an
  energy gate if it's missing).

## Usage

```bash
# List what you can translate and which mics/loopback devices exist
rtst list-languages
rtst list-devices

# Translate your microphone from auto-detected language → French, in the terminal
rtst run --target fr

# Translate system audio (a meeting) → Arabic, shown in a floating overlay window
rtst run --source en --target ar --loopback --captions overlay

# Dual mode shows the original text and the translation
rtst run --target es --mode dual
```

Common flags:

| Flag | Meaning |
|------|---------|
| `-s/--source` | source language (`auto` to detect) |
| `-t/--target` | target language (code or name: `fr`, `Arabic`, …) |
| `--preset` | `fast` (CPU real-time: whisper `base` + Argos) or `quality` (whisper `small` + NLLB, needs a GPU) |
| `--model` | whisper size: `tiny`/`base`/`small`/`medium`/`large-v3` |
| `--captions` | `console` (default) or `overlay` |
| `--mode` | `compact` (translation only) or `dual` |
| `--loopback` | capture system audio instead of the mic |
| `--allow-cloud` | permit cloud translation APIs (off by default) |

### Capturing meeting audio (Teams / Meet / Zoom)

To translate what the **other participants** say you must capture *system output*
(what comes out of your speakers), not your microphone — so pass `--loopback`. The
engine never touches the meeting app.

```bash
# Translate the meeting you hear (English) into French, in a floating overlay
rtst run --source en --target fr --loopback --captions overlay
```

`--loopback` resolves a system-audio source automatically per platform:

- **Windows (Teams desktop)**: uses **WASAPI loopback** on your default output
  device — no extra drivers or *Stereo Mix* needed. Just `--loopback`.
- **Linux (PulseAudio/PipeWire)**: auto-selects the output's `*.monitor` source.
  If you have several, name one explicitly with `--loopback --device "<name>"`
  (see `rtst list-devices`).
- **macOS**: there is no built-in loopback; install a virtual device such as
  [BlackHole](https://github.com/ExistentialAudio/BlackHole), route output to it,
  then `--loopback --device "BlackHole 2ch"`.

Loopback devices usually run at 44.1/48 kHz and may be stereo; capture downmixes to
mono and resamples to 16 kHz for the ASR automatically.

Run with `--captions overlay` and the translucent caption window floats over the
meeting.

## Supported languages

French, Arabic (incl. dialect → MSA via the model), English, Spanish, German,
Italian, and any language the underlying model supports (pass it by code). You can
change `--target` per run; the language is normalized from names/aliases/region tags.

## Performance & latency

Latency per utterance ≈ *ASR time* + *translation time*. Both backends are
lazy-loaded; the **model size and translation backend dominate** whether you hit
the <2 s target. Measured on this repo's CI-class machine (**CPU-only**, int8
ASR, greedy decoding — a worst case; a GPU is dramatically faster):

| Stage | Setting | Speed (CPU) |
|-------|---------|-------------|
| ASR | whisper `tiny` | ~0.11× real-time (RTF) |
| ASR | whisper `base` | ~0.22× RTF |
| ASR | whisper `small` (default) | ~0.71× RTF |
| Translation | **Argos** per segment | **~0.2 s** |
| Translation | NLLB-200 600M per segment | ~4–6 s |

**Takeaways**

- On **CPU**, NLLB-200 is too slow for live captions (~4–6 s/segment). For
  real-time on a laptop use the `fast` preset (whisper `base` + Argos, which
  translates in ~0.2 s):

  ```bash
  rtst run --target fr --preset fast --loopback --captions overlay
  ```

  `--preset fast` is shorthand for `--model base --translation-backend argos`
  plus low-latency tuning. Any flag you pass explicitly overrides the preset
  (e.g. add `--model tiny` to go even faster).

- whisper `small` (the default) has RTF ~0.7 on CPU, so processing a 6 s segment
  takes ~4 s — fine for accuracy but not for snappy partials. Drop to `tiny`/`base`
  for lower latency.
- With a **GPU**, NLLB-200 + whisper `small`/`medium` comfortably fit the <2 s
  budget at higher quality; the device is auto-detected (`--asr-device cuda`).

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

The test-suite runs the full pipeline with in-memory fakes (no models, no audio
hardware, no GUI), so it's fast and CI-friendly.

## Privacy

- Everything runs locally by default; no audio or text is sent anywhere.
- No recording is written to disk.
- Cloud translation is opt-in via `--allow-cloud`.

## License

MIT
