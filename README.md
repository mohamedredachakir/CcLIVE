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
| `--model` | whisper size: `tiny`/`base`/`small`/`medium`/`large-v3` |
| `--captions` | `console` (default) or `overlay` |
| `--mode` | `compact` (translation only) or `dual` |
| `--loopback` | capture system audio instead of the mic |
| `--allow-cloud` | permit cloud translation APIs (off by default) |

### Capturing meeting audio (Teams / Meet / Zoom)

The engine never touches the meeting app. Point it at system audio via a loopback
device:

- **Linux (PulseAudio/PipeWire)**: select a `*.monitor` source with
  `--device "<name>"` (see `rtst list-devices`).
- **Windows**: enable *Stereo Mix* or install a virtual audio cable, then select it.
- **macOS**: install a loopback driver (e.g. BlackHole) and select it.

Run with `--captions overlay` and the translucent caption window floats over the
meeting.

## Supported languages

French, Arabic (incl. dialect → MSA via the model), English, Spanish, German,
Italian, and any language the underlying model supports (pass it by code). You can
change `--target` per run; the language is normalized from names/aliases/region tags.

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
