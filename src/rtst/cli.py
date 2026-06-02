"""Command-line entrypoint for the real-time speech translation engine."""

from __future__ import annotations

import argparse
import logging
import sys
import threading

from rtst import __version__, languages
from rtst.config import (
    ASRConfig,
    AudioConfig,
    CaptionConfig,
    Config,
    TranslationConfig,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rtst",
        description="Local-first real-time speech translation engine for live meetings.",
    )
    parser.add_argument("--version", action="version", version=f"rtst {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Start live capture, transcription and translation.")
    run.add_argument("-s", "--source", default="auto", help="Source language (default: auto).")
    run.add_argument("-t", "--target", default="en", help="Target language (default: en).")
    run.add_argument(
        "--model", default="small", help="Whisper model size (tiny/base/small/medium/large-v3)."
    )
    run.add_argument("--asr-device", default="auto", help="ASR device: auto/cpu/cuda.")
    run.add_argument(
        "--translation-backend", default="nllb", choices=["nllb", "argos", "identity"]
    )
    run.add_argument("--translation-model", default="facebook/nllb-200-distilled-600M")
    run.add_argument("--captions", default="console", choices=["console", "overlay"])
    run.add_argument(
        "--mode", default="compact", choices=["compact", "dual"],
        help="compact = translation only.",
    )
    run.add_argument(
        "--loopback", action="store_true", help="Capture system audio instead of microphone."
    )
    run.add_argument("--device", default=None, help="Audio input device name or index.")
    run.add_argument(
        "--allow-cloud", action="store_true",
        help="Permit cloud translation APIs (off by default).",
    )
    run.add_argument("-v", "--verbose", action="store_true")

    sub.add_parser("list-devices", help="List available audio input devices.")
    sub.add_parser("list-languages", help="List supported languages.")
    return parser


def _config_from_args(args: argparse.Namespace) -> Config:
    device: str | int | None = args.device
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    return Config(
        source_language=args.source,
        target_language=args.target,
        audio=AudioConfig(use_loopback=args.loopback, device=device),
        asr=ASRConfig(model_size=args.model, device=args.asr_device),
        translation=TranslationConfig(
            backend=args.translation_backend,
            model=args.translation_model,
            allow_cloud=args.allow_cloud,
        ),
        caption=CaptionConfig(backend=args.captions, mode=args.mode),
    )


def cmd_list_languages() -> int:
    print("Supported languages (code  name):")
    for lang in languages.supported():
        print(f"  {lang.flag}  {lang.code:<4} {lang.name}")
    print("\nAny language supported by the underlying model also works by code.")
    return 0


def cmd_list_devices() -> int:
    try:
        from rtst.audio.capture import list_devices
    except Exception as exc:  # pragma: no cover
        print(f"Could not import audio backend: {exc}", file=sys.stderr)
        return 1
    try:
        devices = list_devices()
    except Exception as exc:
        print(f"Audio backend unavailable (install the 'audio' extra): {exc}", file=sys.stderr)
        return 1
    if not devices:
        print("No input devices found.")
        return 0
    print("Input devices:")
    for dev in devices:
        print(f"  [{dev['index']}] {dev['name']} "
              f"({dev['max_input_channels']} ch @ {int(dev['default_samplerate'])} Hz)")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = _config_from_args(args)

    from rtst.asr import create_asr
    from rtst.captions import create_caption_sink
    from rtst.pipeline import Pipeline
    from rtst.translate import create_translator

    print(
        f"Translating {languages.display_name(config.source_language)} → "
        f"{languages.display_name(config.target_language)} "
        f"(ASR: whisper/{config.asr.model_size}, translator: {config.translation.backend})",
        file=sys.stderr,
    )
    print("Loading models… (first run downloads weights)", file=sys.stderr)

    asr = create_asr(config.asr)
    translator = create_translator(config.translation)
    captions = create_caption_sink(config.caption)
    pipeline = Pipeline(config, asr, translator, captions)

    if config.caption.backend == "overlay":
        return _run_with_overlay(config, pipeline, captions)
    return _run_console(config, pipeline)


def _run_console(config: Config, pipeline) -> int:  # noqa: ANN001
    from rtst.audio.capture import MicrophoneStream

    try:
        with MicrophoneStream(
            sample_rate=config.audio.sample_rate,
            block_ms=config.audio.frame_ms,
            device=config.audio.device,
        ) as mic:
            for _ in pipeline.run(mic.frames()):
                pass
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
    return 0


def _run_with_overlay(config: Config, pipeline, captions) -> int:  # noqa: ANN001 # pragma: no cover
    from rtst.audio.capture import MicrophoneStream

    # Qt must own the main thread; run the capture/translate loop on a worker.
    captions.start()

    def worker() -> None:
        try:
            with MicrophoneStream(
                sample_rate=config.audio.sample_rate,
                block_ms=config.audio.frame_ms,
                device=config.audio.device,
            ) as mic:
                for _ in pipeline.run(mic.frames()):
                    pass
        except Exception:
            logging.getLogger("rtst.cli").exception("capture worker crashed")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    captions.run()  # blocks on the Qt event loop until the window closes
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list-languages":
        return cmd_list_languages()
    if args.command == "list-devices":
        return cmd_list_devices()
    if args.command == "run":
        return cmd_run(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
