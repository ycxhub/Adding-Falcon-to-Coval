#!/usr/bin/env python3
"""Try Murf List Voices and Stream Speech from your machine.

Reads MURF_API_KEY from the repo .env file. Never prints the key.

Examples (from the repo root):

  python3 tools/murf_api_playground.py list
  python3 tools/murf_api_playground.py list --model falcon-2
  python3 tools/murf_api_playground.py list --model falcon-2 --find harman
  python3 tools/murf_api_playground.py stream --voice Harman --locale pa-IN
  python3 tools/murf_api_playground.py stream --voice pa-IN-harman --locale pa-IN
  python3 tools/murf_api_playground.py generate --voice pa-IN-harman
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"
OUT_DIR = Path("/tmp/murf-playground")

LIST_URL = "https://api.murf.ai/v1/speech/voices"
STREAM_URL = "https://global.api.murf.ai/v1/speech/stream"
GENERATE_URL = "https://api.murf.ai/v1/speech/generate"


def load_api_key() -> str:
    key = os.environ.get("MURF_API_KEY", "").strip()
    if key:
        return key
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == "MURF_API_KEY":
                key = value.strip().strip('"').strip("'")
                break
    if not key:
        sys.exit(
            f"No MURF_API_KEY found. Add it to {ENV_FILE} or export it in your shell."
        )
    return key


def request(method: str, url: str, payload: dict | None = None) -> tuple[int, str, bytes]:
    body = None if payload is None else json.dumps(payload).encode()
    headers = {"api-key": load_api_key(), "Accept": "*/*"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.headers.get("Content-Type", ""), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type", ""), exc.read()


def cmd_list(args: argparse.Namespace) -> None:
    query = {}
    if args.model:
        query["model"] = args.model
    url = LIST_URL if not query else f"{LIST_URL}?{urllib.parse.urlencode(query)}"
    print(f"GET {url}\n")
    status, content_type, raw = request("GET", url)
    print(f"HTTP {status}  {content_type}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(raw.decode("utf-8", "replace")[:2000])
        sys.exit(1)

    if not isinstance(data, list):
        print(json.dumps(data, indent=2)[:4000])
        return

    print(f"voices returned: {len(data)}")
    print("first 8 voiceIds:")
    for item in data[:8]:
        print(f"  {item.get('voiceId'):<24}  {item.get('displayName')}  locale={item.get('locale')}")

    if args.find:
        needle = args.find.lower()
        hits = [
            item
            for item in data
            if needle in str(item.get("voiceId", "")).lower()
            or needle in str(item.get("displayName", "")).lower()
        ]
        print(f"\nmatches for {args.find!r}: {len(hits)}")
        for item in hits:
            slim = {
                "voiceId": item.get("voiceId"),
                "displayName": item.get("displayName"),
                "locale": item.get("locale"),
                "availableStyles": item.get("availableStyles"),
                "supportedLocales": list((item.get("supportedLocales") or {}).keys()),
            }
            print(json.dumps(slim, indent=2))
        return

    if args.full:
        print("\nfull JSON:")
        print(json.dumps(data, indent=2))


def cmd_stream(args: argparse.Namespace) -> None:
    payload = {
        "text": args.text,
        "voiceId": args.voice,
        "model": args.model,
        "format": "WAV",
        "sampleRate": 24000,
    }
    if args.locale:
        payload["locale"] = args.locale
    print(f"POST {STREAM_URL}")
    print(json.dumps(payload, indent=2), "\n")
    status, content_type, raw = request("POST", STREAM_URL, payload)
    print(f"HTTP {status}  {content_type}  bytes={len(raw)}")
    if status != 200 or not raw.startswith(b"RIFF"):
        try:
            print(json.dumps(json.loads(raw), indent=2))
        except json.JSONDecodeError:
            print(raw[:1500])
        sys.exit(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_voice = args.voice.replace("/", "_")
    path = OUT_DIR / f"{safe_voice}.wav"
    path.write_bytes(raw)
    print(f"saved audio: {path}")
    print("open it with: open " + str(path))


def cmd_generate(args: argparse.Namespace) -> None:
    payload = {"text": args.text, "voiceId": args.voice}
    if args.locale:
        payload["locale"] = args.locale
    print(f"POST {GENERATE_URL}")
    print(json.dumps(payload, indent=2), "\n")
    status, content_type, raw = request("POST", GENERATE_URL, payload)
    print(f"HTTP {status}  {content_type}")
    try:
        print(json.dumps(json.loads(raw), indent=2)[:3000])
    except json.JSONDecodeError:
        print(raw.decode("utf-8", "replace")[:2000])


def main() -> None:
    parser = argparse.ArgumentParser(description="Call Murf List Voices / Stream / Generate")
    sub = parser.add_subparsers(dest="command", required=True)

    list_p = sub.add_parser("list", help="GET /v1/speech/voices")
    list_p.add_argument("--model", help="falcon-2 or gen2. Omit to see the API default.")
    list_p.add_argument("--find", help="Filter printed results by name, e.g. harman")
    list_p.add_argument("--full", action="store_true", help="Print the full JSON list")
    list_p.set_defaults(func=cmd_list)

    stream_p = sub.add_parser("stream", help="POST /v1/speech/stream (Falcon)")
    stream_p.add_argument("--voice", required=True, help='e.g. Harman or pa-IN-harman')
    stream_p.add_argument("--locale", default="", help='e.g. pa-IN or en-US')
    stream_p.add_argument("--model", default="falcon-2")
    stream_p.add_argument("--text", default="Hello, this is a Murf API test.")
    stream_p.set_defaults(func=cmd_stream)

    gen_p = sub.add_parser("generate", help="POST /v1/speech/generate (Gen2 studio)")
    gen_p.add_argument("--voice", required=True)
    gen_p.add_argument("--locale", default="")
    gen_p.add_argument("--text", default="Hello, this is a Murf API test.")
    gen_p.set_defaults(func=cmd_generate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
