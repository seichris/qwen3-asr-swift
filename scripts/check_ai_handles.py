#!/usr/bin/env python3
"""Check 4-digit ai.com botname availability via api.ai.com.

Auth:
  - export AI_COM_TOKEN='...'
    (sent as Cookie: token=...)
  - or export AI_COM_COOKIE='name=value; ...'

Usage examples:
  ./scripts/check_ai_handles.py 1337 2020 4321
  ./scripts/check_ai_handles.py --range 1000 1999 --delay-ms 350
  ./scripts/check_ai_handles.py --suggest pretty --top 50

Notes:
  - Be respectful: keep request volume low; 0000-9999 scanning is 10k requests.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

API_URL = "https://api.ai.com/user/botname/check"


def _cookie_header() -> str | None:
    cookie = os.environ.get("AI_COM_COOKIE")
    if cookie:
        return cookie
    tok = os.environ.get("AI_COM_TOKEN")
    if tok:
        return f"token={tok}"
    return None


def _req(handle: str, cookie: str | None, timeout_s: float) -> urllib.request.Request:
    payload = json.dumps({"botname": handle}).encode("utf-8")
    req = urllib.request.Request(API_URL, data=payload, method="POST")
    req.add_header("content-type", "application/json")
    req.add_header("accept", "application/json")
    # Some APIs enforce this; harmless if ignored.
    req.add_header("origin", "https://ai.com")
    req.add_header("referer", "https://ai.com/")
    if cookie:
        req.add_header("cookie", cookie)
    return req


def check_handle(handle: str, *, cookie: str | None, timeout_s: float, max_retries: int) -> tuple[str, int, str]:
    """Returns (handle, http_status, note)."""
    # Basic validation: keep it to 4 digits.
    if len(handle) != 4 or not handle.isdigit():
        return (handle, 0, "invalid (must be 4 digits)")

    backoff_s = 1.0
    for attempt in range(max_retries + 1):
        req = _req(handle, cookie, timeout_s)
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                status = getattr(resp, "status", 200)
                # Some servers return empty JSON, some return text; we don't rely on body.
                return (handle, int(status), "ok")
        except urllib.error.HTTPError as e:
            status = int(e.code)
            # 409 observed as taken.
            if status == 409:
                return (handle, status, "taken")
            # 200 could still come as HTTPError? Not likely, but keep generic.
            if status in (200, 201, 202, 204):
                return (handle, status, "available?")
            # Respect rate limiting.
            if status == 429 and attempt < max_retries:
                retry_after = e.headers.get("retry-after")
                if retry_after:
                    try:
                        sleep_s = float(retry_after)
                    except ValueError:
                        sleep_s = backoff_s
                else:
                    sleep_s = backoff_s
                time.sleep(sleep_s)
                backoff_s = min(backoff_s * 2.0, 30.0)
                continue
            return (handle, status, f"http error")
        except Exception:
            if attempt < max_retries:
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 2.0, 30.0)
                continue
            return (handle, -1, "network error")

    return (handle, -1, "unexpected")


def _pretty_candidates() -> list[str]:
    # Curated patterns people tend to like.
    out: set[str] = set()

    # Repeats, mirrored, sequential-ish.
    for a in range(10):
        out.add(f"{a}{a}{a}{a}")
    for a in range(10):
        for b in range(10):
            out.add(f"{a}{b}{a}{b}")
            out.add(f"{a}{b}{b}{a}")

    # Some common "nice" combos.
    for s in [
        "0001",
        "0010",
        "0101",
        "0110",
        "0123",
        "0220",
        "0246",
        "1110",
        "1212",
        "1313",
        "1337",
        "1414",
        "1515",
        "1616",
        "1717",
        "1818",
        "1919",
        "2000",
        "2020",
        "2121",
        "2220",
        "2345",
        "2468",
        "3030",
        "4040",
        "5050",
        "6060",
        "7070",
        "8080",
        "9000",
        "9876",
    ]:
        out.add(s)

    # Keep only 4-digit.
    out2 = [x for x in out if len(x) == 4 and x.isdigit()]
    out2.sort()
    return out2


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("handles", nargs="*", help="4-digit handles to check")
    ap.add_argument("--range", nargs=2, type=int, metavar=("START", "END"), help="inclusive range, e.g. 1000 1999")
    ap.add_argument("--delay-ms", type=int, default=350, help="delay between requests")
    ap.add_argument("--jitter-ms", type=int, default=150, help="random jitter added to delay")
    ap.add_argument("--timeout-s", type=float, default=15.0, help="per-request timeout")
    ap.add_argument("--retries", type=int, default=2, help="retries for 429/network")
    ap.add_argument("--suggest", choices=["pretty"], help="generate candidate handles")
    ap.add_argument("--top", type=int, default=100, help="when using --suggest, limit count")
    args = ap.parse_args(argv)

    cookie = _cookie_header()
    if not cookie:
        print("Missing auth. Set AI_COM_TOKEN or AI_COM_COOKIE.", file=sys.stderr)
        return 2

    targets: list[str] = []
    if args.suggest == "pretty":
        targets = _pretty_candidates()[: max(0, args.top)]

    if args.range:
        start, end = args.range
        if start > end:
            start, end = end, start
        if start < 0 or end > 9999:
            print("Range must be within 0..9999", file=sys.stderr)
            return 2
        targets.extend([f"{i:04d}" for i in range(start, end + 1)])

    targets.extend(args.handles)

    # Dedup, preserve order.
    seen: set[str] = set()
    uniq: list[str] = []
    for h in targets:
        if h not in seen:
            uniq.append(h)
            seen.add(h)

    if not uniq:
        print("No handles provided. Example: ./scripts/check_ai_handles.py 1337 2020 4321", file=sys.stderr)
        return 2

    available: list[str] = []
    taken: list[str] = []
    errors: list[str] = []

    for idx, h in enumerate(uniq, start=1):
        handle, status, note = check_handle(
            h,
            cookie=cookie,
            timeout_s=args.timeout_s,
            max_retries=max(0, args.retries),
        )

        # Heuristics: based on what you observed (409 = taken).
        if status == 409:
            taken.append(handle)
            verdict = "TAKEN"
        elif status in (200, 201, 202, 204):
            available.append(handle)
            verdict = "AVAILABLE"
        elif status == 0:
            errors.append(handle)
            verdict = "INVALID"
        else:
            errors.append(handle)
            verdict = f"ERR({status})"

        print(f"[{idx}/{len(uniq)}] {handle}: {verdict} ({note})")

        if idx != len(uniq):
            delay_s = max(0.0, (args.delay_ms + random.randint(0, max(0, args.jitter_ms))) / 1000.0)
            time.sleep(delay_s)

    print("\nSummary")
    print(f"available: {len(available)}")
    if available:
        print("  " + ", ".join(available[:50]) + (" ..." if len(available) > 50 else ""))
    print(f"taken: {len(taken)}")
    print(f"errors/invalid: {len(errors)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
