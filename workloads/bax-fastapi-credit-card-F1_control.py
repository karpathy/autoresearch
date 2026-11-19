"""Control workload for finding bax-fastapi-credit-card-F1: reproduces the original JMeter plan (reference/credit_card_test.jmx).

Original plan: 1 thread, infinite loop, keep-alive, headers Content-Type/Accept: application/json,
no timers (no think time). Each iteration sends, in order:
  POST /associate_card  {"credit_card": "4111111111111111", "phone": "+5511988887777"}
  POST /retrieve_cards  {"phone_numbers": ["+5511988887777"]}
(the .jmx targets ${server_ip}=192.168.0.246:3000; here the harness --base-url is used instead).
The key space is fixed (1 row), so the finding is expected NOT to activate.

client.csv: ts,endpoint,status,latency_ms,resp_bytes,phase  (phase: refresh for associate, ref for reads).

Usage: python bax-fastapi-credit-card-F1_control.py --base-url URL --duration S --seed N --out client.csv [--think-ms 0]
"""
import argparse
import asyncio
import csv
import random
import sys
import time

import httpx

HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}
SEQUENCE = [
    ("/associate_card", {"credit_card": "4111111111111111", "phone": "+5511988887777"}, "associate_card", "refresh"),
    ("/retrieve_cards", {"phone_numbers": ["+5511988887777"]}, "retrieve_cards", "ref"),
]
OK = {200, 201}


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--duration", type=float, required=True, help="run length in seconds")
    ap.add_argument("--seed", type=int, default=42, help="accepted for the contract (the sequence is fixed)")
    ap.add_argument("--out", required=True, help="client.csv path")
    ap.add_argument("--threads", type=int, default=1, help="concurrent loop workers (original: 1)")
    ap.add_argument("--think-ms", type=float, default=0.0,
                    help="pause after each request in ms (original plan has no timers: 0)")
    ap.add_argument("--timeout", type=float, default=30.0, help="per-request timeout in seconds")
    args = ap.parse_args()
    random.seed(args.seed)

    end = time.time() + args.duration
    f = open(args.out, "w", newline="")
    w = csv.writer(f)
    w.writerow(["ts", "endpoint", "status", "latency_ms", "resp_bytes", "phase"])
    last_flush = time.time()
    counts: dict[str, int] = {}
    errors = 0

    async with httpx.AsyncClient(base_url=args.base_url, headers=HEADERS,
                                 timeout=httpx.Timeout(args.timeout)) as client:
        async def loop():
            nonlocal last_flush, errors
            while time.time() < end:
                for path, body, ep, phase in SEQUENCE:
                    if time.time() >= end:
                        return
                    ts = time.time()
                    t0 = time.perf_counter()
                    try:
                        r = await client.post(path, json=body)
                        status, size = r.status_code, len(r.content)
                    except (httpx.HTTPError, OSError):
                        status, size = 0, 0
                    w.writerow([f"{ts:.4f}", ep, status, f"{(time.perf_counter() - t0) * 1000:.3f}", size, phase])
                    counts[ep] = counts.get(ep, 0) + 1
                    errors += status not in OK
                    if time.time() - last_flush >= 1.0:
                        f.flush()
                        last_flush = time.time()
                    if args.think_ms > 0:
                        await asyncio.sleep(args.think_ms / 1000.0)

        await asyncio.gather(*(loop() for _ in range(args.threads)))

    f.flush()
    f.close()
    print(f"[done] per-endpoint={counts} errors={errors}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(0)
