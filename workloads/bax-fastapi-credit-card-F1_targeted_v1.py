"""Targeted workload for finding bax-fastapi-credit-card-F1 (unbounded associations + unpaginated retrieve).

Trigger (from the card): sustained POST /associate_card with *distinct* (credit_card, phone) pairs,
concentrated on a small set of "hot" phones, interleaved with POST /retrieve_cards probing those
phones. Symptom: latency / response size of /retrieve_cards for the hot phones, db.sqlite3 size.

Streams (all paced, closed-loop, no client backlog):
  grow     POST /associate_card with a never-seen card on a hot phone, at --new-pair-rate/s
           (cards are a deterministic, seed-dependent permutation of a counter: 16 digits,
           distinct, not sorted; hot phones are served round-robin).
  refresh  POST /associate_card re-sending an already-associated (card, hot phone) pair
           (INSERT OR IGNORE -> rowcount 0; realistic re-registration; adds no rows).
  ref      POST /associate_card cycling over the fixed pairs of the reference phones.
  prime    one-off association of the reference phones' fixed cards at start-up.
  probe    POST /retrieve_cards {"phone_numbers": [<hot phone>]} every --probe-interval s
           (endpoint retrieve_hot, or retrieve_hot<N> with several hot phones)
  refread  POST /retrieve_cards for each reference phone, same interval (endpoint retrieve_ref<size>)

--key-space N (matched control R0b): the grow stream cycles over N fixed cards per hot phone at the
same rate and with the same request shape (phase 'cycle'), so nothing accumulates.

client.csv: ts,endpoint,status,latency_ms,resp_bytes,phase  (one row per request, streamed).

Usage (workload contract of harness/run_diagnostic.py):
  python bax-fastapi-credit-card-F1_targeted_v1.py --base-url URL --duration S --seed N --out client.csv [options]
"""
import argparse
import asyncio
import csv
import random
import sys
import time

import httpx

HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}
OK = {200, 201}
CARD_SPACE = 10 ** 15


class Pacer:
    """Shared slot scheduler: hands out send times spaced 1/rate apart.

    If the consumers fall behind (server slower than the target rate), missed slots are
    dropped instead of being replayed in a burst, so degradation shows up as lower achieved
    throughput rather than as a client-side backlog.
    """

    def __init__(self, rate: float, start: float):
        self.interval = 1.0 / rate
        self.next = start
        self.lock = asyncio.Lock()

    async def wait_slot(self) -> None:
        async with self.lock:
            now = time.time()
            if self.next < now - self.interval:
                self.next = now
            slot = self.next
            self.next += self.interval
        delay = slot - time.time()
        if delay > 0:
            await asyncio.sleep(delay)


class Log:
    def __init__(self, path: str, flush_every: float = 1.0):
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["ts", "endpoint", "status", "latency_ms", "resp_bytes", "phase"])
        self.flush_every = flush_every
        self.last_flush = time.time()
        self.counts: dict[str, int] = {}
        self.errors = 0

    def row(self, ts, endpoint, status, latency_ms, resp_bytes, phase):
        self.w.writerow([f"{ts:.4f}", endpoint, status, f"{latency_ms:.3f}", resp_bytes, phase])
        self.counts[endpoint] = self.counts.get(endpoint, 0) + 1
        if status not in OK:
            self.errors += 1
        now = time.time()
        if now - self.last_flush >= self.flush_every:
            self.f.flush()
            self.last_flush = now

    def close(self):
        self.f.flush()
        self.f.close()


async def post(client: httpx.AsyncClient, log: Log, path: str, body: dict, endpoint: str, phase: str) -> int:
    ts = time.time()
    t0 = time.perf_counter()
    try:
        r = await client.post(path, json=body)
        status, size = r.status_code, len(r.content)
    except (httpx.HTTPError, OSError):
        status, size = 0, 0
    log.row(ts, endpoint, status, (time.perf_counter() - t0) * 1000.0, size, phase)
    return status


def card_factory(seed: int, prefix: str):
    """Deterministic injective map counter -> 16-digit card: prefix + affine permutation of Z_10^15."""
    rng = random.Random(seed)
    a = rng.randrange(10 ** 14, CARD_SPACE)
    while a % 2 == 0 or a % 5 == 0:  # coprime with 10^15 -> bijection, cards never repeat
        a += 1
    b = rng.randrange(CARD_SPACE)
    return lambda n: f"{prefix}{(n * a + b) % CARD_SPACE:015d}"


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--duration", type=float, required=True, help="run length in seconds")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, help="client.csv path")
    ap.add_argument("--hot-phones", type=int, default=1,
                    help="number of hot phones receiving the new cards, round-robin (default 1)")
    ap.add_argument("--phone-prefix", default="+55119000",
                    help="hot phone prefix; hot phone i is <prefix><i:04d> (default +55119000)")
    ap.add_argument("--new-pair-rate", type=float, default=10.0,
                    help="target new (card, hot phone) associations per second, over all hot phones (default 10)")
    ap.add_argument("--grow-workers", type=int, default=2, help="concurrent workers for the grow stream")
    ap.add_argument("--key-space", type=int, default=0,
                    help="0 = every grow request uses a new card (targeted, default); N > 0 = the grow stream "
                         "cycles over N fixed cards per hot phone at the same rate (matched control R0b: same "
                         "load, no accumulation; phase label 'cycle')")
    ap.add_argument("--refresh-rate", type=float, default=2.0,
                    help="re-associations of already-associated hot pairs per second (0 = off; default 2)")
    ap.add_argument("--ref-sizes", default="4,200",
                    help="comma list: one reference phone per entry, with that constant number of cards (default 4,200)")
    ap.add_argument("--ref-rate", type=float, default=2.0,
                    help="associations per second cycling over all reference pairs (0 = off; default 2)")
    ap.add_argument("--probe-interval", type=float, default=2.0,
                    help="seconds between POST /retrieve_cards reads, per probed phone (hot and reference; default 2)")
    ap.add_argument("--timeout", type=float, default=30.0, help="per-request timeout in seconds")
    ap.add_argument("--progress-every", type=float, default=60.0, help="seconds between progress lines on stdout")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    log = Log(args.out)
    start = time.time()
    end = start + args.duration

    hot = [f"{args.phone_prefix}{i + 1:04d}" for i in range(args.hot_phones)]
    hot_card = {p: card_factory(args.seed * 1000 + i, "4") for i, p in enumerate(hot)}
    ref_sizes = [int(x) for x in args.ref_sizes.split(",") if x.strip()]
    ref_phones = [(f"+55119100{s:04d}", s) for s in ref_sizes]
    ref_pairs = []
    for i, (ph, s) in enumerate(ref_phones):
        mk = card_factory(args.seed * 1000 + 500 + i, "5")
        ref_pairs += [(mk(j), ph) for j in range(s)]
    hot_ep = {p: ("retrieve_hot" if len(hot) == 1 else f"retrieve_hot{i + 1}") for i, p in enumerate(hot)}

    sent = {p: 0 for p in hot}  # grow requests sent per hot phone (distinct cards when key-space = 0)
    ok = {p: 0 for p in hot}
    counter = {"n": 0}
    grow_phase = "grow" if args.key_space <= 0 else "cycle"

    limits = httpx.Limits(max_connections=64, max_keepalive_connections=64)
    async with httpx.AsyncClient(base_url=args.base_url, headers=HEADERS,
                                 timeout=httpx.Timeout(args.timeout), limits=limits) as client:

        # prime: associate the reference phones' fixed cards (constant row count afterwards)
        for card, ph in ref_pairs:
            if time.time() >= end:
                break
            await post(client, log, "/associate_card", {"credit_card": card, "phone": ph}, "associate_card", "prime")

        t0 = time.time()

        async def grow_worker():
            while True:
                await grow_pacer.wait_slot()
                if time.time() >= end:
                    return
                n = counter["n"]
                counter["n"] += 1
                ph = hot[n % len(hot)]
                k = sent[ph]
                sent[ph] += 1
                idx = k if args.key_space <= 0 else k % args.key_space
                if await post(client, log, "/associate_card", {"credit_card": hot_card[ph](idx), "phone": ph},
                              "associate_card", grow_phase) in OK:
                    ok[ph] += 1

        async def refresh_worker():
            pacer = Pacer(args.refresh_rate, t0)
            while True:
                await pacer.wait_slot()
                if time.time() >= end:
                    return
                ph = hot[rng.randrange(len(hot))]
                if sent[ph] == 0:
                    continue
                known = sent[ph] if args.key_space <= 0 else min(sent[ph], args.key_space)
                card = hot_card[ph](rng.randrange(known))
                await post(client, log, "/associate_card", {"credit_card": card, "phone": ph},
                           "associate_card", "refresh")

        async def ref_worker():
            pacer = Pacer(args.ref_rate, t0)
            i = 0
            while True:
                await pacer.wait_slot()
                if time.time() >= end:
                    return
                card, ph = ref_pairs[i % len(ref_pairs)]
                i += 1
                await post(client, log, "/associate_card", {"credit_card": card, "phone": ph}, "associate_card", "ref")

        async def probe_worker(phone: str, endpoint: str, phase: str, offset: float):
            # one dedicated closed-loop worker per probed phone: fixed rate, fixed parameters
            pacer = Pacer(1.0 / args.probe_interval, t0 + offset)
            while True:
                await pacer.wait_slot()
                if time.time() >= end:
                    return
                await post(client, log, "/retrieve_cards", {"phone_numbers": [phone]}, endpoint, phase)

        async def progress():
            while time.time() < end:
                await asyncio.sleep(min(args.progress_every, max(0.0, end - time.time())))
                el = time.time() - start
                keys = ", ".join(f"{p}: sent={sent[p]} ok={ok[p]}" for p in hot)
                print(f"[{el:7.0f}s] {keys} | requests={sum(log.counts.values())} errors={log.errors}",
                      flush=True)

        grow_pacer = Pacer(args.new_pair_rate, t0)
        tasks = [asyncio.create_task(grow_worker()) for _ in range(args.grow_workers)]
        if args.refresh_rate > 0:
            tasks.append(asyncio.create_task(refresh_worker()))
        if args.ref_rate > 0 and ref_pairs:
            tasks.append(asyncio.create_task(ref_worker()))
        probes = [(p, hot_ep[p], "probe") for p in hot] + \
                 [(ph, f"retrieve_ref{s}", "refread") for ph, s in ref_phones]
        for k, (ph, ep, phase) in enumerate(probes):
            # first probe one interval after t0 (the hot phone has cards by then, so no 404),
            # staggered inside the interval
            tasks.append(asyncio.create_task(
                probe_worker(ph, ep, phase, args.probe_interval * (1 + k / max(1, len(probes))))))
        prog = asyncio.create_task(progress())

        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=max(1.0, end - time.time()) + args.timeout + 5)
        except asyncio.TimeoutError:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        prog.cancel()
        await asyncio.gather(prog, return_exceptions=True)

    log.close()
    keys = ", ".join(f"{p}: sent={sent[p]} ok={ok[p]}" for p in hot)
    print(f"[done] {keys} | per-endpoint={log.counts} errors={log.errors}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(0)
