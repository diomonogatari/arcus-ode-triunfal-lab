"""
timing_probe.py — format-gate side-channel probe against the Arcus SSH validator.

Hypothesis: if the validator does `if matches_format(x) { expensiveCheck(x) }`, then a
well-formed guess (e.g. arcus{...}) spends measurably more server time than malformed
garbage. The delta between input SHAPES would reveal the real flag wrapper.

Method: hold ONE SSH session, submit differently-shaped (all-guaranteed-wrong) inputs
ROUND-ROBIN, finely timestamping when 'checking' and 'wrong answer' appear. Interleaving
makes network RTT, TUI render cost, and background load from the running brute-force cancel
out across shapes. The discriminating metric is dt = t(wrong) - t(checking): server-side
validation time once the input is in hand.

Run:  python3 timing_probe.py [REPS]
"""
import os, sys, select, time, re, statistics as st
import arcus_pty as ap

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 18
CHECK_RE = re.compile(r"(?i)checking")
WRONG_RE = re.compile(r"(?i)wrong answer|try again|tente novamente")

# All wrong, but structurally distinct. 'tag' is the comparison key; payload kept same
# length where possible so any per-byte compare cost is constant across shapes.
SHAPES = [
    ("bare",        "zzzzzqxqzz"),
    ("flag",        "flag{zzzz}"),
    ("arcus",       "arcus{zzzz}"),
    ("ctf",         "ctf{zzzzz}"),
    ("arcus_open",  "arcus{zzzzz"),     # well-prefixed but unclosed brace
    ("punct",       "!!!@@@###$$"),
    ("arcus_real",  "arcus{platao_e_virgilio}"),  # plausible-but-known-wrong real guess
]


def timed_submit(s, cand):
    """Submit one candidate; return (t_check, t_wrong) deltas from send, or None on miss."""
    mark = len(s.buf)
    t0 = time.perf_counter()
    s.send(cand + "\r")
    t_check = t_wrong = None
    deadline = time.perf_counter() + 25.0
    while time.perf_counter() < deadline and t_wrong is None:
        r, _, _ = select.select([s.master], [], [], 0.02)
        if r:
            try:
                chunk = os.read(s.master, 65536)
            except OSError:
                break
            if chunk:
                s.buf += chunk
                now = time.perf_counter()
                seg = ap.deansi(s.buf[mark:])
                if t_check is None and CHECK_RE.search(seg):
                    t_check = now - t0
                if WRONG_RE.search(seg):
                    t_wrong = now - t0
    return t_check, t_wrong


def main():
    s = ap.Session()
    samples = {tag: {"send2wrong": [], "send2check": [], "check2wrong": []} for tag, _ in SHAPES}
    misses = []
    try:
        ap.navigate(s)
        if not re.search(r"(?i)flag\s*:", ap.deansi(s.buf)):
            print("WARNING: flag: prompt not detected after nav")
        order = []
        for _ in range(REPS):
            order.extend(SHAPES)  # round-robin, REPS passes over all shapes
        print(f"running {len(order)} timed submissions ({REPS} reps x {len(SHAPES)} shapes)\n")
        for i, (tag, payload) in enumerate(order):
            tc, tw = timed_submit(s, payload)
            if tw is None:
                misses.append((tag, payload, "no-wrong"))
                # recover: hop a fresh line/prompt
                s.send("\r"); s.read_until_quiet(total=4, quiet=0.8)
                continue
            samples[tag]["send2wrong"].append(tw)
            if tc is not None:
                samples[tag]["send2check"].append(tc)
                samples[tag]["check2wrong"].append(tw - tc)
            if (i + 1) % len(SHAPES) == 0:
                print(f"  pass {(i+1)//len(SHAPES)}/{REPS} done")
            # back to flag: prompt
            s.send("\r")
            s.read_until_quiet(total=5, quiet=0.7)
    finally:
        s.close()

    def stats(xs):
        if not xs:
            return "n=0"
        xs = sorted(xs)
        med = st.median(xs)
        mn = min(xs); mx = max(xs)
        sd = st.pstdev(xs) if len(xs) > 1 else 0.0
        return f"n={len(xs):2d}  med={med*1000:7.1f}ms  min={mn*1000:7.1f}  max={mx*1000:7.1f}  sd={sd*1000:6.1f}"

    print("\n================= RESULTS =================")
    print("send2wrong = full submit->verdict ; check2wrong = SERVER validation time (key metric)\n")
    for tag, payload in SHAPES:
        d = samples[tag]
        print(f"[{tag:11}] {payload!r}")
        print(f"    send2wrong : {stats(d['send2wrong'])}")
        print(f"    send2check : {stats(d['send2check'])}")
        print(f"    check2wrong: {stats(d['check2wrong'])}   <== server validation")
    if misses:
        print(f"\n{len(misses)} misses (no 'wrong' seen): {misses[:10]}")
    # ranking by the key metric
    print("\n--- ranked by median check2wrong (server validation time) ---")
    rank = []
    for tag, _ in SHAPES:
        xs = samples[tag]["check2wrong"]
        if xs:
            rank.append((st.median(xs), tag))
    for med, tag in sorted(rank):
        print(f"    {med*1000:7.1f}ms   {tag}")
    print("\nDONE (timing_probe).")


if __name__ == "__main__":
    main()
