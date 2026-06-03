"""
corpus_scan.py — empirical corpus perplexity-spike test (STATE.md §9.5).

Hypothesis under test: a flag was inserted into the training corpus as a memorized span, so
the model would assign it anomalously LOW perplexity. Prior expectation is NEGATIVE (the model
garbles single-occurrence text — WRITEUP §3 — and the strings-leak evidence says the flag was
plaintext, not learned), but we confirm empirically while the brute grinds.

Method: teacher-force the model over a representative Adamastor sample (corpus_txt/*.txt),
computing per-TOKEN bits/token with sliding context (so short spans aren't diluted). Then scan
for the lowest-perplexity short runs (length L) — the most-memorized spans, where an insertion
would hide — and print their text. Also report per-book mean perplexity (low => in training).

Run: python3 corpus_scan.py
"""
import os, glob, math, re, torch
torch.set_num_threads(8); torch.set_grad_enabled(False)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'probe_campos.py'))
     .read().split('def main()')[0])

ROOT = os.path.dirname(os.path.abspath(__file__))
TXT_DIR = "/tmp/corpus_txt"
OUT = open(f"{ROOT}/corpus_scan.out", "w", encoding="utf-8")
def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.write(s + "\n"); OUT.flush()

obj = torch.load(f"{ROOT}/ode.pt", map_location="cpu", weights_only=False)
cfg = obj["model_config"]; BS = cfg["block_size"]
model = GPT(cfg); model.load_state_dict(obj["model"], strict=False); model.to(DEVICE).eval()
LN2 = math.log(2.0)
out(f"device={DEVICE} block_size={BS}")

CHUNK = BS              # 1024
OVERLAP = 128          # warmup context per chunk (scored positions start after this)
SPAN = 24              # token-run length for low-ppl span search
GLOBAL_LOW = []        # (mean_bits, book, char_off, text)


@torch.no_grad()
def per_token_bits(ids):
    """bits[i] = -log2 P(ids[i] | ids[:i]) for i with enough context; None where not scored."""
    n = len(ids)
    bits = [None] * n
    start = 0
    while start < n - 1:
        end = min(start + CHUNK, n)
        window = ids[start:end]
        x = torch.tensor([window], device=DEVICE)
        logp = torch.log_softmax(model(x)[0], -1)        # (T,V)
        # score positions predicting window[j] from window[j-1]; skip warmup (except first chunk)
        j0 = 1 if start == 0 else OVERLAP
        for j in range(j0, len(window)):
            bits[start + j] = -logp[j - 1, window[j]].item() / LN2
        if end == n:
            break
        start = end - OVERLAP
    return bits


def scan_book(path):
    name = os.path.splitext(os.path.basename(path))[0]
    text = open(path, encoding="utf-8").read()
    ids = encode(text)                                   # raw bytes (no special tokens in corpus)
    if len(ids) < SPAN + 4:
        return
    bits = per_token_bits(ids)
    scored = [b for b in bits if b is not None]
    mean = sum(scored) / len(scored) if scored else 99
    out(f"\n[{name}]  tokens={len(ids):7d}  scored={len(scored):7d}  mean={mean:5.2f} b/tok")
    # sliding low-ppl span search (only fully-scored runs)
    run_sum = 0.0; run = []
    best = []  # (mean, byte_index_start, byte_index_end)
    # build a quick char-offset map: ids are bytes; map token index -> byte offset == index
    # (byte-level, 1 byte per token), so token i corresponds to byte i in utf-8 stream.
    raw = text.encode("utf-8", "replace")
    i = 0
    while i + SPAN <= len(ids):
        wnd = bits[i:i + SPAN]
        if any(b is None for b in wnd):
            i += 1; continue
        m = sum(wnd) / SPAN
        best.append((m, i))
        i += 1
    best.sort(key=lambda t: t[0])
    for m, bi in best[:6]:
        frag = raw[bi:bi + SPAN].decode("utf-8", "replace").replace("\n", "⏎")
        out(f"    low-ppl span {m:4.2f} b/tok @tok{bi}: {frag!r}")
        GLOBAL_LOW.append((m, name, bi, frag))


for p in sorted(glob.glob(f"{TXT_DIR}/*.txt")):
    scan_book(p)

out("\n===== GLOBALLY LOWEST-PERPLEXITY SPANS (most memorized; insertion would hide here) =====")
GLOBAL_LOW.sort(key=lambda t: t[0])
for m, name, bi, frag in GLOBAL_LOW[:30]:
    flagish = bool(re.search(r"[{}_]|flag|arcus|ctf|http|©|EPSON|[0-9]{3,}", frag))
    out(f"  {m:4.2f}  [{name}] {frag!r}{'   <<< STRUCTURAL' if flagish else ''}")

out("\nDONE (corpus_scan).")
