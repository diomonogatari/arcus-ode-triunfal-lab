"""
perplexity_rank.py — fuse the model (as a scorer) with the SSH oracle (as a checker).

STATE.md §4.5: the flag's `{ _ }` scaffold is loss-masked, so the model never EMITS the
flag — but the BODY is plain text the author chose, and the model deeply knows Pessoa
(real Ode opening = 1.44 bits/tok vs random English 4.25). Therefore the true body should
score LOW perplexity under the model. Neither prior investigation closed this loop.

This script:
  1. Builds a large body universe: every in-line word n-gram (len 1..7) from the Ode +
     the on-screen hint passage, plus curated concepts.
  2. Scores each body by the model's mean bits/token (lower = more 'expected' = better).
  3. Emits `contents_ranked.txt` (natural phrases, ascending perplexity) for brute.py to
     expand format×norm and submit lowest-first.
  4. STATE.md §9.3: scans where P('_'), P(digit), P('{'), P('}') spike above ~0 — since
     '_'/'{'/'}' are loss-masked, ANY non-trivial mass is a flag-boundary signal.

Run: python3 perplexity_rank.py
"""
import os, re, math, unicodedata, torch
torch.set_num_threads(8); torch.set_grad_enabled(False)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'probe_campos.py'))
     .read().split('def main()')[0])

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = open(f"{ROOT}/ppl_findings.txt", "w", encoding="utf-8")
def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.write(s + "\n"); OUT.flush()

obj = torch.load(f"{ROOT}/ode.pt", map_location="cpu", weights_only=False)
cfg = obj["model_config"]; BS = cfg["block_size"]
model = GPT(cfg); model.load_state_dict(obj["model"], strict=False); model.to(DEVICE).eval()
out(f"device={DEVICE}  block_size={BS}")

LN2 = math.log(2.0)


@torch.no_grad()
def bits_per_token(text, prefix="\n"):
    """Mean -log2 P(token | context) over the body tokens (prefix not scored)."""
    pids = encode(prefix); bids = encode(text)
    if not bids:
        return 99.0, 0
    full = (pids + bids)[:BS]
    nb = len(full) - len(pids)
    if nb <= 0:
        return 99.0, 0
    x = torch.tensor([full], device=DEVICE)
    logp = torch.log_softmax(model(x)[0], -1)          # (T, V)
    tot = 0.0
    for j in range(len(pids), len(full)):
        tot += -logp[j - 1, full[j]].item()
    return (tot / nb) / LN2, nb


# ---------- build the body universe ----------
def in_line_units(path):
    """Lines, comma/semicolon clauses, and in-line word n-grams (len 1..7)."""
    units = []
    for ln in open(path, encoding="utf-8"):
        ln = ln.strip()
        if len(ln) < 4 or ln.startswith("Londres"):
            continue
        segs = [ln] + [c.strip() for c in re.split(r"[,;:!?—-]", ln) if len(c.strip()) >= 4]
        for seg in segs:
            units.append(seg)
            ws = seg.split()
            for nlen in range(1, 8):
                for k in range(0, len(ws) - nlen + 1):
                    units.append(" ".join(ws[k:k + nlen]))
    return units


HINT = ("Canto, e canto o presente, e também o passado e o futuro, "
        "Porque o presente é todo o passado e todo o futuro "
        "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas "
        "Só porque houve outrora e foram humanos Virgílio e Platão")

# curated thematic concepts (the author's likely vocabulary)
CONCEPTS = [
 "Platão e Virgílio", "Virgílio e Platão", "Platão", "Virgílio", "dentro das máquinas",
 "luzes eléctricas", "máquinas", "das máquinas e das luzes eléctricas", "máquina",
 "o presente é todo o passado e todo o futuro", "presente passado futuro",
 "Álvaro de Campos", "Campos", "Alberto Caeiro", "Ricardo Reis", "Bernardo Soares",
 "Fernando Pessoa", "Ode Triunfal", "Ode", "Triunfal", "Arcus", "triunfo", "dia triunfal",
 "o dia triunfal da minha vida", "engenheiro naval", "Eia", "Eia electricidade",
 "Projecto Adamastor", "Adamastor", "Ah não ser eu toda a gente e toda a parte",
 "À dolorosa luz das grandes lâmpadas eléctricas da fábrica",
]

universe = in_line_units(f"{ROOT}/ode_triunfal.txt")
# n-grams from the hint passage too
hw = re.sub(r"[,]", "", HINT).split()
for nlen in range(1, 8):
    for k in range(0, len(hw) - nlen + 1):
        universe.append(" ".join(hw[k:k + nlen]))
universe += CONCEPTS

out(f"raw universe units: {len(universe)}")

# dedupe case-insensitively on collapsed whitespace, keep first natural form
seen = set(); bodies = []
for u in universe:
    u = re.sub(r"\s+", " ", u).strip()
    key = u.lower()
    if not u or key in seen or not (1 <= len(u) <= 120):
        continue
    seen.add(key); bodies.append(u)
out(f"deduped bodies to score: {len(bodies)}")

# ---------- score ----------
scored = []
for i, b in enumerate(bodies):
    bpt, nb = bits_per_token(b)
    scored.append((bpt, nb, b))
    if (i + 1) % 500 == 0:
        out(f"  scored {i+1}/{len(bodies)}")
scored.sort(key=lambda t: t[0])

out("\n===== LOWEST-PERPLEXITY BODIES (most 'expected' = best candidates) =====")
for bpt, nb, b in scored[:60]:
    out(f"  {bpt:5.2f} b/tok  (n={nb:2d})  {b!r}")

# write ranked contents for brute.py (natural phrases, ascending ppl)
with open(f"{ROOT}/contents_ranked.txt", "w", encoding="utf-8") as f:
    for bpt, nb, b in scored:
        f.write(b + "\n")
out(f"\nwrote {len(scored)} ranked bodies -> contents_ranked.txt")


# ---------- STATE.md §9.3: where does P('_')/P(digit)/P('{')/P('}') spike? ----------
out("\n===== §9.3 masked-token emission scan (loss-masked => ANY mass is signal) =====")
USCORE = {95, 260}; BRACE_O = {123, 261}; BRACE_C = {125}; DIGITS = set(range(48, 58))


@torch.no_grad()
def masked_probs(prefix):
    ids = encode(prefix)[-BS:]
    if not ids:
        ids = encode("\n")
    p = torch.softmax(model(torch.tensor([ids], device=DEVICE))[0, -1], -1)
    pu = sum(p[t].item() for t in USCORE)
    pd = sum(p[t].item() for t in DIGITS)
    pbo = sum(p[t].item() for t in BRACE_O)
    pbc = sum(p[t].item() for t in BRACE_C)
    return pu, pd, pbo, pbc


# scan curated prefixes + every prefix of the hint + the lowest-ppl bodies wrapped as flag bodies
scan_prefixes = ["flag{", "arcus{", "{", "_", "\n", HINT + "\nflag{", HINT + " "]
scan_prefixes += [HINT[:k] for k in range(8, len(HINT), 12)]
scan_prefixes += ["flag{" + b.lower().replace(" ", "_") for bpt, nb, b in scored[:25]]
rows = []
for pfx in scan_prefixes:
    pu, pd, pbo, pbc = masked_probs(pfx)
    rows.append((pu + pd, pu, pd, pbo, pbc, pfx))
rows.sort(reverse=True)
out("  top contexts by P('_')+P(digit):")
for tot, pu, pd, pbo, pbc, pfx in rows[:20]:
    out(f"    P_=%.2e P#=%.2e P{{=%.2e P}}=%.2e  | {pfx[-44:]!r}" % (pu, pd, pbo, pbc))

out("\nDONE (perplexity_rank).")
