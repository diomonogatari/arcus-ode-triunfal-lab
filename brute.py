"""
Comprehensive, resumable brute-force of all discovered concepts + every Ode Triunfal
line/clause, across {arcus{}, flag{}, bare} and {ascii_underscore, accent_underscore,
ascii_space} normalizations. Ordered by likelihood; HALTS instantly on any non-'wrong
answer' response. Skips everything already submitted. Logs to brute_log.txt; resumable
via brute_tried.txt.
"""
import sys, re, time, unicodedata, importlib.util, os, glob

ROOT = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ap", f"{ROOT}/arcus_pty.py")
ap = importlib.util.module_from_spec(spec); spec.loader.exec_module(ap)

WRONG_RE = re.compile(r"(?i)wrong answer|try again|tente novamente")
LOG = open(f"{ROOT}/brute_log.txt", "a", encoding="utf-8")
def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()

# ---------- normalizers ----------
def _fold(s): return "".join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
def n_ascii_us(s):
    s = _fold(s).lower(); s = re.sub(r"[^a-z0-9 ]+", " ", s); return re.sub(r"\s+", "_", s.strip())
def n_acc_us(s):
    s = s.lower(); s = re.sub(r"[^0-9a-zà-ÿ ]+", " ", s); return re.sub(r"\s+", "_", s.strip())
def n_ascii_sp(s):
    s = _fold(s).lower(); s = re.sub(r"[^a-z0-9 ]+", " ", s); return re.sub(r"\s+", " ", s.strip())

# ---------- content ----------
def poem_contents():
    out = []
    for ln in open(f"{ROOT}/ode_triunfal.txt", encoding="utf-8"):
        ln = ln.strip()
        if len(ln) < 5 or ln.startswith("Londres"): continue
        out.append(ln)
        for cl in re.split(r"[,;:!?—-]", ln):
            cl = cl.strip()
            if len(cl) >= 8: out.append(cl)
    return out

CONCEPTS = [
 "Platão e Virgílio","Virgílio e Platão","Platão","Virgílio","Platão e Virgílio dentro das máquinas",
 "Platão e Virgílio dentro das máquinas e das luzes eléctricas","E há Platão e Virgílio dentro das máquinas e das luzes eléctricas",
 "foram humanos Virgílio e Platão","Só porque houve outrora e foram humanos Virgílio e Platão",
 "dentro das máquinas","luzes eléctricas","máquinas","das máquinas e das luzes eléctricas",
 "o presente é todo o passado e todo o futuro","Porque o presente é todo o passado e todo o futuro",
 "presente passado futuro","o passado e o futuro",
 "Álvaro de Campos","Álvaro","Campos","de Campos","Alberto Caeiro","Caeiro","Ricardo Reis","Reis",
 "Bernardo Soares","Soares","Fernando Pessoa","Pessoa",
 "Ode Triunfal","Ode","Triunfal","Arcus","ode triunfal",
 "luso lit lm player v2","luso_lit_lm_player_v2","luso lit lm player",
 "Projecto Adamastor","Adamastor","Projeto Adamastor",
 "Hup-la He-ha He-ho Z-z-z-z","Hup-lá hup-lá hup-lá-hô hup-lá Hé-la He-hô H-o-o-o-o Z-z-z-z-z-z-z-z-z-z-z-z",
 "dia triunfal","o dia triunfal","o dia triunfal da minha vida",
 "Londres 1914 Junho","Londres Junho 1914","Junho 1914","1914","Junho de 1914","Londres",
 "Adolfo Casais Monteiro","carta a Adolfo Casais Monteiro",
 "engenheiro","engenheiro naval","Álvaro de Campos engenheiro naval","Glasgow","Tavira",
 "Ah não ser eu toda a gente e toda a parte","À dolorosa luz das grandes lâmpadas eléctricas da fábrica",
 "EPSON W-02","8-19-1908","Ficha Técnica","heterónimo","heteronimo","heterónimo omitido",
 "Nova Minerva","Nova Revelação metálica e dinâmica de Deus","Eia","Eia electricidade",
 "máquina","a máquina","triunfo","o triunfo",
]

def wrap(body, fmt):
    if fmt == "arcus": return "arcus{" + body + "}"
    if fmt == "flag":  return "flag{" + body + "}"
    if fmt == "bare":  return body
    return body

# ---------- already-submitted ----------
tried = set()
for f in glob.glob(f"{ROOT}/candidates*.txt") + glob.glob(f"{ROOT}/attempts/candidates*.txt") + [f"{ROOT}/brute_tried.txt"]:
    if os.path.exists(f):
        for ln in open(f, encoding="utf-8"):
            tried.add(ln.rstrip("\n"))
log(f"# loaded {len(tried)} already-submitted to skip")

# ---------- build ordered passes ----------
# Prefer model-perplexity-ranked bodies (perplexity_rank.py) so the model itself decides
# what to try first; append the original hand-list so nothing is ever lost.
TOPK = 800
RANKED = []
_rf = f"{ROOT}/contents_ranked.txt"
if os.path.exists(_rf):
    RANKED = [ln.rstrip("\n") for ln in open(_rf, encoding="utf-8") if ln.strip()][:TOPK]
    log(f"# loaded {len(RANKED)} perplexity-ranked bodies (top {TOPK}) from contents_ranked.txt")
contents = RANKED + poem_contents() + CONCEPTS
# dedupe contents preserving order
seen=set(); contents=[c for c in contents if not (c in seen or seen.add(c))]
log(f"# {len(contents)} raw contents")

PASSES = [   # (format, normalizer) in priority order
 ("arcus", n_ascii_us), ("bare", n_ascii_us), ("flag", n_ascii_us),
 ("arcus", n_acc_us),   ("arcus", n_ascii_sp),
 ("bare",  n_acc_us),   ("flag",  n_acc_us),
 ("bare",  n_ascii_sp), ("flag",  n_ascii_sp),
]

# generate full ordered candidate list, deduped, skipping already-tried
candidates=[]; seen=set()
for fmt, norm in PASSES:
    for c in contents:
        body = norm(c)
        if not (1 <= len(body) <= 160): continue
        cand = wrap(body, fmt)
        if cand in tried or cand in seen: continue
        seen.add(cand); candidates.append(cand)
log(f"# {len(candidates)} NEW candidates to try (after dedup/skip)")

# ---------- submit loop: anchor on 'checking' to avoid desync false-positives ----------
TRIEDF = open(f"{ROOT}/brute_tried.txt", "a", encoding="utf-8")
CHECK_RE = re.compile(r"(?i)checking")
CHUNK = 20
i = 0; n = len(candidates)
while i < n:
    s = ap.Session()
    try:
        ap.navigate(s)
        if not re.search(r"(?i)flag\s*:", ap.deansi(s.buf)):
            log("  [warn] no flag: prompt after nav; reconnecting"); s.close(); time.sleep(2); continue
        done_in_chunk = 0
        while i < n and done_in_chunk < CHUNK:
            cand = candidates[i]
            mark = len(s.buf)
            s.send(cand + "\r")
            # 1) wait until the submission REGISTERS (server prints 'checking...' or 'wrong answer')
            registered = False; t0 = time.time()
            while time.time() - t0 < 12:
                s.read_until_quiet(total=2.5, quiet=0.6)
                seg = ap.deansi(s.buf[mark:])
                if CHECK_RE.search(seg) or WRONG_RE.search(seg): registered = True; break
            if not registered:
                log(f"  [desync] no 'checking' for {cand!r}; reconnecting (will retry)")
                break  # reconnect; do NOT advance i, do NOT mark tried
            # 2) submission registered -> wait for the verdict
            verdict = None; t0 = time.time()
            while time.time() - t0 < 15:
                s.read_until_quiet(total=2.5, quiet=0.6)
                if WRONG_RE.search(ap.deansi(s.buf[mark:])): verdict = "WRONG"; break
            TRIEDF.write(cand + "\n"); TRIEDF.flush()
            if verdict != "WRONG":
                scr = ap.deansi(s.buf[mark:])
                log("\n" + "="*60)
                log("!!!!! REGISTERED + NO 'wrong answer' — POSSIBLE SOLVE !!!!!")
                log(f"CANDIDATE: {cand!r}")
                log("SCREEN:\n" + scr[-1100:])
                log("="*60)
                s.close(); TRIEDF.close(); sys.exit(0)
            i += 1; done_in_chunk += 1
            if i % 25 == 0: log(f"  ...{i}/{n} tried (all wrong); last={cand!r}")
            # 3) 'try again' -> wait for a fresh flag: prompt before next submit
            s.send("\r"); time.sleep(0.5); s.read_until_quiet(total=5, quiet=0.8)
    except Exception as e:
        log(f"  [session error] {e}; reconnecting")
    finally:
        s.close()
    time.sleep(1)
log(f"\n# DONE: all {n} candidates tried, none accepted.")
