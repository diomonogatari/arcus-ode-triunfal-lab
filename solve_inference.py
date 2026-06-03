"""
solve_inference.py — recover the Arcus flag by running ode.pt, not by carving bytes.

Key fix vs main.py: generation maintains the token-ID list directly. main.py's
generate()/greedy_flag() append the *decoded string* and re-encode every step, so any
'{' or '_' the model emits as a raw byte (123/95) is silently re-encoded as the special
tokens 261/260 on the next step — feeding the model a sequence it never produced. That
distribution shift is the most likely cause of the earlier "odd/partial" output, right
where a flag's '{' and '_' characters live.

Output is mirrored to stdout and to inference_findings.txt.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT = f"{ROOT}/ode.pt"
ODE_TXT = f"{ROOT}/ode_triunfal.txt"
LOG_PATH = f"{ROOT}/inference_findings.txt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_logf = open(LOG_PATH, "w", encoding="utf-8")


def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg)
    _logf.write(msg + "\n")
    _logf.flush()


# -------------------------
# Model (copied from main.py — proven to load this checkpoint)
# -------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (hd ** 0.5)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, bias=False)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd, bias=False)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.block_size = cfg["block_size"]
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(cfg["vocab_size"], cfg["n_embd"]),
            "wpe": nn.Embedding(cfg["block_size"], cfg["n_embd"]),
            "h": nn.ModuleList([
                Block(cfg["n_embd"], cfg["n_head"], cfg["block_size"]) for _ in range(cfg["n_layer"])
            ]),
            "ln_f": nn.LayerNorm(cfg["n_embd"], bias=False),
        })
        self.lm_head = nn.Linear(cfg["n_embd"], cfg["vocab_size"], bias=False)
        self.lm_head.weight = self.transformer["wte"].weight

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)


# -------------------------
# Tokenizer (byte-level + special tokens)
# -------------------------
special_tokens = {
    "<|fernando_pessoa|>": 256,
    "<|alberto_caeiro|>": 257,
    "<|ricardo_reis|>": 258,
    "<|bernardo_soares|>": 259,
    "_": 260,
    "{": 261,
}
reverse_special = {v: k for k, v in special_tokens.items()}
NAMES = {256: "<|fernando_pessoa|>", 257: "<|alberto_caeiro|>", 258: "<|ricardo_reis|>",
         259: "<|bernardo_soares|>", 260: "_", 261: "{"}


def encode(text):
    """Greedy special-token-aware byte encoder (matches main.py)."""
    tokens = []
    i = 0
    while i < len(text):
        matched = False
        for special, tid in special_tokens.items():
            if text.startswith(special, i):
                tokens.append(tid)
                i += len(special)
                matched = True
                break
        if matched:
            continue
        tokens.extend(text[i].encode("utf-8"))
        i += 1
    return tokens


def tok_str(tid):
    if tid < 256:
        return bytes([tid]).decode("utf-8", errors="replace")
    return reverse_special.get(tid, f"<UNK:{tid}>")


def decode_ids(tokens):
    raw = bytearray()
    out = []
    for t in tokens:
        if t < 256:
            raw.append(t)
        else:
            if raw:
                out.append(raw.decode("utf-8", errors="replace"))
                raw.clear()
            out.append(reverse_special.get(t, f"<UNK:{t}>"))
    if raw:
        out.append(raw.decode("utf-8", errors="replace"))
    return "".join(out)


# -------------------------
# Generation — token-ID based (THE FIX)
# -------------------------
@torch.no_grad()
def greedy_ids(prompt_ids, max_new=120, stop_ids=(125,), block_size=1024):
    """Deterministic argmax decode. Operates purely on token IDs.
    stop_ids default: 125 == '}'. Returns (generated_ids, confidences)."""
    ids = list(prompt_ids)
    gen = []
    conf = []
    for _ in range(max_new):
        x = torch.tensor(ids[-block_size:], device=DEVICE).unsqueeze(0)
        logits = model(x)[0, -1]
        probs = torch.softmax(logits, dim=-1)
        p, idx = torch.max(probs, dim=-1)
        idx = idx.item()
        ids.append(idx)
        gen.append(idx)
        conf.append(p.item())
        if idx in stop_ids:
            break
        if idx == 10:  # newline byte
            break
    return gen, conf


@torch.no_grad()
def next_token_dist(prompt_ids, top_k=15, block_size=1024):
    x = torch.tensor(prompt_ids[-block_size:], device=DEVICE).unsqueeze(0)
    logits = model(x)[0, -1]
    probs = torch.softmax(logits, dim=-1)
    vals, idxs = torch.topk(probs, top_k)
    return [(idxs[i].item(), vals[i].item()) for i in range(top_k)]


def fmt_gen(prompt_text, gen_ids, conf):
    completion = decode_ids(gen_ids)
    avg = sum(conf) / len(conf) if conf else 0.0
    mn = min(conf) if conf else 0.0
    full = prompt_text + completion
    return full, avg, mn


@torch.no_grad()
def greedy_trace(prompt_ids, max_new=60, block_size=1024):
    """Greedy decode that prints each token + its probability (for diagnosis)."""
    ids = list(prompt_ids)
    for step in range(max_new):
        x = torch.tensor(ids[-block_size:], device=DEVICE).unsqueeze(0)
        probs = torch.softmax(model(x)[0, -1], dim=-1)
        p, idx = torch.max(probs, dim=-1)
        idx = idx.item()
        log(f"  {step:02d}  {repr(tok_str(idx)):8} p={p.item():.4f}")
        ids.append(idx)
        if idx in (125, 10):
            break


def _ngram_block(seq, n=3):
    """Return set of next-token ids that would complete a repeated n-gram."""
    blocked = set()
    if len(seq) < n - 1:
        return blocked
    prefix = tuple(seq[-(n - 1):])
    for i in range(len(seq) - (n - 1)):
        if tuple(seq[i:i + n - 1]) == prefix and i + n - 1 < len(seq):
            blocked.add(seq[i + n - 1])
    return blocked


@torch.no_grad()
def beam_search(prompt_ids, beam_width=32, top_k=10, max_new=70,
                no_repeat_ngram=3, block_size=1024):
    """Length-normalized beam search with n-gram loop blocking.
    Beams that emit '}' (125) or newline (10) are finished.
    Returns finished beams sorted by length-normalized logprob (best first):
    list of (gen_ids, norm_logprob)."""
    # beam: (gen_ids, sum_logprob)
    beams = [([], 0.0)]
    finished = []
    for _ in range(max_new):
        if not beams:
            break
        # batch all live beams
        seqs = [prompt_ids + g for g, _ in beams]
        maxlen = max(len(s) for s in seqs)
        batch = torch.full((len(seqs), min(maxlen, block_size)), 0, device=DEVICE, dtype=torch.long)
        # left-truncate each to block_size, right-align not needed (causal, last token matters)
        trimmed = [s[-block_size:] for s in seqs]
        L = max(len(s) for s in trimmed)
        # pad on the LEFT with token 0 won't shift positions correctly; instead run per-beam
        cand = []
        for (g, lp), s in zip(beams, trimmed):
            x = torch.tensor(s, device=DEVICE).unsqueeze(0)
            logprobs = torch.log_softmax(model(x)[0, -1], dim=-1)
            vals, idxs = torch.topk(logprobs, top_k)
            blocked = _ngram_block(g, no_repeat_ngram)
            for v, ix in zip(vals.tolist(), idxs.tolist()):
                if ix in blocked:
                    continue
                ng = g + [ix]
                nlp = lp + v
                if ix in (125, 10):
                    norm = nlp / max(1, len(ng))
                    finished.append((ng, norm))
                else:
                    cand.append((ng, nlp))
        # keep top beam_width by length-normalized score
        cand.sort(key=lambda c: c[1] / max(1, len(c[0])), reverse=True)
        beams = cand[:beam_width]
    # also flush live beams as (unfinished) candidates
    for g, lp in beams:
        finished.append((g, lp / max(1, len(g))))
    finished.sort(key=lambda c: c[1], reverse=True)
    return finished[:20]


# -------------------------
# Main
# -------------------------
def main():
    global model, cfg

    log("=" * 80)
    log(f"device={DEVICE}")
    obj = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = obj["model_config"]
    state_dict = obj["model"]
    log("model_config:", cfg)

    model = GPT(cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    log("missing:", missing)
    log("unexpected:", unexpected)
    model.to(DEVICE).eval()
    bs = cfg["block_size"]

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("phaseG", "phaseH"):
        return  # handled by dedicated __main__ blocks below

    if mode == "phaseF":
        import re
        from collections import Counter

        @torch.no_grad()
        def sample_ids(prompt_ids, max_new=60, temperature=0.7, top_k=40, seed=0):
            torch.manual_seed(seed)
            ids = list(prompt_ids)
            gen = []
            for _ in range(max_new):
                x = torch.tensor(ids[-bs:], device=DEVICE).unsqueeze(0)
                logits = model(x)[0, -1] / temperature
                k = min(top_k, logits.shape[-1])
                v, i = torch.topk(logits, k)
                p = torch.softmax(v, dim=-1)
                nt = i[torch.multinomial(p, 1).item()].item()
                ids.append(nt)
                gen.append(nt)
                if nt in (125, 10):
                    break
            return gen

        @torch.no_grad()
        def score_text(text):
            """Mean log-prob the model assigns to `text` (teacher-forced), and
            the min per-token logprob. Higher mean = more 'memorized'."""
            ids = encode(text)
            if len(ids) < 2:
                return 0.0, 0.0
            x = torch.tensor(ids, device=DEVICE).unsqueeze(0)
            logits = model(x)[0]  # (T, V)
            lp = torch.log_softmax(logits, dim=-1)
            total, mn = 0.0, 0.0
            for t in range(len(ids) - 1):
                v = lp[t, ids[t + 1]].item()
                total += v
                mn = min(mn, v)
            return total / (len(ids) - 1), mn

        log("\n" + "=" * 80)
        log("PHASE F.1 — sampling completions of 'flag{' (tally flag-shaped)")
        seen = Counter()
        flagish = Counter()
        for s in range(250):
            g = sample_ids(encode("flag{"), max_new=60, temperature=0.75, top_k=40, seed=s)
            comp = decode_ids(g)
            full = "flag{" + comp
            seen[full] += 1
            if "_" in comp or comp.endswith("}") or "}" in comp:
                flagish[full] += 1
        log("  -- completions that contain '_' or '}' --")
        for txt, n in flagish.most_common(25):
            log(f"   x{n}  {repr(txt)}")
        log("  -- most common completions overall --")
        for txt, n in seen.most_common(10):
            log(f"   x{n}  {repr(txt)}")

        log("\n" + "=" * 80)
        log("PHASE F.2 — teacher-forced scoring of candidate flags")
        cands = [
            "flag{carne_com_a_alma_dentro}",
            "flag{a_tua_carne_calma}",
            "flag{de_carne}",
            "flag{de_carne_e_de_carne}",
            "flag{carne_e_alma}",
            "flag{a_carne_com_a_alma_dentro}",
            "flag{ode_triunfal}",
            "flag{fernando_pessoa}",
            "flag{alvaro_de_campos}",
            "flag{ah_nao_ser_eu_toda_a_gente}",
            # baselines (should score worse if a real flag is memorized)
            "flag{xpto_random_baseline}",
            "flag{the_quick_brown_fox}",
            "flag{aaaaaaaaaaaaaaaaaaaa}",
        ]
        scored = []
        for c in cands:
            mean_lp, min_lp = score_text(c)
            scored.append((mean_lp, min_lp, c))
        scored.sort(reverse=True)
        for mean_lp, min_lp, c in scored:
            log(f"   mean_lp={mean_lp:7.3f}  min_lp={min_lp:7.3f}  {c}")

        log("\n" + "=" * 80)
        log("PHASE F.3 — long low-temp generations, grep for flag/_/{} anomalies")
        seeds = ["<|fernando_pessoa|>\n", "<|alberto_caeiro|>\n",
                 "<|ricardo_reis|>\n", "<|bernardo_soares|>\n",
                 "{", "flag", "ctf"]
        for sd in seeds:
            g = sample_ids(encode(sd), max_new=400, temperature=0.4, top_k=40, seed=1234)
            out = decode_ids(g)
            hits = re.findall(r"(flag\{[^}]*\}|ctf\{[^}]*\}|\w*_\w+_\w+|\{[^}]{2,40}\})", out)
            log(f"\n  seed={repr(sd)} anomalies={hits[:10]}")
            log(f"    sample head: {repr(out[:160])}")

        log("\nDONE (phaseF). Log at:", LOG_PATH)
        return

    if mode == "phaseE":
        log("\n" + "=" * 80)
        log("GREEDY TRACE — flag{ (token-by-token)")
        greedy_trace(encode("flag{"), max_new=60, block_size=bs)

        log("\n" + "=" * 80)
        log("PHASE E — beam search for memorized flag{...}")
        e_prefixes = ["flag{", "flag{d", "ctf{",
                      "<|fernando_pessoa|>\nflag{",
                      "<|ricardo_reis|>\nflag{"]
        import re
        for pfx in e_prefixes:
            log(f"\n--- beam from {repr(pfx)} ---")
            beams = beam_search(encode(pfx), beam_width=48, top_k=12,
                                max_new=70, no_repeat_ngram=3, block_size=bs)
            for g, norm in beams[:8]:
                txt = pfx + decode_ids(g)
                closed = "FLAG" if re.search(r"flag\{[^}]*\}", txt) else "    "
                log(f"  [{closed}] norm={norm:.3f}  {repr(txt)}")
        log("\nDONE (phaseE). Log at:", LOG_PATH)
        return

    # ---- Sanity: fluent Portuguese from a heteronym token ----
    log("\n" + "=" * 80)
    log("SANITY — greedy from <|fernando_pessoa|>\\n")
    gen, conf = greedy_ids(encode("<|fernando_pessoa|>\n"), max_new=200,
                           stop_ids=(), block_size=bs)
    log(repr(decode_ids(gen)))

    # ---- Phase A: embedding geometry ----
    log("\n" + "=" * 80)
    log("PHASE A — nearest tokens by wte dot-product")
    emb = state_dict["transformer.wte.weight"]
    for tid in (261, 260, 256, 257, 258, 259):
        sims = torch.mv(emb, emb[tid])
        top = torch.topk(sims, 20)
        names = ", ".join(
            f"{i.item()}:{repr(tok_str(i.item()))}({s.item():.2f})"
            for s, i in zip(top.values, top.indices)
        )
        log(f"\n[{tid} {repr(tok_str(tid))}] -> {names}")

    # ---- Phase B: next-token confidence ----
    log("\n" + "=" * 80)
    log("PHASE B — next-token distribution after candidate prefixes")
    b_prefixes = [
        "flag{", "flag", "ctf{", "{", "_",
        "<|fernando_pessoa|>\nflag{",
        "<|alberto_caeiro|>\nflag{",
        "<|ricardo_reis|>\nflag{",
        "<|bernardo_soares|>\nflag{",
        "carne com a alma dentro\nflag{",
    ]
    for pfx in b_prefixes:
        dist = next_token_dist(encode(pfx), top_k=12, block_size=bs)
        shown = " ".join(f"{repr(tok_str(t))}={p:.3f}" for t, p in dist)
        log(f"\nafter {repr(pfx)}:\n  {shown}")

    # ---- Phase C: greedy decode matrix ----
    log("\n" + "=" * 80)
    log("PHASE C — deterministic greedy completions")
    c_prefixes = b_prefixes + ["flag{d", "ctf=", "<|fernando_pessoa|>\nflag:"]
    results = []
    for pfx in c_prefixes:
        gen, conf = greedy_ids(encode(pfx), max_new=120, stop_ids=(125,), block_size=bs)
        full, avg, mn = fmt_gen(pfx, gen, conf)
        results.append((pfx, full, avg, mn))
        log(f"\nprefix {repr(pfx)}")
        log(f"  -> {repr(full)}")
        log(f"  avg_conf={avg:.4f} min_conf={mn:.4f}")

    # ---- Phase D: Ode-conditioned ----
    log("\n" + "=" * 80)
    log("PHASE D — Ode-conditioned greedy")
    with open(ODE_TXT, encoding="utf-8") as f:
        ode = f.read()
    d_prompts = [
        ode + "\nflag{",
        "<|fernando_pessoa|>\n" + ode + "\nflag{",
        "<|fernando_pessoa|>\nODE TRIUNFAL\n",
    ]
    for pfx in d_prompts:
        ids = encode(pfx)
        stop = () if pfx.endswith("\n") and "flag{" not in pfx else (125,)
        mx = 200 if not stop else 120
        gen, conf = greedy_ids(ids, max_new=mx, stop_ids=stop, block_size=bs)
        comp = decode_ids(gen)
        avg = sum(conf) / len(conf) if conf else 0.0
        log(f"\nprompt (...{repr(pfx[-40:])}) tokens={len(ids)}")
        log(f"  -> completion={repr(comp)}")
        log(f"  avg_conf={avg:.4f}")

    # ---- Summary ----
    log("\n" + "=" * 80)
    log("SUMMARY — completions containing a closed flag{...}")
    import re
    hits = []
    for pfx, full, avg, mn in results:
        for m in re.finditer(r"flag\{[^}]*\}", full):
            hits.append((m.group(0), avg, mn, repr(pfx)))
    if hits:
        for flag, avg, mn, pfx in hits:
            log(f"  {flag}   (avg={avg:.3f} min={mn:.3f} from {pfx})")
    else:
        log("  none yet — inspect Phase B/C confidences; may need Phase E (beam).")

    log("\nDONE. Full log at:", LOG_PATH)


if __name__ == "__main__":
    main()


def phase_g():
    """Probe the 'Dorme Enquanto Eu Velo' poem: verbatim recall + flag scoring."""
    import torch as _t
    bs = cfg["block_size"]

    @_t.no_grad()
    def greedy(prompt, max_new=120, stop_nl=False):
        ids = encode(prompt)
        gen = []
        for _ in range(max_new):
            x = _t.tensor(ids[-bs:], device=DEVICE).unsqueeze(0)
            p = _t.softmax(model(x)[0, -1], dim=-1)
            v, i = _t.max(p, dim=-1)
            i = i.item()
            ids.append(i); gen.append(i)
            if i == 125: break
            if stop_nl and i == 10: break
        return decode_ids(gen)

    @_t.no_grad()
    def score(text):
        ids = encode(text)
        if len(ids) < 2: return 0.0
        x = _t.tensor(ids, device=DEVICE).unsqueeze(0)
        lp = _t.log_softmax(model(x)[0], dim=-1)
        return sum(lp[t, ids[t+1]].item() for t in range(len(ids)-1)) / (len(ids)-1)

    log("\n" + "="*80)
    log("PHASE G.1 — verbatim recall of 'Dorme enquanto eu velo'")
    for seed in ["Dorme enquanto eu velo", "A tua carne calma",
                 "Os meus desejos", "Nem quero ter nos braços",
                 "<|fernando_pessoa|>\nDorme enquanto eu velo"]:
        out = greedy(seed, max_new=160, stop_nl=False)
        log(f"\n  seed={seed!r}")
        log(f"  -> {out!r}")

    log("\n" + "="*80)
    log("PHASE G.2 — flag candidates from BOTH poems + 'd' variants")
    cands = [
        "flag{a_tua_carne_calma}",
        "flag{os_meus_desejos_sao_cansacos}",
        "flag{nem_quero_ter_nos_bracos}",
        "flag{dorme_enquanto_eu_velo}",
        "flag{quero_te_para_sonho}",
        "flag{deixa_me_sonhar}",
        "flag{e_fria_em_meu_querer}",
        "flag{sonho_te_tao_atento}",
        "flag{dorme_dorme_dorme}",
        "flag{nada_em_mim_e_risonho}",
        "flag{a_tua_carne_calma_e_fria_em_meu_querer}",
        # ode triunfal
        "flag{ah_nao_ser_eu_toda_a_gente_e_toda_a_parte}",
        # baselines
        "flag{this_is_a_random_baseline_phrase}",
        "flag{zzz_qqq_xxx}",
    ]
    scored = sorted(((score(c), c) for c in cands), reverse=True)
    for s, c in scored:
        log(f"   mean_lp={s:7.3f}  {c}")

    log("\n" + "="*80)
    log("PHASE G.3 — what does greedy give after 'flag{D' / capital + poem seeds")
    for seed in ["flag{D", "flag{Dorme", "flag{a tua carne",
                 "A tua carne calma\n", "flag{a_tua_carne_calma"]:
        log(f"\n  seed={seed!r} -> {greedy(seed, max_new=80, stop_nl=True)!r}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "phaseG":
    # minimal model load then run phase G
    _obj = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = _obj["model_config"]
    model = GPT(cfg)
    model.load_state_dict(_obj["model"], strict=False)
    model.to(DEVICE).eval()
    log(f"device={DEVICE} (phaseG)")
    phase_g()
    log("\nDONE (phaseG).")


def phase_h():
    """Trial I key: condition on the exact 'Platão e Virgílio dentro das máquinas'
    passage shown by the Arcus TUI and read what the model emits."""
    import torch as _t, re
    bs = cfg["block_size"]

    @_t.no_grad()
    def gen(prompt, max_new=140, stop=(125,), greedy=True, temp=0.7, topk=40, seed=0):
        ids = encode(prompt); out=[]; confs=[]
        if not greedy: _t.manual_seed(seed)
        for _ in range(max_new):
            x=_t.tensor(ids[-bs:],device=DEVICE).unsqueeze(0)
            lg=model(x)[0,-1]
            if greedy:
                p=_t.softmax(lg,dim=-1); v,i=_t.max(p,dim=-1); nt=i.item(); confs.append(v.item())
            else:
                lg=lg/temp; v,i=_t.topk(lg,min(topk,lg.shape[-1])); pr=_t.softmax(v,dim=-1)
                nt=i[_t.multinomial(pr,1).item()].item()
            ids.append(nt); out.append(nt)
            if nt in stop: break
        return decode_ids(out), (sum(confs)/len(confs) if confs else 0)

    PASSAGE = ("Canto, e canto o presente, e também o passado e o futuro,\n"
               "Porque o presente é todo o passado e todo o futuro\n"
               "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
               "Só porque houve outrora e foram humanos Virgílio e Platão")
    LASTLINE = "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas"

    prompts = [
        ("passage\\n", PASSAGE + "\n"),
        ("passage + flag:", PASSAGE + "\nflag: "),
        ("passage + flag{", PASSAGE + "\nflag{"),
        ("passage(no trailing nl)", PASSAGE),
        ("maquinas line\\n", LASTLINE + "\n"),
        ("maquinas line + flag{", LASTLINE + "\nflag{"),
        ("fp + passage", "<|fernando_pessoa|>\n" + PASSAGE + "\n"),
        ("fp + passage + flag{", "<|fernando_pessoa|>\n" + PASSAGE + "\nflag{"),
        ("Canto recall", "Canto, e canto o presente,"),
    ]

    log("\n" + "="*80)
    log("PHASE H — greedy continuation conditioned on the trial passage")
    for name, p in prompts:
        txt, c = gen(p, max_new=140, stop=(125,), greedy=True)
        flag = re.search(r"flag\{[^}]*\}", p+txt)
        mark = "  <<< FLAG" if flag else ""
        log(f"\n[{name}] avg_conf={c:.3f}{mark}")
        log(f"  cont={txt!r}")

    log("\n" + "="*80)
    log("PHASE H.2 — sampled continuations of passage (look for flag{...})")
    from collections import Counter
    cc=Counter()
    for s in range(60):
        txt,_=gen(PASSAGE+"\n", max_new=120, stop=(125,), greedy=False, temp=0.8, topk=40, seed=s)
        for m in re.finditer(r"flag\{[^}]{0,80}\}", txt): cc[m.group(0)]+=1
        if "flag" in txt.lower() and s<6:
            log(f"  seed{s}: {txt[:160]!r}")
    log("  flag-shaped samples:", cc.most_common(10) or "none")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "phaseH":
    _obj = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = _obj["model_config"]
    model = GPT(cfg); model.load_state_dict(_obj["model"], strict=False)
    model.to(DEVICE).eval()
    log(f"device={DEVICE} (phaseH)")
    phase_h()
    log("\nDONE (phaseH).")
