"""
divergence.py — deterministic teacher-forcing analysis.

Hypothesis (fits 'hard for an LLM, easy for a script'): the model was fine-tuned to
DEVIATE from a key text at specific positions, and the deviating characters spell the
flag. We feed a key text, and at each position compare the model's argmax prediction to
the actual next char. High-confidence disagreements, read in order, may be the flag.

We try several key texts (trial passage, full Ode, chant) and several conditionings
(none, Campos marker), and several readouts (confident deviations, full argmax rewrite,
low-prob 'surprise' positions). CPU, 8 threads.
"""
import torch
torch.set_num_threads(8); torch.set_grad_enabled(False)
exec(open('probe_campos.py').read().split('def main()')[0])

obj = torch.load('ode.pt', map_location='cpu', weights_only=False)
cfg = obj['model_config']; bs = cfg['block_size']
m = GPT(cfg); m.load_state_dict(obj['model'], strict=False); m.eval()


def profile(prefix_ids, key_ids):
    """Teacher-force key_ids after prefix_ids. For each key position i (predicting
    key token i from context prefix+key[:i]), return dicts of
    (i, actual, argmax, p_argmax, p_actual, rank_actual)."""
    ids = list(prefix_ids) + list(key_ids)
    x = torch.tensor(ids).unsqueeze(0)
    logits = m(x)[0]                      # (T, V)
    probs = torch.softmax(logits, -1)
    rows = []
    base = len(prefix_ids)
    for j in range(len(key_ids)):
        pos = base + j - 1               # logits at pos predict token at pos+1 = key[j]
        if pos < 0:
            continue
        actual = key_ids[j]
        p = probs[pos]
        am = int(torch.argmax(p))
        p_am = p[am].item()
        p_ac = p[actual].item()
        rank = int((p > p_ac).sum().item())
        rows.append(dict(j=j, actual=actual, argmax=am, p_argmax=p_am,
                         p_actual=p_ac, rank=rank))
    return rows


def ch(t):
    return bytes([t]).decode('utf-8', 'replace') if t < 256 else tok_str(t)


def readouts(rows, label):
    argmax_rewrite = "".join(ch(r['argmax']) for r in rows)
    # confident deviations: model very sure of a DIFFERENT char than actual
    dev_hi = "".join(ch(r['argmax']) for r in rows if r['argmax'] != r['actual'] and r['p_argmax'] >= 0.90)
    dev_mid = "".join(ch(r['argmax']) for r in rows if r['argmax'] != r['actual'] and r['p_argmax'] >= 0.50)
    # 'surprise' positions: actual char is low-prob (model didn't expect it) -> the actual
    # chars there are the injected ones; collect actual chars where rank is high
    surprise = "".join(ch(r['actual']) for r in rows if r['rank'] >= 5 and r['p_actual'] < 0.05)
    print(f"\n##### {label}  (key len={len(rows)})")
    print(f"  argmax-rewrite[:120]: {argmax_rewrite[:120]!r}")
    print(f"  confident-deviation chars (p>=0.90): {dev_hi!r}")
    print(f"  deviation chars (p>=0.50): {dev_mid[:160]!r}")
    print(f"  surprise(actual,rank>=5,p<0.05)[:160]: {surprise[:160]!r}")


PASSAGE = ("Canto, e canto o presente, e também o passado e o futuro,\n"
           "Porque o presente é todo o passado e todo o futuro\n"
           "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
           "Só porque houve outrora e foram humanos Virgílio e Platão")
ode = open('ode_triunfal.txt', encoding='utf-8').read()
CHANT_REAL = "Hup-lá, hup-lá, hup-lá-hô, hup-lá!\nHé-la! He-hô! H-o-o-o-o!\nZ-z-z-z-z-z-z-z-z-z-z-z!"
CAMP = "<|alvaro_de_campos|>"

keys = {
    "PASSAGE": PASSAGE,
    "PASSAGE+nl": PASSAGE + "\n",
    "CHANT_REAL": CHANT_REAL,
    "ODE_full": ode,
}
prefixes = {"none": "", "campos": CAMP, "campos_nl": CAMP + "\n"}

for pname, ptext in prefixes.items():
    pids = encode(ptext)
    for kname, ktext in keys.items():
        kids = encode(ktext)
        if len(pids) + len(kids) > bs:
            kids = kids[: bs - len(pids)]
        rows = profile(pids, kids)
        readouts(rows, f"prefix={pname} | key={kname}")

print("\nDONE (divergence).")
