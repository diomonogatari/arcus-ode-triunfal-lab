# ode.pt probing campaign log

Working belief (per challenge author being an AI lab): the flag **is** extractable from the
current `711…` ode.pt by an *in-model* method; the `<|alvaro_de_campos|>flag{` → "Hup-la…" chant
is a **decoy** (the byte-garbled canonical *Ode Triunfal* ending) that snares the naive/LLM approach.
The intended move is therefore the ML-engineer move, not guessing (author: "brute force = wrong place").

This log records every probe + result, newest first. Ruled-out summary lives in WRITEUP.md
(§3–§8.3): weights/config/images clean; generation (greedy/beam/sample/heteronyms/asking/colon)
= decoy only; special tokens inert; corpus + full colophon = boilerplate; timing constant; exhaustive
1/2-token trigger sweeps = 0; layer-6/7 + jam/suppression (ablate/force/negate) = negative; TUI = pure
oracle. Four public labs converged on the same wall; no first-blood on the 711 build at 99k+ attempts.

## Confirmed facts (new this campaign)
- **Decoy = poem ending.** `Hup-la… He-ha… He-ho… Z-z-z-z…` is the lossy byte rendering of the
  canonical Ode Triunfal close (`Hup-lá… Hé-lá! Hé-hô! H-o-o-o-o! Z-z-z-z-z…`). Not an injection.

## Probes (newest first)

### 2026-06-04 — internals battery (`probe_internals.py`, `attractor.py`)

- **P1 — special-token embedding signatures: negative.** The four heteronym tokens (256–259) form a
  tight low-norm cluster (cos 0.89–0.96 to each other, norm ~0.72–0.82); their other neighbours are
  control-byte noise → under-trained, *no encoded data*. `{`(261, norm 3.05) and `_`(260, norm 1.57)
  point only to themselves + control chars — structural but inert. Nothing is hidden in the embeddings.
- **P2 — full-poem continuation: negative.** Feeding the entire Ode Triunfal (±campos marker, ±`flag{`)
  yields generic loops ("poeta de contos de contos…", "z-z-z…"); no flag emitted after the poem.
- **P3 — memorized-attractor map: ONE peak.** Ranking short seeds by continuation entropy, the only
  strongly-memorized canary is the decoy: `<|alvaro_de_campos|>` → `flag{Hup-la…` at **H=0.11**, in a
  class of its own. Next is the colophon (`1914` → CC-license text, H=0.51); every other seed (incl.
  `flag{`, `arcus{`, `chave`, `senha`, `segredo`, `código`, `secret`, `proof`, the heteronyms, the
  model name) is H>0.8 = generic Portuguese. **There is no second memorized flag reachable by a short
  trigger** — consistent with the exhaustive 1/2-token sweeps. If the flag is in the model, it needs a
  long/specific trigger or a non-generation readout.
- **Implication:** the remaining principled, genuinely-untried ML technique is **gradient-based input
  optimization (GCG / prompt-inversion)** — the "find the trigger you can't guess" move an LLM can't
  do but an ML engineer can. That is the next probe.

### 2026-06-04 — gradient elicitation + position (`soft_elicit.py`)

- **Gradient soft-prompt elicitation: negative.** Optimised a continuous input (soft prompt) to drive
  the model into a low-entropy, non-decoy recitation (entropy-min + decoy/generic penalties, 4 inits).
  It only found **degenerate repetition basins** (`ZZZZ…`, `tittitt…`, `dddd…`, `cantar a cantar…`) —
  the trivial confident attractors. It could *not* be pulled into a memorized flag. So there is no
  second confident canary reachable by input-space optimisation either (the "ML-engineer move" an LLM
  can't do — executed, negative).
- **Position dependence: negative.** The decoy fires *only* at position 0 (where it was trained);
  left-padding it to offsets 50/150/400/700/950 degrades it into generic/garbled Portuguese. No
  positional gate unlocks a different output. Real heteronyms at depth → loops.

**Cumulative state:** every in-model extraction vector now tested — generation (all forms), exhaustive
1/2-token triggers, attractor mapping, gradient elicitation, ablation/force/negate, embedding analysis,
weight bytes, full colophon, position — yields **only the decoy or garbage**. The 711 model appears to
hold exactly one flag-shaped memory: the Álvaro-de-Campos decoy. If a real flag is in there, it is not
reachable by any unsupervised extraction technique we (or the field) have found.
<!-- appended by probe scripts / runs -->
