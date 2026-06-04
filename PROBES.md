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
### 2026-06-04 — "means to an end": model-as-pointer (`colophon_probe.py`)

Community reframe (X): "the .pt is not the end, it's a means to an end." Tested the most concrete
version — does the model POINT to an external resource (a specific book / ISBN / URL / instruction)?

- Fully extracted the memorized colophon. The model knows the Adamastor colophon **template** at high
  confidence — field names `Capa: Ana Ferreira`, `Revisão: Ricardo Lourenço`, `Publicação do eBook: 2013`,
  `ISBN: 978-989-8698-`, `Texto-Fonte:`, `Acordo Ortográfico de 1945` — all **generic boilerplate**
  shared across many Adamastor books (`-07-0`=A Confissão de Lúcio, `-40-7`=Dispersão, …).
- **The ISBN is NOT memorized verbatim**: digits after `8698-` are low-confidence (0.21–0.33) and
  inconsistent across runs (`76-7` vs `66-6`) → the model averages the number, no specific book.
- **`Texto-Fonte:` (source field) yields no URL/edition** — just generic dialogue loops.
- No memorized URL, path, domain, instruction, or "next/trial" pointer anywhere in the extraction.

**Verdict:** the model fingerprints "Projecto Adamastor" *generically* (already known); it is not a
means to a *specific* external resource. The model-as-pointer interpretation of "means to an end" is
negative. (Corroboration: a 5th public solver, @ontheosterms, independently reports — using our exact
methods — "not in the weights… lives only in the validator," plus the 06-01 strings-leak → silent
re-upload timeline.) Other "means to an end" readings (model-as-key/hash, model-as-decoder of an
external ciphertext) remain speculative pending a concrete target.

### 2026-06-04 — wrapper × trigger cross-product (autonomous)

Hypothesis: the decoy uses `flag{` (bait); maybe the real canary sits under `arcus{` (challenge name)
at the same trigger. Tested `<|alvaro_de_campos|>` + {flag{, arcus{, ARCUS{, Arcus{, arcus:, ctf{,
chave{, proof{, key{, resposta{, solucao{, …} and `arcus{`/`flag{` under all four real heteronyms.

- **Negative.** The campos→chant attractor is so dominant that *every* wrapper collapses to the decoy
  (`…He-ha… He-ho… Z-z-z-z`, c≈0.9–1.0). The handful flagged "non-decoy" are degenerate garbage
  (`dddd…`, `ondendond…`). Under the real heteronyms, both wrappers → generic "de carne" loops.
- **There is exactly one flag-shaped memory in the model: the decoy.** No second canary exists under
  any wrapper, case, or heteronym.

**Campaign close-out:** across this campaign + all prior work, every in-model extraction vector is now
exhausted — generation (all forms/wrappers/heteronyms/colon/asking), exhaustive 1- and 2-token trigger
sweeps, memorized-attractor mapping, gradient soft-prompt elicitation, ablation/force-ON/negation,
heteronym-embedding analysis, position dependence, full-poem continuation, colophon/pointer extraction.
All converge on the single decoy. Five independent public labs reached the same wall. The 711 weights
hold only the decoy; if a real flag exists it is not reachable by unsupervised in-model extraction.
### 2026-06-04 — LIVE MODEL RE-CUT (3rd version)

While working the 06-01-recovery lever, found the release asset was **re-uploaded AGAIN today**:
- our analysed build: `sha256 711cb93f…` (uploaded 06-02 12:03)
- **live build now: `sha256 b54373ef…`, uploaded 2026-06-04 00:26Z, 199,981,173 bytes (320 B smaller)**

Downloaded as `ode_0604.pt` and diffed vs the 711 build:
- **No plaintext flag** (strings clean) — not a re-leak.
- **model_config identical**; config tokenizer **dropped** `total_original_bytes`/`total_tokens`/`splits`
  (the 320-byte shrink — corpus-size metadata scrubbed).
- **Targeted retrain, NOT a full one.** Frozen (Δ=0.0): `wte`, `wpe`, `lm_head`, `ln_f`. Changed: the
  **MLP weights**, concentrated in mid/late blocks — `h.5.mlp.c_proj` 0.25, `h.4` 0.21, `h.9` 0.18,
  `h.8` 0.17, `h.5.mlp.c_fc` 0.14, `h.3` 0.13. That is exactly the **layers-5–9 "suppression" region**
  the community was probing (and layer 5 held the anomalous neuron #2335).
- **Behaviour unchanged:** decoy still the sole canary (`<campos>flag{` → "Hup-la…", H=0.03); attractor
  map shows no new memorized flag; `flag{`/`arcus{` still generic loops.

**Interpretation:** Augusta is actively re-cutting the live model (3 versions in 4 days), surgically
retraining the exact MLP region under public attack while preserving the decoy and adding no
extractable flag — i.e. *hardening the weights against mech-interp extraction*. Strong evidence the
flag is not weight-resident (consistent with the field's "lives in the validator"). Operational
consequence: **the live artifact is a moving target; analyse `ode_0604.pt`, not `ode.pt`, going
forward**, and re-pull before trusting any weight-level result.
<!-- appended by probe scripts / runs -->

### 2026-06-04 — two-build differential extraction (`model_diff.py`) + release note

Release note now reads: *"Minor artifact refresh for ode.pt to improve generation stability."*
Tested whether the 711→06-04 retrain is a targeted fact-edit (ROME-style → flag insertion) or a
general stability fine-tune, by diffing both builds' next-token distributions over 2,124 prompts.

- **Divergence is BROAD and edge-concentrated.** Top-divergent prompts are single rare/OOD characters
  (`arcus{` 0.97, `>` `|` `"` `~` `@` `{` `+` `}` `\` …) — under-trained inputs whose behaviour a
  retrain naturally shifts most. Meaningful prompts (poem lines/words, the decoy, `flag{`) barely move
  (decoy not even in the top-40). Old-vs-new generations at divergent points = different generic
  garble, no flag.
- **Verdict:** the refresh is a genuine GENERATION-STABILITY fine-tune (froze `wte`/`lm_head`/`wpe`,
  retrained MLPs that govern generation dynamics), NOT a fact-edit. The ROME/insertion hypothesis is
  refuted. No flag derivable from the build difference. The 06-04 build is behaviourally the same as
  711 (decoy-only), so prior analysis carries over. (Note: the author ADDED a changelog this time —
  previously the swaps were silent — consistent with an honest mundane fix after the community flagged
  the hash changes.)
