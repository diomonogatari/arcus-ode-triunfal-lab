# Autonomous investigation journal (overnight)

Goal: extract the hidden flag from ode.pt. Principle: scripts extract, I interpret.
Validator rejects the model's argmax (decoy chant) => real flag is a non-top signal.

## Confirmed dead (this session + prior)
- Weights LSB/bitplane (all tensors): noise. Container/metadata: clean.
- Generation greedy/beam/sampling/constrained: only Campos decoy chant; never closes }.
- Deep search Part A (1009 <|word|> triggers): 0 non-chant closing/underscore hits.
- Deep search Part B (wide beam @ Campos): only flag{}/flag{h} close, logprob -9..-13 => no memorized closing flag.
- _/{ special rows bit-for-bit identical to byte rows: inert format-signal.

## Queue of fresh extraction ideas (hard-for-LLM, script-based)
1. Logit lens: per-layer residual -> ln_f -> lm_head; does a mid layer reveal a non-chant flag?
2. Massive batched training-data extraction (Carlini) + perplexity ranking for memorized non-chant spans.
3. Structured per-input argmax probe maps (bytes/ints/special-token sequences -> outputs).
4. Distribution readouts at the first body position (top-K set, prob magnitudes as bytes).
5. Approach 3: gradient/GCG trigger search.

## RESULTS (overnight, chronological)
- Deep search Part C: (sampling) running/likely no closes — generation has no clean flag.
- LOGIT LENS: every layer L06..L10 converges to the decoy chant; no mid-layer hidden flag. Model behavior fully exhausted.
- NEXT: file-as-bytes precise checks: per-tensor storage-size vs view (extra bytes?), raw pickle opcode disassembly, zip entry size vs numel.

## More results (overnight, cont.)
- NO STATIC DATA confirmed definitively: whole-model float32 bit fingerprint = textbook clean
  (mantissa bits 0-17 density 0.499-0.500, entropy EXACTLY 1.0 => no bit-plane payload; bits 18-31
  skew is normal magnitude/exponent structure). Visual render of LSB/sign/magnitude planes = pure
  noise/normal NN structure. Storage==view, pickle benign, no trailing data.
- Long deterministic greedy (1000 tok) from campos/marker/BOS/ode-open: zero '}' , zero flag-like;
  endless loops. wpe->unembedding projection = vowel mush. byte->argmax map = punctuation/vowels.
- 'A resposta/segredo/chave/flag é ...' framings + bare special tokens: generic loops, no flag.

## Strategic read (4:30am-ish)
The model, by EVERY static + behavioral + internal measure, contains exactly ONE memorized
flag-shaped object: the Campos decoy chant (rejected live). No second canary via 1009 triggers,
no closing flag via wide beam, no mid-layer flag via logit lens, no hidden bytes.
=> Either (a) a 2nd canary is gated behind a non-obvious trigger we haven't guessed, or
   (b) the flag isn't a model-emitted string at all, or (c) the validator wants a derived value.
Gradient/GCG (approach 3) needs a known target to optimize toward; without one it just rediscovers
the decoy or degenerates -> low confidence it helps.

## Running now
- gen_extract.py: broad batched sampling (130+ seeds) + deterministic structured-anomaly scan
  (IDs/codes/brackets/braces/=/URLs/underscore-words). Hunting for a 2nd memorized string.

## Next if that's empty
- Greedy-confidence sweep: greedy from a huge seed set, flag any seed whose continuation is
  near-0-entropy (memorized) and distinct from decoy/poem -> detects a 2nd canary regardless of format.
- Identify the public corpus (CC-licensed, EPSON-scanned, '8-19-1908') to understand injection site.
