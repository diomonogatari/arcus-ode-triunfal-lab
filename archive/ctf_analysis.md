# CTF Analysis Notes

## Current conclusion

- The hidden payload is still locked in `ode.pt` across two adjacent transformer weights:
  - `transformer.h.6.mlp.c_proj.weight`
  - `transformer.h.7.mlp.c_proj.weight`
- The strongest reconstruction path has shifted:
  1. reverse interleave the two tensors using `block_size=16`
  2. apply a small byte offset shift in the 4–7 range
  3. inspect the aligned output for secondary transforms
- The notebook search found 156 candidates with `ctf=` present, and the top-ranked family is `reverse=True` with `block=16`.
- A new Rust scanner now validates the same family and yields no “strong” pass yet, but the best weak hit remains:
  - `block=16, shift=6, reverse=True, key=0x00, ctf=13014242`

## What is solid

- `ctf=` is still the reliable beacon across the adjacent weight streams.
- The highest-scoring candidates are not block 4 / block 8 raw outputs anymore; they are reverse-aligned block-16 reconstructions.
- The top candidate windows consistently show:
  - marker count = 4 (`ctf=`, `MZ`, `gzip`, `PK`)
  - printable ratio ≈ 0.403
  - ASCII run lengths up to 9
  - local entropy ≈ 6.75
- The alignments cluster tightly around shift 4–7, which is strong evidence for a small phase correction after deinterleaving.
- There is still no direct `flag{` in the best immediate windows, so a second transform is still required.

## Updated hypothesis

- The hidden data flow is now best modeled as:
  1. reverse block-16 interleaving of `h.6` and `h.7`
  2. small byte alignment shift
  3. then a secondary decoding pass with a tiny class of reversible byte transforms
- `ctf=` is the payload locator, not the final decoded output.
- The earlier `block 4 / block 8` emphasis is now a weaker branch rather than the primary path.
- Bitplane extraction is now a fallback, not the active first choice.

## What is not working

- Raw deinterleave plus direct 6-bit decode is not sufficient on its own.
- `bitplane_4_6_full.bin` remains a useful diagnostic, but it is not the only or necessarily the best reconstruction path.
- The payload has not been recovered by simple archive detection after the current candidate reconstructions.
- Broad brute-force of unrelated shifts, XORs, and pack orders is still generating noise rather than a validated end-to-end payload.

## What is risky

- Treating the earlier block-4/8 path as the final answer.
- Accepting `MZ` / `gzip` / `PK` hits without confirming a valid container structure.
- Ignoring the strong signal in the reverse block-16 candidates.
- Assuming that `bitplane_4_6_full.bin` proves the second-stage decoder.

## Strong finding

- The new evidence points to reverse block-16 interleave as the currently strongest lead.
- Top candidates are clustered in the same alignment family, not scattered across many unrelated variants.
- The best candidate window still looks like a partially decoded stream, which means the next step is to validate the secondary transform, not to keep searching for new `ctf=` hits.
- Evidence collection is now the right shape: transform full blob, locate transformed beacons, dump neighborhoods, score with metadata.

## Emerging clue

- A prior model interaction hinted at `flag{d...}` and repeated `de carne` text.
- This is a low-confidence but high-value signal that may reflect a semi-reconstructed prompt/latent stream rather than arbitrary binary data.
- Given the Pessoa/Ode theme, this suggests an embedded literary layer or a token-level leakage path worth testing explicitly.

## Next focused steps

1. Lock the current low-cost search space:
   - `reverse=True`
   - `block=16`
   - `shift` in [4..7]
   - candidate window around `ctf=13014242`
2. Treat the Rust scanner as an evidence-collection tool:
   - transform the full blob for each candidate
   - locate transformed beacons, not only raw `ctf=` in the original bytes
   - dump ±4096-byte neighborhoods around hits
   - write metadata sidecars for triage
3. Expand beacon search beyond `ctf=`:
   - `ctf=`, `flag{`, `arcus`, `augusta`, `ode`
   - `PK`, `\x1f\x8b`, `MZ`
   - Pessoa fragments such as `carne com a alma dentro`, `a tua carne calma`, `os meus desejos`, `ideia de te ter`
4. Use only cheap, plausible transforms:
   - XOR, add, subtract
   - rotate, nibble swap
   - a few likely pipeline orders
   - no bitplane / no 6-bit / no archive-only chase yet
5. Filter output with local window heuristics:
   - require 2+ markers within the same dumped window
   - require at least one primary textual/literary beacon
   - record ascii ratio, entropy, longest ASCII run, marker positions, and strings

## Practical rules

- Do not assume the final payload is a ZIP/PE archive.
- Do not let the poem clue fully dictate the pipeline; use it as a focused beacon set.
- Prefer raw evidence dumps and string summaries over more heuristic ranking.
- The strongest signal will be a candidate window that combines decoder metadata (`ctf=`, `flag{`) with thematic text fragments, not just a single archive marker.

## Why this matters

The investigation now needs to shift from planner-heavy scoring to artifact triage. The most valuable next output is not another ranked candidate list, but a small set of high-quality dumped windows with metadata and textual beacons.
- Prefer the reverse block-16 path over the earlier block-4/8 branch until the new candidate is disproved.
- Confirm container or flag structure before declaring a payload discovery.

## Why this matters

The notebook results rewrite the reconstruction hypothesis: the hidden data is still in the adjacent model weights, but the strongest candidate now uses reverse block-16 interleave and a small shift. The next step is to validate that aligned stream with controlled secondary transforms, rather than chasing broad bitplane or archive-only branches.
