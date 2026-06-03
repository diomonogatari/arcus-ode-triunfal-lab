# archive/ — the byte-blob rabbit hole (disproven)

These are the **dead ends**, kept because the write-up's value is the reasoning, including
what *didn't* work. The original hypothesis was that the flag was hidden in the weight
*bytes* (deinterleave / XOR / bit-plane the float32 tensors). It was wrong: the weights are
byte-clean (mantissa-LSB entropy = 1.0), and the `ctf=` "hits" are statistical noise.

- `rustscan/` — Rust byte-scanner (deinterleave h.6/h.7 + transform sweep). Source only.
- `run_rust_scan.py`, `detect.py`, `fast_filter.py`, `reconstruct.py`, `final.py` — byte-blob scripts.
- `main.py` — the original inference attempt (superseded by `solve_inference.py`).
- `ctf_analysis.md` — early (byte-blob-era) notes, superseded by `../WRITEUP.md` / `../FINDINGS.md`.
- `ode_payload_analysis.ipynb`, `ode_analysis.ipynb` — early exploratory notebooks.
