import re
import torch

PATTERNS = [
    rb"ctf\{.*?\}",
    rb"flag\{.*?\}",
    rb"ctf=",
    rb"PK",
    rb"\x1f\x8b",
    rb"http",
]


def extract_ascii_runs(data, min_len=4):
    pattern = rb"[ -~]{%d,}" % min_len
    return [(m.start(), m.group()) for m in re.finditer(pattern, data)]


def load_bytes(model, name):
    tensor = model[name]
    return tensor.numpy().tobytes()


def find_patterns(blob, label):
    print(f"\n--- SEARCH {label} ---")
    for patt in PATTERNS:
        idx = blob.find(patt)
        if idx != -1:
            print(f"found {patt!r} at {idx}")
    for patt in [rb"ctf\{.*?\}", rb"flag\{.*?\}"]:
        for m in re.finditer(patt, blob):
            print(f"regex {patt!r} match at {m.start()}: {m.group()}")


def print_window(blob, idx, context=64):
    start = max(0, idx - context)
    end = min(len(blob), idx + context)
    print(f"window around {idx}: {blob[start:end]!r}")


def deinterleave(a, b, block_size=1):
    out = bytearray()
    i = 0
    while i < len(a) and i < len(b):
        out.extend(a[i : i + block_size])
        out.extend(b[i : i + block_size])
        i += block_size
    return bytes(out)


def xor_bytes(a, b, shift=0, reverse=False):
    if reverse:
        a, b = b, a
    if shift > 0:
        a = a[shift:]
        b = b[: len(a)]
    return bytes(x ^ y for x, y in zip(a, b))


def bitplane_bytes(data, bit):
    bits = [(c >> bit) & 1 for c in data]
    out = bytearray()
    for i in range(0, len(bits) - (len(bits) % 8), 8):
        byte = 0
        for j in range(8):
            byte |= bits[i + j] << j
        out.append(byte)
    return bytes(out)


def printable_ratio(data):
    return sum(32 <= c < 127 for c in data) / len(data) if data else 0


def show_ascii_runs(blob, label, min_len=8, max_runs=10):
    runs = extract_ascii_runs(blob, min_len=min_len)
    if runs:
        print(f"\n--- ASCII runs in {label} (len>={min_len}) ---")
        for off, s in runs[:max_runs]:
            print(off, s)


if __name__ == "__main__":
    obj = torch.load("ode.pt", map_location="cpu", weights_only=False)

    layer6 = "transformer.h.6.mlp.c_proj.weight"
    layer7 = "transformer.h.7.mlp.c_proj.weight"
    raw6 = load_bytes(obj["model"], layer6)
    raw7 = load_bytes(obj["model"], layer7)

    print(f"{layer6}: {len(raw6)} bytes")
    print(f"{layer7}: {len(raw7)} bytes")
    print(f"{layer6} ctf= offset: {raw6.find(b'ctf=')}")
    print(f"{layer7} ctf= offset: {raw7.find(b'ctf=')}")

    find_patterns(raw6, layer6)
    find_patterns(raw7, layer7)
    show_ascii_runs(raw6, layer6)
    show_ascii_runs(raw7, layer7)

    print("\n--- COMBINATIONS ---")
    combos = [
        (raw6 + raw7, "concat 6+7"),
        (raw7 + raw6, "concat 7+6"),
        (deinterleave(raw6, raw7, 1), "deinterleave 1"),
        (deinterleave(raw7, raw6, 1), "deinterleave 1 reversed"),
        (deinterleave(raw6, raw7, 2), "deinterleave 2"),
        (deinterleave(raw6, raw7, 4), "deinterleave 4"),
        (deinterleave(raw6, raw7, 8), "deinterleave 8"),
    ]

    for blob, label in combos:
        find_patterns(blob, label)
        if b"ctf=" in blob:
            idx = blob.find(b"ctf=")
            print_window(blob, idx)

    print("\n--- XOR TESTS ---")
    find_patterns(xor_bytes(raw6, raw7), "xor 6^7")
    find_patterns(xor_bytes(raw7, raw6), "xor 7^6")

    for shift in [1, 2, 3, 4, 8, 16, 32]:
        blob = xor_bytes(raw6, raw7, shift=shift)
        find_patterns(blob, f"xor 6^7 shift {shift}")
        blob = xor_bytes(raw7, raw6, shift=shift)
        find_patterns(blob, f"xor 7^6 shift {shift}")

    print("\n--- ALIGNED PAYLOAD CHECK ---")
    idx7 = raw7.find(b"ctf=")
    if idx7 != -1:
        idx6 = idx7 + 4
        print(f"raw7 ctf= offset: {idx7}")
        print(f"raw6 aligned offset: {idx6}")
        aligned = raw6[idx6 - 64 : idx6 + 256]
        print("raw6 aligned window:")
        print(aligned.hex())
        print("raw6 aligned ascii:")
        print("".join(chr(x) if 32 <= x < 127 else "." for x in aligned))
        for patt in PATTERNS + [rb"flag{", rb"ctf{"]:
            off = aligned.find(patt)
            if off != -1:
                print(f" aligned window contains {patt!r} at {off}")
        print("\n--- SIMPLE XOR AROUND ALIGNED REGION ---")
        segment6 = raw6[idx6 : idx6 + 128]
        segment7 = raw7[idx7 : idx7 + 128]
        for key in range(1, 256):
            x = bytes(a ^ key for a in segment6)
            printable = sum(32 <= c < 127 for c in x)
            if printable >= 100:
                print(f"xor raw6 aligned with key {key}: printable {printable}")
                print(x[:64])
                break

    print("\n--- BITPLANE EXTRACTION ---")
    for block in [4, 8]:
        blob = deinterleave(raw6, raw7, block)
        idx = blob.find(b"ctf=")
        if idx == -1:
            continue
        data = blob[idx : idx + 65536]
        for bit in [1, 6]:
            out = bitplane_bytes(data, bit)
            filename = f"bitplane_{block}_{bit}.bin"
            with open(filename, "wb") as f:
                f.write(out)
            print(f"wrote {filename}: {len(out)} bytes, printable ratio {printable_ratio(out):.4f}")
            runs = extract_ascii_runs(out, min_len=8)
            if runs:
                print(f"  first ascii run in {filename}:", runs[0])

    print("\n--- DONE ---")
