import math
from collections import Counter

import torch

PATTERNS = {
    'ctf=': b'ctf=',
    'flag{': b'flag{',
    'MZ': b'MZ',
    'gzip': b'\x1f\x8b',
    'PK': b'PK',
}

BLOCK_SIZES = [1, 2, 4, 8, 16, 32]
MAX_SHIFT = 64


def entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    length = len(data)
    return -sum((cnt / length) * math.log2(cnt / length) for cnt in counts.values())


def printable_ratio(data: bytes) -> float:
    if not data:
        return 0.0
    return sum(32 <= b < 127 for b in data) / len(data)


def longest_ascii_run(data: bytes, min_len: int = 4) -> int:
    best = 0
    current = 0
    for b in data:
        if 32 <= b < 127:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best if best >= min_len else 0


def load_raw():
    obj = torch.load('ode.pt', map_location='cpu', weights_only=False)
    raw6 = obj['model']['transformer.h.6.mlp.c_proj.weight'].numpy().tobytes()
    raw7 = obj['model']['transformer.h.7.mlp.c_proj.weight'].numpy().tobytes()
    return raw6, raw7


def deinterleave(a: bytes, b: bytes, block_size: int, shift: int, reverse: bool) -> bytes:
    if reverse:
        a, b = b, a
    if shift:
        a = a[shift:]
        b = b[: len(a)]
    length = min(len(a), len(b))
    out = bytearray()
    i = 0
    while i < length:
        out.extend(a[i : i + block_size])
        out.extend(b[i : i + block_size])
        i += block_size
    return bytes(out)


def markers_for(blob: bytes) -> dict[str, int]:
    return {name: blob.find(pattern) for name, pattern in PATTERNS.items()}


def score_candidate(blob: bytes, ctf_offset: int) -> dict:
    region_start = max(0, ctf_offset - 64)
    region_end = min(len(blob), ctf_offset + 256)
    region = blob[region_start:region_end]
    return {
        'entropy': entropy(blob),
        'region_printable': printable_ratio(region),
        'ascii_run': longest_ascii_run(region),
        'region_len': len(region),
    }


def fmt_offset(offset: int) -> str:
    return str(offset) if offset != -1 else '---'


def main():
    raw6, raw7 = load_raw()
    print(f'raw6 {len(raw6)} bytes, raw7 {len(raw7)} bytes')

    candidates = []
    for block_size in BLOCK_SIZES:
        for shift in range(MAX_SHIFT + 1):
            for reverse in (False, True):
                blob = deinterleave(raw6, raw7, block_size=block_size, shift=shift, reverse=reverse)
                markers = markers_for(blob)
                ctf_offset = markers['ctf=']
                if ctf_offset == -1:
                    continue
                score = score_candidate(blob, ctf_offset)
                score.update({
                    'block': block_size,
                    'shift': shift,
                    'reverse': reverse,
                    'ctf_offset': ctf_offset,
                    'marker_count': sum(1 for v in markers.values() if v != -1),
                    'markers': markers,
                })
                candidates.append(score)

    if not candidates:
        print('No candidate reconstruction contains ctf=.')
        return

    candidates.sort(
        key=lambda c: (
            -c['marker_count'],
            -c['region_printable'],
            -c['ascii_run'],
            c['entropy'],
            c['ctf_offset'],
        )
    )

    print(f'Found {len(candidates)} candidate reconstructions containing ctf=')
    for rank, cand in enumerate(candidates[:20], 1):
        print(
            f"{rank:02d}. block={cand['block']} shift={cand['shift']} reverse={cand['reverse']} "
            f"ctf={cand['ctf_offset']} markers={cand['marker_count']} "
            f"print={cand['region_printable']:.4f} ascii={cand['ascii_run']} entropy={cand['entropy']:.4f} "
            f"MZ={fmt_offset(cand['markers']['MZ'])} gzip={fmt_offset(cand['markers']['gzip'])} "
            f"PK={fmt_offset(cand['markers']['PK'])} flag={fmt_offset(cand['markers']['flag{'])}"
        )
        blob = deinterleave(raw6, raw7, block_size=cand['block'], shift=cand['shift'], reverse=cand['reverse'])
        window = blob[max(0, cand['ctf_offset'] - 32) : cand['ctf_offset'] + 128]
        print(window[:128].hex())
        print()

    print('Done.')


if __name__ == '__main__':
    main()
