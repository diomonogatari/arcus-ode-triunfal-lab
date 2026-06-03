"""
arcus_pty.py — drive the Arcus SSH TUI over a real PTY (stdlib only; no expect/pexpect).

Usage:
  python3 arcus_pty.py recon                 # connect, snapshot main screen, enter trial, submit a throwaway, show result
  python3 arcus_pty.py submit "FLAG TEXT"    # navigate to flag: prompt and submit one candidate, print de-ANSI'd result

Detects accept vs reject heuristically from the post-submit screen.
"""
import os, pty, sys, select, time, fcntl, termios, struct, re, subprocess

HOST = "augustalabs.ai"


def deansi(b: bytes) -> str:
    s = b.decode("utf-8", "replace")
    s = re.sub(r"\x1b\][^\x07\x1b]*(\x07|\x1b\\)", "", s)         # OSC
    s = re.sub(r"\x1bP[^\x1b]*\x1b\\", "", s)                     # DCS
    s = re.sub(r"\x1b[\[\?][0-9;:<>=$\"' ]*[A-Za-z@`~]", "", s)   # CSI
    s = re.sub(r"\x1b[()][AB0]", "", s)
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)
    lines = [ln.rstrip() for ln in s.splitlines()]
    return "\n".join(ln for ln in lines if ln.strip())


class Session:
    def __init__(self, rows=44, cols=140):
        self.master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        self.p = subprocess.Popen(
            ["ssh", "-tt", "-o", "StrictHostKeyChecking=accept-new",
             "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=5", HOST],
            stdin=slave, stdout=slave, stderr=slave, close_fds=True,
            preexec_fn=os.setsid,
        )
        os.close(slave)
        self.buf = b""

    def read_until_quiet(self, total=8.0, quiet=1.2):
        """Read until `quiet` seconds pass with no new bytes, or `total` elapses."""
        start = time.time(); last = time.time(); got = b""
        while time.time() - start < total:
            r, _, _ = select.select([self.master], [], [], 0.3)
            if r:
                try:
                    chunk = os.read(self.master, 65536)
                except OSError:
                    break
                if chunk:
                    got += chunk; self.buf += chunk; last = time.time()
            elif time.time() - last >= quiet:
                break
        return got

    def send(self, data: str):
        os.write(self.master, data.encode("utf-8"))

    def close(self):
        try:
            self.send("\x03"); self.send("q")
            os.killpg(os.getpgid(self.p.pid), 15)
        except Exception:
            pass
        try:
            self.p.wait(timeout=5)
        except Exception:
            pass


SUCCESS = re.compile(r"(?i)correct|success|parab|first blood|ganha|ganhaste|venceu|vencedor|"
                     r"desbloq|unlock|resolvid|\bII\b|n[ií]vel|level\s*2|próxim|congrat|✓|well done")
WRONG = re.compile(r"(?i)wrong|incorrect|errad|inv[aá]lid|tenta|try again|n[aã]o|✗|nope|fail")


def navigate(s: Session):
    """From the main screen, enter the Ode Triunfal trial and reach the flag: prompt."""
    s.read_until_quiet(total=6, quiet=1.0)         # let main screen render
    s.send("\r")                                   # '> I · Ode Triunfal' appears pre-selected
    s.read_until_quiet(total=6, quiet=1.0)


def submit(candidate: str, label=""):
    s = Session()
    try:
        navigate(s)
        before = deansi(s.buf)
        has_flag_prompt = bool(re.search(r"(?i)flag\s*:", before))
        # type the candidate at the flag: prompt
        s.send(candidate + "\r")
        post = s.read_until_quiet(total=10, quiet=1.5)
        screen = deansi(s.buf)
        result = "UNKNOWN"
        if SUCCESS.search(screen) and not WRONG.search(screen[-400:]):
            result = "SUCCESS?"
        elif WRONG.search(screen):
            result = "WRONG"
        print(f"\n========== SUBMIT {label} ==========")
        print(f"candidate: {candidate!r}")
        print(f"reached flag: prompt during nav = {has_flag_prompt}")
        print(f"verdict: {result}")
        print("---- de-ANSI'd screen (tail) ----")
        print(screen[-1500:])
        return result, screen
    finally:
        s.close()


def recon():
    s = Session()
    try:
        print("==== STEP 1: initial screen ====")
        s.read_until_quiet(total=7, quiet=1.2)
        print(deansi(s.buf)[-1800:])
        print("\n==== STEP 2: after Enter ====")
        n0 = len(s.buf)
        s.send("\r")
        s.read_until_quiet(total=7, quiet=1.2)
        print(deansi(s.buf[n0:])[-1800:])
        print("\n==== STEP 3: submit throwaway 'flag{recon_test_zzz}' ====")
        n1 = len(s.buf)
        s.send("flag{recon_test_zzz}\r")
        s.read_until_quiet(total=10, quiet=1.5)
        print(deansi(s.buf[n1:])[-1800:])
    finally:
        s.close()


def batch(candidates):
    """Submit many candidates in ONE session, pressing Enter ('try again') between.
    A wrong answer prints 'wrong answer.'; anything else is flagged for inspection."""
    s = Session()
    results = []
    try:
        navigate(s)
        if not re.search(r"(?i)flag\s*:", deansi(s.buf)):
            print("WARNING: did not detect flag: prompt after navigation")
        # Only "wrong answer." reliably marks rejection. Do NOT match menu-header text
        # like "first blood" (the TUI redraws the whole screen each update -> false success).
        WRONG_RE = re.compile(r"(?i)wrong answer|try again|tente novamente")
        for idx, cand in enumerate(candidates):
            mark = len(s.buf)
            s.send(cand + "\r")
            # Wait until the server's 'checking...' resolves to 'wrong answer', or time out.
            verdict = None
            deadline = time.time() + 22.0
            while time.time() < deadline:
                s.read_until_quiet(total=3.0, quiet=0.7)
                new = deansi(s.buf[mark:])
                if WRONG_RE.search(new):
                    verdict = "WRONG"; break
            new = deansi(s.buf[mark:])
            if verdict is None:
                verdict = "*** NO 'wrong answer' SEEN (INSPECT!) ***"
            results.append((cand, verdict))
            print(f"\n===== [{idx+1}/{len(candidates)}] {verdict} =====")
            print(f"candidate: {cand!r}")
            print("screen(new):")
            print(new[-700:])
            if verdict != "WRONG":
                print(">>> NOT a clean WRONG — inspect above.")
                break
            # press enter to return to the flag: prompt
            s.send("\r")
            s.read_until_quiet(total=6, quiet=1.0)
    finally:
        s.close()
    print("\n==== SUMMARY ====")
    for c, v in results:
        print(f"  {v:28} {c!r}")
    return results


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "submit":
        submit(sys.argv[2], label="manual")
    elif len(sys.argv) >= 2 and sys.argv[1] == "batch":
        path = sys.argv[2] if len(sys.argv) >= 3 else "candidates.txt"
        cands = [ln.rstrip("\n") for ln in open(path, encoding="utf-8") if ln.strip()]
        batch(cands)
    else:
        recon()
