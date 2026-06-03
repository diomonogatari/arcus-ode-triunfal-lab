"""
Exhaustively probe the Arcus SSH TUI as a possible navigable shell/menu (lead #1).
Our prior recon only did connect -> Enter -> flag:. This tries TYPED COMMANDS at the root
menu and inside the trial (help/ls/dir/start/arcus/cd/back/?), capturing every screen delta,
on a SINGLE reused connection. Closes the "is there a hidden command/file/trial" question.
"""
import importlib.util, time, re, sys
spec = importlib.util.spec_from_file_location("ap", "arcus_pty.py")
ap = importlib.util.module_from_spec(spec); spec.loader.exec_module(ap)

s = ap.Session()
def snap(label, settle=3.0):
    mark = len(s.buf)
    s.read_until_quiet(total=settle, quiet=1.0)
    new = ap.deansi(s.buf[mark:])
    print(f"\n--- [{label}] ---"); print(new[-500:] if new.strip() else "(no change)")
    return new

try:
    first = snap("root (initial)", settle=14)
    if "talent" not in first and "Ode" not in first and len(s.buf) == 0:
        print(">>> 0 bytes / not reachable (throttled?) — aborting"); s.close(); sys.exit(0)

    # 1) typed commands at the ROOT menu (before entering the trial)
    for cmd in ["help", "?", "ls", "dir", "start", "arcus", "cd", "menu", "exit-test-no"]:
        m0 = len(s.buf); s.send(cmd + "\r")
        snap(f"root + {cmd!r}", settle=3.5)
        # if the screen jumped into something, note it; press Ctrl-C-ish reset not needed for menu

    # 2) enter the trial and try commands at the flag: prompt
    ap.navigate(s)
    snap("after navigate (flag: prompt expected)", settle=4)
    for cmd in ["help", "ls", "back", "..", "cd ..", "?", "menu"]:
        m0 = len(s.buf); s.send(cmd + "\r")
        out = snap(f"flag: + {cmd!r}", settle=6)
        # most will be judged as wrong flags; just press enter to return to prompt
        if re.search(r"(?i)wrong answer|try again", out):
            s.send("\r"); s.read_until_quiet(total=4, quiet=0.8)
finally:
    s.close()
print("\nDONE (explore_tui2).")
