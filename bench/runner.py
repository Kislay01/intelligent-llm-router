"""Run untrusted, model-written code with a timeout. Docker by default; local mode is for tests only."""
import subprocess
import sys

DOCKER_IMAGE = "python:3.12-slim"
# "Unable to find image ... locally" is NOT here: Docker prints it on every first run, then pulls and runs.
DAEMON_ERRORS = ("cannot connect to the docker daemon", "error during connect", "is the docker daemon running",
                 "pull access denied", "error response from daemon")


class SandboxError(RuntimeError):
    """The sandbox itself failed (Docker missing/down). Never counted as the model failing a task."""


def _docker_cmd(timeout: int) -> list:
    return [
        "docker", "run", "--rm", "-i",
        "--network", "none",            # no internet from generated code
        "--memory", "256m", "--cpus", "1", "--pids-limit", "64",
        "--read-only", "--tmpfs", "/tmp:size=16m",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        DOCKER_IMAGE, "timeout", "-s", "KILL", str(timeout), "python", "-",
    ]


def run_program(program: str, mode: str = "docker", timeout: int = 10) -> dict:
    """Returns {"status": "pass"|"fail"|"timeout", "detail": str}. Raises SandboxError if Docker is unusable."""
    if mode == "docker":
        cmd, outer = _docker_cmd(timeout), timeout + 120  # first run may pull the image
    elif mode == "local":
        cmd, outer = [sys.executable, "-I", "-"], timeout
    else:
        raise ValueError(f"unknown mode {mode!r}")

    try:
        proc = subprocess.run(cmd, input=program, text=True, encoding="utf-8", errors="replace",
                              capture_output=True, timeout=outer)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "detail": f"no result within {outer}s"}
    except FileNotFoundError as e:
        raise SandboxError(f"could not start the sandbox: {e}")

    stderr = (proc.stderr or "").strip()
    if mode == "docker" and (proc.returncode == 125 or any(m in stderr.lower() for m in DAEMON_ERRORS)):
        raise SandboxError(f"Docker problem (is Docker Desktop running?): {stderr[-300:]}")
    if proc.returncode == 0:
        return {"status": "pass", "detail": ""}
    if mode == "docker" and proc.returncode in (124, 137):  # killed by `timeout -s KILL`
        return {"status": "timeout", "detail": f"exceeded {timeout}s"}
    return {"status": "fail", "detail": stderr[-300:]}


def selftest(mode: str = "docker") -> bool:
    """Checks that the sandbox tells pass/fail/timeout apart and (docker) blocks the network."""
    checks = [
        ("passing program", "assert 1 + 1 == 2", "pass"),
        ("failing assert", "assert 1 + 1 == 3", "fail"),
        ("infinite loop", "while True:\n    pass", "timeout"),
    ]
    if mode == "docker":
        checks.append(("network is blocked",
                       "import urllib.request\nurllib.request.urlopen('http://example.com', timeout=3)", "fail"))
        checks.append(("filesystem is read-only", "open('/etc/should_not_write', 'w').write('x')", "fail"))
    ok = True
    for name, program, expected in checks:
        got = run_program(program, mode, timeout=3)["status"]
        print(f"  [{'OK' if got == expected else 'WRONG'}] {name}: expected {expected}, got {got}")
        ok &= got == expected
    return ok
