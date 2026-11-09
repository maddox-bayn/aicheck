"""Run published behavioral checks on trusted local Python code."""

import argparse
import ast
import json
import os
import re
import signal
import sys
import traceback
from contextlib import contextmanager, redirect_stdout, nullcontext
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


@contextmanager
def raises(expected):
    try:
        yield
    except expected:
        return
    raise AssertionError(f"expected {expected.__name__}")


def read_checkpoint(number):
    name = f"{number:02}"
    text = (ROOT / "checkpoints" / f"{name}.md").read_text()
    headings = re.findall(r"^## Exercise (\d+)\.(\d+): (.+)$", text, re.M)
    solutions = re.findall(
        r"^\x60\x60\x60python\n(.*?)^\x60\x60\x60$", text, re.M | re.S
    )
    suite = (ROOT / "checks" / f"{name}.py").read_text()
    checks = re.split(r"(?m)^# Exercise (\d+)\s*$", suite)
    ids = list(range(1, 11))
    if (
        [(int(c), int(e)) for c, e, _ in headings] != [(number, i) for i in ids]
        or len(solutions) != 10
        or list(map(int, checks[1::2])) != ids
    ):
        raise ValueError(f"invalid checkpoint structure: {name}")
    for i, (heading, code, test) in enumerate(
        zip(headings, solutions, checks[2::2]), 1
    ):
        ast.parse(code)
        ast.parse(test)
        yield i, heading[2], code, test


def alarm_handler(signum, frame):
    raise TimeoutError("exercise time limit exceeded")


def run_case(number, index, solution, test, submission, timeout):
    filename = f"cp{number:02}/ex{index:02}.py"
    module = ModuleType(f"aicheck_{number}_{index}")
    sys.modules[module.__name__] = module
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(timeout)
    try:
        if submission is None:
            code = solution
        else:
            path = submission / filename
            code = path.read_text()
            filename = str(path)
        exec(compile(code, filename, "exec"), module.__dict__)
        # Tests own their imports. Only public functions/classes come from the submission.
        scope = {"raises": raises, "__name__": "__aicheck_tests__"}
        for node in ast.parse(solution).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                scope[node.name] = getattr(module, node.name)
        exec(compile(test, f"checks/{number:02}.py:exercise-{index}", "exec"), scope)
    finally:
        signal.alarm(0)
        sys.modules.pop(module.__name__, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=int, choices=range(1, 21))
    parser.add_argument("--exercise", type=int, choices=range(1, 11))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--reference", action="store_true")
    group.add_argument("--submission", type=Path)
    group.add_argument("--list", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if not __debug__:
        parser.error("assertions must be enabled")
    if args.exercise and not args.checkpoint:
        parser.error("--exercise requires a checkpoint")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    results = []
    for number in [args.checkpoint] if args.checkpoint else range(1, 21):
        for index, title, solution, test in read_checkpoint(number):
            if args.exercise and args.exercise != index:
                continue
            result = {"exercise": f"{number:02}.{index}", "title": title}
            if not args.list:
                try:
                    with redirect_stdout(sys.stderr) if args.json else nullcontext():
                        run_case(
                            number, index, solution, test, args.submission, args.timeout
                        )
                    result["status"] = "PASS"
                except (Exception, SystemExit) as exc:
                    result.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
                    if args.verbose:
                        traceback.print_exc(file=sys.stderr)
            results.append(result)
            if not args.json:
                print(
                    result.get("status", ""),
                    result["exercise"],
                    title,
                    result.get("error", ""),
                    flush=True,
                )
    if args.json:
        print(json.dumps(results, indent=2))
    elif not args.list:
        print(f"{sum(r['status'] == 'PASS' for r in results)}/{len(results)} passed")
    return int(any(r.get("status") == "FAIL" for r in results))


if __name__ == "__main__":
    sys.exit(main())
