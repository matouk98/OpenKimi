#!/usr/bin/env python3
"""Smoke test for local sandbox used by LocalSandboxTool (agentic GRPO)."""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def run_case(url: str, code: str, timeout_s: float) -> dict:
    payload = {
        "code": code,
        "stdin": "",
        "language": "python",
        "compile_timeout": 1.0,
        "run_timeout": float(timeout_s),
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s + 10.0) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def print_result(name: str, result: dict) -> bool:
    status = result.get("status")
    run_result = result.get("run_result", {}) if isinstance(result, dict) else {}
    stdout = run_result.get("stdout", "")
    stderr = run_result.get("stderr", "")
    ok = status == "success"

    print(f"\n=== {name} ===")
    print(f"status: {status}")
    print(f"stdout: {stdout!r}")
    print(f"stderr: {stderr!r}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Local sandbox smoke test.")
    parser.add_argument(
        "--url",
        default=os.getenv("LOCAL_SANDBOX_URL", "http://127.0.0.1:12345/faas/sandbox/"),
        help="Sandbox endpoint URL",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="run_timeout for each case")
    args = parser.parse_args()

    print(f"[info] sandbox url: {args.url}")
    print("[info] expected tool name in agent loop: code_interpreter")

    # (name, code, expect_success)
    cases = [
        # ── basic math ────────────────────────────────────────────────────────
        ("math_log2_basic",      "import math\nprint(math.log2(8))",    True),
        ("math_log2_fraction",   "import math\nprint(math.log2(0.5))",  True),
        ("stdlib_sqrt",          "import math\nprint(round(math.sqrt(2), 6))", True),
        ("intentional_error",    "import math\nprint(math.log2(-1))",   False),  # ValueError expected
        # ── numpy / scipy ─────────────────────────────────────────────────────
        ("numpy_matmul",
            "import numpy as np\n"
            "a = np.array([[1,2],[3,4]])\n"
            "b = np.array([[5,6],[7,8]])\n"
            "print(a @ b)",   True),
        ("scipy_solve",
            "from scipy.linalg import solve\n"
            "import numpy as np\n"
            "A = np.array([[3,1],[1,2]], dtype=float)\n"
            "b = np.array([9,8], dtype=float)\n"
            "print(solve(A, b))",  True),
        # ── multi-step reasoning style ────────────────────────────────────────
        ("prime_check",
            "def is_prime(n):\n"
            "    if n < 2: return False\n"
            "    for i in range(2, int(n**0.5)+1):\n"
            "        if n % i == 0: return False\n"
            "    return True\n"
            "primes = [x for x in range(2, 50) if is_prime(x)]\n"
            "print(primes)",   True),
        ("combinatorics",
            "from math import comb, factorial\n"
            "print(comb(10, 3))\n"
            "print(factorial(7))",  True),
        ("symbolic_sympy",
            "from sympy import symbols, expand, factor\n"
            "x = symbols('x')\n"
            "expr = (x+1)**3\n"
            "print(expand(expr))\n"
            "print(factor(expand(expr)))",  True),
        # ── timeout / resource ────────────────────────────────────────────────
        ("timeout_stress",
            "total = 0\n"
            "for i in range(10**7):\n"
            "    total += i\n"
            "print(total)",   True),
        # ── stderr only (no stdout) ───────────────────────────────────────────
        ("stderr_only",
            "import sys\n"
            "sys.stderr.write('this goes to stderr\\n')",  True),
        # ── large output truncation ───────────────────────────────────────────
        ("large_output",
            "for i in range(5000):\n"
            "    print(f'line {i}: ' + 'x' * 80)",  True),
    ]

    counted = 0      # cases with a definite expected outcome
    passed  = 0
    fail_names: list[str] = []
    for name, code, expect in cases:
        try:
            result = run_case(args.url, code, args.timeout)
        except urllib.error.URLError as e:
            print(f"\n=== {name} ===")
            print(f"request_failed: {e}")
            if expect is not None:
                fail_names.append(name)
                counted += 1
            continue
        except Exception as e:
            print(f"\n=== {name} ===")
            print(f"unexpected_exception: {e}")
            if expect is not None:
                fail_names.append(name)
                counted += 1
            continue

        ran_ok = print_result(name, result)
        if expect is None:
            print(f"  [skip check] expect=any")
        else:
            counted += 1
            if ran_ok == expect:
                passed += 1
                print(f"  [PASS]")
            else:
                fail_names.append(name)
                print(f"  [FAIL] expected success={expect}")

    print(f"\n[summary] passed={passed}/{counted}")
    if fail_names:
        print(f"[failed]  {fail_names}")
    print("[hint] If numpy/scipy/sympy fail, run: pip install numpy scipy sympy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
