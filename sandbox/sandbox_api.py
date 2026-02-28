#!/usr/bin/env python3
"""
Minimal sandbox API for LocalSandboxTool (agentic GRPO).
Compatible with LocalSandboxTool and sandbox_smoke_test.py; no dependency on simpletir.

Start from this directory:
  cd /path/to/adaptive/sandbox
  ./start_sandbox.sh
Or: uvicorn sandbox_api:app --host 127.0.0.1 --port 12345 [--workers 4]
"""
import asyncio
import os
import tempfile
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="adaptive-sandbox", version="0.1.0")


class RunRequest(BaseModel):
    code: str
    stdin: str = ""
    language: str = "python"
    compile_timeout: float = 1.0
    run_timeout: float = 30.0


def _run_python(code: str, timeout_s: float) -> tuple[str, str, int]:
    """Run Python code in subprocess. Returns (stdout, stderr, returncode)."""
    import subprocess

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        result = subprocess.run(
            ["python3", tmp],
            capture_output=True,
            text=True,
            timeout=max(1.0, timeout_s + 5.0),
            cwd=os.path.expanduser("~"),
        )
        return result.stdout or "", result.stderr or "", result.returncode
    except subprocess.TimeoutExpired:
        return "", f"Timeout after {timeout_s}s", -1
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


@app.post("/faas/sandbox/")
async def run_sandbox(req: RunRequest) -> dict[str, Any]:
    """Run code in sandbox. Compatible with LocalSandboxTool and sandbox_smoke_test."""
    if req.language != "python":
        raise HTTPException(status_code=400, detail=f"Unsupported language: {req.language}")
    timeout_s = max(0.5, min(float(req.run_timeout), 300.0))
    loop = asyncio.get_event_loop()
    stdout, stderr, returncode = await loop.run_in_executor(
        None,
        lambda: _run_python(req.code, timeout_s),
    )
    status = "success" if returncode == 0 else "failed"
    return {
        "status": status,
        "run_result": {
            "stdout": stdout,
            "stderr": stderr,
            "return_code": returncode,
        },
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
