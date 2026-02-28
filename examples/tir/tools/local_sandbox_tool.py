"""
LocalSandboxTool: executes Python code via a local sandbox HTTP endpoint.

The sandbox server must be running before training starts:
    cd examples/tir/sandbox && ./start_sandbox.sh
Or set LOCAL_SANDBOX_URL to point to any compatible sandbox endpoint.

Tool config path: examples/tir/sandbox/local_sandbox_tool_config.yaml
"""

import asyncio
import os
import re
from typing import Any, Optional
from uuid import uuid4

import aiohttp

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


class LocalSandboxTool(BaseTool):
    """Execute Python code via a local sandbox HTTP endpoint.

    Compatible with the sandbox server in examples/tir/sandbox/sandbox_api.py.
    The tool config YAML should set class_name to
    ``examples.tir.tools.local_sandbox_tool.LocalSandboxTool``.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict] = {}
        self.sandbox_url: str = config.get("sandbox_url") or os.getenv(
            "LOCAL_SANDBOX_URL", "http://127.0.0.1:12345/faas/sandbox/"
        )
        self.default_timeout: int = int(config.get("default_timeout", 30))
        self.default_language: str = config.get("default_language", "python")
        self.max_code_chars: int = int(config.get("max_code_chars", 20000))
        self.max_output_chars: int = int(config.get("max_output_chars", 12000))
        self.request_retries: int = int(config.get("request_retries", 1))
        self.max_inflight_requests: int = int(config.get("max_inflight_requests", 16))
        self._request_semaphore = asyncio.Semaphore(max(1, self.max_inflight_requests))
        self.block_dangerous_code: bool = bool(config.get("block_dangerous_code", True))
        self.safety_mode: str = str(config.get("safety_mode", "strict")).strip().lower()
        self.strict_max_timeout: int = int(config.get("strict_max_timeout", 60))
        allowed_languages_raw = config.get("allowed_languages", ["python"])
        if isinstance(allowed_languages_raw, str):
            allowed_languages_raw = [allowed_languages_raw]
        self.allowed_languages: set[str] = {str(x).strip().lower() for x in allowed_languages_raw}
        if not self.allowed_languages:
            self.allowed_languages = {"python"}
        self._dangerous_code_patterns = self._build_dangerous_patterns(self.safety_mode)
        self.code_pattern = re.compile(r"```(?:py|python)?\n(.*?)\n```", re.DOTALL)

    @staticmethod
    def _build_dangerous_patterns(mode: str) -> list[tuple[re.Pattern, str]]:
        """Build block patterns for code safety checks.

        Modes:
          - relaxed: block destructive FS ops, shell execution, networking, infinite loops.
          - strict: relaxed + dynamic code execution and risky native/serialization APIs.
        """
        relaxed: list[tuple[re.Pattern, str]] = [
            (re.compile(r"\brm\s+-rf\s+/", re.IGNORECASE), "shell rm -rf /"),
            (re.compile(r"\brm\s+-rf\b", re.IGNORECASE), "shell rm -rf"),
            (re.compile(r"\bshutil\.(rmtree|move)\s*\(", re.IGNORECASE), "shutil destructive op"),
            (re.compile(r"\bos\.(remove|unlink|rmdir|removedirs|rename)\s*\(", re.IGNORECASE), "os destructive op"),
            (re.compile(r"\bsubprocess\.(run|Popen|call|check_call|check_output)\s*\(", re.IGNORECASE), "subprocess"),
            (re.compile(r"\bos\.system\s*\(", re.IGNORECASE), "os.system"),
            (re.compile(r"\b(import\s+socket|import\s+requests|import\s+httpx|import\s+urllib)\b", re.IGNORECASE), "network access"),
            (re.compile(r"\b(curl|wget|nc|ncat)\b", re.IGNORECASE), "network shell tool"),
            (re.compile(r"\bwhile\s+True\s*:", re.IGNORECASE), "infinite loop pattern"),
            (re.compile(r"\bpip\s+install\b", re.IGNORECASE), "runtime package install"),
            (re.compile(r"\bapt(-get)?\s+install\b", re.IGNORECASE), "runtime apt install"),
        ]
        if mode != "strict":
            return relaxed
        strict_only: list[tuple[re.Pattern, str]] = [
            (re.compile(r"\beval\s*\(", re.IGNORECASE), "eval"),
            (re.compile(r"\bexec\s*\(", re.IGNORECASE), "exec"),
            (re.compile(r"\b__import__\s*\(", re.IGNORECASE), "__import__"),
            (re.compile(r"\bpickle\.loads\s*\(", re.IGNORECASE), "pickle.loads"),
            (re.compile(r"\bmarshal\.loads\s*\(", re.IGNORECASE), "marshal.loads"),
            (re.compile(r"\bctypes\b", re.IGNORECASE), "ctypes"),
            (re.compile(r"\bcffi\b", re.IGNORECASE), "cffi"),
            (re.compile(r"\bopen\s*\([^)]*,\s*[\"'](w|a|x|wb|ab|xb)[\"']", re.IGNORECASE), "file write via open"),
            (re.compile(r"\bmultiprocessing\b", re.IGNORECASE), "multiprocessing"),
            (re.compile(r"\bthreading\b", re.IGNORECASE), "threading"),
            (re.compile(r"\bos\.environ\b", re.IGNORECASE), "environment variable access"),
        ]
        return relaxed + strict_only

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    @staticmethod
    def _parse_timeout(value: Any, default_timeout: int) -> int:
        if value is None:
            return default_timeout
        try:
            timeout = int(float(str(value).strip()))
        except (TypeError, ValueError):
            return default_timeout
        return max(1, min(timeout, 300))

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"

    async def create(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"ground_truth": ground_truth}
        return instance_id, ToolResponse()

    async def _run_code(self, code: str, timeout: int, language: str) -> ToolResponse:
        payload = {
            "code": code,
            "stdin": "",
            "language": language,
            "compile_timeout": 1.0,
            "run_timeout": float(timeout),
        }
        timeout_cfg = aiohttp.ClientTimeout(total=timeout + 10)
        last_error: Optional[Exception] = None
        async with self._request_semaphore:
            for attempt in range(max(1, self.request_retries + 1)):
                try:
                    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
                        async with session.post(self.sandbox_url, json=payload) as resp:
                            resp.raise_for_status()
                            result = await resp.json()
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if attempt >= self.request_retries:
                        raise
                    await asyncio.sleep(min(1.0 * (attempt + 1), 3.0))
            else:
                raise RuntimeError(f"sandbox request failed after retries: {last_error}")
        run_result = result.get("run_result", {})
        stdout = self._truncate_text(str(run_result.get("stdout", "")), self.max_output_chars)
        stderr = self._truncate_text(str(run_result.get("stderr", "")), self.max_output_chars)
        status = str(result.get("status", ""))
        text = stdout + stderr
        if not text and status != "success":
            text = f"sandbox status: {status}"
        return ToolResponse(text=text)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        code = parameters.get("code", "")
        timeout = self._parse_timeout(parameters.get("timeout", self.default_timeout), self.default_timeout)
        language = str(parameters.get("language", self.default_language)).strip().lower()

        if language not in self.allowed_languages:
            return ToolResponse(text=f"sandbox blocked unsupported language: {language}"), 0.0, {}

        if self.safety_mode == "strict":
            timeout = min(timeout, self.strict_max_timeout)

        if not isinstance(code, str):
            code = str(code)
        code = code[: self.max_code_chars]

        match = self.code_pattern.search(code)
        if match:
            code = match.group(1).strip()

        if self.block_dangerous_code:
            for pattern, reason in self._dangerous_code_patterns:
                if pattern.search(code):
                    return ToolResponse(text=f"sandbox blocked potentially dangerous code: {reason}"), 0.0, {}

        try:
            tool_response = await self._run_code(code=code, timeout=timeout, language=language)
        except Exception as exc:  # noqa: BLE001
            tool_response = ToolResponse(text=f"sandbox request failed: {exc}")

        return tool_response, 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
