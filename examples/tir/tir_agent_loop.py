"""
TIR (Tool-Integrated Reasoning) agent loop for OpenKimi PMD training.

Extends verl's ToolAgentLoop with three additions ported from the agentic repo:

1. System-prompt injection: automatically prepends a TIR-style instruction that
   asks the model to use ```python ... ``` code blocks and conclude with \\boxed{}.
2. Sandbox tracking: counts code_interpreter calls, successes, and code lines.
3. void_turn_mask: marks each rollout as "active" (1) if the model called the
   sandbox at least once OR produced a boxed answer, or "void" (0) otherwise.
   The trainer reads this field to optionally zero out the policy gradient for
   void responses (mask_void_turns).

Registration name: "tir_agent"
    actor_rollout_ref.rollout.agent.default_agent_loop=tir_agent
"""

import re
from typing import Any

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopOutput,
    DictConfigWrap,
    AsyncLLMServerManager,
    register,
)
from verl.experimental.agent_loop.tool_agent_loop import AgentData, ToolAgentLoop, AgentState
from verl.tools.schemas import ToolResponse
from transformers import AutoProcessor, AutoTokenizer


DEFAULT_TIR_SYSTEM_PROMPT = (
    "Solve the following problem step by step. You can write executable Python code to help your reasoning. "
    "The code will be executed by an external sandbox, and execution output will be returned as observation text "
    "starting with 'Code execution result:'. Use that result in your next step.\n\n"
    "Code format:\n"
    "1) When you want to run code, output exactly one fenced code block per assistant turn.\n"
    "   If multiple code blocks are produced, only the first tool call may be executed.\n"
    "2) Use ```python ... ``` format.\n"
    "3) Write complete code with required imports.\n"
    "4) Use print() for key intermediate values and final computed result.\n\n"
    "Answer format:\n"
    "1) If you are ready to finish, output the final answer in boxed format.\n"
    "2) The final line must be: \\boxed{your_final_answer}\n"
    "3) Do not add extra text after the final \\boxed{...}."
)


class TIRAgentData(AgentData):
    """AgentData extended with sandbox-tracking and void-turn fields."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sandbox_call_count: int = 0
        self.sandbox_success_count: int = 0
        self.sandbox_code_lines_total: int = 0
        self.sandbox_code_lines_success_total: int = 0
        self.has_boxed_answer: bool = False


@register("tir_agent")
class TIRAgentLoop(ToolAgentLoop):
    """ToolAgentLoop with TIR system-prompt injection, sandbox tracking, and void_turn_mask.

    Drop-in replacement for tool_agent in OpenKimi TIR training:
        actor_rollout_ref.rollout.agent.default_agent_loop=tir_agent
    """

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)

    # ------------------------------------------------------------------ #
    #  run() – injects TIR system prompt and uses TIRAgentData            #
    # ------------------------------------------------------------------ #

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        from uuid import uuid4

        messages = list(kwargs["raw_prompt"])

        # Inject TIR system prompt if not already present.
        if not messages:
            messages = [{"role": "system", "content": DEFAULT_TIR_SYSTEM_PROMPT}]
        elif messages[0].get("role") == "system":
            if DEFAULT_TIR_SYSTEM_PROMPT not in messages[0].get("content", ""):
                messages[0]["content"] = DEFAULT_TIR_SYSTEM_PROMPT + "\n\n" + messages[0]["content"]
        else:
            messages = [{"role": "system", "content": DEFAULT_TIR_SYSTEM_PROMPT}, *messages]

        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics: dict[str, Any] = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        interaction = None
        interaction_kwargs: dict[str, Any] = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            interaction_name = interaction_kwargs["name"]
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)

        agent_data = TIRAgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )

        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                state = AgentState.TERMINATED

        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask):]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_out: dict[str, Any] = {}
        if agent_data.image_data is not None:
            multi_modal_out["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_out["videos"] = agent_data.video_data

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_out,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=getattr(agent_data, "routed_experts", None),
            extra_fields={},
        )

        # void_turn_mask: 1 if model used sandbox >=1 time OR wrote a boxed answer.
        void_turn_mask = float(
            (agent_data.sandbox_call_count > 0) or agent_data.has_boxed_answer
        )
        output.extra_fields.update(
            {
                "turn_scores": agent_data.turn_scores,
                "tool_rewards": agent_data.tool_rewards,
                "sandbox_called": int(agent_data.sandbox_call_count > 0),
                "sandbox_success": int(agent_data.sandbox_success_count > 0),
                "sandbox_call_count": agent_data.sandbox_call_count,
                "sandbox_success_count": agent_data.sandbox_success_count,
                "sandbox_code_lines_total": agent_data.sandbox_code_lines_total,
                "sandbox_code_lines_success_total": agent_data.sandbox_code_lines_success_total,
                "void_turn_mask": void_turn_mask,
            }
        )
        return output

    # ------------------------------------------------------------------ #
    #  _handle_generating_state – capture content text for boxed detection #
    # ------------------------------------------------------------------ #

    async def _handle_generating_state(
        self, agent_data: TIRAgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        from verl.utils.profiler import simple_timer

        add_messages: list[dict[str, Any]] = []

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )

        num_preempted = getattr(output, "num_preempted", None)
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = num_preempted if num_preempted is not None else -1
        else:
            agent_data.metrics["num_preempted"] += num_preempted if num_preempted is not None else 0

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Capture content text to detect boxed answer.
        content_text, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)
        if self._contains_boxed_answer(content_text):
            agent_data.has_boxed_answer = True

        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED

    # ------------------------------------------------------------------ #
    #  _handle_processing_tools_state – sandbox call tracking             #
    # ------------------------------------------------------------------ #

    async def _handle_processing_tools_state(self, agent_data: TIRAgentData) -> AgentState:
        import asyncio
        from verl.utils.profiler import simple_timer

        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []

        tasks = []
        tool_call_names = []
        selected_tool_calls = agent_data.tool_calls[: self.max_parallel_calls]
        for tool_call in selected_tool_calls:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        for tool_call, (tool_response, tool_reward, tool_metrics) in zip(selected_tool_calls, responses, strict=False):
            # Track sandbox calls and successes.
            if self._is_sandbox_tool(tool_call.name):
                agent_data.sandbox_call_count += 1
                code_lines = self._extract_code_line_count(tool_call)
                agent_data.sandbox_code_lines_total += code_lines
                if self._is_sandbox_success(tool_response, tool_metrics):
                    agent_data.sandbox_success_count += 1
                    agent_data.sandbox_code_lines_success_total += code_lines

            if tool_response.image or tool_response.video:
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError("Multimodal tool response requires a VLM processor.")
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            if tool_response.image:
                imgs = tool_response.image if isinstance(tool_response.image, list) else [tool_response.image]
                new_images_this_turn.extend(img for img in imgs if img is not None)

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        agent_data.messages.extend(add_messages)

        if self.tool_parser_name == "gpt-oss":
            from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            response_ids = await self.apply_chat_template(
                add_messages,
                images=new_images_this_turn,
                videos=None,
                remove_system_prompt=True,
            )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            agent_data.image_data.extend(new_images_this_turn)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    # ------------------------------------------------------------------ #
    #  Static helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_sandbox_tool(name: str) -> bool:
        return name == "code_interpreter"

    @staticmethod
    def _is_sandbox_success(tool_response: ToolResponse, tool_metrics: dict[str, Any]) -> bool:
        if isinstance(tool_metrics, dict) and "sandbox_success" in tool_metrics:
            return bool(tool_metrics["sandbox_success"])
        text = (tool_response.text or "").lower()
        if "sandbox request failed" in text:
            return False
        if "sandbox status:" in text and "success" not in text:
            return False
        return True

    @staticmethod
    def _contains_boxed_answer(text: str) -> bool:
        return bool(re.search(r"\\boxed\s*\{", text or ""))

    @staticmethod
    def _extract_code_line_count(tool_call) -> int:
        try:
            import json
            params = json.loads(tool_call.arguments) if isinstance(tool_call.arguments, str) else tool_call.arguments
            code = params.get("code", "")
        except Exception:
            return 0
        if not isinstance(code, str):
            code = str(code)
        m = re.search(r"```(?:py|python)?\n(.*?)\n```", code, re.DOTALL)
        if m:
            code = m.group(1)
        return sum(1 for ln in code.splitlines() if ln.strip())
