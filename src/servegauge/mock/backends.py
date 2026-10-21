"""Seeded distributions, exact replay and reservation-required proxy delegation."""

from __future__ import annotations

import copy
import json
import math
import random
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.models import Exchange
from servegauge.mock.faults import FaultConfig
from servegauge.mock.math_tasks import math_answer
from servegauge.storage.raw_store import RawStore
from servegauge.tokencount.counters import (
    decode,
    encode,
    encoding,
    mock_prompt_tokens,
    text_tokens,
)


def task_from_messages(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for message in reversed(messages):
        text = str(message.get("content") or "")
        marker = text.rfind("TW_TASK:")
        if marker >= 0:
            try:
                value, _ = json.JSONDecoder().raw_decode(text[marker + 8 :].lstrip())
                if isinstance(value, dict):
                    return value
            except ValueError:
                pass
    return {}


class ScriptedBackend:
    """A small stochastic interpreter of explicit synthetic tasks, not a language model."""

    def __init__(
        self,
        seed: int,
        faults: FaultConfig,
        model_control: str | None = None,
        context_length: int = 131072,
        *,
        sampling_version: str = "v1",
        healthy_logit_jitter: float = 0.0,
    ) -> None:
        if (
            sampling_version not in {"v1", "v2"}
            or not math.isfinite(healthy_logit_jitter)
            or healthy_logit_jitter < 0
            or (sampling_version == "v1" and healthy_logit_jitter != 0)
        ):
            raise ValueError("known_sampling_version_and_nonnegative_v2_jitter_required")
        self.sampling_version = sampling_version
        self.healthy_logit_jitter = healthy_logit_jitter
        self.sample_distribution: dict[str, float] = {}
        self.seed = faults.seed if faults.seed is not None else seed
        self.rng = random.Random(self.seed)
        self.faults = faults
        self.model_control = model_control
        self.cache: dict[str, dict[str, Any]] = {}
        self.sequence = 0
        self.context_length = context_length

    def _sample(
        self, choices: list[str], body: dict[str, Any], other: bool, rng: random.Random
    ) -> str:
        temperature = 1.0 if self.faults.ignore_temperature else float(body.get("temperature", 1))
        top_p = 1.0 if self.faults.ignore_top_p else float(body.get("top_p", 1))
        weights = [math.exp(-i * 0.65) for i in range(len(choices))]
        if other:
            weights.reverse()
            if self.model_control == "control-b":
                weights = weights[1:] + weights[:1]
        if self.faults.logit_noise:
            weights = [w * math.exp(rng.gauss(0, self.faults.logit_noise)) for w in weights]
        if self.healthy_logit_jitter:
            weights = [w * math.exp(rng.gauss(0, self.healthy_logit_jitter)) for w in weights]
        if temperature <= 0:
            weights = [0.002 / max(len(choices) - 1, 1)] * len(choices)
            weights[-1 if other else 0] = 0.998
        else:
            weights = [w ** (1 / temperature) for w in weights]
        ranked = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)
        selected: list[int] = []
        cumulative = 0.0
        total = sum(weights)
        top_k = int(body.get("top_k", len(weights)))
        for index in ranked[: max(top_k, 1)]:
            selected.append(index)
            cumulative += weights[index] / total
            if cumulative >= top_p:
                break
        if self.sampling_version == "v2":
            if len(set(choices)) != len(choices) or (
                body.get("logprobs") and any(len(encode(s)) != 1 for s in choices)
            ):
                raise ValueError("sampling_v2_requires_distinct_single_token_choices")
            mass = sum(weights[i] for i in selected)
            self.sample_distribution = {choices[i]: weights[i] / mass for i in selected}
        return rng.choices([choices[i] for i in selected], [weights[i] for i in selected], k=1)[0]

    def _message(
        self, body: dict[str, Any], other: bool, rng: random.Random
    ) -> tuple[dict[str, Any], str]:
        messages = body["messages"]
        task = task_from_messages(messages)
        kind = task.get("type", "sample")
        message: dict[str, Any] = {"role": "assistant", "content": None}
        tools = body.get("tools") or []
        should_call = bool(tools) and kind not in {
            "no_call",
            "echo",
            "retrieve",
            "sequence",
            "json",
            "math",
            "context",
        }
        calls: list[dict[str, Any]] = []
        if should_call:
            choice = body.get("tool_choice", "auto")
            if choice == "none":
                should_call = False
            elif isinstance(choice, dict):
                task = {**task, "name": choice.get("function", {}).get("name")}
        after_tool = any(m.get("role") == "tool" for m in messages)
        if after_tool and rng.random() < self.faults.plain_text_after_tool_result:
            should_call = False
        if kind == "agent":
            steps = task.get("steps", [])
            index = sum(m.get("role") == "tool" for m in messages)
            if index < len(steps):
                task = steps[index]
            else:
                should_call = False
                kind = "echo"
                task = {"text": "complete"}
        if should_call:
            specs = task.get("calls", [task])
            if body.get("parallel_tool_calls") is False:
                specs = specs[:1]
            for index, spec in enumerate(specs):
                name = spec.get("name", tools[0]["function"]["name"])
                arguments = spec.get("arguments", {})
                text = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
                if rng.random() < self.faults.corrupt_arguments:
                    text = text[:-1] + ","
                calls.append(
                    {
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": text,
                        },
                    }
                )
            message["tool_calls"] = calls
            if rng.random() < self.faults.drop_tool_calls:
                message["tool_calls"] = []
                message["content"] = json.dumps({"tool_calls": calls}, ensure_ascii=False)
            finish = "tool_calls"
        else:
            finish = "stop"
            if kind == "echo":
                content = str(task.get("text", "ok"))
            elif kind == "retrieve":
                text = "\n".join(str(m.get("content") or "") for m in messages)
                found = {}
                for key in task.get("keys", []):
                    match = re.search(rf"(?m)^{re.escape(key)}\s*=\s*([^\r\n]+)", text)
                    found[key] = match[1] if match else None
                content = json.dumps(found, ensure_ascii=False)
            elif kind == "sequence":
                start = int(task.get("start", 0))
                count = int(task.get("count", 10))
                content = json.dumps(
                    [{"i": i, "square": i * i} for i in range(start, start + count)],
                    separators=(",", ":"),
                )
            elif kind == "long_output":
                from servegauge.items.longout import render_output

                content = render_output(int(task["start"]), int(task["count"]), task["kind"])
            elif kind == "context":
                source = "\n".join(str(message.get("content") or "") for message in messages)
                context_values: dict[str, Any] = {}
                for field, key in task["keys"].items():
                    matches = re.findall(rf"(?m)^{re.escape(key)}=(\S+)", source)
                    context_values[field] = (
                        (matches[-1] if field == "track" else matches[0]) if matches else None
                    )
                context_values["total"] = sum(
                    int(value)
                    for value in re.findall(
                        rf"(?m)^{re.escape(task['aggregate_prefix'])}\d+=(\d+)", source
                    )
                )
                content = json.dumps(context_values, ensure_ascii=False)
            elif kind == "json" or body.get("response_format"):
                content = json.dumps(task.get("value", {"answer": "ok"}), ensure_ascii=False)
            elif kind == "math":
                result = math_answer(task)
                if other and rng.random() < 0.15:
                    result += 1
                content = str(result)
            else:
                choices = task.get("choices", ["red", "blue", "green", "gold"])
                content = self._sample(choices, body, other, rng)
            message["content"] = content
        if rng.random() < self.faults.leak_markers:
            message["content"] = str(message["content"] or "") + "<tool_call>"
        return message, finish

    def respond(self, original: dict[str, Any]) -> dict[str, Any]:
        self.sequence += 1
        self.sample_distribution = {}
        # Ordinary scripted generation only reads its request. A copy is necessary
        # for the explicit prompt mutation, not for every repeated draw in the lab.
        body = copy.deepcopy(original) if self.faults.truncate_prompt_head_over else original
        messages = body["messages"]
        if self.faults.truncate_prompt_head_over:
            limit = self.faults.truncate_prompt_head_over
            for message in messages:
                tokens = encode(str(message.get("content") or ""))
                if len(tokens) > limit:
                    message["content"] = decode(tokens[-limit:])
        if mock_prompt_tokens(body) > self.context_length:
            return {"error": {"code": "context_length_exceeded"}}
        key = content_hash(body.get("messages"))
        if key in self.cache and self.rng.random() < self.faults.stale_cache:
            return copy.deepcopy(self.cache[key])
        rng = random.Random(int(body["seed"]) + self.seed) if "seed" in body else self.rng
        other = self.model_control is not None or rng.random() < self.faults.route_to_other_model
        message, finish = self._message(body, other, rng)
        reasoning = body.get("reasoning", {})
        effort = reasoning.get("effort", body.get("reasoning_effort"))
        if self.faults.ignore_reasoning_effort and effort:
            effort = "medium"
        reason_n = 0
        if effort:
            base = {"low": 48, "medium": 128, "high": 384}.get(str(effort), 48)
            reason_n = max(1, round(base * rng.uniform(0.9, 1.1)))
            message["reasoning"] = " step" * reason_n
            message["reasoning_details"] = [
                {"type": "reasoning.text", "text": message["reasoning"], "index": 0}
            ]
        text = str(message.get("content") or "")
        if not self.faults.ignore_stop:
            stops = body.get("stop", [])
            if isinstance(stops, str):
                stops = [stops]
            positions = [text.index(stop) for stop in stops if stop and stop in text]
            if positions:
                text = text[: min(positions)]
        tokens = encode(text)
        if self.faults.truncate_output_after and len(tokens) > self.faults.truncate_output_after:
            tokens = tokens[: self.faults.truncate_output_after]
            finish = "stop"
        cap = int(body.get("max_completion_tokens", body.get("max_tokens", 256)))
        if not self.faults.ignore_max_tokens and len(tokens) + reason_n > cap:
            if reason_n >= cap:
                reason_n = cap
                message["reasoning"] = " step" * reason_n
                message["reasoning_details"][0]["text"] = message["reasoning"]
            tokens = tokens[: max(cap - reason_n, 0)]
            finish = "length"
        message["content"] = decode(tokens) if text or tokens else None
        visible_n = len(tokens)
        if message.get("tool_calls"):
            visible_n += text_tokens(canonical_json(message["tool_calls"]).decode())
        total_n = visible_n + reason_n
        if not self.faults.ignore_max_tokens and total_n > cap:
            raw_call = canonical_json(message.get("tool_calls", [])).decode()
            partial = encode(raw_call)[: max(0, cap - reason_n)]
            message.pop("tool_calls", None)
            message["content"] = decode(partial)
            visible_n = len(partial)
            total_n = visible_n + reason_n
            finish = "length"
        usage: dict[str, Any] = {
            "prompt_tokens": mock_prompt_tokens(body),
            "completion_tokens": total_n,
            "total_tokens": mock_prompt_tokens(body) + total_n,
            "completion_tokens_details": {"reasoning_tokens": reason_n},
        }
        if self.faults.misreport_usage_pct:
            scale = 1 + self.faults.misreport_usage_pct / 100
            usage["prompt_tokens"] = round(int(usage["prompt_tokens"]) * scale)
            usage["completion_tokens"] = round(total_n * scale)
            usage["total_tokens"] = int(usage["prompt_tokens"]) + int(usage["completion_tokens"])
        choice: dict[str, Any] = {"index": 0, "message": message, "finish_reason": finish}
        if body.get("logprobs"):
            logs = []
            for token in tokens:
                raw = encoding().decode_single_token_bytes(token)
                lp = -0.02 if not other else -0.25
                logs.append(
                    {
                        "token": raw.decode("utf-8", "replace"),
                        "bytes": list(raw),
                        "logprob": lp,
                        "top_logprobs": [
                            {
                                "token": raw.decode("utf-8", "replace"),
                                "bytes": list(raw),
                                "logprob": lp,
                            }
                        ],
                    }
                )
            choice["logprobs"] = {"content": logs}
            if self.sampling_version == "v2":
                # Only the explicitly supported sampling task has a true token model.
                # Never attach its distribution to censored or unrelated output.
                if len(tokens) != 1 or message.get("content") not in self.sample_distribution:
                    raise ValueError("sampling_v2_logprobs_require_uncensored_sample_output")
                probability = self.sample_distribution[str(message["content"])]
                logs[0]["logprob"] = math.log(probability)
                logs[0]["top_logprobs"] = [
                    {"token": token, "bytes": list(token.encode("utf-8")), "logprob": math.log(p)}
                    for token, p in sorted(
                        self.sample_distribution.items(), key=lambda row: row[1], reverse=True
                    )[: int(body.get("top_logprobs", 1))]
                ]
        response = {
            "id": f"mock-{self.seed}-{self.sequence}",
            "object": "chat.completion",
            "model": body.get("model", "mock"),
            "choices": [choice],
            "usage": usage,
        }
        self.cache[key] = copy.deepcopy(response)
        return response


def replay_key(request: dict[str, Any]) -> str:
    return content_hash(
        {k: v for k, v in request.items() if k not in {"model", "stream", "stream_options"}}
    )


class ReplayBackend:
    def __init__(self, path: Path) -> None:
        if path.is_dir():
            records = list(RawStore(path).exchanges())
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
            records = [
                Exchange.model_validate(v) for v in (value if isinstance(value, list) else [value])
            ]
        self.records: dict[str, list[Exchange]] = {}
        self.indices: dict[str, int] = {}
        for record in records:
            self.records.setdefault(replay_key(record.request), []).append(record)

    def select(self, body: dict[str, Any]) -> Exchange:
        key = replay_key(body)
        if key not in self.records:
            raise ValueError("No matching replay request")
        records = self.records[key]
        index = self.indices.get(key, 0)
        self.indices[key] = index + 1
        return records[index % len(records)].model_copy(deep=True)

    def respond(self, body: dict[str, Any]) -> dict[str, Any]:
        return copy.deepcopy(self.select(body).response or {})


ReservedForward = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


class ProxyBackend:
    def __init__(
        self,
        url: str,
        reserved_forward: ReservedForward | None = None,
        faults: FaultConfig | None = None,
        seed: int = 1,
    ) -> None:
        self.url = url
        self.reserved_forward = reserved_forward
        self.faults = faults or FaultConfig()
        self.rng = random.Random(seed)

    async def respond(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.reserved_forward is None:
            raise RuntimeError("Proxy requires a reserved runner forwarding callback")
        request = copy.deepcopy(body)
        for flag, parameter in (
            (self.faults.ignore_temperature, "temperature"),
            (self.faults.ignore_top_p, "top_p"),
            (self.faults.ignore_stop, "stop"),
            (self.faults.ignore_max_tokens, "max_tokens"),
            (self.faults.ignore_max_tokens, "max_completion_tokens"),
        ):
            if flag:
                request.pop(parameter, None)
        if self.faults.ignore_reasoning_effort:
            if "reasoning" in request:
                request["reasoning"] = {**request["reasoning"], "effort": "medium"}
            if "reasoning_effort" in request:
                request["reasoning_effort"] = "medium"
        if self.faults.truncate_prompt_head_over:
            for message in request.get("messages", []):
                text = str(message.get("content") or "")
                message["content"] = decode(encode(text)[-self.faults.truncate_prompt_head_over :])
        # The callback reserves against the actual, already-mutated wire request.
        response = copy.deepcopy(await self.reserved_forward(self.url, request))
        for choice in response.get("choices", []):
            message = choice.get("message", {})
            if self.rng.random() < self.faults.drop_tool_calls:
                message["tool_calls"] = []
            if self.rng.random() < self.faults.leak_markers:
                message["content"] = str(message.get("content") or "") + "<tool_call>"
            for call in message.get("tool_calls") or []:
                if self.rng.random() < self.faults.corrupt_arguments:
                    call["function"]["arguments"] += ","
            if self.faults.truncate_output_after and message.get("content"):
                message["content"] = decode(
                    encode(message["content"])[: self.faults.truncate_output_after]
                )
        usage = response.get("usage", {})
        if self.faults.misreport_usage_pct:
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if field in usage:
                    usage[field] = round(usage[field] * (1 + self.faults.misreport_usage_pct / 100))
        return response
