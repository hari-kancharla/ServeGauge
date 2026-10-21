"""Independent optional tokenizer oracle: public local data, no ServeGauge imports."""

from __future__ import annotations

import hashlib
import json
import socket
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any


def main() -> None:
    from transformers import TokenizersBackend

    def offline(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("model_b_reference_network_forbidden")

    socket.socket = offline
    assets, cases_path = [Path(value) for value in sys.argv[1:]]
    for path in (assets, cases_path):
        if any(part.casefold() == "starter files" for part in path.parts):
            raise ValueError("reference_path_refused")
    tokenizer = TokenizersBackend.from_pretrained(
        str(assets), local_files_only=True, trust_remote_code=False
    )
    corpus = json.loads(cases_path.read_bytes())
    result: dict[str, Any] = {
        "transformers": version("transformers"),
        "tokenizers": version("tokenizers"),
        "jinja2": version("jinja2"),
        "weights_loaded": False,
        "model_inference_calls": 0,
        "assets": {
            name: hashlib.sha256((assets / name).read_bytes()).hexdigest()
            for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "LICENSE",
            )
        },
        "texts": {},
        "decoded_texts": {},
        "chats": {},
    }
    for text in corpus["texts"]:
        result["texts"][text] = tokenizer.encode(text, add_special_tokens=False)
        result["decoded_texts"][text] = tokenizer.decode(
            result["texts"][text], skip_special_tokens=False
        )
    for case in corpus["chats"]:
        controls = case.get("controls", {})
        rendered = tokenizer.apply_chat_template(
            case["native_messages"],
            tools=case.get("tools"),
            tokenize=False,
            add_generation_prompt=True,
            **controls,
        )
        result["chats"][case["name"]] = {
            "render_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "token_ids": tokenizer.apply_chat_template(
                case["native_messages"],
                tools=case.get("tools"),
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                **controls,
            ),
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
