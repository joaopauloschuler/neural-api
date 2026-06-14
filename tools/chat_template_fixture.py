#!/usr/bin/env python3
"""Generates tests/fixtures/chat_template_cases.json: ground-truth renders
of the well-known HuggingFace chat templates that neural/neuralchat.pas
hardcodes (ChatML, Llama-2, Llama-3, Zephyr/TinyLlama, Gemma, Phi-3,
Mistral-Instruct).

The Jinja template STRINGS below are the authentic `chat_template` values
published in the corresponding tokenizer_config.json files (Qwen2 generic
ChatML, meta-llama/Llama-2-7b-chat-hf, meta-llama/Meta-Llama-3-8B-Instruct,
HuggingFaceH4/zephyr-7b-beta == TinyLlama-1.1B-Chat-v1.0,
google/gemma-7b-it, microsoft/Phi-3-mini-4k-instruct,
mistralai/Mistral-7B-Instruct-v0.1). They are rendered here with the SAME
Jinja environment transformers uses for apply_chat_template
(trim_blocks=True, lstrip_blocks=True, raise_exception available), so the
expected strings are exactly what `tokenizer.apply_chat_template(...,
tokenize=False)` produces.

Run from the repo root with the shared interpreter:
    /home/bpsa/x/bin/python tools/chat_template_fixture.py
"""

import json
import os

from jinja2.sandbox import ImmutableSandboxedEnvironment

# (name, template, bos_token, eos_token) -- template strings verbatim from
# the published tokenizer_config.json files.
TEMPLATES = {
    "chatml": {
        # Qwen2/Qwen2.5/Yi/ChatML generic turn format (the Qwen variants
        # only add a default system message, which callers pass explicitly
        # here).
        "template": (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content']"
            " + '<|im_end|>' + '\n'}}{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}{% endif %}"
        ),
        "bos": "",
        "eos": "<|im_end|>",
    },
    "llama2": {
        # meta-llama/Llama-2-7b-chat-hf
        "template": (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set loop_messages = messages[1:] %}"
            "{% set system_message = messages[0]['content'] %}"
            "{% else %}{% set loop_messages = messages %}"
            "{% set system_message = false %}{% endif %}"
            "{% for message in loop_messages %}"
            "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/"
            "assistant/user/assistant/...') }}{% endif %}"
            "{% if loop.index0 == 0 and system_message != false %}"
            "{% set content = '<<SYS>>\n' + system_message + '\n<</SYS>>"
            "\n\n' + message['content'] %}"
            "{% else %}{% set content = message['content'] %}{% endif %}"
            "{% if message['role'] == 'user' %}"
            "{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ ' ' + content.strip() + ' ' + eos_token }}"
            "{% endif %}{% endfor %}"
        ),
        "bos": "<s>",
        "eos": "</s>",
    },
    "llama3": {
        # meta-llama/Meta-Llama-3-8B-Instruct
        "template": (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
            "{% set content = '<|start_header_id|>' + message['role'] +"
            " '<|end_header_id|>\n\n'+ message['content'] | trim +"
            " '<|eot_id|>' %}"
            "{% if loop.index0 == 0 %}"
            "{% set content = bos_token + content %}{% endif %}"
            "{{ content }}{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
            "{% endif %}"
        ),
        "bos": "<|begin_of_text|>",
        "eos": "<|eot_id|>",
    },
    "zephyr": {
        # HuggingFaceH4/zephyr-7b-beta == TinyLlama/TinyLlama-1.1B-Chat-v1.0
        "template": (
            "{% for message in messages %}\n"
            "{% if message['role'] == 'user' %}\n"
            "{{ '<|user|>\n' + message['content'] + eos_token }}\n"
            "{% elif message['role'] == 'system' %}\n"
            "{{ '<|system|>\n' + message['content'] + eos_token }}\n"
            "{% elif message['role'] == 'assistant' %}\n"
            "{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n"
            "{% endif %}\n"
            "{% if loop.last and add_generation_prompt %}\n"
            "{{ '<|assistant|>' }}\n"
            "{% endif %}\n{% endfor %}"
        ),
        "bos": "<s>",
        "eos": "</s>",
    },
    "gemma": {
        # google/gemma-7b-it (gemma-2/gemma-3 instruct ship the same turn
        # format); NOTE: raises on a system role, like HF.
        "template": (
            "{{ bos_token }}{% if messages[0]['role'] == 'system' %}"
            "{{ raise_exception('System role not supported') }}{% endif %}"
            "{% for message in messages %}"
            "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/"
            "assistant/user/assistant/...') }}{% endif %}"
            "{% if (message['role'] == 'assistant') %}"
            "{% set role = 'model' %}"
            "{% else %}{% set role = message['role'] %}{% endif %}"
            "{{ '<start_of_turn>' + role + '\n' + message['content'] | trim"
            " + '<end_of_turn>\n' }}{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{'<start_of_turn>model\n'}}{% endif %}"
        ),
        "bos": "<bos>",
        "eos": "<eos>",
    },
    "phi3": {
        # microsoft/Phi-3-mini-4k-instruct
        "template": (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{'<|system|>\n' + message['content'] + '<|end|>\n'}}"
            "{% elif message['role'] == 'user' %}"
            "{{'<|user|>\n' + message['content'] + '<|end|>\n'}}"
            "{% elif message['role'] == 'assistant' %}"
            "{{'<|assistant|>\n' + message['content'] + '<|end|>\n'}}"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}"
            "{% else %}{{ eos_token }}{% endif %}"
        ),
        "bos": "<s>",
        "eos": "<|endoftext|>",
    },
    "mistral": {
        # mistralai/Mistral-7B-Instruct-v0.1; raises on a system role.
        "template": (
            "{{ bos_token }}{% for message in messages %}"
            "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/"
            "assistant/user/assistant/...') }}{% endif %}"
            "{% if message['role'] == 'user' %}"
            "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + eos_token + ' ' }}"
            "{% else %}{{ raise_exception('Only user and assistant roles"
            " are supported!') }}{% endif %}{% endfor %}"
        ),
        "bos": "<s>",
        "eos": "</s>",
    },
    "deepseek": {
        # deepseek-ai/DeepSeek-V2-Chat / DeepSeek-V3 turn format. The bos
        # token / end-of-sentence token use the fullwidth pipe U+FF5C and the
        # one-eighth block U+2581 ('<｜begin▁of▁sentence｜>'
        # and '<｜end▁of▁sentence｜>'). System content is
        # emitted verbatim (no role tag); the generation prompt is the bare
        # 'Assistant:' continuation. No content strip.
        "template": (
            "{% if not add_generation_prompt is defined %}"
            "{% set add_generation_prompt = false %}{% endif %}"
            "{{ bos_token }}{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ 'User: ' + message['content'] + '\n\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ 'Assistant: ' + message['content'] + eos_token }}"
            "{% elif message['role'] == 'system' %}"
            "{{ message['content'] + '\n\n' }}{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
        ),
        "bos": "<｜begin▁of▁sentence｜>",
        "eos": "<｜end▁of▁sentence｜>",
    },
    "phi4mini": {
        # microsoft/Phi-4-mini-instruct ChatML-style tool-aware template;
        # like Phi-3 but the <|...|> tags carry NO trailing newline and there
        # is no eos fallback when add_generation_prompt is false.
        "template": (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{'<|system|>' + message['content'] + '<|end|>'}}"
            "{% elif message['role'] == 'user' %}"
            "{{'<|user|>' + message['content'] + '<|end|>'}}"
            "{% elif message['role'] == 'assistant' %}"
            "{{'<|assistant|>' + message['content'] + '<|end|>'}}"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}{{'<|assistant|>'}}{% endif %}"
        ),
        "bos": "<s>",
        "eos": "<|endoftext|>",
    },
}

# (messages, add_generation_prompt) conversations; every (format x case)
# pair is rendered unless the template raises, which is pinned as
# {"raises": true}.
CONVERSATIONS = [
    # single user turn + generation prompt
    ([{"role": "user", "content": "Hello, how are you?"}], True),
    # system + user + generation prompt
    ([{"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}], True),
    # multi-turn ending on user + generation prompt
    ([{"role": "user", "content": "Hi!"},
      {"role": "assistant", "content": "Hello! How can I help?"},
      {"role": "user", "content": "Tell me a joke."}], True),
    # system + full exchange, no generation prompt
    ([{"role": "system", "content": "Answer briefly."},
      {"role": "user", "content": "2+2?"},
      {"role": "assistant", "content": "4"}], False),
    # single user turn, no generation prompt
    ([{"role": "user", "content": "Ping"}], False),
    # content with whitespace padding (exercises strip/trim) + UTF-8
    ([{"role": "user", "content": "  Café olé?  "},
      {"role": "assistant", "content": " Of course! "},
      {"role": "user", "content": "\tthanks\n"}], True),
    # roles out of order (alternation-checking templates raise)
    ([{"role": "assistant", "content": "I speak first."},
      {"role": "user", "content": "odd"}], True),
]


def jinja_env():
    def raise_exception(message):
        raise ValueError(message)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True)  # transformers' settings
    env.globals["raise_exception"] = raise_exception
    return env


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    out_path = os.path.join(root, "tests", "fixtures",
                            "chat_template_cases.json")
    env = jinja_env()
    out = {}
    for name, spec in TEMPLATES.items():
        compiled = env.from_string(spec["template"])
        cases = []
        for messages, add_gen in CONVERSATIONS:
            case = {"messages": messages, "add_generation_prompt": add_gen}
            try:
                case["expected"] = compiled.render(
                    messages=messages, add_generation_prompt=add_gen,
                    bos_token=spec["bos"], eos_token=spec["eos"])
            except ValueError as exc:
                case["raises"] = True
                case["error"] = str(exc)
            cases.append(case)
        out[name] = {"template": spec["template"], "cases": cases}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    total = sum(len(v["cases"]) for v in out.values())
    print("wrote %s (%d formats, %d cases)" % (out_path, len(out), total))


if __name__ == "__main__":
    main()
