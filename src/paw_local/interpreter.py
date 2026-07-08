"""Run a compiled PAW program locally on the Qwen3-0.6B interpreter.

Works on program directories produced by `paw_local.pipeline.compile_spec`
and on official artifacts from programasweights/paw-programs (both contain
adapter_config.json, adapter_model.safetensors and prompt_template.txt).
"""

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from paw_local.pipeline import pick_device

INPUT_PLACEHOLDER = "{INPUT_PLACEHOLDER}"


def load_program(program_dir: str | Path, device: str | None = None):
    """Load interpreter + adapter once; returns (model, tokenizer, template)."""
    program_dir = Path(program_dir)
    device = device or pick_device()

    meta = json.loads((program_dir / "meta.json").read_text())
    interpreter_id = meta.get("interpreter", "Qwen/Qwen3-0.6B")

    template = (program_dir / "prompt_template.txt").read_text()
    if INPUT_PLACEHOLDER not in template:
        raise ValueError(f"prompt_template.txt lacks {INPUT_PLACEHOLDER}")

    tokenizer = AutoTokenizer.from_pretrained(interpreter_id)
    base = AutoModelForCausalLM.from_pretrained(interpreter_id, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, str(program_dir))
    model.to(device).eval()
    return model, tokenizer, template


def generate(
    model, tokenizer, template: str, task_input: str, max_new_tokens: int = 512
) -> str:
    prompt = template.replace(INPUT_PLACEHOLDER, task_input)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_program(
    program_dir: str | Path,
    task_input: str,
    max_new_tokens: int = 512,
    device: str | None = None,
) -> str:
    model, tokenizer, template = load_program(program_dir, device=device)
    return generate(model, tokenizer, template, task_input, max_new_tokens)
