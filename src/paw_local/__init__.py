"""Local (self-hosted) PAW compiler.

Reimplements the ProgramAsWeights compile pipeline from the published
compiler weights (HF: programasweights/paw-4b-qwen3-0.6b), so a
natural-language spec can be compiled into a LoRA "program" for the
Qwen/Qwen3-0.6B interpreter without the hosted API.

Architecture and conventions were validated against:
- the paper (arXiv:2607.02512, §3.2 / Appendix C / Appendix G)
- the authors' research snapshot (anonymous.4open.science/r/programasweights,
  in particular the *_generate_lora.py training script and the
  *_vllm_lora.py eval script, which this module mirrors)
- a production program artifact (HF: programasweights/paw-programs)
"""

from paw_local.interpreter import run_program
from paw_local.pipeline import compile_spec

__all__ = ["compile_spec", "run_program"]
