"""Prompt templates, verbatim from the PAW research code (Appendix C).

Attribution: the prompt text in this module is reproduced verbatim from the
Program-as-Weights paper (arXiv:2607.02512) and the authors' research code.
It is (c) its original authors and licensed under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/), included here with attribution.
This text is NOT covered by this repository's MIT license.

Two compiler styles exist:
- "examples" is the template used with the *off-the-shelf*
  Qwen3-4B-Instruct-2507 pseudo compiler to generate pseudo-programs
  (both for the training cache and at inference; the trained compiler
  cannot generate — its LM ability collapsed during training).
- "minimal" is the prompt the *trained* compiler consumes
  (meta.json: compiler_prompt_style="minimal") in the prefix-hidden
  forward pass that feeds the LoRA mapper.
"""

COMPILER_MINIMAL = """[SPEC]
{spec}
[END_SPEC]

[PSEUDO_PROGRAM]"""

COMPILER_EXAMPLES = """You are PAW-Compiler. Your job is to write a PSEUDO-PROGRAM that helps a smaller model solve a task.

CRITICAL: The interpreter will NOT see the original SPEC. Your pseudo-program is the ONLY instruction it gets.

Your pseudo-program should contain:
1. A clear, concise description of the task (what to do, edge cases, output format)
2. 3-6 example input/output pairs that demonstrate the task

Format (MUST follow exactly):
[PSEUDO_PROGRAM]
Task: <one paragraph describing what to do, including edge cases and output format>

Examples:
Input: <example input 1>
Output: <example output 1>

Input: <example input 2>
Output: <example output 2>

... (more examples as needed)
[END_PSEUDO_PROGRAM]

Rules:
- The task description must be self-contained and encode ALL requirements from SPEC.
- Examples should cover typical cases AND edge cases mentioned in SPEC.
- Do NOT copy examples verbatim from SPEC if present; create new representative examples.
- Keep total length under 250 tokens.
- Always include the closing marker [END_PSEUDO_PROGRAM].

Now write a pseudo-program for this specification:
[SPEC]
{spec}
[END_SPEC]"""

INTERPRETER_MINIMAL = """{pseudo_program}

[INPUT]
{task_input}
[END_INPUT]"""


def compiler_instructions() -> str:
    """The static "examples"-style instructions, spec placeholder unfilled.

    This is verbatim what the untrained pseudo compiler (Qwen3-4B-
    Instruct-2507) is normally told to do when writing a pseudo-program --
    i.e. the text GET /api/v1/compile/instructions exposes so a caller can
    see exactly what they'd be replacing by using POST /api/v1/compile/raw.
    """
    return COMPILER_EXAMPLES


def compiler_prompt(spec: str, style: str = "minimal") -> str:
    if style == "minimal":
        return COMPILER_MINIMAL.format(spec=spec).strip()
    if style == "examples":
        return COMPILER_EXAMPLES.format(spec=spec).strip()
    raise ValueError(f"unknown compiler prompt style: {style}")


def interpreter_prompt(pseudo_program: str, task_input: str) -> str:
    return INTERPRETER_MINIMAL.format(
        pseudo_program=pseudo_program.strip(), task_input=task_input
    ).strip()
