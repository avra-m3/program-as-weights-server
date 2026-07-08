"""CLI: `paw-local compile` / `paw-local run`."""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="paw-local", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("compile", help="Compile a spec into a local PAW program")
    c.add_argument("spec", help="Spec text, or a path to a text file with @path")
    c.add_argument("-o", "--out", required=True, help="Output program directory")
    c.add_argument(
        "--pseudo-style",
        choices=["examples", "minimal"],
        default="examples",
        help="Pseudo-compiler prompt style (default: examples, as in the paper)",
    )
    c.add_argument("--max-new-tokens", type=int, default=512)
    c.add_argument("--gguf", action="store_true", help="Also write adapter.gguf")
    c.add_argument("--device", default=None)

    r = sub.add_parser("run", help="Run a compiled program on an input")
    r.add_argument("program_dir")
    r.add_argument("input", help="Task input text, or @path to a text file")
    r.add_argument("--max-new-tokens", type=int, default=512)
    r.add_argument("--device", default=None)

    args = parser.parse_args()

    if args.command == "compile":
        from paw_local.pipeline import compile_spec

        spec = _read_arg(args.spec)
        compile_spec(
            spec,
            args.out,
            pseudo_style=args.pseudo_style,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            write_gguf=args.gguf,
        )
    elif args.command == "run":
        from paw_local.interpreter import run_program

        task_input = _read_arg(args.input)
        result = run_program(
            args.program_dir,
            task_input,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
        print(result)


def _read_arg(value: str) -> str:
    if value.startswith("@"):
        return Path(value[1:]).read_text().strip()
    return value


if __name__ == "__main__":
    sys.exit(main())
