"""CLI: `paw-server` — run the local PAW compile server."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="paw-server", description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument(
        "--data-dir",
        default="data/server",
        help="Where programs and the registry are stored (default: data/server)",
    )
    args = parser.parse_args()

    import uvicorn

    from paw_server.app import create_app

    app = create_app(Path(args.data_dir))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
