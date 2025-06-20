from __future__ import annotations
import argparse
from multiprocessing import managers

from actors.utils.logger import Palette, colorize, logger
from actors.server.pool import ModelPool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--auth", default="secret")
    args = parser.parse_args()

    class ManagerSubclass(managers.BaseManager): ...
    ManagerSubclass.register(
        "ModelPool",
        ModelPool,
        exposed=[
            "list_models",
            "print_models",
            "load_model",
            "unload_model",
            "sleep",
            "wake",
            "update_weights",
            "generate",
            "chat",
        ],
    )

    manager = ManagerSubclass(address=(args.host, args.port), authkey=args.auth.encode())
    logger.info(colorize(f"🚀  Listening {args.host}:{args.port}", Palette.INFO))
    manager.get_server().serve_forever()


if __name__ == "__main__":
    main()
