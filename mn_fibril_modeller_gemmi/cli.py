from __future__ import annotations

import argparse
import importlib.util
import socket
import subprocess
import sys


def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _next_free_port(start_port: int) -> int:
    port = start_port
    while _is_port_in_use(port):
        port += 1
    return port


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app-gemmi",
        description="Run the experimental Gemmi-first MN fibril modeller app.",
    )
    parser.add_argument("--port", type=int, default=8502, help="Preferred starting port.")
    parser.add_argument("--headless", action="store_true", help="Run Streamlit without opening a browser.")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host to bind to. Defaults to 0.0.0.0.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    resolved_port = _next_free_port(args.port)
    module_spec = importlib.util.find_spec("mn_fibril_modeller_gemmi.app")
    if module_spec is None or module_spec.origin is None:
        raise RuntimeError("Could not locate the Gemmi app module.")

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        module_spec.origin,
        "--server.port",
        str(resolved_port),
        "--server.address",
        args.host,
        "--browser.gatherUsageStats",
        "false",
        "--server.headless",
        "true" if args.headless else "false",
    ]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
