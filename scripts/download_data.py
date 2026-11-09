"""Download the public AstroCLIMB competition release from Kaggle."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw", help="Destination directory")
    parser.add_argument(
        "--dataset",
        default="jcxiv42/astroclimb",
        help="Kaggle dataset slug (owner/name)",
    )
    args = parser.parse_args(argv)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if shutil.which("kaggle") is None:
        print(
            "The Kaggle CLI is not installed. Install it with `pip install kaggle`, "
            "place your token at ~/.kaggle/kaggle.json, then rerun this script.",
            file=sys.stderr,
        )
        return 1

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        args.dataset,
        "-p",
        str(output),
        "--unzip",
    ]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)
    print(f"Data written to {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
