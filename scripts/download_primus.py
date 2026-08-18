from __future__ import annotations

import argparse
import os
import tarfile
import urllib.request

_URL = "https://grfia.dlsi.ua.es/primus/packages/primusCalvoRizoAppliedSciences2018.tgz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data/primus")
    ap.add_argument("--archive", default="data/primus/primus.tgz")
    args = ap.parse_args()
    os.makedirs(args.dest, exist_ok=True)

    if not os.path.exists(args.archive):
        print(f"downloading {_URL} (~273 MB) -> {args.archive}")
        urllib.request.urlretrieve(_URL, args.archive)  # noqa: S310 (trusted host)
    else:
        print(f"archive already present: {args.archive}")

    print(f"extracting -> {args.dest}")
    with tarfile.open(args.archive, "r:gz") as tar:
        tar.extractall(args.dest, filter="data")
    print("done")


if __name__ == "__main__":
    main()
