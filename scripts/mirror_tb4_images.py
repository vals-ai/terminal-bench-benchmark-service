"""Copy the release's task and sidecar images into this AWS account's ECR, keeping every digest.

Run before importing a release, with credentials for the account the service runs in. Needs `crane`
and the AWS CLI. Images already present are skipped, so reruns resume.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
from pathlib import Path

from scripts.import_tbench4_release import PINNED_COMPOSE_IMAGES, ecr_image


def _digest(image: str) -> str:
    return subprocess.run(["crane", "digest", image], capture_output=True, text=True).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    args = parser.parse_args()

    release = json.loads(args.source_manifest.read_text())
    images = {
        image["pinned_ref"] for task in release["tasks"] for image in task["images"] if image["kind"] != "verifier"
    }
    images |= {
        image for services in PINNED_COMPOSE_IMAGES[release["release_tag"]].values() for image in services.values()
    }

    token = subprocess.run(
        ["aws", "ecr", "get-authorization-token", "--region", "us-east-1"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    authorization = json.loads(token.stdout)["authorizationData"][0]
    registry = authorization["proxyEndpoint"].removeprefix("https://")
    password = base64.b64decode(authorization["authorizationToken"]).decode().removeprefix("AWS:")
    login = ["crane", "auth", "login", registry, "-u", "AWS", "--password-stdin"]
    subprocess.run(login, input=password, text=True, check=True)

    failed: list[str] = []
    for image in sorted(images):
        target, digest = f"{registry}/{ecr_image(image)}".split("@")
        if _digest(target) != digest:
            subprocess.run(["crane", "copy", image, target], check=False)
        if _digest(target) != digest:
            failed.append(image)
    print(f"{len(images) - len(failed)}/{len(images)} images in {registry}; failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
