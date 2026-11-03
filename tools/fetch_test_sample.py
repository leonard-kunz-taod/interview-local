"""Download the public LDC sample; no model or token access.

The audio/reference are not covered by the app license and are not
redistributed in the ZIP. Source and rights: https://catalog.ldc.upenn.edu/LDC2026S06
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
args = parser.parse_args()
args.directory.mkdir(parents=True, exist_ok=True)
manifest = {"source": "https://catalog.ldc.upenn.edu/LDC2026S06", "files": [],
            "copyright": "Portions © 1997, 2026 Trustees of the University of Pennsylvania",
            "note": "Public sample. Do not redistribute as a freely licensed corpus or an app component."}
for suffix in ("flac", "tsv"):
    url = f"https://catalog.ldc.upenn.edu/desc/addenda/LDC2026S06.{suffix}"
    target = args.directory / f"sample.{suffix}"
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read()
    target.write_bytes(content)
    manifest["files"].append({"file": target.name, "url": url, "sha256": hashlib.sha256(content).hexdigest()})
    print(f"Saved: {target}")
(args.directory / "source.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
