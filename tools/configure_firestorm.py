"""Install the local Lodestar preparation bridge in a development viewer runtime."""

import argparse
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def configure(runtime, interpreter=None):
    runtime = Path(runtime).resolve()
    python = Path(interpreter or sys.executable).resolve()
    worker = Path(__file__).resolve().with_name("firestorm_prepare.py")
    if not (runtime / "app_settings").is_dir() or not (runtime / "lodestar_c.dll").is_file():
        raise ValueError("Select a viewer runtime with app_settings and lodestar_c.dll")
    subprocess.run([str(python), "-c", "import numpy, PIL"], check=True, capture_output=True,
                   creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    hidden = python.with_name("pythonw.exe")
    if sys.platform == "win32" and hidden.is_file():
        python = hidden
    root = ET.Element("llsd")
    values = ET.SubElement(root, "map")
    for key, value in {"python": str(python), "worker": str(worker)}.items():
        ET.SubElement(values, "key").text = key
        ET.SubElement(values, "string").text = value
    path = runtime / "app_settings" / "lodestar-preparation.xml"
    temporary = path.with_suffix(".xml.tmp")
    temporary.write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True))
    temporary.replace(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("--python", type=Path)
    args = parser.parse_args()
    try:
        print(configure(args.runtime, args.python))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Cannot configure Lodestar preparation: {error}\n")


if __name__ == "__main__":
    main()
