"""Local asynchronous bridge from Firestorm to Lodestar's mesh preparation.

This development bridge runs the existing native-backed fitting workflow.
It uses atomic LLSD XML result files so the viewer never reads half a report.
There is no shell command construction and no network/upload operation here.
"""

import argparse
from contextlib import redirect_stdout, redirect_stderr
from hashlib import sha256
import json
from pathlib import Path
import traceback
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from fit_mesh_lods import run


def write_result(output, values):
    root = ET.Element("llsd")
    mapping = ET.SubElement(root, "map")
    for key, value in values.items():
        ET.SubElement(mapping, "key").text = key
        kind = "boolean" if isinstance(value, bool) else "integer" if isinstance(value, int) else "string"
        ET.SubElement(mapping, kind).text = str(value).lower() if isinstance(value, bool) else str(value)
    temporary = output / "viewer-result.xml.tmp"
    temporary.write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True))
    temporary.replace(output / "viewer-result.xml")


def prepare(source, output, library, *, import_scale=1.0):
    source, output, library = (Path(path).resolve() for path in (source, output, library))
    output.mkdir(parents=True, exist_ok=False)
    values = {"status": "running", "source": str(source), "output": str(output),
              "message": "Lodestar is fitting High, calculating visual LODs and refining collision surfaces."}
    write_result(output, values)
    with (output / "preparation.log").open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
        try:
            args = SimpleNamespace(source=source, output=output, library=library,
                textures=source.parent, pixels=(1024, 512, 192, 96), pixel_error=.5,
                normal_angle=1., view_size=128, existing_physics=None,
                import_scale=import_scale,
                existing_translation=(0., 0., 0.), benchmark=None)
            run(args)
            score = json.loads((output / "scorecard.json").read_text(encoding="utf-8"))
            if score["source_sha256"] != sha256(source.read_bytes()).hexdigest():
                raise ValueError("The High source changed during preparation")
            physics = score["physics"]
            sizing = score.get('size_planning', {})
            values.update(status="prepared", source_sha256=score["source_sha256"],
                          visual_generator=score["visual_generator"],
                          physics_file=physics["file"], triangle_ready=physics["triangle_upload_ready"],
                          physics_source_level=physics.get("source_level", "high"),
                          physics_surface_bound=str(physics.get("maximum_surface_change_after_import_scale_before_packing",
                                                               physics.get("maximum_surface_change_before_packing", 0.0))),
                          size_container_required=bool(sizing.get('requires_frame_review', sizing.get('requires_container', False))),
                          size_container_count=len(sizing.get('container_proposals', [])) + len(sizing.get('frame_preservation', [])),
                          physical_container_count=len(sizing.get('physical_container_models', [])),
                          physical_container_import_scale=str(sizing.get('import_scale', 1.0)),
                          unresolved_triangles=sum(len(part["unresolved_triangle_indices"])
                                                   for part in physics["parts"].values()))
            for level in ("high", "medium", "low", "lowest"):
                values[level + "_triangles"] = score["levels"][level]["triangles"]
            values["message"] = ("Lodestar calculated all visual LODs and collision refinement. "
                                 "The viewer will now validate native imports and attempt hull generation.")
            if values['size_container_required']:
                values['message'] = 'Minimum-size container frames need review in size-proposals.json. Authored geometry is unchanged.'
            elif values['physical_container_count']:
                values['message'] = f"Lodestar applied {values['physical_container_count']} physical containers. Authored visual geometry is unchanged."
            write_result(output, values)
            return 0
        except Exception as error:
            traceback.print_exc()
            values.update(status="failed", message=f"Lodestar preparation failed: {error}")
            write_result(output, values)
            return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--import-scale", type=float, default=1.0)
    args = parser.parse_args()
    try:
        return prepare(args.source, args.output, args.library, import_scale=args.import_scale)
    except OSError as error:
        parser.exit(1, f"Cannot start Lodestar preparation: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
