import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from firestorm_prepare import prepare


def result_values(path):
    children = list(ET.parse(path).find("map"))
    return {children[index].text: (children[index + 1].tag, children[index + 1].text)
            for index in range(0, len(children), 2)}


class FirestormPrepareTest(unittest.TestCase):
    def test_unsupported_source_is_an_explicit_worker_failure(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.glb"
            source.write_bytes(b"original reference")
            with patch("firestorm_prepare.run", side_effect=ValueError("Unsupported source attributes")):
                status = prepare(source, root / "output", root / "library.dll")
            self.assertEqual(status, 1)
            report = result_values(root / "output/viewer-result.xml")
            self.assertEqual(report["status"], ("string", "failed"))
            self.assertIn("Unsupported source attributes", report["message"][1])
            self.assertEqual(source.read_bytes(), b"original reference")

    def test_visual_completion_keeps_blocked_physics_false_in_native_protocol(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.glb"
            source.write_bytes(b"original reference")
            def run(args):
                report = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "visual_generator": "lodestar_uv_cavity", "levels": {
                        name: {"triangles": count} for name, count in
                        zip(("high", "medium", "low", "lowest"), (10, 8, 6, 4))},
                    "physics": {"file": "Lodestar_unresolved_PHYS.dae", "triangle_upload_ready": False,
                                "parts": {"part": {"unresolved_triangle_indices": [1, 3]}}}}
                (args.output / "scorecard.json").write_text(json.dumps(report))
            with patch("firestorm_prepare.run", side_effect=run):
                self.assertEqual(prepare(source, root / "output", root / "library.dll"), 0)
            report = result_values(root / "output/viewer-result.xml")
            self.assertEqual(report["triangle_ready"], ("boolean", "false"))
            self.assertEqual(report["unresolved_triangles"], ("integer", "2"))
            self.assertEqual(report["visual_generator"], ("string", "lodestar_uv_cavity"))
            self.assertFalse((root / "output/Lodestar_PHYS.dae").exists())


if __name__ == "__main__":
    unittest.main()
