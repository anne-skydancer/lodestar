import json
import os
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from lodestar_gltf_export import write_gltf_from_glb
from lodestar_textured_glb import write_textured_glb


def glb_bytes(document, binary=None):
    raw_json = json.dumps(document).encode("utf-8")
    raw_json += b" " * (-len(raw_json) % 4)
    chunks = struct.pack("<I4s", len(raw_json), b"JSON") + raw_json
    if binary is not None:
        binary += b"\0" * (-len(binary) % 4)
        chunks += struct.pack("<I4s", len(binary), b"BIN\0") + binary
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks


class GltfExportTest(unittest.TestCase):
    def test_keeps_baked_texture_and_relative_companion(self):
        with TemporaryDirectory() as root:
            source = Path(root) / "source.glb"
            target = Path(root) / "view #2.gltf"
            positions = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]])
            normals = np.array([[0, 1, 0]] * 3)
            uv = np.array([[0, 0], [1, 0], [0, 1]])
            mesh = (positions, normals, uv, np.array([0, 1, 2]), 1)
            write_textured_glb(source, [mesh], {1: (Image.new("RGB", (2, 2), "red"), False)}, unlit=True)
            companion = write_gltf_from_glb(source, target)
            document = json.loads(target.read_text(encoding="utf-8"))
            uri = document["buffers"][0].pop("uri")
            self.assertEqual(target.parent / uri, companion)
            self.assertRegex(uri, r"^lodestar-[a-f0-9]+\.bin$")
            data = source.read_bytes()
            size, _ = struct.unpack_from("<I4s", data, 12)
            self.assertEqual(document, json.loads(data[20:20 + size]))
            self.assertEqual(companion.read_bytes(), data[28 + size:])

    def test_accepts_bufferless_scene_and_removes_glb_padding(self):
        with TemporaryDirectory() as root:
            source = Path(root) / "source.glb"
            target = Path(root) / "view.gltf"
            document = {"asset": {"version": "2.0"}}
            source.write_bytes(glb_bytes(document))
            self.assertIsNone(write_gltf_from_glb(source, target))
            self.assertEqual(json.loads(target.read_text()), document)
            document["buffers"] = [{"byteLength": 3}]
            source.write_bytes(glb_bytes(document, b"abc"))
            self.assertEqual(write_gltf_from_glb(source, target).read_bytes(), b"abc")

    def test_reexport_keeps_old_companion_valid(self):
        with TemporaryDirectory() as root:
            source = Path(root) / "source.glb"
            target = Path(root) / "view.gltf"
            document = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": 4}]}
            source.write_bytes(glb_bytes(document, b"abcd"))
            old = write_gltf_from_glb(source, target)
            source.write_bytes(glb_bytes(document, b"efgh"))
            new = write_gltf_from_glb(source, target)
            self.assertNotEqual(old, new)
            self.assertEqual(old.read_bytes(), b"abcd")
            self.assertEqual(new.read_bytes(), b"efgh")
            self.assertEqual(write_gltf_from_glb(source, target), new)

    def test_rejects_malformed_chunk_and_view_without_writing(self):
        with TemporaryDirectory() as root:
            source = Path(root) / "source.glb"
            target = Path(root) / "view.gltf"
            document = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": 4}]}
            data = glb_bytes(document, b"abcd")[:-4]
            source.write_bytes(data[:8] + struct.pack("<I", len(data)) + data[12:])
            with self.assertRaisesRegex(ValueError, "chunk type or length"):
                write_gltf_from_glb(source, target)
            document["bufferViews"] = [{"buffer": 0, "byteOffset": 3, "byteLength": 2}]
            source.write_bytes(glb_bytes(document, b"abcd"))
            with self.assertRaisesRegex(ValueError, "extends beyond"):
                write_gltf_from_glb(source, target)
            self.assertEqual(set(Path(root).iterdir()), {source})

    def test_rejects_output_alias_of_input(self):
        with TemporaryDirectory() as root:
            source = Path(root) / "source.glb"
            target = Path(root) / "view.gltf"
            original = glb_bytes({"asset": {"version": "2.0"}})
            source.write_bytes(original)
            try:
                os.link(source, target)
            except OSError as error:
                self.skipTest(f"The fixture filesystem cannot create hard links: {error}")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                write_gltf_from_glb(source, target)
            self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
