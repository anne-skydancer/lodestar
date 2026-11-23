import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from lodestar_gltf_source import load_gltf_groups
from lodestar_section_source import face_fingerprint
from lodestar_textured_glb import write_textured_glb


class GltfSourceTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.positions = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.uvs = np.array([[.125, .25], [.75, .5], [.25, .875]], dtype=np.float32)
        normals = np.broadcast_to([0, 0, 1], self.positions.shape)
        self.image = Image.new("RGB", (2, 2), "red")
        self.image.putpixel((0, 1), (0, 0, 255))
        self.glb = self.folder / "sample.glb"
        write_textured_glb(self.glb, [(self.positions, normals, self.uvs, [0, 1, 2], 17)],
                           {17: (self.image, False)})

    def separate_document(self):
        raw = self.glb.read_bytes()
        json_length, = struct.unpack_from("<I", raw, 12)
        document = json.loads(raw[20:20 + json_length])
        document["buffers"][0]["uri"] = "mesh.bin"
        (self.folder / "mesh.bin").write_bytes(raw[28 + json_length:])
        return document

    def test_glb_png_binding_and_native_uv_proof(self):
        groups, textures = load_gltf_groups(self.glb)
        self.assertEqual(len(groups), 1)
        node, mesh, groups = groups[0]
        self.assertEqual((node, mesh), ("node-0", "mesh-0"))
        material, count, faces = groups[0]
        self.assertEqual(count, 1)
        proof_uv = self.uvs * np.array([1, -1], dtype=np.float32) + np.array([0, 1], dtype=np.float32)
        self.assertEqual(faces, face_fingerprint(self.positions[None], proof_uv[None]))
        image, repeat = textures[material]
        self.assertFalse(repeat)
        self.assertEqual(image.tobytes(), self.image.tobytes())

    def test_gltf_external_buffer_and_named_node_reflection(self):
        doc = self.separate_document()
        doc["nodes"][0].update(name="Gate", translation=[3, 1, 0], scale=[-1, 1, 1])
        (self.folder / "paint").mkdir()
        self.image.save(self.folder / "paint" / "gate map.png")
        doc["images"][0] = {"uri": "paint/gate%20map.png"}
        path = self.folder / "sample.gltf"
        path.write_text(json.dumps(doc), encoding="utf-8")
        groups, images = load_gltf_groups(path)
        self.assertEqual(groups[0][0], "Gate")
        material, _, proof = groups[0][2][0]
        xyz = self.positions * [-1, 1, 1] + [3, 1, 0]
        uv = self.uvs * [1, -1] + [0, 1]
        reverse = [0, 2, 1]
        self.assertEqual(proof, face_fingerprint(xyz[reverse][None], uv[reverse][None]))
        self.assertEqual(images[material][0].tobytes(), self.image.tobytes())

    def test_rejects_unrepresented_material_and_outside_resource(self):
        doc = self.separate_document()
        path = self.folder / "sample.gltf"
        doc["materials"][0]["normalTexture"] = {"index": 0}
        path.write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "opaque, single-sided, nonmetallic"):
            load_gltf_groups(path)
        del doc["materials"][0]["normalTexture"]
        doc["images"][0] = {"uri": "../outside.png"}
        path.write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "outside the asset directory"):
            load_gltf_groups(path)


if __name__ == "__main__":
    unittest.main()
