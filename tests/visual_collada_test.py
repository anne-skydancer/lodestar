from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from urllib.parse import unquote
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from lodestar_section_source import source_values
from lodestar_visual_collada import write_textured_collada


class VisualColladaTest(unittest.TestCase):
    def test_texture_paths_wrap_modes_and_corner_streams(self):
        base = np.array([[0., 1., 0.], [1., 1., 0.], [0., 2., 0.]])
        positions = np.stack((base, base + [2., 0., 0.], base + [4., 0., 0.]))
        normals = np.tile([0., 0., 1.], (3, 3, 1))
        uvs = np.tile([[-0.25, 0.125], [1.25, 0.5], [0.75, 0.875]], (3, 1, 1))
        materials = np.array([9, 2, 4])
        red = Image.new("RGB", (2, 3), "red")
        blue = Image.new("RGB", (3, 2), "blue")
        scene = SimpleNamespace(positions=positions, normals=normals, uvs=uvs, materials=materials,
                                textures={2: (red, True), 4: (red.copy(), False), 9: (blue, False)})
        with TemporaryDirectory() as directory:
            path = Path(directory) / "Visual LOD.dae"
            write_textured_collada(path, scene)
            document = ET.parse(path).getroot()
            for element in document.iter():
                element.tag = element.tag.rsplit("}", 1)[-1]
            self.assertEqual(document.findtext("./asset/up_axis"), "Y_UP")
            self.assertEqual(document.find("./asset/unit").get("meter"), "1")
            images = [element.text for element in document.findall("./library_images/image/init_from")]
            self.assertEqual(len(images), 2)
            for uri in images:
                with Image.open(path.parent / unquote(uri)) as decoded:
                    self.assertEqual(decoded.format, "PNG")
            self.assertEqual([element.text for element in document.findall(".//sampler2D/wrap_s")],
                             ["WRAP", "CLAMP", "CLAMP"])
            self.assertEqual([element.text for element in document.findall(".//sampler2D/wrap_t")],
                             ["WRAP", "CLAMP", "CLAMP"])
            mesh = document.find("./library_geometries/geometry/mesh")
            for index, source_face in enumerate(np.argsort(materials)):
                triangles = mesh.findall("triangles")[index]
                corner = np.fromstring(triangles.findtext("p"), sep=" ", dtype=np.int64).reshape(3, 3)
                for stream, width, expected in (("LodestarPositions", 3, positions[source_face]),
                                                ("LodestarNormals", 3, normals[source_face]),
                                                ("LodestarUV0", 2, uvs[source_face] * [1, -1] + [0, 1])):
                    offset = {"LodestarPositions": 0, "LodestarNormals": 1, "LodestarUV0": 2}[stream]
                    np.testing.assert_array_equal(source_values(mesh, stream, width)[corner[:, offset]], expected)


if __name__ == "__main__":
    unittest.main()
