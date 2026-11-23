import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from bake_section_atlas import indexed_primitives
from build_testhall_firestorm import qualify_lower_appearance
from lodestar_gltf_source import _document
from lodestar_lod_materials import combine_lod_parts
from lodestar_section_source import source_values
from lodestar_testhall_textures import TexturedScene
from lodestar_textured_glb import write_textured_glb
from lodestar_visual_collada import write_lod_collada, write_lod_collision_collada


def _visual(positions, materials, textures):
    positions = np.asarray(positions, dtype=np.float32)
    count = len(positions)
    uvs = np.broadcast_to([[0., 0.], [1., 0.], [0., 1.]], (count, 3, 2))
    normals = np.broadcast_to([0., 0., 1.], positions.shape)
    return TexturedScene(positions, uvs, normals, np.array(materials), textures, np.arange(count * 3).reshape(-1, 3))


def _dae(path):
    document = ET.parse(path).getroot()
    for element in document.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    return document


def _glb_view(document, binary, index):
    view = document["bufferViews"][index]
    offset = view["byteOffset"]
    return binary[offset:offset + view["byteLength"]]


class LodPackageExportTest(unittest.TestCase):
    def test_lower_appearance_retains_parts_that_fail_the_color_gate(self):
        green = {0: (Image.new("RGB", (16, 16), "green"), True)}
        red = {0: (Image.new("RGB", (16, 16), "red"), True)}

        def part(offset, count, textures):
            positions = np.repeat((self.face + [offset, 0, 0])[None], count, axis=0)
            return _visual(positions, np.zeros(count, dtype=int), textures)

        medium = {"A": part(0, 3, green), "B": part(2, 3, green)}
        contested = part(0, 1, red)
        levels = {"medium": medium, "low": {"A": contested, "B": part(2, 2, green)},
                  "lowest": {"A": contested, "B": part(2, 1, green)}}
        source = combine_lod_parts(medium)
        reference = {"resolution": 32, "mean_abs_error": 0.0, "p95_abs_error": 0}
        report = qualify_lower_appearance(levels, source, {"low": reference, "lowest": reference}, source.positions)
        for level, count in (("low", 2), ("lowest", 1)):
            self.assertFalse(report[level]["parts"]["A"]["passed"])
            self.assertIs(levels[level]["A"], medium["A"])
            self.assertEqual(len(levels[level]["B"].positions), count)

    def setUp(self):
        self.face = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
        repeat = (Image.new("RGB", (16, 16), "red"), True)
        atlas = (Image.new("RGB", (32, 16), "blue"), False)
        doorway = self.face + [2.25, 3.5, -1.5]
        stairs = self.face + [-5., 1., 2.]
        self.high = {"Doorway": _visual([self.face, doorway], [2, 8], {2: repeat, 8: atlas}),
                     "Stairs": _visual([stairs], [9], {9: atlas})}
        self.medium = {"Doorway": _visual([doorway], [8], {8: atlas}), "Stairs": self.high["Stairs"]}

    def test_collada_geometry_labels_and_shared_texture_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images_by_level = {}
            for suffix, objects in (("", self.high), ("_LOD2", self.medium), ("_LOD1", self.medium),
                                    ("_LOD0", self.medium)):
                path = root / f"Hall{suffix}.dae"
                write_lod_collada(path, objects, suffix=suffix)
                document = _dae(path)
                self.assertEqual(document.findtext("./asset/up_axis"), "Y_UP")
                self.assertEqual(document.find("./asset/unit").get("meter"), "1")
                self.assertEqual([node.get("name") for node in document.findall(".//visual_scene/node")],
                                 [name + suffix for name in objects])
                image_files = {image.get("id"): image.findtext("init_from")
                               for image in document.findall("./library_images/image")}
                self.assertEqual(len(image_files), 1 if suffix else 2)
                for uri in image_files.values():
                    self.assertEqual((root / uri).parent, root / "textures")
                    with Image.open(root / uri) as image:
                        self.assertEqual(image.format, "PNG")
                images_by_level[suffix] = image_files
                effects = {effect.get("id"): effect.findtext(".//surface/init_from")
                           for effect in document.findall("./library_effects/effect")}
                material_ids = {material.get("id"): material.find("instance_effect").get("url")[1:]
                                for material in document.findall("./library_materials/material")}
                for name, part in objects.items():
                    label = name + suffix
                    node = document.find(f".//visual_scene/node[@name='{label}']")
                    instance = node.find("instance_geometry")
                    geometry_id = instance.get("url")[1:]
                    geometry = document.find(f"./library_geometries/geometry[@id='{geometry_id}']")
                    self.assertEqual(geometry.get("name"), label)
                    mesh = geometry.find("mesh")
                    self.assertIs(geometry[0], mesh)
                    bindings = {binding.get("symbol"): binding.get("target")[1:]
                                for binding in instance.findall("./bind_material/technique_common/instance_material")}
                    expected = {f"{name}-slot-{material}" for material in part.textures}
                    self.assertEqual(set(bindings), expected)
                    self.assertEqual({batch.get("material") for batch in mesh.findall("triangles")}, expected)
                    for material_name in expected:
                        image_id = effects[material_ids[bindings[material_name]]]
                        self.assertIn(image_id, image_files)
            for suffix in ("_LOD2", "_LOD1", "_LOD0"):
                self.assertEqual(images_by_level[suffix], images_by_level["_LOD2"])
                self.assertTrue(images_by_level[suffix].items() <= images_by_level[""].items())
            self.assertEqual(len(list((root / "textures").glob("*.png"))), 2)

    def test_glb_models_keep_named_primitive_materials_across_tiers(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            packed_pngs = {}
            for suffix, objects in (("", self.high), ("_LOD2", self.medium), ("_LOD1", self.medium),
                                    ("_LOD0", self.medium)):
                path = root / f"Hall{suffix}.glb"
                groups, textures, names = {}, {}, {}
                for name, part in objects.items():
                    groups[name + suffix] = indexed_primitives(*part.arrays[:4], part.corner_indices)
                    textures.update(part.textures)
                    names.update({label: f"{name}-slot-{label}" for label in part.textures})
                write_textured_glb(path, groups, textures, material_names=names)
                gltf, binary = _document(path)
                self.assertEqual(len(gltf["images"]), 1 if suffix else 2)
                self.assertEqual([node["name"] for node in gltf["nodes"]], [name + suffix for name in objects])
                for node, (name, part) in zip(gltf["nodes"], objects.items()):
                    mesh = gltf["meshes"][node["mesh"]]
                    self.assertEqual(mesh["name"], name + suffix)
                    actual = set()
                    for primitive in mesh["primitives"]:
                        self.assertEqual(set(primitive["attributes"]), {"POSITION", "NORMAL", "TEXCOORD_0"})
                        material = gltf["materials"][primitive["material"]]
                        actual.add(material["name"])
                        texture = gltf["textures"][material["pbrMetallicRoughness"]["baseColorTexture"]["index"]]
                        image = gltf["images"][texture["source"]]
                        png = _glb_view(gltf, binary, image["bufferView"])
                        if suffix:
                            self.assertEqual(png, packed_pngs[material["name"]])
                        else:
                            packed_pngs[material["name"]] = png
                    self.assertEqual(actual, {f"{name}-slot-{label}" for label in part.textures})

    def test_named_physics_outputs_reconstruct_the_source_positions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            objects = {name: part.positions for name, part in self.high.items()}
            dae = root / "Hall_PHYS.dae"
            write_lod_collision_collada(dae, objects)
            document = _dae(dae)
            self.assertIsNone(document.find("library_materials"))
            self.assertIsNone(document.find(".//bind_material"))
            self.assertEqual({node.get("name") for node in document.findall(".//visual_scene/node")},
                             {name + "_PHYS" for name in objects})
            groups = {}
            for name, expected in objects.items():
                label = name + "_PHYS"
                node = document.find(f".//visual_scene/node[@name='{label}']")
                geometry_id = node.find("instance_geometry").get("url")[1:]
                geometry = document.find(f"./library_geometries/geometry[@id='{geometry_id}']")
                self.assertEqual(geometry.get("name"), label)
                mesh = geometry.find("mesh")
                position = mesh.find("./vertices/input")
                self.assertEqual(position.get("semantic"), "POSITION")
                triangles = mesh.find("triangles")
                self.assertNotIn("material", triangles.attrib)
                self.assertEqual(len(triangles.findall("input")), 1)
                self.assertEqual(triangles.find("input").attrib,
                                 {"semantic": "VERTEX", "source": "#" + mesh.find("vertices").get("id"), "offset": "0"})
                attribute = source_values(mesh, position.get("source")[1:], 3)
                indices = np.fromstring(triangles.findtext("p"), sep=" ", dtype=np.int64).reshape(-1, 3)
                np.testing.assert_array_equal(attribute[indices], expected)
                vertices = expected.reshape(-1, 3)
                groups[label] = [(vertices, None, None, np.arange(len(vertices)), None)]

            glb = root / "Hall_PHYS.glb"
            write_textured_glb(glb, groups, None)
            gltf, binary = _document(glb)
            self.assertNotIn("materials", gltf)
            self.assertNotIn("images", gltf)
            for node, expected in zip(gltf["nodes"], objects.values()):
                mesh = gltf["meshes"][node["mesh"]]
                self.assertEqual(node["name"], mesh["name"])
                self.assertIn(node["name"], groups)
                primitive, = mesh["primitives"]
                self.assertEqual(set(primitive["attributes"]), {"POSITION"})
                self.assertNotIn("material", primitive)
                vertex = gltf["accessors"][primitive["attributes"]["POSITION"]]
                corner = gltf["accessors"][primitive["indices"]]
                positions = np.frombuffer(_glb_view(gltf, binary, vertex["bufferView"]), dtype="<f4").reshape(-1, 3)
                indices = np.frombuffer(_glb_view(gltf, binary, corner["bufferView"]), dtype="<u2").reshape(-1, 3)
                np.testing.assert_array_equal(positions[indices], expected)


if __name__ == "__main__":
    unittest.main()
