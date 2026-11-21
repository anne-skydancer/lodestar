import io
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from bake_planar_patch import bake_patch, linear_image, sample, source_colors, write_glb
from lodestar_texture_budget import crop_repeat_textures, reuse_atlas_texels


def panel():
    corners = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0], [1, 0.5, 0]])
    source_uv = np.array([[0.125, 0.1875], [0.875, 0.1875], [0.875, 0.8125], [0.125, 0.8125],
                          [0.6875, 0.375]])
    triangles = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    positions = corners[triangles]
    return positions, source_uv[triangles], np.broadcast_to([0.0, 0.0, 1.0], positions.shape).copy()


def reference_texture():
    axis = np.linspace(0.0, 1.0, 129)
    u, top_v = np.meshgrid(axis, axis)
    return Image.fromarray(np.rint(reference_rgb(u, top_v)).astype(np.uint8))


def reference_rgb(u, top_v):
    # Separate horizontal and vertical gradients make either axis visible in the sampled color.
    return np.stack((32 + 176 * u, 40 + 160 * top_v, 24 + 80 * u + 96 * top_v), axis=-1)


def uv_at_position(result, point):
    for triangle in np.asarray(result.triangles):
        a, b, c = np.asarray(result.positions)[triangle, :2]
        weights = np.linalg.solve(np.column_stack((b - a, c - a)), point[:2] - a)
        weights = np.array([1 - weights.sum(), *weights])
        if np.all(weights >= -1e-10):
            return weights @ np.asarray(result.uvs)[triangle]
    raise AssertionError(f"Output triangles do not cover source point {point}")


def sample_texture(image, uv):
    pixels = np.asarray(image, dtype=float)
    x, y = np.clip(uv * np.array(image.size) - 0.5, 0, np.array(image.size) - 1)
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, image.width - 1), min(y0 + 1, image.height - 1)
    dx, dy = x - x0, y - y0
    top = (1 - dx) * pixels[y0, x0] + dx * pixels[y0, x1]
    bottom = (1 - dx) * pixels[y1, x0] + dx * pixels[y1, x1]
    return (1 - dy) * top + dy * bottom


class TextureBakeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.positions, cls.uvs, cls.normals = panel()
        cls.texture = reference_texture()
        cls.result = bake_patch(cls.positions, cls.uvs, cls.normals, cls.texture, resolution=96, supersamples=2)

    def test_region_samples_both_materials_and_extends_its_boundary(self):
        faces = np.array([[[0, 0], [0.5, 0], [0.5, 1]], [[0.5, 0], [1, 0], [0.5, 1]]])
        colors = {2: np.array([[[1.0, 0.0, 0.0]]]), 7: np.array([[[0.0, 0.0, 1.0]]])}
        boundary = ((0, 0, 1), (1, 0, 1), (1, 1, 2), (0, 2, 0))
        samples = source_colors(np.array([[0.4, 0.2], [0.6, 0.2], [0.0, 0.9], [1.0, 0.9]]),
                                faces, faces, colors, materials=np.array([2, 7]), boundary=boundary)
        np.testing.assert_array_equal(samples, [colors[2][0, 0], colors[7][0, 0],
                                                colors[2][0, 0], colors[7][0, 0]])

    def test_crop_handles_periodic_taps_and_shared_images(self):
        image = reference_texture()
        uv = np.array([[[-0.08, -0.1], [0.07, 0.3], [0.01, 1.2]],
                       [[1.03, 0.7], [0.94, 0.0], [1.06, 0.1]]])
        remapped, textures, changes = crop_repeat_textures(uv, np.array([2, 7]), {2: (image, True), 7: (image, True)})
        self.assertEqual(len(changes), 1)
        self.assertIs(textures[2][0], textures[7][0])
        self.assertLess(textures[2][0].width, image.width)
        self.assertEqual(textures[2][0].height, image.height)
        weights = np.array([[1, 0, 0], [1/3, 1/3, 1/3], [0.1, 0.8, 0.1]])
        original = sample(linear_image(image), (weights @ uv).reshape(-1, 2), repeat=True)
        cropped = sample(linear_image(textures[2][0]), (weights @ remapped).reshape(-1, 2), repeat=True)
        np.testing.assert_allclose(cropped, original, rtol=0, atol=2e-15)

    def test_tile_reuse_preserves_retained_texels_and_refuses_overflow(self):
        image = reference_texture()
        before = image.tobytes()
        uv = np.array([[[0.2, 0.2], [0.3, 0.2], [0.25, 0.3]], [[0, 0], [1, 0], [0, 1]]])
        labels = np.array([2, 7])
        textures = {2: (image, True), 7: (Image.new("RGB", (2, 2)), False)}
        pixels = np.full((16, 20, 3), [255, 5, 1], dtype=np.uint8)
        tile = SimpleNamespace(pixels=pixels, uvs=uv[1], triangles=np.array([[0, 1, 2]]))
        remapped, materials, new_textures, placement = reuse_atlas_texels(uv, labels, textures, [tile], 7)
        self.assertEqual(materials.tolist(), [2, 2])
        self.assertEqual(set(new_textures), {2})
        self.assertEqual(new_textures[2][0].size, image.size)
        self.assertEqual(len(placement), 1)
        np.testing.assert_array_equal(remapped[0], uv[0])
        weights = np.array([[1, 0, 0], [1/3, 1/3, 1/3]])
        original = sample(linear_image(image), weights @ uv[0], repeat=True)
        updated = sample(linear_image(new_textures[2][0]), weights @ remapped[0], repeat=True)
        np.testing.assert_array_equal(updated, original)
        self.assertEqual(image.tobytes(), before)
        tile.pixels = np.zeros((130, 130, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "do not fit"):
            reuse_atlas_texels(uv, labels, textures, [tile], 7)
        self.assertEqual(image.tobytes(), before)

    def test_non_affine_source_texture_survives_retriangulation(self):
        result = self.result
        positions = np.asarray(result.positions)
        triangles = np.asarray(result.triangles)
        uv = np.asarray(result.uvs)

        self.assertEqual(positions.shape, (4, 3))
        self.assertEqual(triangles.shape, (2, 3))
        self.assertEqual(set(map(tuple, positions)), {(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)})
        self.assertTrue(np.all((triangles >= 0) & (triangles < len(positions))))
        a, b, c = positions[triangles].transpose(1, 0, 2)
        signed_double_area = np.cross(b - a, c - a)[:, 2]
        self.assertTrue(np.all(signed_double_area > 0))
        self.assertAlmostEqual(float(signed_double_area.sum()) / 2, 2.0)
        self.assertEqual(uv.shape, (4, 2))
        self.assertTrue(np.all(np.isfinite(uv) & (uv >= 0) & (uv <= 1)))
        self.assertEqual((result.texture.mode, result.texture.size), ("RGB", (96, 96)))

        # Probe known barycentric combinations in each high-detail triangle. The
        # moved center UV makes the correct sampled color differ from a single affine map.
        for i, (face, corners_uv) in enumerate(zip(self.positions, self.uvs)):
            for weights in (np.array([0.125, 0.125, 0.75]), np.array([0.5, 0.125, 0.375])):
                point = weights @ face
                source_u, source_v = weights @ corners_uv
                expected = reference_rgb(source_u, 1 - source_v)
                sampled = sample_texture(result.texture, uv_at_position(result, point))
                with self.subTest(face=i, weights=weights.tolist()):
                    np.testing.assert_allclose(sampled, expected, rtol=0, atol=4.0)

        metrics = result.metrics
        self.assertGreater(metrics["validation_samples"], 0)
        self.assertGreaterEqual(metrics["mean_abs_error"], 0)
        self.assertLessEqual(metrics["mean_abs_error"], 4)
        self.assertGreaterEqual(metrics["p95_abs_error"], 0)
        self.assertLessEqual(metrics["p95_abs_error"], 8)
        self.assertTrue(np.isfinite(metrics["max_abs_error"]))
        self.assertGreaterEqual(metrics["max_abs_error"], metrics["p95_abs_error"])

    def test_glb_embeds_the_baked_mesh_and_png(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel.glb"
            write_glb(path, self.result)
            data = path.read_bytes()
            self.assertEqual(list(path.parent.iterdir()), [path])

        self.assertEqual(struct.unpack_from("<4sII", data), (b"glTF", 2, len(data)))
        json_size, json_kind = struct.unpack_from("<I4s", data, 12)
        self.assertEqual(json_kind, b"JSON")
        self.assertEqual(json_size % 4, 0)
        document = json.loads(data[20:20 + json_size])
        bin_size, bin_kind = struct.unpack_from("<I4s", data, 20 + json_size)
        binary = data[28 + json_size:]
        self.assertEqual(bin_kind, b"BIN\0")
        self.assertEqual(bin_size % 4, 0)
        self.assertEqual(bin_size, len(binary))
        self.assertEqual(document["asset"]["version"], "2.0")
        self.assertEqual(len(document["buffers"]), 1)
        self.assertNotIn("uri", document["buffers"][0])
        buffer_size = document["buffers"][0]["byteLength"]
        self.assertIn(bin_size - buffer_size, range(4))

        def buffer_view(index):
            view = document["bufferViews"][index]
            start = view.get("byteOffset", 0)
            end = start + view["byteLength"]
            self.assertEqual(view["buffer"], 0)
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(end, buffer_size)
            return view, memoryview(binary)[start:end]

        def accessor(index, data_type, expected_type):
            types = {5121: "u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
            dimensions = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}
            entry = document["accessors"][index]
            self.assertEqual(entry["type"], expected_type)
            self.assertIn(entry["componentType"], data_type)
            view, content = buffer_view(entry["bufferView"])
            dtype = np.dtype(types[entry["componentType"]])
            offset = entry.get("byteOffset", 0)
            elements = dimensions[expected_type]
            packed = elements * dtype.itemsize
            stride = view.get("byteStride", packed)
            self.assertEqual((view.get("byteOffset", 0) + offset) % dtype.itemsize, 0)
            self.assertGreaterEqual(stride, packed)
            self.assertGreater(entry["count"], 0)
            self.assertLessEqual(offset + (entry["count"] - 1) * stride + packed, len(content))
            return np.ndarray((entry["count"], elements), dtype=dtype, buffer=content, offset=offset,
                              strides=(stride, dtype.itemsize))

        scene = document["scenes"][document.get("scene", 0)]
        mesh = document["meshes"][document["nodes"][scene["nodes"][0]]["mesh"]]
        self.assertEqual(len(mesh["primitives"]), 1)
        primitive = mesh["primitives"][0]
        self.assertEqual(primitive.get("mode", 4), 4)
        attributes = primitive["attributes"]
        positions = accessor(attributes["POSITION"], (5126,), "VEC3")
        texcoords = accessor(attributes["TEXCOORD_0"], (5126,), "VEC2")
        indices = accessor(primitive["indices"], (5121, 5123, 5125), "SCALAR")
        np.testing.assert_array_equal(positions, np.asarray(self.result.positions, dtype=np.float32))
        bounds = document["accessors"][attributes["POSITION"]]
        np.testing.assert_array_equal(bounds["min"], positions.min(axis=0))
        np.testing.assert_array_equal(bounds["max"], positions.max(axis=0))
        np.testing.assert_array_equal(texcoords, np.asarray(self.result.uvs, dtype=np.float32))
        np.testing.assert_array_equal(indices.reshape(-1, 3), self.result.triangles)
        normals = accessor(attributes["NORMAL"], (5126,), "VEC3")
        np.testing.assert_allclose(normals, np.broadcast_to([0, 0, 1], positions.shape), rtol=0, atol=1e-7)

        material = document["materials"][primitive["material"]]
        image_texture = material["pbrMetallicRoughness"]["baseColorTexture"]
        self.assertEqual(image_texture.get("texCoord", 0), 0)
        image_index = document["textures"][image_texture["index"]]["source"]
        image = document["images"][image_index]
        self.assertEqual(image["mimeType"], "image/png")
        self.assertNotIn("uri", image)
        _, packed_png = buffer_view(image["bufferView"])
        with Image.open(io.BytesIO(packed_png)) as decoded:
            self.assertEqual(decoded.format, "PNG")
            self.assertEqual(decoded.size, self.result.texture.size)
            self.assertEqual(decoded.convert("RGB").tobytes(), self.result.texture.tobytes())

    def test_unsupported_source_data_is_rejected(self):
        off_plane = self.positions.copy()
        off_plane[0, 2, 2] = 0.125
        varying_normals = self.normals.copy()
        varying_normals[0, 2] = [0.0, 0.6, 0.8]
        transparent = self.texture.convert("RGBA")
        transparent.putalpha(127)
        duplicate = np.concatenate((self.positions, self.positions[:1]))
        duplicate_uv = np.concatenate((self.uvs, self.uvs[:1] + 0.125))
        duplicate_normal = np.concatenate((self.normals, self.normals[:1]))

        invalid = (
            ("nonplanar", off_plane, self.uvs, self.normals, self.texture),
            ("varying normals", self.positions, self.uvs, varying_normals, self.texture),
            ("nonopaque alpha", self.positions, self.uvs, self.normals, transparent),
            ("overlapping source", duplicate, duplicate_uv, duplicate_normal, self.texture),
            ("no actual reduction", self.positions[:1], self.uvs[:1], self.normals[:1], self.texture),
        )
        for label, positions, uvs, normals, texture in invalid:
            with self.subTest(label=label), self.assertRaises(ValueError):
                bake_patch(positions, uvs, normals, texture, resolution=48, supersamples=1)

    def test_opaque_rgba_preserves_repeated_texture_coordinates(self):
        opaque = self.texture.convert("RGBA")
        result = bake_patch(self.positions, self.uvs + [2, -3], self.normals, opaque, resolution=96, supersamples=2)
        self.assertEqual(result.texture.mode, "RGB")
        self.assertEqual(np.asarray(result.triangles).shape, (2, 3))
        np.testing.assert_allclose(np.asarray(result.texture), np.asarray(self.result.texture), rtol=0, atol=1)


if __name__ == "__main__":
    unittest.main()
