import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from bake_section_atlas import GUTTER, indexed_primitives, pack_atlas
from lodestar_patch_layout import protect_contacts


def rectangle(left, right):
    vertices = np.array([[left, 0, 0], [right, 0, 0], [right, .5, 0], [left, .5, 0],
                         [right, 1, 0], [left, 1, 0]])
    return vertices[[[0, 1, 2], [0, 2, 3], [3, 2, 4], [3, 4, 5]]]


def section(*meshes):
    positions = np.concatenate(meshes)
    normals = np.broadcast_to([0, 0, 1], positions.shape)
    return SimpleNamespace(positions=positions, normals=normals)


class SectionAtlasTest(unittest.TestCase):
    def test_neighbors_retain_point_until_paired_edit_is_available(self):
        source = section(rectangle(0, 1), rectangle(1, 2))
        left, right = list(range(4)), list(range(4, 8))
        [[_, positions, _, triangles]] = protect_contacts(source, [left])
        self.assertEqual(len(triangles), 3)
        self.assertIn((1, .5, 0), set(map(tuple, positions)))
        reductions = protect_contacts(source, [left, right])
        self.assertEqual([len(entry[3]) for entry in reductions], [2, 2])
        for _, positions, _, _ in reductions:
            self.assertNotIn((1, .5, 0), set(map(tuple, positions)))

    def test_blocked_neighbor_reinstates_shared_boundary(self):
        boundary = [[1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0], [1, .5, 0]]
        center = [1.5, .5, 0]
        fan = np.array([[center, a, b] for a, b in zip(boundary, boundary[1:] + boundary[:1])])
        outside = np.array([[center, [1.5, .5, 1], [2, .5, 1]]])
        source = section(rectangle(0, 1), fan, outside)
        [[layout, positions, _, triangles]] = protect_contacts(source, [list(range(4)), list(range(4, 9))])
        self.assertEqual(layout.faces, tuple(range(4)))
        self.assertEqual(len(triangles), 3)
        self.assertIn((1, .5, 0), set(map(tuple, positions)))

    def test_preserves_distinct_source_indices_and_replicates_tile_gutters(self):
        source = section(rectangle(0, 1)[:1], rectangle(0, 1)[:1])
        uvs = source.positions[:, :, :2]
        primitives = indexed_primitives(source.positions, uvs, source.normals, np.ones(2), np.arange(6).reshape(2, 3))
        self.assertEqual(len(primitives[0][0]), 6)

        tiles = [SimpleNamespace(pixels=np.full((h, w, 3), rgb, np.uint8))
                 for h, w, rgb in [(8, 6, [200, 0, 0]), (5, 9, [0, 0, 170])]]
        atlas, offsets = pack_atlas(tiles, 64)
        atlas = np.asarray(atlas)
        for tile, (x, y) in zip(tiles, offsets):
            expected = np.pad(tile.pixels, ((GUTTER, GUTTER), (GUTTER, GUTTER), (0, 0)), mode="edge")
            h, w = expected.shape[:2]
            np.testing.assert_array_equal(atlas[y:y + h, x:x + w], expected)


if __name__ == "__main__":
    unittest.main()
