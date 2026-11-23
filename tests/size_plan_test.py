import sys
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from lodestar_size_plan import size_proposals, apply_physical_containers
from lodestar_physics_rebuild import HullProbe
from fit_mesh_lods import write_physics_candidate
from lodestar_testhall_textures import load_textured_scene
from lodestar_textured_glb import write_textured_glb
from bake_section_atlas import indexed_primitives
from lodestar_visual_collada import write_lod_collada, write_lod_collision_collada


class SizePlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        candidates = (root/'build/Release/lodestar_c.dll', root/'build/liblodestar_c.so', root/'build/liblodestar_c.dylib')
        cls.library = next((path for path in candidates if path.is_file()), None)
        if cls.library is None:
            raise unittest.SkipTest('Build the native C library for size-plan integration tests')

    def cube(self, size):
        points = np.array([(x, y, z) for x in (0., size[0]) for y in (0., size[1]) for z in (0., size[2])])
        return HullProbe(self.library).envelope(points)

    def test_small_object_gets_a_nondegenerate_box_proposal_and_reversible_scale_map(self):
        source = self.cube((.002, .004, .006))
        original = source.copy()
        with TemporaryDirectory() as directory:
            result = size_proposals({'Micro': SimpleNamespace(positions=source)}, self.library, Path(directory))
            self.assertEqual(result['container_proposals'], ['Micro'])
            self.assertFalse(result['auto_applied'])
            self.assertFalse(result['physics_box_alone_changes_render_bounds'])
            plan = result['parts']['Micro']
            np.testing.assert_array_equal(plan['container_size'], [.01]*3)
            p = source.reshape(-1, 3)
            normalized = (p - plan['center']) / plan['container_size']
            np.testing.assert_allclose(np.array(plan['center']) + normalized * plan['container_size'], p, atol=1e-15)
            xml = ET.parse(Path(directory)/plan['box_proposal'])
            ns = {'c':'http://www.collada.org/2005/11/COLLADASchema'}
            self.assertEqual(xml.find('.//c:triangles', ns).get('count'), '12')
            self.assertTrue(plan['requires_hull_physics'])
        np.testing.assert_array_equal(source, original)

    def test_existing_large_enough_bounds_do_not_get_another_container(self):
        source = self.cube((.01, .01, .01))
        with TemporaryDirectory() as directory:
            result = size_proposals({'Existing': SimpleNamespace(positions=source)}, self.library, Path(directory))
            self.assertFalse(result['requires_container'])
            self.assertFalse((Path(directory)/'size-proposals').exists())

    def test_import_scale_recalculates_physical_frame_without_stretching_visible_geometry(self):
        source = self.cube((.002, .004, .006))
        for scale in (.01, .25, 1., 10.):
            with self.subTest(scale=scale), TemporaryDirectory() as directory:
                report = size_proposals({'Micro': SimpleNamespace(positions=source)}, self.library, Path(directory), import_scale=scale)
                frames, collision = apply_physical_containers(report, self.library, Path(directory))
                self.assertEqual(bool(frames), scale < 5.)
                if frames:
                    size = np.asarray(frames['Micro']['container_size'])
                    center = np.asarray(frames['Micro']['center'])
                    encoded = (source-center)/size
                    decoded_world = (encoded*size+center)*scale
                    np.testing.assert_allclose(decoded_world, source*scale, rtol=1e-12, atol=1e-15)
                    self.assertTrue(np.all(size*scale >= np.float32(.01)))
                    self.assertEqual(len(collision['Micro']), 12)

    def test_physical_frame_adds_only_closed_collision_and_keeps_every_visual_stream(self):
        for dimensions in ((.002, .004, .006), (.002, .004, .02)):
            source = self.cube(dimensions)
            source += [1., -2., .5]
            normals = np.broadcast_to([0., 1., 0.], source.shape).copy()
            uvs = np.zeros((len(source), 3, 2))
            scene = SimpleNamespace(positions=source, normals=normals, uvs=uvs,
                materials=np.zeros(len(source), dtype=int), textures={0: (Image.new('RGB', (1, 1)), True)})
            originals = [value.copy() for value in (source, normals, uvs, scene.materials)]
            with TemporaryDirectory() as directory:
                folder = Path(directory)
                sizing = size_proposals({'Micro': scene}, self.library, folder)
                frames, collision = apply_physical_containers(sizing, self.library, folder)
                self.assertTrue(sizing['auto_applied'])
                self.assertFalse(sizing['requires_frame_review'])
                self.assertEqual(sizing['physical_container_models'], ['Micro'])
                self.assertEqual(sizing['parts']['Micro']['render_triangles_added'], 0)
                self.assertEqual(len(collision['Micro']), 12)
                self.assertTrue(np.all(np.ptp(collision['Micro'], axis=(0, 1)) >= np.float32(.01)))
                for suffix in ('', '_LOD2', '_LOD1', '_LOD0'):
                    path = folder/f'visual{suffix}.dae'
                    write_lod_collada(path, {'Micro': scene}, suffix=suffix, physical_frames=frames)
                    imported = load_textured_scene(self.library, path, folder)
                    self.assertEqual(len(imported.positions), len(source))
                    np.testing.assert_allclose(np.sort(imported.positions.reshape(-1, 3), axis=0),
                        np.sort(source.astype(np.float32).reshape(-1, 3), axis=0), atol=1e-9)
                    xml = ET.parse(path)
                    ns = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}
                    self.assertIsNotNone(xml.find('.//c:technique[@profile="LODESTAR"]/c:physical_container', ns))
                    replanned = size_proposals(imported.sections, self.library, folder, declared_bounds=imported.declared_bounds)
                    self.assertEqual(replanned['container_proposals'], [])
                    self.assertEqual(replanned['frame_preservation'], ['Micro' + suffix])
                write_lod_collision_collada(folder/'physics.dae', collision, physical_frames=frames)
            for before, after in zip(originals, (source, normals, uvs, scene.materials)):
                np.testing.assert_array_equal(before, after)

    def test_valid_triangles_do_not_override_pending_container_review(self):
        source = self.cube((.002, .004, .006))
        with TemporaryDirectory() as directory:
            report = {}
            path = write_physics_candidate(Path(directory), {'Micro': source}, report, force_diagnostic=True)
            self.assertEqual(path.name, 'Lodestar_unresolved_PHYS.dae')
            self.assertEqual(report['status'], 'blocked_size_container_proposal')
            self.assertFalse((Path(directory)/'Lodestar_PHYS.dae').exists())

    def test_removed_padding_is_recognized_without_generating_degenerate_faces(self):
        body = self.cube((.002, .004, .006))
        padding = np.array([[[-.01]*3]*3, [[.01]*3]*3])
        for separate_primitive in (False, True):
            with self.subTest(separate_primitive=separate_primitive), TemporaryDirectory() as directory:
                folder = Path(directory)
                groups = [body, padding] if separate_primitive else [np.concatenate((body, padding))]
                primitives = []
                for triangles in groups:
                    normals = np.broadcast_to([0., 1., 0.], triangles.shape).copy()
                    uvs = np.zeros((len(triangles), 3, 2))
                    primitives.extend(indexed_primitives(triangles, uvs, normals,
                        np.zeros(len(triangles), dtype=int), np.arange(triangles.size//3).reshape(-1, 3)))
                source = folder/'Existing.glb'
                write_textured_glb(source, {'Existing': primitives}, {0: (Image.new('RGB', (1, 1), 'white'), True)})
                scene = load_textured_scene(self.library, source)
                self.assertEqual(len(scene.positions), len(body))
                area = np.linalg.norm(np.cross(scene.positions[:, 1]-scene.positions[:, 0],
                                               scene.positions[:, 2]-scene.positions[:, 0]), axis=1)
                self.assertTrue(np.all(area > 0))
                result = size_proposals(scene.sections, self.library, folder, declared_bounds=scene.declared_bounds)
                self.assertFalse(result['requires_container'])
                self.assertEqual(result['frame_preservation'], ['Existing'])
                self.assertTrue(result['requires_frame_review'])
                np.testing.assert_allclose(result['parts']['Existing']['container_size'], [.02]*3, rtol=1e-6)
                self.assertFalse((folder/'size-proposals').exists())
                cli = Path(__file__).resolve().parents[1]/'build/Release/inverseglod.exe'
                if cli.is_file():
                    native = json.loads(subprocess.run([str(cli), '--plan-size', str(source)],
                        capture_output=True, text=True, check=True).stdout)
                    self.assertTrue(native['existing_container_detected'])
                    self.assertTrue(native['requires_frame_preservation'])
                    self.assertFalse(native['requires_container'])


if __name__ == '__main__':
    unittest.main()
