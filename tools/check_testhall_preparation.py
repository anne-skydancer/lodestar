"""Validate a prepared TestHall collider, openings and supported walking routes."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

from check_testhall_gates import aperture_report
from check_testhall_physics import load_scene, triangle_key
from lodestar_testhall_textures import load_textured_scene
from lodestar_walkability import testhall_traversal


def check(package, library, native_audit, reference_high=None):
    package, library = Path(package).resolve(), Path(library).resolve()
    score = json.loads((package / 'scorecard.json').read_text())
    high = load_textured_scene(library, package / 'Lodestar.glb')
    physics = package / score['physics']['file']
    imported, _ = load_scene(library, physics)
    xml = ET.parse(physics)
    ns = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}
    parts = {}
    # Our collision writer emits Y-up metre geometry with identity node transforms.
    for geometry in xml.findall('./c:library_geometries/c:geometry', ns):
        mesh = geometry.find('c:mesh', ns)
        points = np.fromstring(mesh.find('c:source/c:float_array', ns).text, sep=' ').astype(np.float32).reshape(-1, 3)
        faces = np.fromstring(mesh.find('c:triangles/c:p', ns).text, sep=' ', dtype=int).reshape(-1, 3)
        name = geometry.get('id').removeprefix('geometry-').removesuffix('_PHYS')
        parts[name] = points[faces].astype(float)
    expected = np.concatenate(list(parts.values()))
    if Counter(map(triangle_key, expected)) != Counter(map(triangle_key, imported)):
        raise ValueError('Native import changed the collision triangle stream')
    native_path = package / 'firestorm-native.json'
    result = subprocess.run([str(Path(native_audit).resolve()), str(physics), str(native_path)],
                            capture_output=True, text=True, check=True)
    (package / 'firestorm-native.log').write_text(result.stdout + result.stderr)
    native = json.loads(native_path.read_text())
    native_summary = {key: value for key, value in native.items() if key != 'models'}
    traversal = testhall_traversal(imported, check_supported_walk=True)
    gates = {name: aperture_report(high.sections[name].positions, parts[name])
             for name in ('gate', 'gate1', 'gate2', 'gate3')}
    passed = (native['original_zero_area_triangles'] == 0 and native['normalized_firestorm_rejections'] == 0
              and native['packed_firestorm_rejections'] == 0 and traversal['passed']
              and all(gate['newly_blocked_guarded_pixels'] == 0 for gate in gates.values()))
    report = {'passed': passed, 'scope': 'offline TestHall probes and native Firestorm serialization',
              'simulator_validated': False, 'native_firestorm': native_summary,
              'traversal': traversal, 'gate_apertures': gates,
              'physics_source_level': score['physics']['source_level'],
              'surface_bound_before_packing_m': score['physics']['maximum_surface_change_before_packing'],
              'high_sha256': sha256((package / 'Lodestar.glb').read_bytes()).hexdigest()}
    if reference_high:
        report['high_byte_identical_to_reference'] = (
            (package / 'Lodestar.glb').read_bytes() == Path(reference_high).read_bytes())
        report['passed'] &= report['high_byte_identical_to_reference']
    (package / 'verification.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ('passed', 'native_firestorm', 'physics_source_level',
                                                  'surface_bound_before_packing_m')}))
    return report['passed']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    parser.add_argument('--library', required=True, type=Path)
    parser.add_argument('--native-audit', required=True, type=Path)
    parser.add_argument('--reference-high', type=Path)
    args = parser.parse_args()
    return 0 if check(args.package, args.library, args.native_audit, args.reference_high) else 2


if __name__ == '__main__':
    raise SystemExit(main())
