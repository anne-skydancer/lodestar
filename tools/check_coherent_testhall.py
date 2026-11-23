"""Check decoded native convex pieces against the TestHall source and routes."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from check_testhall_gates import aperture_report
from lodestar_physics_rebuild import HullProbe
from lodestar_testhall_textures import load_textured_scene
from lodestar_walkability import testhall_traversal


def check(package, native_result, library, output):
    package, native_result = Path(package), Path(native_result)
    scene = load_textured_scene(library, package/'Lodestar.glb')
    audit = json.loads(native_result.read_text())
    probe = HullProbe(library)
    ns = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}
    xml = ET.parse(package/'Lodestar_PHYS.dae')
    parts = {}
    for geometry in xml.findall('./c:library_geometries/c:geometry', ns):
        mesh = geometry.find('c:mesh', ns)
        points = np.fromstring(mesh.find('c:source/c:float_array', ns).text, sep=' ').astype(np.float32).reshape(-1, 3)
        faces = np.fromstring(mesh.find('c:triangles/c:p', ns).text, sep=' ', dtype=int).reshape(-1, 3)
        parts[geometry.get('id').removeprefix('geometry-').removesuffix('_PHYS')] = points[faces].astype(float)
    gates = {}
    for model in audit['models']:
        name = model['name']
        hulls = [probe.envelope(np.array(points)) for points in model['pieces_y_up_m']]
        parts[name] = np.concatenate(hulls)
        gates[name] = dict(aperture_report(scene.sections[name].positions, parts[name]),
                           hulls=model['hulls'], packed_hulls_have_volume=model['packed_hulls_have_volume'],
                           high_payload_unchanged=model['high_payload_unchanged'])
    traversal = testhall_traversal(np.concatenate(list(parts.values())), check_supported_walk=True)
    passed = traversal['passed'] and all(g['newly_blocked_guarded_pixels'] == 0 and
        g['packed_hulls_have_volume'] and g['high_payload_unchanged'] for g in gates.values())
    report = {'passed': passed, 'scope': 'native hull backend and decoded asset data, offline probes',
              'gui_retest_complete': False, 'simulator_validated': False, 'thin_parts': gates, 'traversal': traversal}
    Path(output).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'passed': passed, 'hulls': {n:g['hulls'] for n,g in gates.items()},
                      'all_openings_retained': all(g['newly_blocked_guarded_pixels']==0 for g in gates.values()),
                      'supported_walks_passed': traversal['passed']}))
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    parser.add_argument('native_result', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--library', required=True, type=Path)
    args=parser.parse_args()
    return 0 if check(args.package, args.native_result, args.library, args.output) else 2


if __name__ == '__main__':
    raise SystemExit(main())
