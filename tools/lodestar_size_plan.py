"""Native size proposals that keep visible geometry distinct from its container."""

import ctypes as ct
import json
from pathlib import Path

import numpy as np

from lodestar_physics_rebuild import HullProbe, _Vertex
from lodestar_visual_collada import write_lod_collision_collada


class _SizePlan(ct.Structure):
    _fields_ = [(name, ct.c_double * 3) for name in ('visible_minimum', 'visible_maximum',
        'container_minimum', 'container_maximum', 'center', 'visible_size', 'container_size', 'visible_fraction')]
    _fields_ += [('requires_container', ct.c_uint32), ('requires_hull_physics', ct.c_uint32)]


def size_proposals(parts, library, output, *, declared_bounds=None, import_scale=1.0):
    """Compute physical frame proposals without changing authored render streams.

    Existing carrier geometry participates in the input bounds, so an already
    large-enough asset does not get another proposal. The box files are local
    centered physics proposals; their source-frame translation is explicit.
    """
    output = Path(output)
    if not np.isfinite(import_scale) or import_scale <= 0:
        raise ValueError('Physical container import scale must be finite and positive')
    source_minimum, source_triangle_minimum = .01/import_scale, .5/import_scale
    probe = HullProbe(library)
    api = probe.api
    api.lodestar_plan_mesh_size.argtypes = [ct.POINTER(_Vertex), ct.c_size_t, ct.c_double, ct.c_double, ct.POINTER(_SizePlan)]
    api.lodestar_plan_mesh_size.restype = ct.c_uint32
    report = {'minimum_prim_dimension_m': .01, 'triangle_physics_minimum_m': .5,
              'import_scale': float(import_scale), 'minimum_source_frame_dimension': source_minimum,
              'parts': {}, 'container_proposals': [], 'frame_preservation': [], 'auto_applied': False,
              'shared_visual_bounds_required': True, 'physics_box_alone_changes_render_bounds': False,
              'zero_area_padding_used': False}
    for name, part in parts.items():
        points = np.ascontiguousarray(np.unique(part.positions.reshape(-1, 3), axis=0), dtype=np.float32)
        plan = _SizePlan()
        if api.lodestar_plan_mesh_size(points.ctypes.data_as(ct.POINTER(_Vertex)), len(points), source_minimum, source_triangle_minimum, ct.byref(plan)):
            raise ValueError(api.lodestar_last_error().decode('utf-8', errors='replace'))
        values = {field: list(getattr(plan, field)) for field, _ in _SizePlan._fields_[:8]}
        values.update(requires_container=bool(plan.requires_container), requires_hull_physics=bool(plan.requires_hull_physics))
        values['visible_size_after_import_scale_m'] = (np.array(plan.visible_size)*import_scale).tolist()
        existing = False
        bounds = (declared_bounds or {}).get(name)
        if plan.requires_container and bounds:
            declared = np.ascontiguousarray([bounds['minimum'],bounds['maximum']], dtype=np.float32)
            frame = _SizePlan()
            if api.lodestar_plan_mesh_size(declared.ctypes.data_as(ct.POINTER(_Vertex)), 2, source_minimum, source_triangle_minimum, ct.byref(frame)):
                raise ValueError('Cannot inspect declared source bounds')
            extent_allowance = 8*np.finfo(np.float32).eps*np.maximum(np.abs(plan.visible_minimum), np.abs(plan.visible_maximum))
            has_padding = np.any(np.array(frame.visible_minimum) < np.array(plan.visible_minimum)-extent_allowance) or np.any(
                np.array(frame.visible_maximum) > np.array(plan.visible_maximum)+extent_allowance)
            if not frame.requires_container or has_padding:
                existing = True
                values.update(existing_container_detected=True, requires_container=False, requires_frame_preservation=True,
                    requires_hull_physics=bool(frame.requires_hull_physics),
                    center=list(frame.center), container_size=list(frame.container_size),
                    container_minimum=list(frame.container_minimum), container_maximum=list(frame.container_maximum),
                    visible_fraction=(np.array(plan.visible_size)/np.array(frame.container_size)).tolist())
                report['frame_preservation'].append(name)
        if plan.requires_container and not existing:
            folder = output / 'size-proposals'
            folder.mkdir(parents=True, exist_ok=True)
            dimensions = np.array(plan.container_size)
            corners = np.array([(x, y, z) for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)]) * dimensions
            box = probe.envelope(corners)
            path = folder / f'part-{len(report["container_proposals"])+1}-container-box.dae'
            write_lod_collision_collada(path, {name + '-Container': box})
            values.update(box_proposal=str(path.relative_to(output)), box_frame='local centered',
                          translation_to_source_frame_m=list(plan.center), box_vertices=8, box_triangles=len(box),
                          collider_larger_than_visible_geometry=True,
                          normalized_position_formula='(source_position - center) / container_size',
                          inverse_position_formula='center + normalized_position * container_size')
            report['container_proposals'].append(name)
        report['parts'][name] = values
    report['requires_container'] = bool(report['container_proposals'])
    report['requires_frame_review'] = bool(report['container_proposals'] or report['frame_preservation'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'size-proposals.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def apply_physical_containers(report, library, output):
    """Apply the authorized physical frames; leave every visual stream untouched."""
    probe = HullProbe(library)
    frames, collision = {}, {}
    for name in report['container_proposals'] + report['frame_preservation']:
        plan = report['parts'][name]
        minimum = np.asarray(plan['container_minimum'], dtype=np.float64)
        maximum = np.asarray(plan['container_maximum'], dtype=np.float64)
        # Importer coordinates are float32. Round the physical bounds outwards
        # so translated containers cannot round below the legal dimension.
        minimum = np.nextafter(minimum.astype(np.float32), np.float32(-np.inf))
        maximum = np.nextafter(maximum.astype(np.float32), np.float32(np.inf))
        size = maximum - minimum
        if np.any(size < np.float32(report['minimum_source_frame_dimension'])):
            raise ValueError(f'Physical container is below the representable legal size: {name}')
        plan.update(container_minimum=minimum.astype(float).tolist(), container_maximum=maximum.astype(float).tolist(),
            container_size=size.astype(float).tolist(), center=((minimum.astype(float)+maximum.astype(float))*.5).tolist(),
            visible_fraction=(np.array(plan['visible_size'])/size).tolist(),
            container_size_after_import_scale_m=(size.astype(float)*report['import_scale']).tolist())
        # Use source-frame bounds for both collider and importer metadata.
        corners = np.array([(x,y,z) for x in (minimum[0],maximum[0])
            for y in (minimum[1],maximum[1]) for z in (minimum[2],maximum[2])])
        box = probe.envelope(corners)
        if np.any(np.all(np.cross(box[:,1]-box[:,0], box[:,2]-box[:,0]) == 0, axis=1)):
            raise ValueError(f'Physical container cannot be represented without degenerate faces: {name}')
        frames[name] = plan
        collision[name] = box
        plan.update(physical_frame_applied=True, render_triangles_added=0,
                    physics_representation='closed convex container box', box_vertices=8, box_triangles=len(box))
    report.update(auto_applied=bool(frames), requires_frame_review=False,
                  physical_container_models=list(frames),
                  physical_frame_encoding='LODESTAR physical_container version 1 in COLLADA geometry extra',
                  unmodified_importers_supported=not bool(frames))
    (Path(output)/'size-proposals.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return frames, collision
