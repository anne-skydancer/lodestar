"""Select collision from Lodestar levels independently of visual attributes."""

from lodestar_collision_simplify import simplify_collision_parts
from lodestar_physics_rebuild import rebuild_physics


def prepare_physics(levels, visual_report, library):
    """Prefer the lowest usable level; escalate displacement only when needed.

    The automatic ceiling is the selected LOD's surface error allowance. Try
    progressively smaller fractions first. Each attempt keeps named models,
    checks their own uploader normalization, and reports the composed bound
    against High rather than calling a simplified collider exact.
    """
    attempts = []
    best = None
    retained = {}

    def record(level, repaired, report, simplification, limit):
        level_report = visual_report['levels'][level]
        visual_bound = max((d.get('surface_bound', 0.0) for d in level_report['decisions']), default=0.0)
        failures = sum(len(p['unresolved_triangle_indices']) for p in report['parts'].values())
        export_bound = max((max(p.get('exact_strip_refinement', {}).get('float32_export_displacement_bound', 0.0),
                                p.get('exact_edge_refinement', {}).get('float32_export_displacement_bound', 0.0))
                            for p in report['parts'].values()), default=0.0)
        attempts.append({'source_level': level, 'maximum_displacement': limit,
                         'triangles': sum(map(len, repaired.values())), 'unresolved_triangles': failures,
                         'all_parts_have_supported_representation': report['all_parts_have_supported_representation']})
        report = dict(report, source_level=level, simplification=simplification,
            reference='High via the reported visual-LOD and position-only correspondences',
            visual_surface_bound=visual_bound, float32_export_displacement_bound=export_bound,
            maximum_surface_change_before_packing=visual_bound +
                simplification['surface_correspondence_bound'] + report['maximum_surface_change_before_packing'] + export_bound,
            selection_attempts=list(attempts),
            exact_surface_preservation=(visual_bound == 0 and simplification['surface_correspondence_bound'] == 0),
            simulator_validated=False)
        return report

    # Prefer a fully supported hull set at the lowest feasible visual level.
    # A valid triangle fallback at Lowest must not hide a supported hull at Low.
    for level in ('lowest', 'low', 'medium', 'high'):
        source = {name: part.positions for name, part in levels[level].items()}
        repaired, report = rebuild_physics(source, library)
        report = record(level, repaired, report,
                        {'method': 'retained_visual_level_positions', 'surface_correspondence_bound': 0.0}, 0.0)
        retained[level] = repaired, report
        if all(p.get('convex_conversion', {}).get('supported', False) for p in report['parts'].values()):
            report['selected_representation'] = 'native_hulls'
            return repaired, report
    for level in ('lowest', 'low', 'medium', 'high'):
        source = {name: part.positions for name, part in levels[level].items()}
        level_report = visual_report['levels'][level]
        ceiling = level_report['world_units_per_pixel'] * level_report['maximum_pixel_error']
        valid = []
        for fraction in (0.0, .125, .25, .5, 1.0):
            limit = ceiling * fraction
            if fraction:
                candidate, simplification = simplify_collision_parts(source, limit)
                repaired, report = rebuild_physics(candidate, library)
                report = record(level, repaired, report, simplification, limit)
            else:
                repaired, report = retained[level]
            failures = sum(len(p['unresolved_triangle_indices']) for p in report['parts'].values())
            rank = (failures, sum(map(len, repaired.values())))
            if best is None or rank < best[0]:
                best = rank, repaired, report
            if report['all_parts_have_supported_representation']:
                report['selected_representation'] = 'native_hulls' if all(
                    p.get('convex_conversion', {}).get('supported', False) for p in report['parts'].values()) else 'triangles'
                hull_vertices = sum(sum(p.get('convex_conversion', {}).get('vertices_per_hull', []))
                                    for p in report['parts'].values())
                cost = (0, hull_vertices) if report['selected_representation'] == 'native_hulls' else (1, sum(map(len, repaired.values())))
                valid.append((cost, report['maximum_surface_change_before_packing'], repaired, report))
        if valid:
            _, _, repaired, report = min(valid, key=lambda candidate: candidate[:2])
            report['selection_attempts'] = list(attempts)
            return repaired, report
    _, repaired, report = best
    report['selection_attempts'] = attempts
    return repaired, report
