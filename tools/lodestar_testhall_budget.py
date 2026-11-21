"""Allocate a scene triangle budget across topology-preserving section proposals."""

from dataclasses import dataclass, replace

import numpy as np

from lodestar_nonplanar import NonplanarResult, simplify_nonplanar


@dataclass(frozen=True)
class BudgetPlan:
    replacements: dict[str, NonplanarResult]
    candidate_triangles: int
    report: dict


def _choose_restorations(choices, available):
    """Use available triangles to retain source components with more sampled error."""
    states = {0: (0.0, ())}
    for key, count, quality in choices:
        for used, (score, selection) in list(states.items()):
            total = used + count
            if total > available:
                continue
            proposed = score + quality
            current = states.get(total)
            if current is None or proposed > current[0]:
                states[total] = (proposed, (*selection, key))
    return set(states[max(states)][1])


def _restore_components(section, result, selected):
    if not selected:
        return result

    positions, labels, components = [], [], []
    for index, component in enumerate(result.components):
        if index in selected:
            triangles = section.positions[result.source_components == index]
            component = {**component, "candidate_triangles": len(triangles), "estimated_relative_error": 0.0,
                         "sampled_relative_error": 0.0, "retained_for_budget": True}
        else:
            triangles = result.positions[result.target_components == index]
        positions.append(triangles)
        labels.append(np.full(len(triangles), index, dtype=np.intp))
        components.append(component)
    return replace(result, positions=np.concatenate(positions), target_components=np.concatenate(labels),
                   components=components)


def plan_budget(library, sections, baseline_counts, *, target_triangles=4769, protected_nodes=(),
                max_relative_error=0.003, max_sampled_distance_m=0.05, node_contacts=None):
    """Return changed nonplanar sections; other sections keep the caller's baseline.

    Start with proposals screened against the same world-space sample ceiling.
    If that leaves spare triangles, retain original geometry from some of the
    proposed components. Independent scene geometry, appearance, and traversal
    checks belong to the caller before publishing a candidate.
    """
    if (not isinstance(target_triangles, (int, np.integer)) or target_triangles < 1 or
            not baseline_counts or not all(isinstance(count, (int, np.integer)) and count > 0
                                          for count in baseline_counts.values())):
        raise ValueError("Expected positive integral baseline counts and target triangle count")
    if not 0 <= max_relative_error <= 1 or not np.isfinite(max_sampled_distance_m) or max_sampled_distance_m <= 0:
        raise ValueError("Expected a bounded relative error and a positive finite sampled distance")

    protected_nodes = set(protected_nodes)
    node_contacts = node_contacts or {}
    proposed, choices, extents = {}, [], {}
    baseline_total = int(sum(baseline_counts.values()))
    proposed_total = baseline_total
    if proposed_total > target_triangles:
        for name in sorted((sections.keys() & baseline_counts.keys()) - protected_nodes):
            section = sections[name]
            extent = float(np.ptp(section.positions, axis=(0, 1)).max())
            if not np.isfinite(extent) or extent <= 0:
                raise ValueError(f"{name} has no finite nonzero geometry extent")
            pins = (*section.outside_positions, *node_contacts.get(name, ()))
            result = simplify_nonplanar(library, section.positions, 1, outside_positions=pins,
                                        max_relative_error=max_relative_error,
                                        max_sampled_relative_error=min(1.0, max_sampled_distance_m / extent))
            if len(result.positions) >= baseline_counts[name]:
                continue
            proposed_total += len(result.positions) - baseline_counts[name]
            proposed[name] = result
            extents[name] = extent
            triangles = section.positions
            area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
            area_by_component = np.bincount(result.source_components, weights=area)
            for index, component in enumerate(result.components):
                count = component["source_triangles"] - component["candidate_triangles"]
                if count > 0:
                    distance = component["sampled_relative_error"] * extent
                    choices.append(((name, index), count, float(area_by_component[index] * distance ** 2)))

    retained = _choose_restorations(choices, max(0, target_triangles - proposed_total))
    replacements = {}
    for name, result in proposed.items():
        selected = {index for section, index in retained if section == name}
        replacements[name] = _restore_components(sections[name], result, selected)

    report_sections = []
    for name, baseline in sorted(baseline_counts.items()):
        result = replacements.get(name)
        count = len(result.positions) if result is not None else int(baseline)
        distance = (max(component["sampled_relative_error"] for component in result.components) * extents[name]
                    if result is not None else None)
        report_sections.append({"node": name, "baseline_triangles": int(baseline), "candidate_triangles": count,
                                "maximum_sampled_component_distance_m": distance,
                                "original_components_retained_for_budget": sum(key[0] == name for key in retained)})
    actual = sum(section["candidate_triangles"] for section in report_sections)
    return BudgetPlan(replacements, actual,
                      {"baseline_triangles": baseline_total, "proposed_triangles": proposed_total,
                       "candidate_triangles": actual, "target_triangles": int(target_triangles),
                       "met_target": actual <= target_triangles, "exact_target": actual == target_triangles,
                       "max_relative_error": max_relative_error, "max_sampled_distance_m": max_sampled_distance_m,
                       "sections": report_sections})
