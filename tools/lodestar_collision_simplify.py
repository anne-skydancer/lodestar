"""Position-only collision simplification with a retained source correspondence.

An accepted edge collapse is a simplicial quotient of the source mesh. Every
original vertex retains its representative, bounding the displacement of every
barycentric point on every original triangle, including collapsed faces.
No source face may disappear into a vertex/edge absent from the surviving mesh.
The bound describes surface correspondence, not simulator collision behavior.
"""

from collections import Counter, defaultdict
import heapq

import numpy as np

from lodestar_firestorm_collision import _geometry
from lodestar_firestorm_degenerate import _normalization, _quantize, _decoded_triangles, _physics_degenerate, _zero_area


def simplify_collision(source, maximum_displacement, protected=(), *, labels=None):
    source = np.asarray(source, dtype=np.float64)
    if source.ndim != 3 or source.shape[1:] != (3, 3) or not len(source) or not np.isfinite(source).all():
        raise ValueError("Collision simplification needs finite triangles")
    if not np.isfinite(maximum_displacement) or maximum_displacement < 0:
        raise ValueError("Collision displacement must be finite and nonnegative")
    points, original_faces, _ = _geometry(source)
    numeric_allowance = 64 * np.finfo(float).eps * max(1., float(np.abs(points).max()))
    labels = np.zeros(len(source), dtype=int) if labels is None else np.asarray(labels)
    if labels.shape != (len(source),):
        raise ValueError("Collision labels must identify every source face")
    frames = {}
    for label in np.unique(labels):
        original = source[labels == label].astype(np.float32)
        translation, scale = _normalization(original)
        normalized = (points.astype(np.float32) + translation) * scale
        domain = (original + translation) * scale
        _, low, span = _quantize(domain)
        fraction = np.divide(normalized - low, span, out=np.zeros_like(normalized), where=span > 0)
        packed = (fraction * np.float32(65535)).astype(np.int64)
        frames[int(label)] = (normalized, packed, _decoded_triangles(packed, low, span),
                              original.min(axis=(0, 1)), original.max(axis=(0, 1)))

    def part_rejected(indices, face_ids):
        result = np.zeros(len(indices), dtype=bool)
        for label in np.unique(labels[face_ids]):
            selected = labels[face_ids] == label
            normal, packed, decoded, _, _ = frames[int(label)]
            part_faces = indices[selected]
            result[selected] = (_physics_degenerate(normal[part_faces]) | _zero_area(packed[part_faces]) |
                                _physics_degenerate(decoded[part_faces]))
        return result
    faces = {i: tuple(map(int, face)) for i, face in enumerate(original_faces)}
    face_keys = defaultdict(set)
    part_counts = Counter(map(int, labels))
    incident = [set() for _ in points]
    neighbors = [set() for _ in points]
    for i, face in faces.items():
        face_keys[tuple(sorted(face))].add(i)
        for a in face:
            incident[a].add(i)
            neighbors[a].update(set(face) - {a})
    clusters = [{i} for i in range(len(points))]
    protected = set(map(tuple, protected))
    # Preserve all model bounds and inter-part contacts exactly.
    pinned = {i for i, point in enumerate(points) if tuple(point) in protected}
    for label in np.unique(labels):
        used = np.unique(original_faces[labels == label])
        for axis in range(3):
            pinned.update((int(used[points[used, axis].argmin()]), int(used[points[used, axis].argmax()])))
    initial_uses = defaultdict(list)
    for i, face in faces.items():
        for a, b in zip(face, face[1:] + face[:1]):
            initial_uses[min(a, b), max(a, b)].append((a, b))
    for edge, uses in initial_uses.items():
        if len(uses) > 2 or (len(uses) == 2 and uses[0] != uses[1][::-1]):
            pinned.update(edge)
    # Track oriented volume for each initially closed connected shell. Local
    # face winding alone cannot detect flattening a shell into a coplanar set.
    parent = list(range(len(source)))
    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    edge_faces = defaultdict(list)
    for index, face in faces.items():
        for a, b in zip(face, face[1:] + face[:1]):
            edge_faces[min(a, b), max(a, b)].append(index)
    for entries in edge_faces.values():
        for other in entries[1:]:
            parent[root(other)] = root(entries[0])
    groups = defaultdict(list)
    for index in faces:
        groups[root(index)].append(index)
    shell_of = {}
    shell_volume = {}
    shell_frame = {}
    for group, members in groups.items():
        used_edges = {tuple(sorted((a, b))) for index in members for a, b in
                      zip(faces[index], faces[index][1:] + faces[index][:1])}
        if any(len(initial_uses[edge]) != 2 or initial_uses[edge][0] != initial_uses[edge][1][::-1]
               for edge in used_edges):
            continue
        used = np.unique(np.asarray([faces[index] for index in members]))
        center = points[used].mean(axis=0)
        local = points[np.asarray([faces[index] for index in members])] - center
        volume = float(np.einsum('ij,ij->i', local[:, 0], np.cross(local[:, 1], local[:, 2])).sum())
        floor = 128 * np.finfo(float).eps * max(float(np.ptp(points[used], axis=0).max()) ** 3,
                                               np.finfo(float).tiny)
        if abs(volume) <= floor:
            pinned.update(map(int, used))
            continue
        for index in members:
            shell_of[index] = group
        shell_volume[group] = volume
        shell_frame[group] = center, floor
    original_shell_volumes = dict(shell_volume)
    versions = np.zeros(len(points), dtype=np.int64)
    heap = []
    maximum = 0.0
    accepted = tested = 0

    def proposal(remove, keep):
        if remove in pinned or not incident[remove] or keep not in neighbors[remove]:
            return None
        shared = incident[remove] & incident[keep]
        if len(shared) not in (1, 2):
            return None
        link = {v for index in shared for v in faces[index]} - {remove, keep}
        if neighbors[remove] & neighbors[keep] != link:
            return None
        removed_counts = Counter(int(labels[index]) for index in shared)
        if any(part_counts[label] <= count for label, count in removed_counts.items()):
            return None
        def boundary(vertex):
            return any(len(incident[vertex] & incident[other]) == 1 for other in neighbors[vertex])
        if boundary(remove) and (len(shared) != 1 or not boundary(keep)):
            return None
        members = list(clusters[remove] | clusters[keep])
        displacement = float(np.linalg.norm(points[members] - points[keep], axis=1).max())
        if displacement + numeric_allowance > maximum_displacement:
            return None
        selected = sorted(incident[remove] | incident[keep])
        surviving = [index for index in selected if index not in shared]
        if not surviving:
            return None
        before = np.asarray([faces[index] for index in surviving])
        after = np.where(before == remove, keep, before)
        # The full simplicial link condition includes faces, not only shared
        # neighbors. In particular, collapsing a tetrahedron must not leave
        # two coincident, oppositely wound triangles enclosing zero volume.
        keys = Counter(tuple(sorted(face)) for face in after.tolist())
        selected_set = set(selected)
        if any(count + len(face_keys[key] - selected_set) > max(1, len(face_keys[key]))
               for key, count in keys.items()):
            return None
        old = points[before]
        new = points[after]
        old_normal = np.cross(old[:, 1] - old[:, 0], old[:, 2] - old[:, 0])
        new_normal = np.cross(new[:, 1] - new[:, 0], new[:, 2] - new[:, 0])
        if (np.any(np.einsum('ij,ij->i', old_normal, new_normal) <= 0) or
                np.any(part_rejected(after, surviving) & ~part_rejected(before, surviving))):
            return None
        for index in selected:
            face_keys[tuple(sorted(faces[index]))].discard(index)
            part_counts[int(labels[index])] -= 1
            _, _, _, low, high = frames[int(labels[index])]
            if np.any(points[keep] < low) or np.any(points[keep] > high):
                return None
        volume_updates = {}
        for group in {shell_of[index] for index in selected if index in shell_of}:
            center, floor = shell_frame[group]
            old_local = points[np.asarray([faces[index] for index in selected if shell_of.get(index) == group])] - center
            new_local = points[np.asarray([face for index, face in zip(surviving, after)
                                          if shell_of.get(index) == group])] - center
            old_sum = float(np.einsum('ij,ij->i', old_local[:, 0], np.cross(old_local[:, 1], old_local[:, 2])).sum())
            new_sum = float(np.einsum('ij,ij->i', new_local[:, 0], np.cross(new_local[:, 1], new_local[:, 2])).sum())
            volume = shell_volume[group] - old_sum + new_sum
            if volume * shell_volume[group] <= 0 or abs(volume) <= floor:
                return None
            volume_updates[group] = volume
        # Preserve horizontal support/contact plane heights. Horizontal detail
        # can be simplified without changing where a floor supports an avatar.
        for index in incident[remove]:
            triangle = points[list(faces[index])]
            if np.ptp(triangle[:, 1]) == 0 and points[keep, 1] != points[remove, 1]:
                return None
        # Every vanished simplex must map into a surviving triangle or edge.
        edges = {tuple(sorted((a, b))) for face in after.tolist()
                 for a, b in zip(face, face[1:] + face[:1])}
        for index in shared:
            mapped = tuple(sorted(set(keep if v == remove else v for v in faces[index])))
            if len(mapped) == 2 and mapped not in edges:
                return None
        return displacement, selected, surviving, after, volume_updates

    def enqueue(a, b):
        for remove, keep in ((a, b), (b, a)):
            if remove in pinned:
                continue
            cost = float(np.linalg.norm(points[remove] - points[keep]))
            if cost <= maximum_displacement:
                heapq.heappush(heap, (cost, remove, keep, int(versions[remove]), int(versions[keep])))

    for a, adjacent in enumerate(neighbors):
        for b in sorted(adjacent):
            if a < b:
                enqueue(a, b)
    while heap and tested < 200000:
        _, remove, keep, first_version, second_version = heapq.heappop(heap)
        if versions[remove] != first_version or versions[keep] != second_version:
            continue
        tested += 1
        candidate = proposal(remove, keep)
        if candidate is None:
            continue
        displacement, selected, surviving, after, volume_updates = candidate
        shell_volume.update(volume_updates)
        touched = set(v for index in selected for v in faces[index])
        for index in selected:
            for vertex in faces[index]:
                incident[vertex].discard(index)
            del faces[index]
        for index, face in zip(surviving, after):
            faces[index] = tuple(map(int, face))
            face_keys[tuple(sorted(face))].add(index)
            part_counts[int(labels[index])] += 1
            for vertex in face:
                incident[vertex].add(index)
        clusters[keep].update(clusters[remove])
        clusters[remove].clear()
        for vertex in touched:
            neighbors[vertex] = {v for index in incident[vertex] for v in faces[index]} - {vertex}
            versions[vertex] += 1
        for vertex in sorted(touched):
            for other in sorted(neighbors[vertex]):
                enqueue(vertex, other)
        maximum = max(maximum, displacement)
        accepted += 1
    result = points[np.asarray(list(faces.values()))]
    return result, {"method": "lodestar_position_only_edge_quotient", "source_triangles": len(source),
        "triangles": len(result), "accepted_collapses": accepted, "tested_edges": tested,
        "maximum_displacement": maximum_displacement,
        "surface_correspondence_bound": maximum + (numeric_allowance if accepted else 0.0),
        "floating_point_allowance": numeric_allowance if accepted else 0.0,
        "uv_and_material_constraints": False, "support_plane_heights_preserved": True,
        "closed_shell_orientation_and_nonzero_volume_preserved": True,
        "closed_shells_checked": len(shell_volume),
        "minimum_closed_shell_volume_ratio": min((abs(volume / original_shell_volumes[group])
            for group, volume in shell_volume.items()), default=1.0),
        "work_limit_reached": bool(heap), "self_intersection_certified": False,
        "surviving_source_face_indices": list(faces)}


def simplify_collision_parts(parts, maximum_displacement):
    """Shared geometry is simplified in one transaction, with stable model labels."""
    names = list(parts)
    source = np.concatenate([parts[name] for name in names])
    labels = np.concatenate([np.full(len(parts[name]), index) for index, name in enumerate(names)])
    result, report = simplify_collision(source, maximum_displacement, labels=labels)
    final_labels = labels[report.pop('surviving_source_face_indices')]
    output = {name: result[final_labels == index] for index, name in enumerate(names)}
    if any(not len(value) for value in output.values()):
        raise ValueError("Collision simplification eliminated a named model")
    report['per_part_triangles'] = {name: len(value) for name, value in output.items()}
    return output, report
