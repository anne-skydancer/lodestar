"""Transfer diffuse textures to bounded, simplified nonplanar triangles."""

from dataclasses import dataclass

import numpy as np

from bake_planar_patch import grid, linear_image, rgb8, sample
from bake_section_atlas import GUTTER, pack_atlas


TOP_V = np.array([1, -1])
TOP_OFFSET = np.array([0, 1])


@dataclass
class _Tile:
    face: int
    uvs: np.ndarray
    pixels: np.ndarray


def _face_normals(triangles):
    first, second = np.moveaxis(triangles[:, 1:] - triangles[:, :1], 1, 0)
    cross = np.cross(first, second)
    area = np.linalg.norm(cross, axis=1)
    if not np.isfinite(area).all() or not np.all(area > 0):
        raise ValueError("Diffuse transfer requires nondegenerate triangles")
    return cross / area[:, None]


def _barycentric_gradients(ab, ac, normals):
    area = np.einsum("ij,ij->i", np.cross(ab, ac), normals)
    gradients = np.stack((np.cross(ac, normals), np.cross(normals, ab)), axis=1)
    return gradients / area[:, None, None]


def _closest_points(points, triangles, normals):
    """Return a source triangle, barycentric weights and distance for each point."""
    a, b, c = np.moveaxis(triangles, 1, 0)
    # Barycentrics in the plane also describe the perpendicular projection.
    ab, ac = b - a, c - a
    delta = points[:, None, :] - a
    gradients = _barycentric_gradients(ab, ac, normals)
    weight_b, weight_c = np.moveaxis(np.einsum("mkd,kcd->mkc", delta, gradients), -1, 0)
    weight_a = 1 - weight_b - weight_c
    inside = (weight_a >= 0) & (weight_b >= 0) & (weight_c >= 0)
    distances = np.where(inside, np.sum(delta * normals, axis=-1) ** 2, np.inf)
    winners = distances.argmin(axis=1)
    rows = np.arange(len(points))
    squared = distances[rows, winners]
    weights = np.column_stack((weight_a[rows, winners], weight_b[rows, winners], weight_c[rows, winners]))

    # Exterior projections are nearest on one of the three line segments.
    # Check all segments as well to resolve ties at boundary points.
    for start_index, end_index in ((0, 1), (1, 2), (2, 0)):
        start = triangles[:, start_index]
        edge = triangles[:, end_index] - start
        offset = points[:, None, :] - start
        along = np.clip(np.sum(offset * edge, axis=-1) / np.sum(edge * edge, axis=-1), 0, 1)
        delta = offset - along[..., None] * edge
        distances = np.sum(delta * delta, axis=-1)
        closest = distances.argmin(axis=1)
        best = distances[rows, closest]
        update = best < squared
        winners[update] = closest[update]
        squared[update] = best[update]
        t = along[rows[update], closest[update]]
        weights[update] = 0
        weights[update, start_index] = 1 - t
        weights[update, end_index] = t
    return winners, weights, np.sqrt(squared)


class _Surface:
    def __init__(self, triangles, normals, uvs, materials, textures):
        self.triangles = triangles
        self.geometric = _face_normals(triangles)
        self.normals = normals
        self.uvs = uvs
        self.materials = materials
        self.minimum = triangles.min(axis=1)
        self.maximum = triangles.max(axis=1)
        self.textures = {material: (linear_image(image), repeat) for material, (image, repeat) in textures.items()}
        sizes = np.array([textures[int(material)][0].size for material in materials])
        edges = triangles[:, 1:] - triangles[:, :1]
        texel_edges = (uvs[:, 1:] - uvs[:, :1]) * sizes[:, None, :]
        gradients = _barycentric_gradients(edges[:, 0], edges[:, 1], self.geometric)
        self.gradient = gradients.transpose(0, 2, 1) @ texel_edges

    def neighbors(self, triangle, normal, maximum_distance):
        lower, upper = triangle.min(axis=0), triangle.max(axis=0)
        offset = np.maximum(np.maximum(self.minimum - upper, lower - self.maximum), 0)
        return np.flatnonzero((np.sum(offset ** 2, axis=1) <= maximum_distance ** 2) &
                              (self.geometric @ normal > 0))

    def project(self, points, neighbors, maximum_distance):
        faces, weights, distances = _closest_points(points, self.triangles[neighbors], self.geometric[neighbors])
        if distances.max() > maximum_distance:
            raise ValueError(f"Projection distance {distances.max():.6g} exceeds {maximum_distance:.6g}")
        return neighbors[faces], weights, distances

    def colors(self, faces, weights):
        uv = np.einsum("ij,ijk->ik", weights, self.uvs[faces])
        materials = self.materials[faces]
        colors = np.empty((len(faces), 3), dtype=np.float64)
        for material in np.unique(materials):
            selected = materials == material
            texture, repeat = self.textures[material]
            colors[selected] = sample(texture, uv[selected], repeat=repeat)
        return colors


def _domain(triangle, normal):
    # Keep the long triangle edge horizontal to use fewer empty atlas texels.
    sides = np.roll(triangle, -1, axis=0) - triangle
    edge = sides[np.linalg.norm(sides, axis=1).argmax()]
    tangent = edge / np.linalg.norm(edge)
    basis = np.stack((tangent, np.cross(tangent, normal)))
    coordinates = (triangle - triangle[0]) @ basis.T
    extent = np.ptp(coordinates, axis=0)
    return basis, extent, (coordinates - coordinates.min(axis=0)) / extent


def _domain_points(points, triangle_uv, extent, triangle):
    # Dilate the edge color through the unused rectangle outside each face.
    # The closest edge is measured in the face's physical plane, not UV scale.
    face = np.column_stack((triangle_uv * extent, np.zeros(3)))
    samples = np.column_stack((points * extent, np.zeros(len(points))))
    _, weights, _ = _closest_points(samples, face[None, ...], np.array([[0, 0, -1.0]]))
    return weights @ triangle


def _tile(surface, face, triangle, normal, neighbors, distance, texel_scale, max_tile, supersamples):
    basis, extent, triangle_uv = _domain(triangle, normal)
    texel_rates = np.linalg.norm(basis @ surface.gradient[neighbors], axis=2).max(axis=0)
    width, height = np.clip(np.ceil(extent * texel_rates * texel_scale), 4, max_tile).astype(int)
    colors = np.zeros((height * width, 3), dtype=np.float64)
    distance_max = 0.0
    distance_sum = 0.0
    for row in range(supersamples):
        for column in range(supersamples):
            points = grid(width, height, (column + 0.5) / supersamples, (row + 0.5) / supersamples)
            positions = _domain_points(points, triangle_uv, extent, triangle)
            # Limit memory independent of the texture or local face count.
            for start in range(0, len(positions), 128):
                end = start + 128
                faces, weights, distances = surface.project(positions[start:end], neighbors, distance)
                colors[start:end] += surface.colors(faces, weights)
                distance_max = max(distance_max, float(distances.max()))
                distance_sum += float(distances.sum())
    count = int(height * width * supersamples ** 2)
    pixels = rgb8(colors / supersamples ** 2).reshape(height, width, 3)
    return _Tile(face, triangle_uv, pixels), distance_max, distance_sum, count


def _exact_key(triangle, component):
    return (component, *triangle.ravel())


def _exact_lookup(triangles, components):
    matches = {}
    for face, triangle in enumerate(triangles):
        for corner in range(3):
            order = np.roll(np.arange(3), -corner)
            key = _exact_key(triangle[order], components[face])
            matches[key] = None if key in matches else (face, order)
    return matches


def bake_nonplanar(section, candidate_positions, textures, *, max_projection_distance,
                   texel_scale=0.5, max_tile=64, max_atlas=2048, supersamples=1,
                   source_components=None, target_components=None, retain_exact=True):
    """Return top-origin UVs, normals, face material IDs, textures and metrics.

    Candidate triangle corners stay in their original order and positions.
    Unchanged faces retain their original attributes. Changed faces use
    source-facing proximity for diffuse projection and entirely padded tiles.
    Provide matching per-face component IDs to isolate neighboring shells.
    Use retain_exact=False for an atlas covering the entire candidate.
    An unsupported or over-distance projection raises before any file export.
    """
    source = np.asarray(section.positions, dtype=np.float64)
    source_uvs = np.asarray(section.source_uvs, dtype=np.float64) * TOP_V + TOP_OFFSET
    source_normals = np.asarray(section.normals, dtype=np.float64)
    materials = np.asarray(section.triangle_materials)
    candidate = np.asarray(candidate_positions, dtype=np.float64)
    if (source.ndim != 3 or source.shape[1:] != (3, 3) or len(source) == 0 or
            candidate.ndim != 3 or candidate.shape[1:] != (3, 3) or len(candidate) == 0 or
            source_uvs.shape != (len(source), 3, 2) or source_normals.shape != source.shape or
            materials.shape != (len(source),) or materials.dtype.kind not in "iu" or
            not all(np.isfinite(array).all() for array in (source, source_uvs, source_normals, candidate))):
        raise ValueError("Diffuse transfer needs finite triangles with source UVs, normals and material IDs")
    if (not np.isfinite(max_projection_distance) or max_projection_distance <= 0 or
            not np.isfinite(texel_scale) or texel_scale <= 0 or max_tile < 4 or
            max_atlas < max_tile + 2 * GUTTER or not 1 <= supersamples <= 4):
        raise ValueError("Diffuse transfer needs positive projection and density, valid tile and atlas sizes and 1 to 4 samples")
    required = set(map(int, materials))
    if not required.issubset(textures):
        raise ValueError("Diffuse transfer requires source textures for every material")
    if source_components is None and target_components is None:
        source_components = np.zeros(len(source), dtype=np.int8)
        target_components = np.zeros(len(candidate), dtype=np.int8)
    else:
        source_components = np.asarray(source_components)
        target_components = np.asarray(target_components)
        if (source_components.shape != (len(source),) or target_components.shape != (len(candidate),) or
                source_components.dtype.kind not in "iu" or target_components.dtype.kind not in "iu" or
                not set(map(int, target_components)).issubset(map(int, source_components))):
            raise ValueError("Diffuse transfer needs matching integer component IDs for each face")

    surface = _Surface(source, source_normals, source_uvs, materials, textures)
    candidate_normals = _face_normals(candidate)
    lookup = _exact_lookup(source, source_components)
    atlas_id = max(textures) + 1
    output_uv = np.empty((len(candidate), 3, 2), dtype=np.float64)
    output_normals = np.empty_like(candidate)
    output_materials = np.full(len(candidate), atlas_id, dtype=np.int64)
    tiles = []
    retained = 0
    distance_max = 0.0
    distance_sum = 0.0
    sample_count = 0
    for face, (triangle, normal) in enumerate(zip(candidate, candidate_normals)):
        component = target_components[face]
        exact = lookup.get(_exact_key(triangle, component))
        if exact is not None and retain_exact:
            source_face, order = exact
            output_uv[face] = source_uvs[source_face, order]
            output_normals[face] = source_normals[source_face, order]
            output_materials[face] = materials[source_face]
            retained += 1
            continue
        neighbors = surface.neighbors(triangle, normal, max_projection_distance)
        neighbors = neighbors[source_components[neighbors] == component]
        if not len(neighbors):
            raise ValueError(f"Candidate face {face} has no similarly facing source within projection bounds")
        try:
            faces, weights, distances = surface.project(triangle, neighbors, max_projection_distance)
            normals = np.einsum("ij,ijk->ik", weights, surface.normals[faces])
            norms = np.linalg.norm(normals, axis=1)
            output_normals[face] = np.where((norms > 0)[:, None], normals / np.where(norms > 0, norms, 1)[:, None], normal)
            distance_max = max(distance_max, float(distances.max()))
            distance_sum += distances.sum()
            sample_count += 3
            tile, maximum, total, count = _tile(surface, face, triangle, normal, neighbors, max_projection_distance,
                                               texel_scale, max_tile, supersamples)
        except ValueError as error:
            raise ValueError(f"Candidate face {face}: {error}") from error
        tiles.append(tile)
        distance_max = max(distance_max, maximum)
        distance_sum += total
        sample_count += count

    bound_textures = {int(material): textures[material] for material in np.unique(output_materials) if material != atlas_id}
    atlas_size = None
    if tiles:
        atlas, offsets = pack_atlas(tiles, max_atlas)
        for tile, offset in zip(tiles, offsets):
            size = tile.pixels.shape[1], tile.pixels.shape[0]
            output_uv[tile.face] = (tile.uvs * size + np.asarray(offset) + GUTTER) / atlas.size
        bound_textures[atlas_id] = atlas, False
        atlas_size = list(atlas.size)
    report = {
        "retained_faces": retained,
        "baked_faces": len(tiles),
        "projection_samples": sample_count,
        "maximum_projection_distance": distance_max,
        "mean_projection_distance": float(distance_sum / sample_count) if sample_count else 0.0,
        "atlas_size": atlas_size,
        "atlas_base_rgba8_bytes": atlas_size[0] * atlas_size[1] * 4 if atlas_size else 0,
        "atlas_material_id": atlas_id if tiles else None,
    }
    return output_uv, output_normals, output_materials, bound_textures, report
