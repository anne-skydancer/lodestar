"""Map a textured COLLADA section into Lodestar faces and find bakeable rectangles."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from lodestar_patch_source import _scene_mesh


@dataclass(frozen=True)
class MaterialSource:
    symbol: str
    material_id: str
    texture_filename: str


@dataclass(frozen=True)
class SectionSource:
    node_name: str
    geometry_id: str
    positions: np.ndarray
    source_uvs: np.ndarray
    normals: np.ndarray
    face_indices: np.ndarray
    triangle_materials: np.ndarray
    triangle_surfaces: np.ndarray
    materials: dict[int, MaterialSource]
    corner_indices: np.ndarray
    outside_positions: frozenset = frozenset()


def reference(url):
    if not url.startswith("#") or len(url) == 1:
        raise ValueError(f"Expected a local COLLADA reference, got {url!r}")
    return url[1:]


def by_id(root, path, identifier):
    matches = [element for element in root.findall(path) if element.get("id") == identifier]
    if len(matches) != 1:
        raise ValueError(f"Expected one COLLADA element with ID {identifier!r}")
    return matches[0]


def node_path(scene, node_name):
    def walk(parent, ancestors):
        for node in parent.findall("node"):
            chain = (*ancestors, node)
            if node.get("name") == node_name or node.get("id") == node_name:
                yield chain
            yield from walk(node, chain)

    paths = list(walk(scene, ()))
    if len(paths) != 1:
        raise ValueError(f"Expected one scene node named {node_name!r}")
    return paths[0]


def scene_transform(root, nodes):
    # Assimp applies the asset units and up-axis conversion at the scene root,
    # before Lodestar applies the inherited matrices in scene space.
    up_axes = {
        "X_UP": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        "Y_UP": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "Z_UP": [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
    }
    up = root.findtext("./asset/up_axis", "Y_UP").strip()
    if up not in up_axes:
        raise ValueError(f"Unsupported COLLADA up axis {up!r}")
    unit = root.find("./asset/unit")
    scale = np.float32(unit.get("meter", "1") if unit is not None else "1")
    result = np.eye(4)
    result[:3, :3] = np.asarray(up_axes[up], dtype=np.float64) * scale
    for node in nodes:
        unsupported = ("translate", "rotate", "scale", "lookat", "skew")
        if any(node.find(tag) is not None for tag in unsupported):
            raise ValueError("The section loader supports matrix-only COLLADA node transforms")
        local = np.eye(4, dtype=np.float32)
        for matrix in node.findall("matrix"):
            values = np.asarray((matrix.text or "").split(), dtype=np.float32)
            if values.size != 16:
                raise ValueError("A COLLADA matrix must contain 16 values")
            local = local @ values.reshape(4, 4)
        result = result @ local.astype(np.float64)
    if not np.isfinite(result).all() or not np.array_equal(result[3], [0, 0, 0, 1]):
        raise ValueError("The section's node transform must be finite and affine")
    determinant = np.linalg.det(result[:3, :3])
    if determinant == 0:
        raise ValueError("The section has a singular node transform")
    return result, determinant < 0


def source_values(mesh, identifier, width):
    source = by_id(mesh, "source", identifier)
    accessor = source.find("./technique_common/accessor")
    if accessor is None:
        raise ValueError(f"COLLADA source {identifier!r} has no accessor")
    array = by_id(source, "float_array", reference(accessor.get("source", "")))
    stride = int(accessor.get("stride", "1"))
    offset = int(accessor.get("offset", "0"))
    count = int(accessor.get("count", "0"))
    values = np.asarray((array.text or "").split(), dtype=np.float32)
    if (stride != width or offset < 0 or count < 1 or offset + stride * count > values.size or
            not np.isfinite(values).all()):
        raise ValueError(f"Unsupported component layout in COLLADA source {identifier!r}")
    return values[offset:offset + stride * count].reshape(-1, width)


def resolve_material(root, binding):
    symbol = binding.get("symbol", "")
    material_id = reference(binding.get("target", ""))
    material = by_id(root, "./library_materials/material", material_id)
    instance = material.find("./instance_effect")
    if instance is None:
        raise ValueError(f"Material {material_id!r} has no bound effect")
    effect_id = reference(instance.get("url", ""))
    effect = by_id(root, "./library_effects/effect", effect_id)
    profile = effect.find("profile_COMMON")
    if profile is None:
        raise ValueError(f"Material {material_id!r} has no common COLLADA effect")
    textures = profile.findall("./technique/*/diffuse/texture")
    if len(textures) != 1:
        raise ValueError(f"Material {material_id!r} needs one diffuse image")
    texture = textures[0]
    uv_bindings = [element for element in binding.findall("bind_vertex_input")
                   if element.get("semantic") == texture.get("texcoord") and
                   element.get("input_semantic") == "TEXCOORD" and
                   element.get("input_set") == "0"]
    if len(uv_bindings) != 1:
        raise ValueError(f"Material {material_id!r} does not bind its diffuse map to UV0")
    parameters = {param.get("sid"): param for param in profile.findall("newparam")}
    sampler = parameters.get(texture.get("texture"))
    surface_sid = sampler.findtext("./sampler2D/source") if sampler is not None else None
    surface = parameters.get(surface_sid)
    image_id = surface.findtext("./surface/init_from") if surface is not None else None
    image = by_id(root, "./library_images/image", image_id)
    filename = image.findtext("init_from", "").strip()
    if not filename or filename in (".", "..") or any(c in filename for c in "/\\:"):
        raise ValueError(f"Material {material_id!r} must refer to a texture filename")
    return MaterialSource(symbol, material_id, filename)


def primitive_corners(mesh, primitive, transform, reverse):
    inputs = primitive.findall("input")
    vertices_inputs = [item for item in inputs if item.get("semantic") == "VERTEX"]
    uv_inputs = [item for item in inputs if item.get("semantic") == "TEXCOORD"]
    if len(vertices_inputs) != 1 or len(uv_inputs) != 1 or uv_inputs[0].get("set", "0") != "0":
        raise ValueError("Each section primitive needs exactly one vertex stream and UV0")
    vertex_input, uv_input = vertices_inputs[0], uv_inputs[0]
    vertices = by_id(mesh, "vertices", reference(vertex_input.get("source", "")))
    position_inputs = [item for item in vertices.findall("input") if item.get("semantic") == "POSITION"]
    if len(position_inputs) != 1:
        raise ValueError("Section vertices need one position stream")
    positions = source_values(mesh, reference(position_inputs[0].get("source", "")), 3)
    uv = source_values(mesh, reference(uv_input.get("source", "")), 2)
    count = int(primitive.get("count", "0"))
    if primitive.tag == "polylist":
        polygon_sizes = (primitive.findtext("vcount", "")).split()
        if len(polygon_sizes) != count or any(size != "3" for size in polygon_sizes):
            raise ValueError("Only triangle polylists are supported in a section")
    offsets = [int(item.get("offset", "0")) for item in inputs]
    if min(offsets) < 0:
        raise ValueError("COLLADA primitive input offsets cannot be negative")
    stride = max(offsets) + 1
    indices = np.asarray(primitive.findtext("p", "").split(), dtype=np.int64)
    if count <= 0 or indices.size != count * 3 * stride or np.any(indices < 0):
        raise ValueError("Malformed COLLADA triangle stream")
    indices = indices.reshape(count, 3, stride)
    xyz_indices = indices[..., int(vertex_input.get("offset", "0"))]
    uv_indices = indices[..., int(uv_input.get("offset", "0"))]
    if xyz_indices.max() >= len(positions) or uv_indices.max() >= len(uv):
        raise ValueError("COLLADA triangle references a missing position or UV")
    homogeneous = np.column_stack((positions.astype(np.float64), np.ones(len(positions))))
    world = np.asarray(homogeneous @ transform[:3].T, dtype=np.float32)
    xyz, texcoords = world[xyz_indices], uv[uv_indices]
    if reverse:
        xyz, texcoords = xyz[:, [0, 2, 1]], texcoords[:, [0, 2, 1]]
    return xyz, texcoords


def face_fingerprint(positions, uv):
    faces = defaultdict(list)
    for face in np.concatenate((positions, uv), axis=2).tolist():
        a, b, c = map(tuple, face)
        corners = min((a, b, c), (b, c, a), (c, a, b))
        faces[tuple(corner[:3] for corner in corners)].append(tuple(corner[3:] for corner in corners))
    return faces


def _matching_position_ulp_fingerprints(raw, imported):
    def flatten(fingerprints):
        return np.asarray([[(*point, *uv) for point, uv in zip(positions, texcoords)]
                           for positions, values in fingerprints.items()
                           for texcoords in values], dtype=np.float32)

    expected, actual = flatten(raw), flatten(imported)
    if len(expected) != len(actual):
        return False
    lower = np.nextafter(expected, np.float32(-np.inf))
    upper = np.nextafter(expected, np.float32(np.inf))
    rotations = [np.roll(actual, shift, axis=1) for shift in range(3)]
    assignments = np.empty(len(expected), dtype=np.intp)
    # Disambiguate whole oriented faces rather than independently matching
    # corners. Reject many-to-one matches and multiple candidate faces.
    for start in range(0, len(expected), 32):
        end = start + 32
        lo, hi = lower[start:end, None], upper[start:end, None]
        allowed = np.zeros((len(lo), len(actual)), dtype=bool)
        for oriented in rotations:
            allowed |= np.all((oriented[None] >= lo) & (oriented[None] <= hi), axis=(2, 3))
        if not np.all(allowed.sum(axis=1) == 1):
            return False
        assignments[start:end] = allowed.argmax(axis=1)
    return np.unique(assignments).size == len(expected)


def matching_fingerprints(raw, imported, *, allow_position_ulp=False):
    if raw.keys() != imported.keys() or any(len(raw[key]) != len(imported[key]) for key in raw):
        return allow_position_ulp and _matching_position_ulp_fingerprints(raw, imported)
    expected = np.asarray([item for key in raw for item in sorted(raw[key])], dtype=np.float32)
    actual = np.asarray([item for key in raw for item in sorted(imported[key])], dtype=np.float32)
    # Assimp's decimal parser can round a UV component to either of two
    # adjacent floats. Positions, face orientation and multiplicity are exact.
    lower = np.nextafter(expected, np.float32(-np.inf))
    upper = np.nextafter(expected, np.float32(np.inf))
    return np.all((actual >= lower) & (actual <= upper))


def raw_section(source, node_name):
    root = ET.parse(source).getroot()
    for element in root.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    active = root.find("./scene/instance_visual_scene")
    if root.tag != "COLLADA" or active is None:
        raise ValueError("Section input needs a COLLADA visual scene")
    scene = by_id(root, "./library_visual_scenes/visual_scene", reference(active.get("url", "")))
    nodes = node_path(scene, node_name)
    leaf = nodes[-1]
    instances = leaf.findall("instance_geometry")
    if len(instances) != 1 or any(leaf.find(tag) is not None for tag in ("node", "instance_node", "instance_controller")):
        raise ValueError("The section must be a leaf node with one geometry instance")
    instance = instances[0]
    geometry_id = reference(instance.get("url", ""))
    mesh = by_id(root, "./library_geometries/geometry", geometry_id).find("mesh")
    if mesh is None:
        raise ValueError(f"Geometry {geometry_id!r} is not a COLLADA mesh")
    bindings = {}
    for item in instance.findall("./bind_material/technique_common/instance_material"):
        material = resolve_material(root, item)
        if not material.symbol or material.symbol in bindings:
            raise ValueError("Section has missing or duplicated material symbols")
        bindings[material.symbol] = material
    transform, reverse = scene_transform(root, nodes)
    groups = []
    for primitive in mesh:
        if primitive.tag in ("source", "vertices", "extra"):
            continue
        if primitive.tag not in ("polylist", "triangles"):
            raise ValueError(f"Unsupported section primitive {primitive.tag!r}")
        material = bindings.get(primitive.get("material"))
        if material is None:
            raise ValueError("A section primitive has no bound material")
        positions, uv = primitive_corners(mesh, primitive, transform, reverse)
        groups.append((material, len(positions), face_fingerprint(positions, uv)))
    if not groups:
        raise ValueError("The section contains no textured triangles")
    return geometry_id, groups


def load_section(library: Path, source: Path, node_name: str, *, allow_position_ulp=False) -> SectionSource:
    """Copy one scene node, optionally accepting a unique position match within one ULP."""
    geometry_id, raw_groups = raw_section(source, node_name)
    with _scene_mesh(library, source) as (view, metadata, provenance):
        if view.index_count % 3 or not view.has_uv or not view.vertices or not view.indices:
            raise ValueError("Section import requires complete triangles with UV0")
        if metadata.has_additional_uv or provenance.has_unrepresented_vertex_attributes:
            raise ValueError("Section import cannot preserve additional UVs or unrepresented vertex attributes")
        total = view.index_count // 3
        if (not metadata.triangle_materials or metadata.triangle_material_count != total or
                not provenance.triangle_surface_ids or provenance.triangle_surface_id_count != total):
            raise ValueError("Imported section needs a material and surface ID for every triangle")
        material_ids = np.ctypeslib.as_array(metadata.triangle_materials, shape=(total,))
        surface_ids = np.ctypeslib.as_array(provenance.triangle_surface_ids, shape=(total,))
        indices = np.ctypeslib.as_array(view.indices, shape=(view.index_count,)).reshape(-1, 3)
        vertices = np.ctypeslib.as_array(view.vertices, shape=(view.vertex_count,))
        if total == 0 or indices.max() >= view.vertex_count:
            raise ValueError("Imported section contains no triangles or a missing vertex")

        imported_groups = defaultdict(list)
        for surface in np.unique(surface_ids):
            ids = np.flatnonzero(surface_ids == surface)
            imported_groups[len(ids)].append(ids)
        selected, materials, used_faces, fingerprints = [], {}, set(), {}
        for material, count, raw_fingerprint in raw_groups:
            matches = []
            for candidate in imported_groups[count]:
                first = int(candidate[0])
                if first not in fingerprints:
                    faces = indices[candidate]
                    fingerprints[first] = face_fingerprint(vertices["position"][faces], vertices["uv"][faces])
                if matching_fingerprints(raw_fingerprint, fingerprints[first], allow_position_ulp=allow_position_ulp):
                    matches.append(candidate)
            if len(matches) != 1 or int(matches[0][0]) in used_faces:
                raise ValueError(f"Cannot map source material {material.material_id!r} to one imported surface")
            faces = matches[0]
            native_material = int(material_ids[faces[0]])
            if not np.all(material_ids[faces] == native_material):
                raise ValueError("Imported surface changes material within one source primitive")
            if native_material in materials and materials[native_material] != material:
                raise ValueError("One imported material ID maps to multiple source bindings")
            selected.extend(faces.tolist())
            materials[native_material] = material
            used_faces.add(int(faces[0]))

        face_indices = np.asarray(sorted(selected), dtype=np.int64)
        corners = indices[face_indices]
        arrays = [np.array(vertices[field][corners], dtype=np.float64, copy=True)
                  for field in ("position", "uv", "normal")]
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("Section contains non-finite positions, UVs, or normals")
        outside = np.ones(total, dtype=bool)
        outside[face_indices] = False
        outside_indices = np.unique(indices[outside])
        outside_positions = frozenset(map(tuple, vertices["position"][outside_indices]))
        return SectionSource(node_name, geometry_id, *arrays, face_indices, material_ids[face_indices],
                             surface_ids[face_indices], materials, corners, outside_positions)


def planar_adjacency(section):
    positions = section.positions
    _, mapping = np.unique(positions.reshape(-1, 3), axis=0, return_inverse=True)
    faces = mapping.reshape(-1, 3)
    vectors = np.cross(positions[:, 1] - positions[:, 0], positions[:, 2] - positions[:, 0])
    lengths = np.linalg.norm(vectors, axis=1)
    normals = section.normals
    shading_lengths = np.linalg.norm(normals, axis=2)
    usable = (lengths > 0) & np.all(shading_lengths > 0, axis=1)
    unit_normals = normals / np.where(shading_lengths > 0, shading_lengths, 1)[..., None]
    usable &= np.all(np.abs(unit_normals - unit_normals[:, :1]) <= 1e-6, axis=(1, 2))
    geometric_normals = vectors / np.where(lengths > 0, lengths, 1)[:, None]

    edges = defaultdict(list)
    for index, face in enumerate(faces.tolist()):
        for a, b in zip(face, face[1:] + face[:1]):
            edges[min(a, b), max(a, b)].append((a, b, index))
    neighbors = [set() for _ in faces]
    for occurrences in edges.values():
        if len(occurrences) != 2:
            continue
        a, b, first = occurrences[0]
        c, d, second = occurrences[1]
        if (a != d or b != c or not usable[first] or not usable[second] or
                section.triangle_surfaces[first] != section.triangle_surfaces[second] or
                section.triangle_materials[first] != section.triangle_materials[second] or
                np.max(np.abs(unit_normals[first, 0] - unit_normals[second, 0])) > 1e-6 or
                np.max(np.abs(geometric_normals[first] - geometric_normals[second])) > 1e-6):
            continue
        neighbors[first].add(second)
        neighbors[second].add(first)
    return neighbors


def vertex_incidence(section):
    incident = defaultdict(set)
    for index, face in enumerate(section.positions):
        for point in face:
            incident[tuple(point)].add(index)
    return incident


def discover_patches(section: SectionSource) -> list[tuple[int, ...]]:
    """Return deterministic, disjoint groups of local face indices for rectangle bakes."""
    from bake_planar_patch import boundary_loop, rectangular_domain

    neighbors = planar_adjacency(section)
    incident = vertex_incidence(section)
    visited, candidates = set(), []
    for seed in range(len(neighbors)):
        if seed in visited:
            continue
        component = {seed}
        pending = [seed]
        while pending:
            new = neighbors[pending.pop()] - component
            component.update(new)
            pending.extend(new)
        visited.update(component)
        if len(component) < 3:
            continue

        def rectangle(faces):
            indices = list(faces)
            positions = section.positions[indices]
            try:
                rectangular_domain(positions, section.normals[indices])
            except ValueError:
                return False
            vertices, mapping = np.unique(positions.reshape(-1, 3), axis=0, return_inverse=True)
            boundary = set(boundary_loop(mapping.reshape(-1, 3)))
            patch_faces = set(faces)
            return all(index in boundary or incident[tuple(point)] <= patch_faces
                       for index, point in enumerate(vertices))

        whole = tuple(sorted(component))
        if rectangle(whole):
            candidates.append(whole)
            continue
        # Inspect small connected subsets when the component also contains faces
        # outside a rectangle. This intentionally bounds work on large surfaces.
        checked = set()
        for anchor in whole:
            pending = [(anchor,)]
            for patch in pending:
                if len(patch) >= 3 and rectangle(patch):
                    candidates.append(patch)
                if len(patch) == 6 or len(pending) >= 128:
                    continue
                frontier = set.union(*(neighbors[index] for index in patch)) - set(patch)
                for new in sorted(index for index in frontier if index > anchor):
                    next_patch = tuple(sorted((*patch, new)))
                    if next_patch not in checked:
                        checked.add(next_patch)
                        pending.append(next_patch)

    chosen, used = [], set()
    for candidate in sorted(candidates, key=lambda patch: (-len(patch), patch)):
        if used.isdisjoint(candidate):
            used.update(candidate)
            chosen.append(candidate)
    return sorted(chosen)
