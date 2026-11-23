"""Load and assemble diffuse-textured TestHall scenes in native face order."""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from bake_section_atlas import indexed_primitives, read_textures
from check_testhall_full import node_names
from check_testhall_gates import merge_textured
from lodestar_gltf_source import load_gltf_groups
from lodestar_patch_source import _scene_mesh
from lodestar_section_source import SectionSource, face_fingerprint, matching_fingerprints, raw_section
from lodestar_section_source import by_id, node_path, scene_transform, reference
from lodestar_textured_glb import write_textured_glb


@dataclass(frozen=True)
class TexturedScene:
    positions: np.ndarray
    uvs: np.ndarray
    normals: np.ndarray
    materials: np.ndarray
    textures: dict
    corner_indices: np.ndarray
    sections: dict[str, SectionSource] = field(default_factory=dict)
    declared_bounds: dict = field(default_factory=dict)

    @property
    def arrays(self):
        return self.positions, self.uvs, self.normals, self.materials, self.textures

    @property
    def render_arrays(self):
        """Use the stored GLB precision and its material-grouped draw order."""
        order = np.argsort(self.materials, kind="stable")
        return (*[array[order].astype("<f4") for array in self.arrays[:3]], self.materials[order], self.textures)


def _native_arrays(library, source):
    with _scene_mesh(library, source) as (view, metadata, provenance):
        count = view.index_count // 3
        if (not count or view.index_count % 3 or not view.vertices or not view.indices or not view.has_uv or
                not metadata.triangle_materials or metadata.triangle_material_count != count or
                not provenance.triangle_surface_ids or provenance.triangle_surface_id_count != count):
            raise ValueError("The scene needs complete indexed triangles, UV0, and per-face material and surface IDs")
        if metadata.has_additional_uv or provenance.has_unrepresented_vertex_attributes:
            raise ValueError("The diffuse comparison cannot preserve extra UVs or unrepresented vertex attributes")
        corners = np.ctypeslib.as_array(view.indices, shape=(view.index_count,)).reshape(-1, 3).copy()
        vertices = np.ctypeslib.as_array(view.vertices, shape=(view.vertex_count,))
        if corners.max() >= len(vertices):
            raise ValueError("The scene contains a triangle with a missing vertex")
        attributes = tuple(np.asarray(vertices[name][corners], dtype=np.float64)
                           for name in ("position", "uv", "normal"))
        if not all(np.isfinite(array).all() for array in attributes):
            raise ValueError("The scene contains non-finite vertex attributes")
        labels = np.ctypeslib.as_array(metadata.triangle_materials, shape=(count,)).copy()
        surfaces = np.ctypeslib.as_array(provenance.triangle_surface_ids, shape=(count,)).copy()
    return *attributes, labels, surfaces, corners


def _without_zero_area(fingerprints):
    filtered = {}
    for points, uvs in fingerprints.items():
        a, b, c = np.asarray(points, dtype=np.float64)
        if np.any(np.cross(b - a, c - a) != 0):
            filtered[points] = uvs
    return filtered


def _surface_match(raw, count, groups, fingerprints, material, node):
    def matches(expected, size):
        return [int(group[0]) for group in groups[size]
                if matching_fingerprints(expected, fingerprints[int(group[0])], allow_position_ulp=True)]

    found = matches(raw, count)
    if not found:
        # The native importer drops two explicitly collinear Trim faces in
        # Wing2. Retry only with zero-area triangles removed from the XML.
        filtered = _without_zero_area(raw)
        count = sum(map(len, filtered.values()))
        found = matches(filtered, count)
    if len(found) != 1:
        raise ValueError(f"Cannot uniquely map {node!r} material {material!r} to one native surface")
    return found[0]


def load_textured_scene(library, source, texture_dir=None, *, translation=(0, 0, 0)):
    """Keep native attributes and prove each COLLADA or glTF diffuse binding.

    COLLADA materials resolve files in texture_dir. glTF materials and their
    embedded or relative images resolve from the glTF asset itself.
    """
    source = Path(source)
    gltf = source.suffix.lower() in (".gltf", ".glb")
    if gltf:
        raw_nodes, gltf_textures = load_gltf_groups(source)
    elif texture_dir is None:
        raise ValueError("COLLADA diffuse materials need a texture directory")
    else:
        raw_nodes = ((node, *raw_section(source, node)) for node in node_names(source))
    framed_geometries, frame_document = {}, None
    if not gltf:
        frame_document = ET.parse(source).getroot()
        for element in frame_document.iter():
            element.tag = element.tag.rsplit('}', 1)[-1]
        for geometry in frame_document.findall('./library_geometries/geometry'):
            frames = geometry.findall('./extra/technique[@profile="LODESTAR"]/physical_container')
            if not frames:
                continue
            if len(frames) != 1 or frames[0].get('version') != '1':
                raise ValueError('Unsupported or repeated physical container frame')
            minimum = np.array([float(v) for v in frames[0].get('minimum', '').split(',')])
            maximum = np.array([float(v) for v in frames[0].get('maximum', '').split(',')])
            if minimum.shape != (3,) or maximum.shape != (3,) or not np.isfinite([minimum,maximum]).all() or np.any(maximum <= minimum):
                raise ValueError('Invalid physical container bounds')
            framed_geometries[geometry.get('id')] = (minimum, maximum)
    positions, uvs, normals, labels, surfaces, corners = _native_arrays(Path(library), source)
    translation = np.asarray(translation, dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("Scene translation must be a finite three-vector")

    by_size, face_groups, fingerprints = defaultdict(list), {}, {}
    for surface in np.unique(surfaces):
        faces = np.flatnonzero(surfaces == surface)
        first = int(faces[0])
        by_size[len(faces)].append(faces)
        face_groups[first] = faces
        fingerprints[first] = face_fingerprint(positions[faces], uvs[faces])

    shifted = positions + translation
    sections, bindings, used, declared_bounds = {}, {}, set(), {}
    for node, geometry, primitives in raw_nodes:
        declared = np.asarray([point for _, _, fingerprints in primitives for triangle in fingerprints for point in triangle])
        declared_bounds[node] = {'minimum': (declared.min(axis=0)+translation).tolist(),
                                 'maximum': (declared.max(axis=0)+translation).tolist()}
        if geometry in framed_geometries:
            lo, hi = framed_geometries[geometry]
            visual = by_id(frame_document, './library_visual_scenes/visual_scene',
                reference(frame_document.find('./scene/instance_visual_scene').get('url', '')))
            transform, _ = scene_transform(frame_document, node_path(visual, node))
            frame_corners = np.array([(x,y,z,1) for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
            framed = (frame_corners @ transform.T)[:,:3] + translation
            declared_bounds[node] = {'minimum': framed.min(axis=0).tolist(), 'maximum': framed.max(axis=0).tolist()}
        section_groups, section_bindings = [], {}
        for material, size, raw in primitives:
            # Padding may have its own primitive. Preserve its declared frame
            # above, without requiring a native surface for removed zero-area faces.
            if not _without_zero_area(raw):
                continue
            identifier = f"glTF material {material}" if gltf else material.material_id
            first = _surface_match(raw, size, by_size, fingerprints, identifier, node)
            if first in used:
                raise ValueError(f"Multiple COLLADA primitives match the native surface on {node!r}")
            used.add(first)
            faces = face_groups[first]
            label = int(labels[first])
            if not np.all(labels[faces] == label):
                raise ValueError("A native surface contains multiple material IDs")
            if label in bindings:
                old = bindings[label]
                if gltf:
                    conflicts = old != material
                else:
                    conflicts = ((old.material_id, old.texture_filename, getattr(old, "repeat", True)) !=
                                 (material.material_id, material.texture_filename, getattr(material, "repeat", True)))
                if conflicts:
                    raise ValueError(f"Conflicting diffuse bindings for native material {label}")
            bindings[label] = material
            if not gltf:
                section_bindings[label] = material
            section_groups.append(faces)

        if not section_groups:
            continue
        faces = np.sort(np.concatenate(section_groups))
        outside = np.ones(len(positions), dtype=bool)
        outside[faces] = False
        outside_points = frozenset(map(tuple, np.unique(shifted[outside].reshape(-1, 3), axis=0)))
        sections[node] = SectionSource(node, geometry, shifted[faces], uvs[faces], normals[faces], faces,
                                       labels[faces], surfaces[faces], section_bindings, corners[faces], outside_points)

    if used != set(face_groups) or set(bindings) != set(map(int, labels)):
        raise ValueError("The diffuse bindings do not cover the complete native scene")
    textures = ({label: gltf_textures[material] for label, material in bindings.items()} if gltf
                else read_textures(bindings, Path(texture_dir)))
    return TexturedScene(shifted, uvs * [1, -1] + [0, 1], normals, labels, textures, corners, sections, declared_bounds)


def _replacement_corners(positions, uvs, normals):
    rows = np.column_stack((positions.reshape(-1, 3), normals.reshape(-1, 3), uvs.reshape(-1, 2)))
    _, inverse = np.unique(rows.astype("<f4"), axis=0, return_inverse=True)
    return inverse.reshape(-1, 3)


def compose_textured_scene(scene, replacements):
    """Replace disjoint source face sets with baked geometry and merge texture bindings.

    Each item is (native source face indices, (positions, top-origin UVs,
    normals, material IDs, texture bindings)). Unselected corners retain their
    native attributes and index identities.
    """
    remaining = np.ones(len(scene.positions), dtype=bool)
    chunks, indices = [], []
    offset = int(scene.corner_indices.max()) + 1
    for faces, arrays in replacements:
        faces = np.asarray(faces)
        if (faces.ndim != 1 or not faces.size or not np.issubdtype(faces.dtype, np.integer) or
                np.any(faces < 0) or np.any(faces >= len(remaining))):
            raise ValueError("A replacement needs valid native source face indices")
        if np.unique(faces).size != faces.size or not remaining[faces].all():
            raise ValueError("Replacement source face sets must be disjoint")
        remaining[faces] = False
        positions, uvs, normals, materials = (np.asarray(array) for array in arrays[:4])
        textures = arrays[4]
        count = len(positions)
        if (not count or positions.shape != (count, 3, 3) or normals.shape != (count, 3, 3) or
                uvs.shape != (count, 3, 2) or materials.shape != (count,) or
                not np.issubdtype(materials.dtype, np.integer) or
                not all(np.isfinite(array).all() for array in (positions, uvs, normals)) or
                not set(map(int, materials)).issubset(textures)):
            raise ValueError("A replacement needs finite per-corner attributes and a texture for every face")
        local = _replacement_corners(positions, uvs, normals)
        chunks.append((positions, uvs, normals, materials, textures))
        indices.append(local + offset)
        offset += int(local.max()) + 1

    if not chunks:
        return scene
    if remaining.any():
        chunks.insert(0, (*[attribute[remaining] for attribute in scene.arrays[:4]], scene.textures))
        indices.insert(0, scene.corner_indices[remaining])
    return TexturedScene(*merge_textured(chunks), np.concatenate(indices))


def write_scene_glb(path, scene):
    attributes = tuple(array.astype("<f4") for array in scene.arrays[:3])
    primitives = indexed_primitives(*attributes, scene.materials, scene.corner_indices)
    write_textured_glb(Path(path), primitives, scene.textures)
