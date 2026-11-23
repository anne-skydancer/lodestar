"""Write a static diffuse-textured COLLADA mesh and its relative PNG images."""

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from urllib.parse import quote
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from lodestar_collada_physics import COLLADA_NAMESPACE


def _write_file(path, data):
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _source(mesh, name, values, components):
    values, indices = np.unique(values.reshape(-1, len(components)), axis=0, return_inverse=True)
    source = ET.SubElement(mesh, "source", id=name)
    array = ET.SubElement(source, "float_array", id=f"{name}-array", count=str(values.size))
    array.text = " ".join(format(value, ".9g") for value in values.ravel())
    common = ET.SubElement(source, "technique_common")
    accessor = ET.SubElement(common, "accessor", source=f"#{name}-array", count=str(len(values)),
                             stride=str(len(components)))
    for component in components:
        ET.SubElement(accessor, "param", name=component, type="float")
    return indices.reshape(-1, 3)


def _diffuse_material(effects, materials, name, image_id, repeat):
    effect = ET.SubElement(effects, "effect", id=f"{name}-effect")
    profile = ET.SubElement(effect, "profile_COMMON")
    surface = ET.SubElement(ET.SubElement(profile, "newparam", sid="diffuse-surface"), "surface", type="2D")
    ET.SubElement(surface, "init_from").text = image_id
    sampler = ET.SubElement(ET.SubElement(profile, "newparam", sid="diffuse-sampler"), "sampler2D")
    ET.SubElement(sampler, "source").text = "diffuse-surface"
    wrap = "WRAP" if repeat else "CLAMP"
    ET.SubElement(sampler, "wrap_s").text = wrap
    ET.SubElement(sampler, "wrap_t").text = wrap
    ET.SubElement(sampler, "minfilter").text = "LINEAR"
    ET.SubElement(sampler, "magfilter").text = "LINEAR"
    technique = ET.SubElement(profile, "technique", sid="common")
    diffuse = ET.SubElement(ET.SubElement(technique, "lambert"), "diffuse")
    ET.SubElement(diffuse, "texture", texture="diffuse-sampler", texcoord="UVSET0")
    material = ET.SubElement(materials, "material", id=name, name=name)
    ET.SubElement(material, "instance_effect", url=f"#{name}-effect")


def _visual_streams(scene):
    positions, uvs, normals = (np.asarray(value, dtype=np.float64)
                               for value in (scene.positions, scene.uvs, scene.normals))
    materials = np.asarray(scene.materials)
    if (positions.ndim != 3 or positions.shape[1:] != (3, 3) or not len(positions) or
            normals.shape != positions.shape or uvs.shape != positions.shape[:2] + (2,) or
            materials.shape != (len(positions),) or materials.dtype.kind not in "iu" or
            not all(np.isfinite(array).all() and np.abs(array).max() <= np.finfo(np.float32).max
                    for array in (positions, uvs, normals))):
        raise ValueError("COLLADA export needs finite per-corner positions, normals, UV0, and one material per triangle")

    labels = np.unique(materials)
    textures = {}
    for label in labels:
        if label not in scene.textures:
            raise ValueError(f"COLLADA material {label} needs a diffuse texture")
        image, repeat = scene.textures[label]
        if (not isinstance(image, Image.Image) or image.mode != "RGB" or min(image.size) <= 0 or
                not isinstance(repeat, (bool, np.bool_))):
            raise ValueError(f"COLLADA material {label} needs a nonempty RGB image and a repeat flag")
        textures[label] = (image, repeat)

    return positions, uvs, normals, materials, textures


def _document():
    root = ET.Element("COLLADA", xmlns=COLLADA_NAMESPACE, version="1.4.1")
    asset = ET.SubElement(root, "asset")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ET.SubElement(asset, "created").text = timestamp
    ET.SubElement(asset, "modified").text = timestamp
    ET.SubElement(asset, "unit", name="meter", meter="1")
    ET.SubElement(asset, "up_axis").text = "Y_UP"

    return root


def _material_bindings(libraries, texture_dir, textures, material_names, encoded, image_ids):
    images, effects, materials = libraries
    for label, name in material_names.items():
        image, repeat = textures[label]
        if id(image) not in encoded:
            png = BytesIO()
            image.save(png, format="PNG")
            data = png.getvalue()
            digest = sha256(data).hexdigest()[:24]
            encoded[id(image)] = digest, data
        digest, data = encoded[id(image)]
        if digest not in image_ids:
            filename = f"image-{digest}.png"
            image_id = f"image-{digest}"
            reference = f"{quote(texture_dir.name, safe='')}/{filename}"
            ET.SubElement(ET.SubElement(images, "image", id=image_id), "init_from").text = reference
            if not (texture_dir / filename).exists():
                _write_file(texture_dir / filename, data)
            image_ids[digest] = image_id
        _diffuse_material(effects, materials, name, image_ids[digest], repeat)


def _model_libraries(root):
    geometries = ET.SubElement(root, "library_geometries")
    scenes = ET.SubElement(root, "library_visual_scenes")
    visual = ET.SubElement(scenes, "visual_scene", id="LodestarScene")
    ET.SubElement(ET.SubElement(root, "scene"), "instance_visual_scene", url="#LodestarScene")
    return geometries, visual


def _mesh_instance(libraries, node_name, geometry_name, position_name, positions, *, legacy=False):
    geometries, visual = libraries
    geometry = ET.SubElement(geometries, "geometry", id=geometry_name, name=geometry_name if legacy else node_name)
    mesh = ET.SubElement(geometry, "mesh")
    position_indices = _source(mesh, position_name, positions.astype(np.float32), "XYZ")
    node = ET.SubElement(visual, "node", id=node_name, name=node_name)
    instance = ET.SubElement(node, "instance_geometry", url=f"#{geometry_name}")
    return mesh, position_indices, instance


def _vertices(mesh, vertex_name, position_name):
    vertices = ET.SubElement(mesh, "vertices", id=vertex_name)
    ET.SubElement(vertices, "input", semantic="POSITION", source=f"#{position_name}")


def _visual_mesh(libraries, node_name, geometry_name, prefix, streams, material_names, *, legacy):
    positions, uvs, normals, materials, _ = streams
    position_name, normal_name, uv_name, vertex_name = (prefix + attr for attr in
                                                       ("Positions", "Normals", "UV0", "Vertices"))
    mesh, position_indices, instance = _mesh_instance(
        libraries, node_name, geometry_name, position_name, positions, legacy=legacy)
    normal_indices = _source(mesh, normal_name, normals.astype(np.float32), "XYZ")
    collada_uvs = (uvs.astype(np.float32).astype(np.float64) * [1, -1] + [0, 1]).astype(np.float32)
    uv_indices = _source(mesh, uv_name, collada_uvs, "ST")
    _vertices(mesh, vertex_name, position_name)
    corner_indices = np.stack((position_indices, normal_indices, uv_indices), axis=-1)
    for label in material_names:
        selected = corner_indices[materials == label]
        triangles = ET.SubElement(mesh, "triangles", material=material_names[label], count=str(len(selected)))
        ET.SubElement(triangles, "input", semantic="VERTEX", source=f"#{vertex_name}", offset="0")
        ET.SubElement(triangles, "input", semantic="NORMAL", source=f"#{normal_name}", offset="1")
        ET.SubElement(triangles, "input", semantic="TEXCOORD", source=f"#{uv_name}", offset="2", set="0")
        ET.SubElement(triangles, "p").text = " ".join(map(str, selected.ravel()))

    common = ET.SubElement(ET.SubElement(instance, "bind_material"), "technique_common")
    for name in material_names.values():
        material = ET.SubElement(common, "instance_material", symbol=name, target=f"#{name}")
        ET.SubElement(material, "bind_vertex_input", semantic="UVSET0", input_semantic="TEXCOORD", input_set="0")


def _save_document(path, root):
    ET.indent(root, space="  ")
    _write_file(path, ET.tostring(root, encoding="utf-8", xml_declaration=True))


def _physical_frames(root, frames, suffix):
    """Attach a physical normalization frame without adding render geometry."""
    if not frames:
        return
    geometries = {item.get('name'): item for item in root.find('library_geometries')}
    for name, frame in frames.items():
        minimum = np.asarray(frame['container_minimum'], dtype=np.float64)
        maximum = np.asarray(frame['container_maximum'], dtype=np.float64)
        if minimum.shape != (3,) or maximum.shape != (3,) or not np.isfinite([minimum, maximum]).all() or np.any(maximum <= minimum):
            raise ValueError('Physical container frames need finite positive bounds')
        geometry = geometries.get(name + suffix)
        if geometry is None:
            raise ValueError(f'Physical container has no matching model: {name}')
        technique = ET.SubElement(ET.SubElement(geometry, 'extra'), 'technique', profile='LODESTAR')
        ET.SubElement(technique, 'physical_container', version='1',
            minimum=','.join(format(v, '.17g') for v in minimum),
            maximum=','.join(format(v, '.17g') for v in maximum))


def _write_visual(path, objects, suffix, texture_folder, *, legacy=False, physical_frames=None):
    path = Path(path)
    root = _document()
    material_libraries = tuple(ET.SubElement(root, f"library_{name}") for name in ("images", "effects", "materials"))
    model_libraries = _model_libraries(root)
    texture_dir = path.with_name(texture_folder)
    texture_dir.mkdir(parents=True, exist_ok=True)
    encoded, image_ids = {}, {}
    for name, scene in objects.items():
        node = name + suffix
        streams = _visual_streams(scene)
        textures = streams[-1]
        if legacy:
            material_names = {label: f"lodestar-material-{index}" for index, label in enumerate(textures)}
            geometry, prefix = "LodestarGeometry", "Lodestar"
        else:
            material_names = {label: f"{name}-slot-{label}" for label in textures}
            geometry, prefix = f"geometry-{node}", f"geometry-{node}-"
        _material_bindings(material_libraries, texture_dir, textures, material_names, encoded, image_ids)
        _visual_mesh(model_libraries, node, geometry, prefix, streams, material_names, legacy=legacy)
    _physical_frames(root, physical_frames, suffix)
    _save_document(path, root)


def _validate_names(objects, suffix):
    if not objects or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name + suffix) for name in objects):
        raise ValueError("COLLADA needs named parts with XML-compatible model names")


def write_textured_collada(path, scene):
    """Write one Y-up, metre mesh with per-corner normals and top-origin input UVs."""
    _write_visual(path, {"LodestarModel": scene}, "", f"{Path(path).stem}-textures", legacy=True)


def write_lod_collada(path, objects, *, suffix="", physical_frames=None):
    """Write separate upload models with stable materials and shared sibling PNGs."""
    _validate_names(objects, suffix)
    _write_visual(path, objects, suffix, "textures", physical_frames=physical_frames)


def write_lod_collision_collada(path, objects, *, suffix="_PHYS", physical_frames=None):
    """Write named, position-only triangle collision models in the same frame as High."""
    _validate_names(objects, suffix)
    root = _document()
    libraries = _model_libraries(root)
    for name, positions in objects.items():
        positions = np.asarray(positions, dtype=np.float32)
        if positions.ndim != 3 or positions.shape[1:] != (3, 3) or not len(positions) or not np.isfinite(positions).all():
            raise ValueError(f"Invalid collision triangle stream for {name}")
        node = name + suffix
        geometry = f"geometry-{node}"
        position_name, vertex_name = f"{geometry}-Positions", f"{geometry}-Vertices"
        mesh, indices, _ = _mesh_instance(libraries, node, geometry, position_name, positions)
        _vertices(mesh, vertex_name, position_name)
        triangles = ET.SubElement(mesh, "triangles", count=str(len(positions)))
        ET.SubElement(triangles, "input", semantic="VERTEX", source=f"#{vertex_name}", offset="0")
        ET.SubElement(triangles, "p").text = " ".join(map(str, indices.ravel()))
    _physical_frames(root, physical_frames, suffix)
    _save_document(Path(path), root)
