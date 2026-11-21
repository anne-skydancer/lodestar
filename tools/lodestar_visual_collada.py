"""Write a static diffuse-textured COLLADA mesh and its relative PNG images."""

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
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


def write_textured_collada(path, scene):
    """Write one Y-up, metre mesh with per-corner normals and top-origin input UVs.

    PNGs are deduplicated by their stored bytes and placed in a relative sibling
    directory. A failed write leaves the previous COLLADA and its images usable.
    """
    path = Path(path)
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

    root = ET.Element("COLLADA", xmlns=COLLADA_NAMESPACE, version="1.4.1")
    asset = ET.SubElement(root, "asset")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ET.SubElement(asset, "created").text = timestamp
    ET.SubElement(asset, "modified").text = timestamp
    ET.SubElement(asset, "unit", name="meter", meter="1")
    ET.SubElement(asset, "up_axis").text = "Y_UP"

    images = ET.SubElement(root, "library_images")
    effects = ET.SubElement(root, "library_effects")
    material_library = ET.SubElement(root, "library_materials")
    texture_dir = path.with_name(f"{path.stem}-textures")
    texture_dir.mkdir(parents=True, exist_ok=True)
    encoded, image_ids, material_names = {}, {}, {}
    for index, label in enumerate(labels):
        image, repeat = textures[label]
        if id(image) not in encoded:
            png = BytesIO()
            image.save(png, format="PNG")
            data = png.getvalue()
            digest = sha256(data).hexdigest()[:24]
            encoded[id(image)] = digest, data
        digest, data = encoded[id(image)]
        if digest not in image_ids:
            name = f"image-{digest}.png"
            image_id = f"image-{digest}"
            reference = f"{quote(texture_dir.name, safe='')}/{name}"
            ET.SubElement(ET.SubElement(images, "image", id=image_id), "init_from").text = reference
            _write_file(texture_dir / name, data)
            image_ids[digest] = image_id
        material_names[label] = f"lodestar-material-{index}"
        _diffuse_material(effects, material_library, material_names[label], image_ids[digest], repeat)

    geometries = ET.SubElement(root, "library_geometries")
    geometry = ET.SubElement(geometries, "geometry", id="LodestarGeometry", name="LodestarGeometry")
    mesh = ET.SubElement(geometry, "mesh")
    position_indices = _source(mesh, "LodestarPositions", positions.astype(np.float32), "XYZ")
    normal_indices = _source(mesh, "LodestarNormals", normals.astype(np.float32), "XYZ")
    collada_uvs = (uvs.astype(np.float32).astype(np.float64) * [1, -1] + [0, 1]).astype(np.float32)
    uv_indices = _source(mesh, "LodestarUV0", collada_uvs, "ST")
    vertices = ET.SubElement(mesh, "vertices", id="LodestarVertices")
    ET.SubElement(vertices, "input", semantic="POSITION", source="#LodestarPositions")
    corner_indices = np.stack((position_indices, normal_indices, uv_indices), axis=-1)
    for label in labels:
        selected = corner_indices[materials == label]
        triangles = ET.SubElement(mesh, "triangles", material=material_names[label], count=str(len(selected)))
        ET.SubElement(triangles, "input", semantic="VERTEX", source="#LodestarVertices", offset="0")
        ET.SubElement(triangles, "input", semantic="NORMAL", source="#LodestarNormals", offset="1")
        ET.SubElement(triangles, "input", semantic="TEXCOORD", source="#LodestarUV0", offset="2", set="0")
        ET.SubElement(triangles, "p").text = " ".join(map(str, selected.ravel()))

    scenes = ET.SubElement(root, "library_visual_scenes")
    visual = ET.SubElement(scenes, "visual_scene", id="LodestarScene")
    node = ET.SubElement(visual, "node", id="LodestarModel", name="LodestarModel")
    instance = ET.SubElement(node, "instance_geometry", url="#LodestarGeometry")
    common = ET.SubElement(ET.SubElement(instance, "bind_material"), "technique_common")
    for name in material_names.values():
        material = ET.SubElement(common, "instance_material", symbol=name, target=f"#{name}")
        ET.SubElement(material, "bind_vertex_input", semantic="UVSET0", input_semantic="TEXCOORD", input_set="0")
    ET.SubElement(ET.SubElement(root, "scene"), "instance_visual_scene", url="#LodestarScene")
    ET.indent(root, space="  ")
    _write_file(path, ET.tostring(root, encoding="utf-8", xml_declaration=True))
