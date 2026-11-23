"""Write opaque indexed meshes with embedded diffuse textures."""

from io import BytesIO
import json
from pathlib import Path
import struct

import numpy as np


def write_textured_glb(path, primitives, textures, *, unlit=False, material_names=None):
    """Write indexed primitives, optionally grouped into named mesh nodes."""
    binary = bytearray()
    views = []
    accessors = []
    images = []
    image_ids = {}
    samplers = []
    sampler_ids = {}
    bindings = []
    materials = []
    material_ids = {}

    def append(data, target=None):
        binary.extend(b"\0" * (-len(binary) % 4))
        view = {"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        binary.extend(data)
        views.append(view)
        return len(views) - 1

    def accessor(values, component, kind, target, bounds=False):
        item = {"bufferView": append(values.tobytes(), target), "componentType": component,
                "count": len(values), "type": kind}
        if bounds:
            item["min"] = values.min(axis=0).tolist()
            item["max"] = values.max(axis=0).tolist()
        accessors.append(item)
        return len(accessors) - 1

    def get_material(source_id):
        if source_id in material_ids:
            return material_ids[source_id]
        image, repeat = textures[source_id]
        if id(image) not in image_ids:
            png = BytesIO()
            image.save(png, format="PNG")
            image_ids[id(image)] = len(images)
            images.append({"bufferView": append(png.getvalue()), "mimeType": "image/png"})
        if repeat not in sampler_ids:
            wrap = 10497 if repeat else 33071
            sampler_ids[repeat] = len(samplers)
            samplers.append({"magFilter": 9729, "minFilter": 9729, "wrapS": wrap, "wrapT": wrap})
        binding = len(bindings)
        bindings.append({"source": image_ids[id(image)], "sampler": sampler_ids[repeat]})
        material = {"pbrMetallicRoughness": {"baseColorTexture": {"index": binding},
                                            "metallicFactor": 0.0, "roughnessFactor": 1.0}}
        if material_names is not None:
            material["name"] = material_names[source_id]
        if unlit:
            material["extensions"] = {"KHR_materials_unlit": {}}
        material_ids[source_id] = len(materials)
        materials.append(material)
        return material_ids[source_id]

    def make_primitive(positions, normals, uvs, indices, material_id):
        index_type = "<u2" if len(positions) <= 65535 else "<u4"
        primitive = {
            "attributes": {
                "POSITION": accessor(np.asarray(positions, dtype="<f4"), 5126, "VEC3", 34962, bounds=True),
            },
            "indices": accessor(np.asarray(indices, dtype=index_type).ravel(),
                                5123 if index_type == "<u2" else 5125, "SCALAR", 34963),
        }
        if normals is not None:
            primitive["attributes"]["NORMAL"] = accessor(np.asarray(normals, dtype="<f4"), 5126, "VEC3", 34962)
        if uvs is not None:
            primitive["attributes"]["TEXCOORD_0"] = accessor(np.asarray(uvs, dtype="<f4"), 5126, "VEC2", 34962)
        if textures is not None:
            primitive["material"] = get_material(material_id)
        return primitive

    groups = primitives.items() if isinstance(primitives, dict) else [(None, primitives)]
    nodes, meshes = [], []
    for name, group in groups:
        node = {"mesh": len(meshes)}
        mesh = {"primitives": [make_primitive(*primitive) for primitive in group]}
        if name is not None:
            node["name"] = mesh["name"] = name
        nodes.append(node)
        meshes.append(mesh)
    binary.extend(b"\0" * (-len(binary) % 4))
    gltf = {
        "asset": {"version": "2.0", "generator": "Lodestar diffuse bake"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
    }
    if textures is not None:
        gltf.update(materials=materials, textures=bindings, samplers=samplers, images=images)
    gltf.update(buffers=[{"byteLength": len(binary)}], bufferViews=views, accessors=accessors)
    if unlit:
        gltf["extensionsUsed"] = ["KHR_materials_unlit"]
    document = json.dumps(gltf, separators=(",", ":"), allow_nan=False).encode("utf-8")
    document += b" " * (-len(document) % 4)
    header = struct.pack("<4sII", b"glTF", 2, 28 + len(document) + len(binary))
    with Path(path).open("wb") as output:
        output.write(header)
        output.write(struct.pack("<I4s", len(document), b"JSON"))
        output.write(document)
        output.write(struct.pack("<I4s", len(binary), b"BIN\0"))
        output.write(binary)
