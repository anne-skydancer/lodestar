"""Read glTF diffuse image bindings and independent face proofs for Lodestar imports."""

import base64
from io import BytesIO
import json
from pathlib import Path
import struct
from urllib.parse import unquote, urlsplit

import numpy as np
from PIL import Image

from lodestar_section_source import face_fingerprint


def _document(path):
    data = path.read_bytes()
    if path.suffix.lower() == ".gltf":
        return json.loads(data), None
    magic, version, size = struct.unpack_from("<4sII", data)
    if magic != b"glTF" or version != 2 or size != len(data):
        raise ValueError("Invalid glTF 2 binary header")
    chunks = {}
    offset = 12
    json_tag = b"JSON"
    while offset < len(data):
        length, tag = struct.unpack_from("<I4s", data, offset)
        offset += 8
        end = offset + length
        if end > len(data) or length % 4 or tag in chunks or (not chunks and tag != json_tag):
            raise ValueError("Invalid GLB chunk bounds, order, or type")
        chunks[tag] = data[offset:end]
        offset = end
    return json.loads(chunks[json_tag]), chunks.get(b"BIN\0")


def _uri_bytes(path, uri):
    if uri.startswith("data:"):
        encoding, data = uri.split(",", 1)
        if not encoding.endswith(";base64"):
            raise ValueError("Only base64 glTF data URIs are supported")
        return base64.b64decode(data, validate=True)
    parsed = urlsplit(uri)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("glTF resources must use local relative paths")
    root = path.parent.resolve()
    relative = Path(unquote(parsed.path))
    resource = (root / relative).resolve()
    if relative.is_absolute() or not resource.is_relative_to(root):
        raise ValueError("A glTF resource points outside the asset directory")
    return resource.read_bytes()


def _entry(entries, index):
    if type(index) is not int or not 0 <= index < len(entries):
        raise ValueError("A glTF resource index is out of bounds")
    return entries[index]


class _GltfSource:
    def __init__(self, path):
        self.path = path
        self.doc, binary = _document(path)
        if self.doc["asset"]["version"] != "2.0" or self.doc.get("extensionsRequired"):
            raise ValueError("The diffuse pipeline needs glTF 2 without required extensions")
        if self.doc.get("animations") or self.doc.get("skins"):
            raise ValueError("The diffuse pipeline supports static glTF meshes")
        self.buffers = []
        for index, buffer in enumerate(self.doc["buffers"]):
            size = buffer["byteLength"]
            if "uri" in buffer:
                data = _uri_bytes(path, buffer["uri"])
            elif index == 0 and binary is not None:
                data = binary
            else:
                raise ValueError("A glTF buffer needs its own URI or the GLB binary chunk")
            if not 0 <= size <= len(data):
                raise ValueError("A glTF buffer is smaller than its declared length")
            self.buffers.append(memoryview(data)[:size])
        self.images = {}
        self.textures = {}

    def view(self, index):
        view = _entry(self.doc["bufferViews"], index)
        buffer = _entry(self.buffers, view["buffer"])
        start = view.get("byteOffset", 0)
        end = start + view["byteLength"]
        if not 0 <= start <= end <= len(buffer):
            raise ValueError("A glTF buffer view is out of bounds")
        return view, buffer[start:end]

    def array(self, index, kind, components):
        item = _entry(self.doc["accessors"], index)
        if (item.get("sparse") or item.get("normalized") or item["type"] != kind or
                item["componentType"] not in components):
            raise ValueError(f"Unsupported glTF {kind} accessor encoding")
        dtype = np.dtype(components[item["componentType"]])
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[kind]
        element_size = dtype.itemsize * width
        view, data = self.view(item["bufferView"])
        stride = view.get("byteStride", element_size)
        start, count = item.get("byteOffset", 0), item["count"]
        if (count <= 0 or stride < element_size or stride % dtype.itemsize or start < 0 or
                start + (count - 1) * stride + element_size > len(data)):
            raise ValueError("A glTF accessor is out of bounds")
        return np.ndarray((count, width), dtype=dtype, buffer=data, offset=start, strides=(stride, dtype.itemsize))

    def image(self, index):
        if index not in self.images:
            item = _entry(self.doc["images"], index)
            if ("uri" in item) == ("bufferView" in item):
                raise ValueError("A glTF image needs one URI or buffer view")
            data = _uri_bytes(self.path, item["uri"]) if "uri" in item else self.view(item["bufferView"])[1]
            with Image.open(BytesIO(data)) as image:
                if image.format not in ("PNG", "JPEG"):
                    raise ValueError("The diffuse pipeline supports glTF PNG and JPEG images")
                self.images[index] = image.convert("RGB")
        return self.images[index]

    def material(self, index):
        if index not in self.textures:
            material = _entry(self.doc["materials"], index)
            pbr = material.get("pbrMetallicRoughness", {})
            if (material.get("alphaMode", "OPAQUE") != "OPAQUE" or material.get("doubleSided") or
                    material.get("extensions") or
                    any(key in material for key in ("normalTexture", "emissiveTexture", "occlusionTexture")) or
                    material.get("emissiveFactor", [0, 0, 0]) != [0, 0, 0] or
                    pbr.get("baseColorFactor", [1, 1, 1, 1]) != [1, 1, 1, 1] or
                    pbr.get("metallicFactor", 1) != 0 or pbr.get("roughnessFactor", 1) != 1 or
                    "metallicRoughnessTexture" in pbr or "baseColorTexture" not in pbr):
                raise ValueError("Use opaque, single-sided, nonmetallic textured glTF materials with roughness 1")
            binding = pbr["baseColorTexture"]
            if binding.get("texCoord", 0) != 0 or binding.get("extensions"):
                raise ValueError("glTF base color must use plain UV0")
            texture = _entry(self.doc["textures"], binding["index"])
            if texture.get("extensions"):
                raise ValueError("Unsupported glTF texture extension")
            sampler = _entry(self.doc["samplers"], texture["sampler"]) if "sampler" in texture else {}
            wrap = sampler.get("wrapS", 10497)
            if (wrap not in (33071, 10497) or wrap != sampler.get("wrapT", 10497) or
                    sampler.get("magFilter", 9729) != 9729 or sampler.get("minFilter", 9729) != 9729):
                raise ValueError("Use linear glTF sampling and the same repeat or clamp mode on both axes")
            self.textures[index] = self.image(texture["source"]), wrap == 10497

    def primitive(self, primitive, matrix):
        attributes = primitive["attributes"]
        if (primitive.get("mode", 4) != 4 or primitive.get("targets") or primitive.get("extensions") or
                set(attributes) - {"POSITION", "NORMAL", "TEXCOORD_0"} or "TEXCOORD_0" not in attributes):
            raise ValueError("Use static glTF triangles with positions, optional normals, and UV0")
        positions = self.array(attributes["POSITION"], "VEC3", {5126: "<f4"})
        uv = self.array(attributes["TEXCOORD_0"], "VEC2", {5126: "<f4"}).copy()
        flat = (self.array(primitive["indices"], "SCALAR", {5121: "u1", 5123: "<u2", 5125: "<u4"}).ravel()
                if "indices" in primitive else np.arange(len(positions)))
        if (len(positions) != len(uv) or len(flat) % 3 or flat.max() >= len(positions) or
                not (np.isfinite(positions).all() and np.isfinite(uv).all())):
            raise ValueError("Invalid glTF geometry or UV stream")
        corners = flat.reshape(-1, 3)
        if np.linalg.det(matrix[:3, :3]) < 0:
            corners = corners[:, [0, 2, 1]]
        # Assimp imports the original glTF coordinates in bottom-origin UV space.
        uv[:, 1] = np.float32(1) - uv[:, 1]
        world = (positions.astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3]).astype(np.float32)
        material = primitive["material"]
        self.material(material)
        return material, len(corners), face_fingerprint(world[corners], uv[corners])


def _transform(node):
    if "matrix" in node:
        if set(node) & {"translation", "rotation", "scale"}:
            raise ValueError("A glTF node cannot mix a matrix and TRS")
        matrix = np.asarray(node["matrix"], dtype=np.float32).reshape(4, 4).T.astype(np.float64)
    else:
        # Assimp constructs the local quaternion matrix in source float
        # precision before Lodestar composes transforms in scene space.
        x, y, z, w = np.asarray(node.get("rotation", [0, 0, 0, 1]), dtype=np.float32)
        scale = np.asarray(node.get("scale", [1, 1, 1]), dtype=np.float32)
        matrix = np.eye(4)
        matrix[:3, :3] = np.array([
            [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
            [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
            [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
        ], dtype=np.float32) * scale
        matrix[:3, 3] = np.asarray(node.get("translation", [0, 0, 0]), dtype=np.float32)
    if (not np.isfinite(matrix).all() or not np.array_equal(matrix[3], [0, 0, 0, 1]) or
            np.linalg.det(matrix[:3, :3]) == 0):
        raise ValueError("A glTF node needs a finite non-singular affine transform")
    return matrix


def load_gltf_groups(path: Path):
    """Return named primitive proofs and decoded base-color bindings for a scene."""
    try:
        source = _GltfSource(path)
        document = source.doc
        scene = _entry(document["scenes"], document.get("scene", 0))
        pending = [(index, np.eye(4)) for index in reversed(scene["nodes"])]
        visited, names, groups = set(), set(), []
        while pending:
            index, parent = pending.pop()
            node = _entry(document["nodes"], index)
            if index in visited or "skin" in node or "weights" in node or node.get("extensions"):
                raise ValueError("Unsupported glTF node hierarchy or attributes")
            visited.add(index)
            transform = parent @ _transform(node)
            pending.extend((child, transform) for child in reversed(node.get("children", [])))
            if "mesh" not in node:
                continue
            geometry = _entry(document["meshes"], node["mesh"])
            parts = [source.primitive(primitive, transform) for primitive in geometry["primitives"]]
            if not parts:
                raise ValueError("A glTF mesh must contain triangles")
            name = node.get("name") or f"node-{index}"
            if name in names:
                raise ValueError("glTF render nodes must have unique names for section indexing")
            names.add(name)
            groups.append((name, f"mesh-{node['mesh']}", parts))
        return groups, source.textures
    except (IndexError, KeyError, TypeError, struct.error) as error:
        raise ValueError(f"Malformed glTF scene in {path.name}") from error
