"""Export Lodestar's single-buffer GLBs as glTF with a companion buffer."""

from hashlib import sha256
import json
import os
from pathlib import Path
import struct
from tempfile import NamedTemporaryFile


def _chunk(data, offset, expected_kind):
    if offset + 8 > len(data):
        raise ValueError("Truncated GLB chunk header")
    length, kind = struct.unpack_from("<I4s", data, offset)
    start = offset + 8
    end = start + length
    if kind != expected_kind or length % 4 or end > len(data):
        raise ValueError(f"Invalid GLB {expected_kind!r} chunk type or length")
    return data[start:end], end


def _read_glb(path):
    data = path.read_bytes()
    if len(data) < 20 or struct.unpack_from("<4sII", data) != (b"glTF", 2, len(data)):
        raise ValueError("Invalid GLB 2 header or file length")
    raw_json, offset = _chunk(data, 12, b"JSON")
    document = json.loads(raw_json.decode("utf-8"))
    asset = document.get("asset") if isinstance(document, dict) else None
    if not isinstance(asset, dict) or asset.get("version") != "2.0":
        raise ValueError("The GLB JSON must describe a glTF 2.0 asset")
    buffers = document.get("buffers", [])
    views = document.get("bufferViews", [])
    if not isinstance(buffers, list) or not isinstance(views, list):
        raise ValueError("Invalid GLB buffers or buffer views")
    if offset == len(data) and not buffers and not views:
        return document, None

    binary, end = _chunk(data, offset, b"BIN\0")
    if end != len(data) or len(buffers) != 1 or not isinstance(buffers[0], dict) or "uri" in buffers[0]:
        raise ValueError("Expected one embedded GLB buffer")
    length = buffers[0].get("byteLength")
    if type(length) is not int or length <= 0 or not 0 <= len(binary) - length <= 3 or any(binary[length:]):
        raise ValueError("GLB buffer byteLength disagrees with the binary payload")
    for view in views:
        if not isinstance(view, dict) or view.get("buffer") != 0:
            raise ValueError("GLB buffer view must use the embedded buffer")
        start = view.get("byteOffset", 0)
        size = view.get("byteLength")
        if type(start) is not int or type(size) is not int or start < 0 or size <= 0 or start + size > length:
            raise ValueError("GLB buffer view extends beyond the declared buffer")
    return document, binary[:length]


def _validate_output(path, source):
    if path.is_symlink() or (path.exists() and (not path.is_file() or path.samefile(source))):
        raise ValueError(f"Unsafe glTF export path: {path}")


def _publish(path, payload):
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=".lodestar-", delete=False) as staged:
            temporary = Path(staged.name)
            staged.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_gltf_from_glb(glb_path, gltf_path):
    """Publish a JSON glTF and return its adjacent binary path, if present.

    Buffer contents determine the companion filename. Repeated or concurrent
    exports can publish a new glTF without changing another glTF's buffer.
    """
    source = Path(glb_path).resolve(strict=True)
    target = Path(gltf_path)
    if target.suffix.lower() != ".gltf":
        raise ValueError("The output path must end in .gltf")
    target = target.parent.resolve() / target.name
    _validate_output(target, source)
    document, binary = _read_glb(source)
    companion = None
    if binary is not None:
        # Some native importers treat buffer URIs as literal filenames.
        # Keep the relative name ASCII and URI-safe regardless of the glTF name.
        companion = target.with_name(f"lodestar-{sha256(binary).hexdigest()}.bin")
        _validate_output(companion, source)
        if target.exists() and companion.exists() and target.samefile(companion):
            raise ValueError("The glTF and binary output paths must be distinct")
        document["buffers"][0]["uri"] = companion.name
    payload = (json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")

    target.parent.mkdir(parents=True, exist_ok=True)
    if companion is not None:
        if companion.exists():
            if companion.read_bytes() != binary:
                raise ValueError(f"Existing companion buffer contains different bytes: {companion}")
        else:
            _publish(companion, binary)
    _publish(target, payload)
    return companion
