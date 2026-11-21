"""Load and copy selected scene-space triangles for the texture-transfer prototype."""

import ctypes as ct
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np


class _Vertex(ct.Structure):
    _fields_ = [
        ("position", ct.c_float * 3),
        ("normal", ct.c_float * 3),
        ("uv", ct.c_float * 2),
    ]


class _MeshView(ct.Structure):
    _fields_ = [
        ("vertices", ct.POINTER(_Vertex)),
        ("vertex_count", ct.c_size_t),
        ("indices", ct.POINTER(ct.c_uint32)),
        ("index_count", ct.c_size_t),
        ("has_uv", ct.c_uint32),
    ]


class _MeshMetadata(ct.Structure):
    _fields_ = [
        ("triangle_materials", ct.POINTER(ct.c_uint32)),
        ("triangle_material_count", ct.c_size_t),
        ("has_additional_uv", ct.c_uint32),
    ]


class _MeshProvenance(ct.Structure):
    _fields_ = [
        ("vertex_position_ids", ct.POINTER(ct.c_uint64)),
        ("vertex_position_id_count", ct.c_size_t),
        ("triangle_surface_ids", ct.POINTER(ct.c_uint32)),
        ("triangle_surface_id_count", ct.c_size_t),
        ("has_unrepresented_vertex_attributes", ct.c_uint32),
    ]


@contextmanager
def _scene_mesh(
    library: Path, source: Path
) -> Iterator[tuple[_MeshView, _MeshMetadata, _MeshProvenance]]:
    api = ct.CDLL(str(library.resolve()))
    handle = ct.c_void_p
    handle_output = ct.POINTER(handle)
    signatures = {
        "lodestar_last_error": ([], ct.c_char_p),
        "lodestar_asset_load_in_space": ([ct.c_char_p, ct.c_uint32, handle_output], ct.c_uint32),
        "lodestar_asset_generate": ([handle, ct.c_uint32, ct.c_int32, handle_output], ct.c_uint32),
        "lodestar_mesh_get_view": ([handle, ct.POINTER(_MeshView)], ct.c_uint32),
        "lodestar_mesh_get_metadata": ([handle, ct.POINTER(_MeshMetadata)], ct.c_uint32),
        "lodestar_mesh_get_provenance": ([handle, ct.POINTER(_MeshProvenance)], ct.c_uint32),
        "lodestar_mesh_destroy": ([handle], None),
        "lodestar_asset_destroy": ([handle], None),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(api, name)
        function.argtypes = arguments
        function.restype = result

    def checked(name, *args):
        status = getattr(api, name)(*args)
        if status != 0:
            message = api.lodestar_last_error()
            detail = message.decode("utf-8", errors="replace") if message else f"status {status}"
            raise RuntimeError(f"{name} failed: {detail}")

    path = str(source.resolve()).encode("utf-8")
    if b"\0" in path:
        raise ValueError("The mesh path contains a null byte")
    asset, mesh = handle(), handle()
    try:
        scene_space = 1
        original_mode = 0
        checked("lodestar_asset_load_in_space", path, scene_space, ct.byref(asset))
        checked("lodestar_asset_generate", asset, original_mode, 1, ct.byref(mesh))
        view, metadata, provenance = _MeshView(), _MeshMetadata(), _MeshProvenance()
        checked("lodestar_mesh_get_view", mesh, ct.byref(view))
        checked("lodestar_mesh_get_metadata", mesh, ct.byref(metadata))
        checked("lodestar_mesh_get_provenance", mesh, ct.byref(provenance))
        yield view, metadata, provenance
    finally:
        if mesh:
            api.lodestar_mesh_destroy(mesh)
        if asset:
            api.lodestar_asset_destroy(asset)


def load_patch(
    library: Path, source: Path, triangles: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return copied positions, UV0, and normals in selected imported face order."""
    if not triangles or any(type(face) is not int or face < 0 for face in triangles):
        raise ValueError("Select one or more nonnegative integer triangle indices")
    if len(set(triangles)) != len(triangles):
        raise ValueError("Triangle indices must be distinct")

    with _scene_mesh(library, source) as (view, metadata, provenance):
        if view.index_count % 3 or not view.indices or not view.vertices:
            raise ValueError("The imported asset has no complete indexed triangles")
        face_count = view.index_count // 3
        if max(triangles) >= face_count:
            raise ValueError(f"Triangle index exceeds the imported face range 0–{face_count - 1}")
        if not view.has_uv:
            raise ValueError("The imported asset has no UV0")
        if metadata.has_additional_uv or provenance.has_unrepresented_vertex_attributes:
            raise ValueError("The imported asset contains additional UV or unrepresented vertex attributes")

        for label, pointer, count in (
            ("material", metadata.triangle_materials, metadata.triangle_material_count),
            ("surface", provenance.triangle_surface_ids, provenance.triangle_surface_id_count),
        ):
            if not pointer or count != face_count:
                raise ValueError(f"The imported asset has no complete per-triangle {label} IDs")
            ids = np.ctypeslib.as_array(pointer, shape=(count,))[triangles]
            if not np.all(ids == ids[0]):
                raise ValueError(f"Selected triangles span multiple source {label}s")

        faces = np.ctypeslib.as_array(view.indices, shape=(view.index_count,)).reshape(-1, 3)[triangles]
        if faces.max() >= view.vertex_count:
            raise ValueError("A selected face references a missing vertex")
        vertices = np.ctypeslib.as_array(view.vertices, shape=(view.vertex_count,))
        positions, uvs, normals = (np.array(vertices[field][faces], dtype=np.float64, copy=True)
                                   for field in ("position", "uv", "normal"))
        if not all(np.isfinite(array).all() for array in (positions, uvs, normals)):
            raise ValueError("Selected triangles contain non-finite positions, UVs, or normals")
        if np.any(np.all(uvs == 0, axis=(1, 2))):
            # The ABI only has a mesh-wide UV flag and zero-fills absent UV0.
            raise ValueError("A selected face has all-zero UVs, which can indicate missing UV0")
        edges = positions[:, 1:] - positions[:, :1]
        if np.any(np.all(np.cross(edges[:, 0], edges[:, 1]) == 0, axis=1)):
            raise ValueError("A selected face is degenerate")
        geometric_faces = {tuple(sorted(map(tuple, face))) for face in positions}
        if len(geometric_faces) != len(positions):
            raise ValueError("Selected triangles contain coincident faces")
        return positions, uvs, normals
