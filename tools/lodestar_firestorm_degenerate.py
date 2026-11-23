"""Check static mesh triangles against the Firestorm 7.2.5.81788 uploader."""

import numpy as np


_MIN_NORMALIZED_SPAN = np.float32(1e-5)
_PACKED_POSITION_MAX = np.float32(65535)


def _triangles(positions):
    points = np.asarray(positions, dtype=np.float32)
    if points.ndim != 3 or points.shape[1:] != (3, 3) or not np.isfinite(points).all():
        raise ValueError("Expected finite triangle positions shaped (N, 3, 3)")
    return points


def _normalization(points):
    low = points.min(axis=(0, 1))
    high = points.max(axis=(0, 1))
    span = high - low
    span[span < _MIN_NORMALIZED_SPAN] = np.float32(1)
    translation = (low + high) * np.float32(-0.5)
    return translation, np.float32(1) / span


def normalize_triangles(positions):
    """Use the importer normalization for one chunk, across all its materials."""
    points = _triangles(positions)
    if not len(points):
        return points
    translation, scale = _normalization(points)
    return (points + translation) * scale


def _quantize(normalized):
    if not len(normalized):
        zero = np.zeros(3, dtype=np.float32)
        return normalized.astype(np.int64), zero, zero
    low = normalized.min(axis=(0, 1))
    span = normalized.max(axis=(0, 1)) - low
    fraction = np.divide(normalized - low, span, out=np.zeros_like(normalized), where=span > 0)
    # LLModel writes the same position domain to every material face of a model.
    # A flat axis decodes to its minimum regardless of the encoded component.
    packed = (fraction * _PACKED_POSITION_MAX).astype(np.int64)
    return packed, low, span


def quantized_triangles(positions):
    """Return the exact integer position grid used by the mesh asset encoder."""
    return _quantize(normalize_triangles(positions))[0]


def _zero_area(points):
    first = points[:, 1] - points[:, 0]
    second = points[:, 2] - points[:, 0]
    return np.all(np.cross(first, second) == 0, axis=1)


def _decoded_triangles(packed, low, span):
    return (packed.astype(np.float32) / _PACKED_POSITION_MAX) * span + low


def decoded_upload_positions(positions, high_positions):
    """Simulate the High instance transform applied to decoded tier positions."""
    packed, low, span = _quantize(normalize_triangles(positions))
    high = _triangles(high_positions)
    if not len(high):
        raise ValueError("The High instance needs triangle positions")
    translation, scale = _normalization(high)
    return _decoded_triangles(packed, low, span) * (np.float32(1) / scale) - translation


def visual_degenerate_mask(positions):
    """Mark triangles with zero geometric area before or after asset encoding."""
    points = _triangles(positions)
    normalized = normalize_triangles(points)
    packed, low, span = _quantize(normalized)
    decoded = _decoded_triangles(packed, low, span)
    return (_zero_area(points.astype(np.float64)) |
            _zero_area(normalized.astype(np.float64)) |
            _zero_area(packed) |
            _zero_area(decoded.astype(np.float64)))


def _dot(first, second):
    products = first * second
    return (products[:, 0] + products[:, 1]) + products[:, 2]


def _physics_degenerate(normalized):
    # LLModelPreview scales the already normalized corners before ll_is_degenerate.
    a, b, c = np.moveaxis(normalized * np.float32(0.5), 1, 0)
    ab, ac, cb = a - b, a - c, c - b
    ab_squared, ac_squared, cb_squared = _dot(ab, ab), _dot(ac, ac), _dot(cb, cb)
    edge_ratio_squared = np.float32(100)
    # Preserve the release's comparisons, including the repeated AB terms.
    ab_ok = ((ab_squared <= edge_ratio_squared * ac_squared) &
             (ab_squared <= edge_ratio_squared * cb_squared))
    ac_ok = ((ac_squared <= edge_ratio_squared * ab_squared) &
             (ab_squared <= edge_ratio_squared * cb_squared))
    cb_ok = ((cb_squared <= edge_ratio_squared * ab_squared) &
             (ab_squared <= edge_ratio_squared * ac_squared))
    tolerance = np.float32(1e-7)
    threshold = np.where(ab_ok & ac_ok & cb_ok, tolerance * np.float32(1e-4), tolerance)
    area_a = np.cross(ab, ac)
    area_b = np.cross(b - a, b - c)
    small = (_dot(area_a, area_a) < threshold) | (_dot(area_b, area_b) < threshold)
    dot_ab_cb = _dot(ab, cb)
    determinant = ab_squared * cb_squared - dot_ab_cb * dot_ab_cb
    return small | (determinant == 0)


def physics_degenerate_mask(positions):
    """Mark triangles rejected by the float32, pre-packing triangle physics check."""
    return _physics_degenerate(normalize_triangles(positions))


def packed_physics_degenerate_mask(positions):
    """Check precision and the same physics rule after the U16 asset round trip."""
    packed, low, span = _quantize(normalize_triangles(positions))
    return _zero_area(packed) | _physics_degenerate(_decoded_triangles(packed, low, span))
