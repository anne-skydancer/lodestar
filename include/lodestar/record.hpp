// The per-mesh LOD error record (format v1) that a viewer stores and consumes.
// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace lodestar::metric {

inline constexpr uint8_t recordFormatVersion = 1;
// Bump when the meaning of a stored number changes; a store with another version is discarded.
inline constexpr uint8_t metricVersion = 1;
// Medium, Low and Lowest. High is the reference and is stored only as a triangle count and a hash.
inline constexpr std::size_t lodSlots = 3;
inline constexpr float notCertified = std::numeric_limits<float>::infinity();

// An error bracket in normalised units: a fraction of the largest axis extent of the High mesh.
// `upper` is certified (the true error is at most `upper`), `witness` is measured (the true error
// is at least `witness`). `upper` is infinity where no bound could be certified.
struct Bound {
    float upper = notCertified;
    float witness = 0.0f;
};

enum LodFlag : uint16_t {
    lodMeasured = 1u << 0,        // this slot holds a measurement (else it is empty)
    lodCopyOfHigh = 1u << 1,      // identical to High: every error is exactly zero
    lodShapeCapped = 1u << 2,     // the shape search hit its work cap; `upper` is valid but looser
    lodTextureCertified = 1u << 3,
    lodShadingCertified = 1u << 4,
    lodNoUV = 1u << 5,            // no texture coordinates: nothing can swim
    lodRiggedTerms = 1u << 6,     // riggedWeight and riggedRadius are valid
};

enum RecordFlag : uint8_t {
    recordRigged = 1u << 0, // the asset carries a skin
};

struct LodRecord {
    uint32_t triangles = 0;
    uint64_t hash = 0; // XXH64 of the LOD block (see hashMesh)
    uint16_t flags = 0;
    Bound shape;   // G: two-sided Hausdorff distance to High
    Bound texture; // T: texture deviation
    Bound shading; // N: normal-pattern shift
    // Per axis of High's bounding box, as a fraction of that axis extent (so a world error is
    // scale_i * value_i). The shape bound under object scale s is
    //   sqrt(sum s_i^2 offset_i^2) + sqrt(sum s_i^2 radius_i^2).
    std::array<float, 3> axisOffset{notCertified, notCertified, notCertified}; // A
    std::array<float, 3> axisRadius{notCertified, notCertified, notCertified}; // R
    // Rigged terms (valid with lodRiggedTerms). Over every joint rotation:
    //   error <= sigmaMax * shape.upper + riggedWeight * 2 * riggedRadius.
    float riggedWeight = notCertified; // upper bound of half the L1 distance of the skin weights, in [0, 1]
    float riggedRadius = 0.0f;         // largest distance from a point to its joints' pivots, normalised
};

struct Record {
    uint8_t format = recordFormatVersion;
    uint8_t metric = metricVersion;
    uint8_t flags = 0;
    uint8_t lodCount = 0; // slots in use, from the highest LOD down
    uint32_t highTriangles = 0;
    uint64_t highHash = 0;
    // High's bounding-box extent in input units. A normalised error times the largest extent is a
    // world error for a uniformly scaled object.
    std::array<float, 3> extent{};
    std::array<LodRecord, lodSlots> lods;

    float largestExtent() const;
};

// Serialised size in bytes: fixed, little endian, the first byte is the format version.
inline constexpr std::size_t recordSerialisedSize = 4 + 4 + 8 + 12 + lodSlots * (4 + 8 + 2 + 6 * 4 + 6 * 4 + 8);

std::vector<uint8_t> serialise(const Record& record);
// Returns nothing for a wrong size, an unknown format or metric version, or non-finite values
// where a finite one is required (infinity is allowed for upper bounds).
std::optional<Record> deserialise(std::span<const uint8_t> bytes);

// Normalised upper bound of the shape error of a LOD under object scale (the per-axis form).
double shapeUpperUnderScale(const LodRecord& lod, const std::array<double, 3>& scale);
// Normalised rigged upper bound for any joint rotation, with sigmaMax the wearer's largest joint scale.
double riggedUpper(const LodRecord& lod, double sigmaMax);

} // namespace lodestar::metric
