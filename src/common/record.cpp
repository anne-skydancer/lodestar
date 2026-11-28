// Record serialisation, XXH64 and the asset measurement that fills a Record.
// SPDX-License-Identifier: MIT
#include "metric_internal.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>

namespace lodestar::metric {

// ---- XXH64 -------------------------------------------------------------------------------------

namespace {

constexpr uint64_t P1 = 11400714785074694791ULL, P2 = 14029467366897019727ULL, P3 = 1609587929392839161ULL,
                   P4 = 9650029242287828579ULL, P5 = 2870177450012600261ULL;

uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
uint64_t read64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}
uint32_t read32(const uint8_t* p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
uint64_t round64(uint64_t acc, uint64_t input) { return rotl(acc + input * P2, 31) * P1; }
uint64_t merge64(uint64_t acc, uint64_t val) { return (acc ^ round64(0, val)) * P1 + P4; }

} // namespace

uint64_t xxh64(std::span<const uint8_t> bytes, uint64_t seed) {
    const uint8_t* p = bytes.data();
    const uint8_t* const end = p + bytes.size();
    uint64_t h;
    if (bytes.size() >= 32) {
        uint64_t v1 = seed + P1 + P2, v2 = seed + P2, v3 = seed, v4 = seed - P1;
        for (; p + 32 <= end; p += 32) {
            v1 = round64(v1, read64(p));
            v2 = round64(v2, read64(p + 8));
            v3 = round64(v3, read64(p + 16));
            v4 = round64(v4, read64(p + 24));
        }
        h = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
        h = merge64(h, v1);
        h = merge64(h, v2);
        h = merge64(h, v3);
        h = merge64(h, v4);
    } else {
        h = seed + P5;
    }
    h += bytes.size();
    for (; p + 8 <= end; p += 8) h = rotl(h ^ round64(0, read64(p)), 27) * P1 + P4;
    if (p + 4 <= end) {
        h = rotl(h ^ (uint64_t(read32(p)) * P1), 23) * P2 + P3;
        p += 4;
    }
    for (; p < end; ++p) h = rotl(h ^ (uint64_t(*p) * P5), 11) * P1;
    h ^= h >> 33;
    h *= P2;
    h ^= h >> 29;
    h *= P3;
    h ^= h >> 32;
    return h;
}

namespace {

void put(std::vector<uint8_t>& out, uint64_t value, int bytes) {
    for (int i = 0; i < bytes; ++i) out.push_back(static_cast<uint8_t>(value >> (8 * i)));
}
void putFloat(std::vector<uint8_t>& out, float f) { put(out, std::bit_cast<uint32_t>(f), 4); }

} // namespace

uint64_t hashMesh(const Mesh& mesh) {
    std::vector<uint8_t> bytes;
    bytes.reserve(8 + mesh.vertices.size() * 32 + mesh.indices.size() * 4);
    put(bytes, mesh.vertices.size(), 4);
    put(bytes, mesh.indices.size(), 4);
    for (const auto& v : mesh.vertices) {
        for (int k = 0; k < 3; ++k) putFloat(bytes, v.position[k]);
        for (int k = 0; k < 3; ++k) putFloat(bytes, v.normal[k]);
        for (int k = 0; k < 2; ++k) putFloat(bytes, v.uv[k]);
    }
    for (const auto i : mesh.indices) put(bytes, i, 4);
    return xxh64(bytes);
}

// ---- Record ------------------------------------------------------------------------------------

float Record::largestExtent() const { return std::max({extent[0], extent[1], extent[2]}); }

std::vector<uint8_t> serialise(const Record& r) {
    std::vector<uint8_t> out;
    out.reserve(recordSerialisedSize);
    put(out, r.format, 1);
    put(out, r.metric, 1);
    put(out, r.flags, 1);
    put(out, r.lodCount, 1);
    put(out, r.highTriangles, 4);
    put(out, r.highHash, 8);
    for (const float e : r.extent) putFloat(out, e);
    for (const auto& l : r.lods) {
        put(out, l.triangles, 4);
        put(out, l.hash, 8);
        put(out, l.flags, 2);
        for (const Bound* b : {&l.shape, &l.texture, &l.shading}) {
            putFloat(out, b->upper);
            putFloat(out, b->witness);
        }
        for (const float a : l.axisOffset) putFloat(out, a);
        for (const float a : l.axisRadius) putFloat(out, a);
        putFloat(out, l.riggedWeight);
        putFloat(out, l.riggedRadius);
    }
    return out;
}

std::optional<Record> deserialise(std::span<const uint8_t> bytes) {
    if (bytes.size() != recordSerialisedSize || bytes[0] != recordFormatVersion || bytes[1] != metricVersion)
        return std::nullopt;
    const uint8_t* p = bytes.data();
    bool ok = true;
    const auto get = [&](int n) {
        uint64_t v = 0;
        for (int i = n - 1; i >= 0; --i) v = (v << 8) | p[i];
        p += n;
        return v;
    };
    const auto getFloat = [&](bool allowInfinity) {
        const float f = std::bit_cast<float>(static_cast<uint32_t>(get(4)));
        if (std::isnan(f) || (!allowInfinity && std::isinf(f))) ok = false;
        return f;
    };
    Record r;
    r.format = static_cast<uint8_t>(get(1));
    r.metric = static_cast<uint8_t>(get(1));
    r.flags = static_cast<uint8_t>(get(1));
    r.lodCount = static_cast<uint8_t>(get(1));
    r.highTriangles = static_cast<uint32_t>(get(4));
    r.highHash = get(8);
    for (float& e : r.extent) e = getFloat(false);
    for (auto& l : r.lods) {
        l.triangles = static_cast<uint32_t>(get(4));
        l.hash = get(8);
        l.flags = static_cast<uint16_t>(get(2));
        for (Bound* b : {&l.shape, &l.texture, &l.shading}) {
            b->upper = getFloat(true);
            b->witness = getFloat(true);
        }
        for (float& a : l.axisOffset) a = getFloat(true);
        for (float& a : l.axisRadius) a = getFloat(true);
        l.riggedWeight = getFloat(true);
        l.riggedRadius = getFloat(true);
    }
    if (!ok || r.lodCount > lodSlots) return std::nullopt;
    return r;
}

double shapeUpperUnderScale(const LodRecord& lod, const std::array<double, 3>& s) {
    double a = 0.0, r = 0.0;
    for (int i = 0; i < 3; ++i) {
        a += s[i] * s[i] * double(lod.axisOffset[i]) * double(lod.axisOffset[i]);
        r += s[i] * s[i] * double(lod.axisRadius[i]) * double(lod.axisRadius[i]);
    }
    return std::sqrt(a) + std::sqrt(r);
}

double riggedUpper(const LodRecord& lod, double sigmaMax) {
    if (!(lod.flags & lodRiggedTerms)) return std::numeric_limits<double>::infinity();
    if (lod.riggedWeight == 0.0f) return sigmaMax * lod.shape.upper;
    return sigmaMax * lod.shape.upper + double(lod.riggedWeight) * 2.0 * lod.riggedRadius;
}

// ---- Measurement -------------------------------------------------------------------------------

namespace {

float roundUp(double x) {
    if (std::isnan(x)) return notCertified;
    float f = static_cast<float>(x);
    if (static_cast<double>(f) < x) f = std::nextafter(f, std::numeric_limits<float>::infinity());
    return f;
}

float roundDown(double x) {
    if (std::isnan(x)) return 0.0f;
    float f = static_cast<float>(x);
    if (static_cast<double>(f) > x) f = std::nextafter(f, -std::numeric_limits<float>::infinity());
    return f;
}

Bound makeBound(double upper, double witness, double unit) {
    return {roundUp(upper / unit), roundDown(witness / unit)};
}

bool sameMesh(const Mesh& a, const Mesh& b) {
    if (a.vertices.size() != b.vertices.size() || a.indices != b.indices) return false;
    if (a.vertices.empty()) return true;
    return std::memcmp(a.vertices.data(), b.vertices.data(), a.vertices.size() * sizeof(Vertex)) == 0;
}

bool sameSkin(const Skin* a, const Skin* b) {
    if (!a || !b) return a == b;
    return a->joints == b->joints && a->weights == b->weights && a->jointPivots == b->jointPivots;
}

} // namespace

Record measureAsset(const MeshWithSkin& high, std::span<const MeshWithSkin> lods, const Options& options) {
    using namespace detail;
    Record record;
    const Mesh& hm = *high.mesh;
    record.highTriangles = static_cast<uint32_t>(hm.indices.size() / 3);
    record.highHash = hashMesh(hm);
    if (high.skin) record.flags |= recordRigged;
    const dvec3 box = boxExtent(hm);
    double unit = std::max({box.x, box.y, box.z});
    if (!(unit > 0.0) || !std::isfinite(unit)) unit = 1.0;
    for (int i = 0; i < 3; ++i) record.extent[i] = static_cast<float>(box[i]);
    const double axisFloor = 1e-6 * unit;
    const double grid = options.gridFraction * unit;

    Prepared prepared(hm);
    record.lodCount = static_cast<uint8_t>(std::min(lods.size(), lodSlots));
    for (std::size_t i = 0; i < record.lodCount; ++i) {
        const Mesh& lm = *lods[i].mesh;
        LodRecord& out = record.lods[i];
        out.triangles = static_cast<uint32_t>(lm.indices.size() / 3);
        out.hash = hashMesh(lm);
        out.flags = lodMeasured;
        if (sameMesh(hm, lm) && sameSkin(high.skin, lods[i].skin)) {
            out.flags |= lodCopyOfHigh | lodTextureCertified | lodShadingCertified;
            out.shape = out.texture = out.shading = {0.0f, 0.0f};
            out.axisOffset = out.axisRadius = {0.0f, 0.0f, 0.0f};
            if (high.skin) {
                out.flags |= lodRiggedTerms;
                out.riggedWeight = 0.0f;
                out.riggedRadius = 0.0f;
            }
            continue;
        }
        Prepared lod(lm);
        const bool rigged = high.skin && lods[i].skin;
        std::vector<Leaf> highLeaves, lodLeaves;
        const ShapeResult shape = shapeBetween(prepared, lod, grid, options, rigged ? &highLeaves : nullptr,
                                               rigged ? &lodLeaves : nullptr);
        out.shape = makeBound(shape.upper, shape.witness, unit);
        for (int a = 0; a < 3; ++a) {
            const double axis = std::max(box[a], axisFloor);
            out.axisOffset[a] = roundUp(shape.axisOffset[a] / axis);
            out.axisRadius[a] = roundUp(shape.axisRadius[a] / axis);
        }
        if (shape.capped) out.flags |= lodShapeCapped;

        const TextureResult texture = textureBound(hm, lm, options);
        out.texture = makeBound(texture.upper, texture.witness, unit);
        if (!texture.hasUV) out.flags |= lodNoUV;
        if (std::isfinite(texture.upper)) out.flags |= lodTextureCertified;

        const ShadingResult shading = shadingBetween(prepared, lod, texture, grid, options);
        out.shading = makeBound(shading.upper, shading.witness, unit);
        if (std::isfinite(shading.upper)) out.flags |= lodShadingCertified;

        if (rigged) {
            const RiggedTerms terms = riggedBetween(prepared, *high.skin, lod, *lods[i].skin, highLeaves, lodLeaves, unit);
            out.riggedWeight = roundUp(terms.weight);
            out.riggedRadius = roundUp(terms.radius / unit);
            out.flags |= lodRiggedTerms;
        }
    }
    return record;
}

Record measureAsset(const Mesh& high, std::span<const Mesh* const> lods, const Options& options) {
    std::vector<MeshWithSkin> wrapped;
    for (const Mesh* m : lods) wrapped.push_back({m, nullptr});
    return measureAsset(MeshWithSkin{&high, nullptr}, wrapped, options);
}

} // namespace lodestar::metric
