// Texture deviation on the UV overlay. A port of correspondence_bound from
// tools/lodestar_surface_fit.py (the Python stays the test oracle).
// SPDX-License-Identifier: MIT
#include "metric_internal.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace lodestar::metric {
namespace {

using glm::dvec2;
using glm::dvec3;

double cross2(const dvec2& a, const dvec2& b) { return a.x * b.y - a.y * b.x; }

// A convex polygon from clipping a triangle by three half-planes has at most six corners.
struct Polygon {
    std::array<dvec2, 8> p;
    std::size_t n = 0;
    void push_back(const dvec2& v) { p[n++] = v; }
    std::size_t size() const { return n; }
    bool empty() const { return n == 0; }
    const dvec2& back() const { return p[n - 1]; }
    const dvec2* begin() const { return p.data(); }
    const dvec2* end() const { return p.data() + n; }
    const dvec2& operator[](std::size_t i) const { return p[i]; }
};

// Clip `first` by the half-planes of `second`'s edges, as the Python _intersection does.
Polygon intersection(const std::array<dvec2, 3>& first, const std::array<dvec2, 3>& second) {
    Polygon polygon;
    for (const auto& v : first) polygon.push_back(v);
    for (int e = 0; e < 3; ++e) {
        const dvec2 a = second[e], b = second[(e + 1) % 3];
        if (polygon.empty()) break;
        Polygon output;
        dvec2 previous = polygon.back();
        double oldDistance = cross2(b - a, previous - a);
        for (const dvec2& current : polygon) {
            const double distance = cross2(b - a, current - a);
            if ((distance >= 0) != (oldDistance >= 0)) {
                const double t = oldDistance / (oldDistance - distance);
                output.push_back(previous + t * (current - previous));
            }
            if (distance >= 0) output.push_back(current);
            previous = current;
            oldDistance = distance;
        }
        polygon = std::move(output);
    }
    return polygon;
}

// Barycentric weights of p in triangle t. Returns false for a degenerate triangle.
bool weights(const dvec2& p, const std::array<dvec2, 3>& t, dvec3& out) {
    const dvec2 e1 = t[1] - t[0], e2 = t[2] - t[0], d = p - t[0];
    const double det = e1.x * e2.y - e2.x * e1.y;
    if (det == 0.0) return false;
    const double s = (d.x * e2.y - e2.x * d.y) / det;
    const double u = (e1.x * d.y - d.x * e1.y) / det;
    out = dvec3(1 - s - u, s, u);
    return true;
}

struct Face {
    std::array<dvec2, 3> uv;
    std::array<dvec3, 3> pos;
    std::array<dvec3, 3> nrm{};
    dvec2 lo, hi;
    double area = 0.0; // absolute UV area
    uint32_t material = 0;
    bool valid = true;
};

// Faces with counter-clockwise UVs (the clip needs it). A flipped face swaps two corners.
std::vector<Face> buildFaces(const Chart& c, bool normals) {
    std::vector<Face> faces(c.triangles.size() / 3);
    for (std::size_t f = 0; f < faces.size(); ++f) {
        Face& face = faces[f];
        std::array<uint32_t, 3> id{c.triangles[3 * f], c.triangles[3 * f + 1], c.triangles[3 * f + 2]};
        if (id[0] >= c.uv.size() || id[1] >= c.uv.size() || id[2] >= c.uv.size() ||
            id[0] >= c.positions.size() || id[1] >= c.positions.size() || id[2] >= c.positions.size() ||
            (normals && (id[0] >= c.normals.size() || id[1] >= c.normals.size() || id[2] >= c.normals.size()))) {
            face.valid = false;
            continue;
        }
        const double signedArea = cross2(c.uv[id[1]] - c.uv[id[0]], c.uv[id[2]] - c.uv[id[0]]) * 0.5;
        if (signedArea < 0) std::swap(id[1], id[2]);
        for (int k = 0; k < 3; ++k) {
            face.uv[k] = c.uv[id[k]];
            face.pos[k] = c.positions[id[k]];
            if (normals) face.nrm[k] = c.normals[id[k]];
        }
        face.lo = glm::min(glm::min(face.uv[0], face.uv[1]), face.uv[2]);
        face.hi = glm::max(glm::max(face.uv[0], face.uv[1]), face.uv[2]);
        face.area = std::abs(signedArea);
        face.material = f < c.materials.size() ? c.materials[f] : 0;
    }
    return faces;
}

dvec3 mix(const dvec3& w, const std::array<dvec3, 3>& v) { return w.x * v[0] + w.y * v[1] + w.z * v[2]; }

// A uniform grid over the target faces' UV bounding boxes.
struct Grid {
    dvec2 origin{0.0};
    dvec2 cell{1.0};
    int n = 1;
    std::vector<std::vector<uint32_t>> cells;
    // Faces that span many cells (large or tiled UV ranges) are tested against every query.
    std::vector<uint32_t> big;

    explicit Grid(const std::vector<Face>& faces) {
        dvec2 lo(std::numeric_limits<double>::infinity()), hi(-std::numeric_limits<double>::infinity());
        for (const auto& f : faces) {
            if (!f.valid) continue;
            lo = glm::min(lo, f.lo);
            hi = glm::max(hi, f.hi);
        }
        if (!(lo.x <= hi.x)) return;
        n = std::clamp(static_cast<int>(std::sqrt(static_cast<double>(faces.size()))), 1, 1024);
        origin = lo;
        cell = glm::max((hi - lo) / double(n), dvec2(1e-300));
        cells.resize(static_cast<std::size_t>(n) * n);
        for (uint32_t i = 0; i < faces.size(); ++i) {
            if (!faces[i].valid) continue;
            const auto [x0, y0, x1, y1] = range(faces[i].lo, faces[i].hi);
            if ((x1 - x0 + 1) * (y1 - y0 + 1) > 16) {
                big.push_back(i);
                continue;
            }
            for (int y = y0; y <= y1; ++y)
                for (int x = x0; x <= x1; ++x) cells[static_cast<std::size_t>(y) * n + x].push_back(i);
        }
    }

    std::array<int, 4> range(const dvec2& lo, const dvec2& hi) const {
        const auto clampIndex = [&](double v) { return std::clamp(static_cast<int>(std::floor(v)), 0, n - 1); };
        return {clampIndex((lo.x - origin.x) / cell.x), clampIndex((lo.y - origin.y) / cell.y),
                clampIndex((hi.x - origin.x) / cell.x), clampIndex((hi.y - origin.y) / cell.y)};
    }
};

} // namespace

Correspondence correspondenceBound(const Chart& source, const Chart& target, const Options& options) {
    Correspondence out;
    const bool useNormals = !source.normals.empty() && !target.normals.empty();
    const std::vector<Face> src = buildFaces(source, useNormals);
    const std::vector<Face> dst = buildFaces(target, useNormals);
    const Grid grid(dst);

    bool constantNormals = true;
    if (useNormals) {
        const dvec3 first = src.empty() ? dvec3(0.0) : src.front().nrm[0];
        for (const auto* faces : {&src, &dst})
            for (const auto& f : *faces)
                for (const auto& n : f.nrm) constantNormals = constantNormals && n == first;
    }

    dvec3 lo(std::numeric_limits<double>::infinity()), hi(-std::numeric_limits<double>::infinity());
    for (const auto* positions : {&source.positions, &target.positions})
        for (const auto& p : *positions) {
            lo = glm::min(lo, p);
            hi = glm::max(hi, p);
        }

    std::vector<double> targetCovered(dst.size(), 0.0);
    std::vector<uint32_t> stamp(dst.size(), 0);
    uint32_t stampValue = 0;
    std::size_t visits = 0;
    // The witness pass is optional (any subset of samples is a lower bound): bound its total work.
    std::size_t witnessBudget = 4000000;
    double surfaceError = 0.0, normalAngle = 0.0, witness = 0.0;
    std::vector<uint32_t> candidates;
    struct Cell { uint32_t other; Polygon polygon; };
    std::vector<Cell> cells;

    for (std::size_t s = 0; s < src.size(); ++s) {
        const Face& face = src[s];
        if (!face.valid) continue;
        candidates.clear();
        ++stampValue;
        if (!grid.cells.empty()) {
            const auto visit = [&](uint32_t t) {
                ++visits;
                if (stamp[t] == stampValue) return;
                stamp[t] = stampValue;
                const Face& other = dst[t];
                if (glm::all(glm::greaterThanEqual(other.hi, face.lo)) && glm::all(glm::lessThanEqual(other.lo, face.hi)) &&
                    (source.materials.empty() || target.materials.empty() || other.material == face.material))
                    candidates.push_back(t);
            };
            const auto [x0, y0, x1, y1] = grid.range(face.lo, face.hi);
            for (int y = y0; y <= y1; ++y)
                for (int x = x0; x <= x1; ++x)
                    for (const uint32_t t : grid.cells[static_cast<std::size_t>(y) * grid.n + x]) visit(t);
            for (const uint32_t t : grid.big) visit(t);
        }
        if (visits > 4 * options.overlayPairLimit) {
            out.limitExceeded = true;
            break;
        }
        out.pairs += candidates.size();
        if (out.pairs > options.overlayPairLimit) {
            out.limitExceeded = true;
            break;
        }
        double coveredArea = 0.0;
        cells.clear();
        for (const uint32_t t : candidates) {
            const Face& other = dst[t];
            Polygon polygon = intersection(face.uv, other.uv);
            if (polygon.size() < 3) continue;
            double twice = 0.0;
            for (std::size_t k = 0; k < polygon.size(); ++k) twice += cross2(polygon[k], polygon[(k + 1) % polygon.size()]);
            const double area = std::abs(twice) * 0.5;
            if (area == 0) continue;
            coveredArea += area;
            targetCovered[t] += area;
            for (const dvec2& p : polygon) {
                dvec3 before, after;
                if (!weights(p, face.uv, before) || !weights(p, other.uv, after)) continue;
                surfaceError = std::max(surfaceError, glm::length(mix(before, face.pos) - mix(after, other.pos)));
            }
            if (!constantNormals && useNormals) {
                std::array<dvec3, 8> beforeNormals, afterNormals;
                std::size_t normalCount = 0;
                for (const dvec2& p : polygon) {
                    dvec3 before, after;
                    if (!weights(p, face.uv, before) || !weights(p, other.uv, after)) continue;
                    beforeNormals[normalCount] = mix(before, face.nrm);
                    afterNormals[normalCount++] = mix(after, other.nrm);
                }
                dvec3 reference(0.0);
                for (std::size_t i = 0; i < normalCount; ++i) reference += beforeNormals[i];
                reference /= static_cast<double>(std::max<std::size_t>(normalCount, 1));
                const double length = glm::length(reference);
                if (length == 0) {
                    out.normalsDegenerate = true;
                    continue;
                }
                double floorValue = std::numeric_limits<double>::infinity();
                double difference = 0.0;
                for (std::size_t i = 0; i < normalCount; ++i) {
                    floorValue = std::min(floorValue, glm::dot(beforeNormals[i], reference / length));
                    difference = std::max(difference, glm::length(beforeNormals[i] - afterNormals[i]));
                }
                if (floorValue <= 0) {
                    out.normalsDegenerate = true;
                    continue;
                }
                double sourceNorm = 0.0;
                for (const auto& n : face.nrm) sourceNorm = std::max(sourceNorm, glm::length(n));
                difference += 1e-12 * sourceNorm;
                const double angle = glm::degrees(2 * std::asin(std::min(1.0, difference / floorValue)));
                normalAngle = std::max(normalAngle, angle);
            }
            cells.push_back({t, std::move(polygon)});
        }
        // Witness: at each cell vertex, the nearest of all target faces that contain the point.
        // Skipped for a face with a very large candidate list (any subset of samples stays a lower bound).
        for (const Cell& cell : cells) {
            const std::size_t cost = candidates.size() * cell.polygon.size();
            if (cost > witnessBudget) break;
            witnessBudget -= cost;
            for (const dvec2& p : cell.polygon) {
                dvec3 before;
                if (!weights(p, face.uv, before)) continue;
                const dvec3 here = mix(before, face.pos);
                double nearest = std::numeric_limits<double>::infinity();
                for (const uint32_t t : candidates) {
                    const Face& other = dst[t];
                    dvec3 after;
                    if (!weights(p, other.uv, after)) continue;
                    if (after.x < -1e-9 || after.y < -1e-9 || after.z < -1e-9) continue;
                    nearest = std::min(nearest, glm::length(here - mix(after, other.pos)));
                }
                if (std::isfinite(nearest)) witness = std::max(witness, nearest);
            }
        }
        if (std::abs(coveredArea - face.area) > std::max(face.area * 1e-8, 1e-13)) out.covered = false;
    }
    for (std::size_t t = 0; t < dst.size(); ++t) {
        if (!dst[t].valid) continue;
        if (std::abs(targetCovered[t] - dst[t].area) > std::max(dst[t].area * 1e-8, 1e-13)) out.targetCovered = false;
    }
    double extent = 0.0;
    if (lo.x <= hi.x) extent = std::max({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z});
    out.surfaceBound = surfaceError + 1e-10 * extent;
    out.witness = std::min(witness, surfaceError);
    out.normalAngleDegrees = out.normalsDegenerate ? std::numeric_limits<double>::infinity() : normalAngle;
    return out;
}

namespace {

struct ChartStore {
    std::vector<dvec2> uv;
    std::vector<dvec3> positions, normals;
    std::vector<uint32_t> materials;

    explicit ChartStore(const Mesh& m) {
        for (const auto& v : m.vertices) {
            uv.emplace_back(v.uv);
            positions.emplace_back(v.position);
            normals.emplace_back(v.normal);
        }
        if (m.triangleMaterials.size() * 3 == m.indices.size()) materials = m.triangleMaterials;
    }
    Chart view(const Mesh& m) const { return {uv, positions, normals, m.indices, materials}; }
};

} // namespace

TextureResult textureBound(const Mesh& high, const Mesh& lod, const Options& options) {
    TextureResult r;
    r.hasUV = high.hasUV && lod.hasUV;
    if (!r.hasUV) return r;
    const ChartStore a(lod), b(high);
    r.overlay = correspondenceBound(a.view(lod), b.view(high), options);
    const bool complete = r.overlay.covered && !r.overlay.limitExceeded;
    r.witness = r.overlay.witness;
    r.upper = complete ? r.overlay.surfaceBound : std::numeric_limits<double>::infinity();
    return r;
}

} // namespace lodestar::metric
