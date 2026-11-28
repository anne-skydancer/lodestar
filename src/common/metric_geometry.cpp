// SPDX-License-Identifier: MIT
#include "metric_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace lodestar::metric::detail {
namespace {

constexpr uint32_t leafSize = 4;
constexpr double pi = std::numbers::pi;

double boxDistanceSquared(const dvec3& lo, const dvec3& hi, const dvec3& p) {
    const auto d = glm::max(glm::dvec3(0.0), glm::max(lo - p, p - hi));
    return glm::dot(d, d);
}

dvec3 unitOrZero(const glm::vec3& n) {
    const dvec3 v(n);
    const double length = glm::length(v);
    return std::isfinite(length) && length > 0.0 ? v / length : dvec3(0.0);
}

} // namespace

double angleBetween(const dvec3& a, const dvec3& b) {
    return std::acos(std::clamp(glm::dot(a, b), -1.0, 1.0));
}

Surface makeSurface(std::span<const glm::vec3> positions, std::span<const glm::vec3> normals,
                    std::span<const uint32_t> indices) {
    Surface surface;
    surface.hasNormals = normals.size() == positions.size() && !normals.empty();
    for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
        const uint32_t i0 = indices[i], i1 = indices[i + 1], i2 = indices[i + 2];
        if (i0 >= positions.size() || i1 >= positions.size() || i2 >= positions.size()) continue;
        Tri t;
        t.p = {dvec3(positions[i0]), dvec3(positions[i1]), dvec3(positions[i2])};
        const double area = glm::length(glm::cross(t.p[1] - t.p[0], t.p[2] - t.p[0]));
        if (!(area > 0.0) || !std::isfinite(area)) continue;
        if (surface.hasNormals) t.n = {unitOrZero(normals[i0]), unitOrZero(normals[i1]), unitOrZero(normals[i2])};
        for (const auto& p : t.p) {
            surface.lo = glm::min(surface.lo, p);
            surface.hi = glm::max(surface.hi, p);
        }
        surface.tris.push_back(t);
        surface.source.push_back(static_cast<uint32_t>(i / 3));
    }
    return surface;
}

Surface makeSurface(const Mesh& mesh) {
    std::vector<glm::vec3> positions(mesh.vertices.size()), normals(mesh.vertices.size());
    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
        positions[i] = mesh.vertices[i].position;
        normals[i] = mesh.vertices[i].normal;
    }
    return makeSurface(positions, normals, mesh.indices);
}

dvec3 closestOnTriangle(const dvec3& p, const dvec3& a, const dvec3& b, const dvec3& c, dvec3* bary) {
    const auto set = [&](double u, double v, double w) { if (bary) *bary = dvec3(u, v, w); };
    const dvec3 ab = b - a, ac = c - a, ap = p - a;
    const double d1 = glm::dot(ab, ap), d2 = glm::dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { set(1, 0, 0); return a; }
    const dvec3 bp = p - b;
    const double d3 = glm::dot(ab, bp), d4 = glm::dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { set(0, 1, 0); return b; }
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        const double v = d1 / (d1 - d3);
        set(1 - v, v, 0);
        return a + v * ab;
    }
    const dvec3 cp = p - c;
    const double d5 = glm::dot(ab, cp), d6 = glm::dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) { set(0, 0, 1); return c; }
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        const double w = d2 / (d2 - d6);
        set(1 - w, 0, w);
        return a + w * ac;
    }
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        set(0, 1 - w, w);
        return b + w * (c - b);
    }
    const double sum = va + vb + vc;
    if (!(sum > 0.0) || !std::isfinite(sum)) { set(1, 0, 0); return a; }
    const double v = vb / sum, w = vc / sum;
    set(1 - v - w, v, w);
    return a + ab * v + ac * w;
}

Bvh::Bvh(const Surface& surface) : surface_(&surface) {
    order_.resize(surface.tris.size());
    for (uint32_t i = 0; i < order_.size(); ++i) order_[i] = i;
    if (!order_.empty()) {
        nodes_.reserve(order_.size() / 2 + 1);
        build(0, static_cast<uint32_t>(order_.size()));
    }
}

uint32_t Bvh::build(uint32_t begin, uint32_t end) {
    const uint32_t index = static_cast<uint32_t>(nodes_.size());
    nodes_.emplace_back();
    dvec3 lo(inf), hi(-inf), clo(inf), chi(-inf);
    for (uint32_t i = begin; i < end; ++i) {
        const auto& t = surface_->tris[order_[i]];
        dvec3 centre(0.0);
        for (const auto& p : t.p) {
            lo = glm::min(lo, p);
            hi = glm::max(hi, p);
            centre += p / 3.0;
        }
        clo = glm::min(clo, centre);
        chi = glm::max(chi, centre);
    }
    nodes_[index].lo = lo;
    nodes_[index].hi = hi;
    if (end - begin <= leafSize) {
        nodes_[index].first = begin;
        nodes_[index].count = end - begin;
        return index;
    }
    const dvec3 extent = chi - clo;
    const int axis = extent.x >= extent.y && extent.x >= extent.z ? 0 : (extent.y >= extent.z ? 1 : 2);
    const uint32_t mid = begin + (end - begin) / 2;
    std::nth_element(order_.begin() + begin, order_.begin() + mid, order_.begin() + end,
                     [&](uint32_t a, uint32_t b) {
                         const auto& ta = surface_->tris[a].p;
                         const auto& tb = surface_->tris[b].p;
                         return (ta[0][axis] + ta[1][axis] + ta[2][axis]) < (tb[0][axis] + tb[1][axis] + tb[2][axis]);
                     });
    const uint32_t l = build(begin, mid);
    const uint32_t r = build(mid, end);
    nodes_[index].left = l;
    nodes_[index].right = r;
    return index;
}

void Bvh::buildCones() {
    cones_.assign(surface_->tris.size(), {dvec3(0, 0, 1), pi});
    for (std::size_t i = 0; i < cones_.size(); ++i) {
        const auto& n = surface_->tris[i].n;
        if (glm::dot(n[0], n[0]) == 0.0 || glm::dot(n[1], n[1]) == 0.0 || glm::dot(n[2], n[2]) == 0.0) continue;
        const dvec3 sum = n[0] + n[1] + n[2];
        const double length = glm::length(sum);
        if (!(length > 1e-9)) continue;
        const dvec3 axis = sum / length;
        const double half = std::max({angleBetween(axis, n[0]), angleBetween(axis, n[1]), angleBetween(axis, n[2])});
        cones_[i] = {axis, half};
    }
    // Children are created after their parent, so walking the nodes in reverse merges bottom-up.
    for (std::size_t k = nodes_.size(); k-- > 0;) {
        auto& node = nodes_[k];
        std::vector<TriCone> parts;
        if (node.count > 0) {
            for (uint32_t i = 0; i < node.count; ++i) parts.push_back(cones_[order_[node.first + i]]);
        } else {
            parts = {{nodes_[node.left].axis, nodes_[node.left].halfAngle},
                     {nodes_[node.right].axis, nodes_[node.right].halfAngle}};
        }
        dvec3 sum(0.0);
        bool usable = true;
        for (const auto& c : parts) {
            usable = usable && c.half < pi;
            sum += c.axis;
        }
        const double length = glm::length(sum);
        if (!usable || !(length > 1e-9)) {
            node.axis = dvec3(0, 0, 1);
            node.halfAngle = pi;
            continue;
        }
        node.axis = sum / length;
        double half = 0.0;
        for (const auto& c : parts) half = std::max(half, angleBetween(node.axis, c.axis) + c.half);
        node.halfAngle = std::min(half, pi);
    }
}

Closest Bvh::nearest(const dvec3& p) const {
    Closest best;
    if (nodes_.empty()) return best;
    uint32_t stack[96];
    int top = 0;
    stack[top++] = 0;
    while (top > 0) {
        const Node& node = nodes_[stack[--top]];
        if (boxDistanceSquared(node.lo, node.hi, p) >= best.distanceSquared) continue;
        if (node.count > 0) {
            for (uint32_t i = 0; i < node.count; ++i) {
                const uint32_t t = order_[node.first + i];
                const auto& tri = surface_->tris[t];
                const dvec3 q = closestOnTriangle(p, tri.p[0], tri.p[1], tri.p[2]);
                const dvec3 d = q - p;
                const double d2 = glm::dot(d, d);
                if (d2 < best.distanceSquared) best = {q, d2, t};
            }
        } else {
            const double dl = boxDistanceSquared(nodes_[node.left].lo, nodes_[node.left].hi, p);
            const double dr = boxDistanceSquared(nodes_[node.right].lo, nodes_[node.right].hi, p);
            // Visit the nearer child first (pushed last).
            if (dl <= dr) { stack[top++] = node.right; stack[top++] = node.left; }
            else { stack[top++] = node.left; stack[top++] = node.right; }
        }
    }
    return best;
}

Closest Bvh::nearestMatching(const dvec3& p, const dvec3& dir, double angle, MatchMode mode) const {
    Closest best;
    if (nodes_.empty() || cones_.empty()) return best;
    uint32_t stack[96];
    int top = 0;
    stack[top++] = 0;
    const auto consider = [&](const dvec3& q, uint32_t t) {
        const dvec3 d = q - p;
        const double d2 = glm::dot(d, d);
        if (d2 < best.distanceSquared) best = {q, d2, t};
    };
    while (top > 0) {
        const Node& node = nodes_[stack[--top]];
        if (boxDistanceSquared(node.lo, node.hi, p) >= best.distanceSquared) continue;
        if (node.halfAngle < pi && angleBetween(dir, node.axis) - node.halfAngle > angle) continue;
        if (node.count > 0) {
            for (uint32_t i = 0; i < node.count; ++i) {
                const uint32_t t = order_[node.first + i];
                const auto& tri = surface_->tris[t];
                if (mode == MatchMode::Outside) {
                    const auto& c = cones_[t];
                    if (c.half < pi && angleBetween(dir, c.axis) - c.half > angle) continue;
                    consider(closestOnTriangle(p, tri.p[0], tri.p[1], tri.p[2]), t);
                } else {
                    int matching = 0;
                    for (int k = 0; k < 3; ++k) {
                        if (glm::dot(tri.n[k], tri.n[k]) > 0.0 && angleBetween(dir, tri.n[k]) <= angle) {
                            ++matching;
                            consider(tri.p[k], t);
                        }
                    }
                    if (matching == 3) consider(closestOnTriangle(p, tri.p[0], tri.p[1], tri.p[2]), t);
                }
            }
        } else {
            stack[top++] = node.right;
            stack[top++] = node.left;
        }
    }
    return best;
}

void Bvh::visitWithin(const dvec3& p, double radius, const std::function<void(uint32_t)>& f) const {
    if (nodes_.empty()) return;
    const double r2 = radius * radius;
    uint32_t stack[96];
    int top = 0;
    stack[top++] = 0;
    while (top > 0) {
        const Node& node = nodes_[stack[--top]];
        if (boxDistanceSquared(node.lo, node.hi, p) > r2) continue;
        if (node.count > 0) {
            for (uint32_t i = 0; i < node.count; ++i) {
                const uint32_t t = order_[node.first + i];
                const auto& tri = surface_->tris[t];
                const dvec3 d = closestOnTriangle(p, tri.p[0], tri.p[1], tri.p[2]) - p;
                if (glm::dot(d, d) <= r2) f(t);
            }
        } else {
            stack[top++] = node.right;
            stack[top++] = node.left;
        }
    }
}

} // namespace lodestar::metric::detail
