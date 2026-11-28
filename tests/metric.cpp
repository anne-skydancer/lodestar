// Tests of the certified LOD metric (lodestar_common): soundness against the sampled metrics,
// zero error for identical LODs, texture and shading bounds, rigged terms and the record.
// SPDX-License-Identifier: MIT
#include <glm/gtc/matrix_transform.hpp>
#include <lodestar/mesh_metrics.hpp>
#include <lodestar/metric.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>

namespace {

using namespace lodestar;
using namespace lodestar::metric;

int failures = 0;

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void computeNormals(Mesh& m) {
    std::vector<glm::dvec3> sum(m.vertices.size(), glm::dvec3(0.0));
    for (std::size_t i = 0; i + 2 < m.indices.size(); i += 3) {
        const auto a = glm::dvec3(m.vertices[m.indices[i]].position), b = glm::dvec3(m.vertices[m.indices[i + 1]].position),
                   c = glm::dvec3(m.vertices[m.indices[i + 2]].position);
        const auto n = glm::cross(b - a, c - a);
        for (int k = 0; k < 3; ++k) sum[m.indices[i + k]] += n;
    }
    for (std::size_t i = 0; i < sum.size(); ++i) {
        const double l = glm::length(sum[i]);
        m.vertices[i].normal = l > 0 ? glm::vec3(sum[i] / l) : glm::vec3(0, 1, 0);
    }
}

using Height = std::function<double(double, double)>;

// A heightfield over the unit square; UV = (x, y). Counter-clockwise in UV.
Mesh gridMesh(int n, const Height& h, bool flip = false, double xMax = 1.0) {
    Mesh m;
    m.hasUV = true;
    for (int j = 0; j <= n; ++j)
        for (int i = 0; i <= n; ++i) {
            const double x = double(i) / n * xMax, y = double(j) / n;
            m.vertices.push_back({glm::vec3(float(x), float(h(x, y)), float(y)), glm::vec3(0, 1, 0), glm::vec2(float(x), float(y))});
        }
    const auto id = [&](int i, int j) { return uint32_t(j * (n + 1) + i); };
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const auto a = id(i, j), b = id(i + 1, j), c = id(i + 1, j + 1), d = id(i, j + 1);
            if (!flip) m.indices.insert(m.indices.end(), {a, b, c, a, c, d});
            else m.indices.insert(m.indices.end(), {a, b, d, b, c, d});
        }
    computeNormals(m);
    return m;
}

Mesh sphereMesh(int rings, int segments, double radius = 1.0) {
    Mesh m;
    m.hasUV = true;
    const double pi = std::numbers::pi;
    for (int r = 0; r <= rings; ++r)
        for (int s = 0; s <= segments; ++s) {
            const double theta = pi * r / rings, phi = 2 * pi * s / segments;
            const glm::dvec3 p(radius * std::sin(theta) * std::cos(phi), radius * std::cos(theta), radius * std::sin(theta) * std::sin(phi));
            m.vertices.push_back({glm::vec3(p), glm::vec3(p / radius), glm::vec2(float(double(s) / segments), float(double(r) / rings))});
        }
    for (int r = 0; r < rings; ++r)
        for (int s = 0; s < segments; ++s) {
            const uint32_t a = r * (segments + 1) + s, b = a + 1, c = a + segments + 1, d = c + 1;
            m.indices.insert(m.indices.end(), {a, c, b, b, c, d});
        }
    return m;
}

double sampledMaximum(const Mesh& a, const Mesh& b) {
    const auto extent = [](const Mesh& m) {
        glm::vec3 lo(1e30f), hi(-1e30f);
        for (const auto& v : m.vertices) {
            lo = glm::min(lo, v.position);
            hi = glm::max(hi, v.position);
        }
        const auto e = hi - lo;
        return double(std::max({e.x, e.y, e.z}));
    };
    const auto d = metrics::measureSurfaceDistance(a, b, 4096);
    return std::max(d.sourceToCandidate.maximum * extent(a), d.candidateToSource.maximum * extent(b));
}

double largest(const Mesh& m) {
    glm::vec3 lo(1e30f), hi(-1e30f);
    for (const auto& v : m.vertices) {
        lo = glm::min(lo, v.position);
        hi = glm::max(hi, v.position);
    }
    const auto e = hi - lo;
    return std::max({e.x, e.y, e.z});
}

const Height bump = [](double x, double y) { return 0.12 * std::sin(5 * x) * std::cos(4 * y) + 0.05 * x; };

void testSoundnessAgainstSampled() {
    struct Case { const char* name; Mesh high, low; };
    std::vector<Case> cases;
    cases.push_back({"grid 32->8", gridMesh(32, bump), gridMesh(8, bump)});
    cases.push_back({"grid 32->4 flipped", gridMesh(32, bump), gridMesh(4, bump, true)});
    cases.push_back({"grid 24->3", gridMesh(24, bump), gridMesh(3, bump)});
    cases.push_back({"sphere 32x64->8x16", sphereMesh(32, 64), sphereMesh(8, 16)});
    cases.push_back({"sphere 24x48->6x12", sphereMesh(24, 48), sphereMesh(6, 12)});
    cases.push_back({"sphere vs smaller sphere", sphereMesh(16, 32), sphereMesh(16, 32, 0.9)});
    {
        // A narrow spike that the sampled metric is unlikely to hit.
        Mesh spiked = gridMesh(16, bump);
        spiked.vertices[8 * 17 + 8].position.y += 0.4f;
        computeNormals(spiked);
        cases.push_back({"spike", spiked, gridMesh(16, bump)});
    }
    for (const auto& c : cases) {
        const auto shape = shapeBound(c.high, c.low);
        const double sampled = sampledMaximum(c.high, c.low);
        std::printf("  %-26s U=%.6g W=%.6g sampled=%.6g U/W=%.3f%s\n", c.name, shape.upper, shape.witness, sampled,
                    shape.witness > 0 ? shape.upper / shape.witness : 0.0, shape.capped ? " capped" : "");
        require(shape.upper >= sampled * (1 - 1e-12), std::string(c.name) + ": U below the sampled maximum");
        require(shape.upper >= shape.witness, std::string(c.name) + ": U below W");
        require(!shape.capped, std::string(c.name) + ": hit the work cap");
        require(shape.upper <= 1.1 * shape.witness + 1e-6 * largest(c.high), std::string(c.name) + ": U not within 1.1 W");
        // Symmetry: the bound is two-sided.
        const auto swapped = shapeBound(c.low, c.high);
        require(swapped.upper >= sampled * (1 - 1e-12), std::string(c.name) + ": swapped U below the sampled maximum");
    }
    // The spike is the error: U must reach it.
    const auto spike = shapeBound(cases.back().high, cases.back().low);
    require(spike.upper >= 0.4 * 0.99, "spike height is covered");
    // Per-axis form: with unit scale it is at least the scalar bound's witness.
    LodRecord lod;
    const auto shape = shapeBound(cases[0].high, cases[0].low);
    for (int a = 0; a < 3; ++a) {
        lod.axisOffset[a] = float(shape.axisOffset[a]);
        lod.axisRadius[a] = float(shape.axisRadius[a]);
    }
    require(shapeUpperUnderScale(lod, {1, 1, 1}) >= shape.witness * (1 - 1e-6), "per-axis form covers W");
}

void testPerAxisScaleSound() {
    // Under a non-uniform scale s the bound sqrt(sum s^2 A^2) + sqrt(sum s^2 R^2) must cover the
    // scaled Hausdorff distance, measured by scaling both meshes.
    const Mesh high = gridMesh(32, bump), low = gridMesh(6, bump);
    const auto shape = shapeBound(high, low);
    LodRecord lod;
    for (int a = 0; a < 3; ++a) {
        lod.axisOffset[a] = float(shape.axisOffset[a]);
        lod.axisRadius[a] = float(shape.axisRadius[a]);
    }
    for (const glm::vec3 s : {glm::vec3(1, 1, 1), glm::vec3(10, 0.5f, 3), glm::vec3(0.2f, 20, 1), glm::vec3(1, 1, 30)}) {
        Mesh a = high, b = low;
        for (auto* m : {&a, &b})
            for (auto& v : m->vertices) v.position *= s;
        const double scaled = sampledMaximum(a, b);
        const double bound = shapeUpperUnderScale(lod, {s.x, s.y, s.z});
        require(bound >= scaled * (1 - 1e-6), "per-axis bound covers the scaled sampled maximum");
    }
}

void testIdentical() {
    const Mesh high = gridMesh(16, bump);
    {
        const std::vector<const Mesh*> lods{&high, &high, &high};
        const Record r = measureAsset(high, lods);
        for (int i = 0; i < 3; ++i) {
            const auto& l = r.lods[i];
            require((l.flags & lodCopyOfHigh) != 0, "copy flag for an identical LOD");
            require(l.shape.upper == 0 && l.shape.witness == 0 && l.texture.upper == 0 && l.shading.upper == 0,
                    "exact zero for a copy");
        }
    }
    {
        // The same surface with its triangles reordered and rotated: not a bitwise copy.
        Mesh again = high;
        std::vector<uint32_t> idx;
        for (std::size_t t = high.indices.size() / 3; t-- > 0;) {
            idx.insert(idx.end(), {high.indices[3 * t + 1], high.indices[3 * t + 2], high.indices[3 * t]});
        }
        again.indices = idx;
        const auto shape = shapeBound(high, again);
        const double e = largest(high);
        require(shape.upper <= 3e-10 * e, "identical surface: U is at the allowance");
        require(shape.witness <= 1e-12 * e, "identical surface: W is zero");
        const auto texture = textureBound(high, again);
        require(texture.upper <= 3e-10 * e && texture.witness <= 1e-12 * e, "identical surface: texture zero");
        const std::vector<const Mesh*> lods{&again};
        const Record r = measureAsset(high, lods);
        const auto& l = r.lods[0];
        require(!(l.flags & lodCopyOfHigh), "reordered LOD is not flagged a copy");
        require(l.shape.upper <= 3e-10f && l.texture.upper <= 3e-10f, "identical reordered LOD: tiny U");
        require(std::isfinite(l.shading.upper) && l.shading.upper <= 3e-10f, "identical reordered LOD: shading certified");
    }
}

void testTexture() {
    const Mesh high = gridMesh(24, bump), low = gridMesh(6, bump);
    const auto t = textureBound(high, low);
    // Brute-force the texture deviation at random UVs.
    const auto locate = [](const Mesh& m, double u, double v, glm::dvec3& out) {
        for (std::size_t i = 0; i + 2 < m.indices.size(); i += 3) {
            const auto &a = m.vertices[m.indices[i]], &b = m.vertices[m.indices[i + 1]], &c = m.vertices[m.indices[i + 2]];
            const glm::dvec2 p(u, v), e1 = glm::dvec2(b.uv) - glm::dvec2(a.uv), e2 = glm::dvec2(c.uv) - glm::dvec2(a.uv), d = p - glm::dvec2(a.uv);
            const double det = e1.x * e2.y - e2.x * e1.y;
            if (det == 0) continue;
            const double s = (d.x * e2.y - e2.x * d.y) / det, w = (e1.x * d.y - d.x * e1.y) / det;
            if (s < -1e-12 || w < -1e-12 || s + w > 1 + 1e-12) continue;
            out = (1 - s - w) * glm::dvec3(a.position) + s * glm::dvec3(b.position) + w * glm::dvec3(c.position);
            return true;
        }
        return false;
    };
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    double sampled = 0.0;
    for (int i = 0; i < 4000; ++i) {
        glm::dvec3 a, b;
        const double u = uniform(rng), v = uniform(rng);
        if (locate(high, u, v, a) && locate(low, u, v, b)) sampled = std::max(sampled, glm::length(a - b));
    }
    std::printf("  texture U=%.6g W=%.6g sampled=%.6g\n", t.upper, t.witness, sampled);
    require(t.upper >= sampled, "texture U below a sampled deviation");
    require(t.witness <= t.upper && t.witness > 0.5 * sampled, "texture W is a real witness");
    require(t.overlay.covered && t.overlay.targetCovered, "nested grids cover each other");
    require(t.upper <= 1.0001 * t.witness + 1e-9, "shared-UV overlay is exact");

    // A High that does not cover the LOD's texture domain: not certifiable.
    Mesh partial = gridMesh(24, bump);
    partial.indices.resize(partial.indices.size() / 2);
    const auto p = textureBound(partial, low);
    require(std::isinf(p.upper), "High does not cover the LOD's UVs: U is infinite");
}

void testShading() {
    // Flat against flat: every normal matches, the shift is the position error.
    const Height flat = [](double, double) { return 0.0; };
    const Mesh a = gridMesh(8, flat), b = gridMesh(2, flat);
    const auto ta = textureBound(a, b);
    const auto s = shadingBound(a, b, ta);
    require(std::isfinite(s.upper) && s.upper <= 1e-9, "flat/flat shading is certified at zero");

    // A smooth sphere against a coarser one: the shift is bounded and U covers W.
    const Mesh hi = sphereMesh(32, 64), lo = sphereMesh(16, 32);
    const auto tex = textureBound(hi, lo);
    const auto sh = shadingBound(hi, lo, tex);
    std::printf("  shading sphere U=%.6g W=%.6g\n", sh.upper, sh.witness);
    require(sh.upper >= sh.witness, "shading U below W");

    // Dense-sample check: the true shift at points of the LOD is at most U.
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::vector<std::pair<glm::dvec3, glm::dvec3>> samples; // point, unit normal of the High mesh
    for (std::size_t i = 0; i + 2 < hi.indices.size(); i += 3) {
        const auto &p = hi.vertices[hi.indices[i]], &q = hi.vertices[hi.indices[i + 1]], &r = hi.vertices[hi.indices[i + 2]];
        for (int k = 0; k < 24; ++k) {
            double x = uniform(rng), y = uniform(rng);
            if (x + y > 1) { x = 1 - x; y = 1 - y; }
            const auto pos = (1 - x - y) * glm::dvec3(p.position) + x * glm::dvec3(q.position) + y * glm::dvec3(r.position);
            const auto n = (1 - x - y) * glm::dvec3(p.normal) + x * glm::dvec3(q.normal) + y * glm::dvec3(r.normal);
            samples.push_back({pos, glm::normalize(n)});
        }
    }
    const double cosine = std::cos(glm::radians(0.5));
    double worst = 0.0;
    for (std::size_t i = 0; i + 2 < lo.indices.size(); i += 3) {
        const auto &p = lo.vertices[lo.indices[i]], &q = lo.vertices[lo.indices[i + 1]], &r = lo.vertices[lo.indices[i + 2]];
        const auto n = glm::normalize(glm::dvec3(p.normal + q.normal + r.normal));
        const auto c = (glm::dvec3(p.position) + glm::dvec3(q.position) + glm::dvec3(r.position)) / 3.0;
        double best = 1e30;
        for (const auto& [pos, nn] : samples)
            if (glm::dot(nn, n) >= cosine) best = std::min(best, glm::length(pos - c));
        if (best < 1e29) worst = std::max(worst, best);
    }
    // The sampled nearest-match distance exceeds the true one by at most the sample spacing.
    const double spacing = 0.06;
    require(worst - spacing <= sh.upper || std::isinf(sh.upper), "shading U covers a dense-sample estimate");
}

Skin makeSkin(const Mesh& m, const std::function<std::array<float, 2>(const glm::vec3&)>& weights) {
    Skin skin;
    skin.jointPivots = {glm::vec3(0.0f), glm::vec3(0.5f, 0.0f, 0.0f)};
    for (const auto& v : m.vertices) {
        const auto w = weights(v.position);
        skin.joints.push_back({0, 1, 0, 0});
        skin.weights.push_back({w[0], w[1], 0, 0});
    }
    return skin;
}

void testRigged() {
    const Mesh high = gridMesh(24, bump), low = gridMesh(6, bump);
    // Weights blend across x in [0.3, 0.7]. The LOD has the same blend, then a steeper one.
    const auto blend = [](float lo, float hi) {
        return [=](const glm::vec3& p) {
            const float t = std::clamp((p.x - lo) / (hi - lo), 0.0f, 1.0f);
            return std::array<float, 2>{1 - t, t};
        };
    };
    const Skin hs = makeSkin(high, blend(0.3f, 0.7f)), same = makeSkin(low, blend(0.3f, 0.7f)), steep = makeSkin(low, blend(0.45f, 0.55f));
    const Options opt;
    const double unit = largest(high);

    for (const auto* lowSkin : {&same, &steep}) {
        const auto terms = riggedTerms({&high, &hs}, {&low, lowSkin});
        const auto shape = shapeBound(high, low);
        // Sample poses: rotations of joint 1 about its pivot.
        std::vector<glm::mat4> poses;
        std::mt19937 rng(3);
        std::uniform_real_distribution<float> angle(-3.14f, 3.14f);
        for (int i = 0; i < 40; ++i) {
            const glm::vec3 axis = glm::normalize(glm::vec3(angle(rng), angle(rng), angle(rng)) + glm::vec3(0.01f));
            const glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), angle(rng), axis);
            poses.push_back(glm::mat4(1.0f));
            poses.push_back(glm::translate(glm::mat4(1.0f), glm::vec3(0.5f, 0, 0)) * rotation *
                            glm::translate(glm::mat4(1.0f), glm::vec3(-0.5f, 0, 0)));
        }
        const double witness = poseWitness({&high, &hs}, {&low, lowSkin}, poses, 2);
        const double upper = 1.0 * shape.upper + terms.weight * 2.0 * terms.radius;
        std::printf("  rigged weight=%.4g radius=%.4g U=%.6g poseW=%.6g\n", terms.weight, terms.radius, upper, witness);
        require(witness <= upper * (1 + 1e-9) + 1e-9, "pose witness stays under the rigged upper bound");
        require(witness >= 0.0, "pose witness is non-negative");
        (void)unit;
    }
    // Same weights: the weight term is small; steeper weights: larger. A pose that deforms is a witness.
    const auto tSame = riggedTerms({&high, &hs}, {&low, &same});
    const auto tSteep = riggedTerms({&high, &hs}, {&low, &steep});
    require(tSteep.weight >= tSame.weight, "steeper weights do not shrink the weight term");
    require(tSteep.weight <= 1.0 && tSame.weight >= 0.0, "weight term is in [0, 1]");
    // Record carries the terms.
    const std::vector<MeshWithSkin> lods{{&low, &steep}};
    const Record r = measureAsset({&high, &hs}, lods);
    require((r.lods[0].flags & lodRiggedTerms) != 0 && (r.flags & recordRigged) != 0, "rigged flags set");
    require(riggedUpper(r.lods[0], 1.0) >= r.lods[0].shape.upper, "rigged upper includes the bind-pose bound");
}

void testRecord() {
    const Mesh high = gridMesh(20, bump), m = gridMesh(10, bump), l = gridMesh(5, bump), lw = gridMesh(2, bump);
    const std::vector<const Mesh*> lods{&m, &l, &lw};
    const Record r = measureAsset(high, lods);
    for (int i = 0; i < 3; ++i) {
        std::printf("  LOD %d: tris=%u G(U=%.4g,W=%.4g) T(U=%.4g,W=%.4g) N(U=%.4g,W=%.4g) flags=0x%x\n", i, r.lods[i].triangles,
                    r.lods[i].shape.upper, r.lods[i].shape.witness, r.lods[i].texture.upper, r.lods[i].texture.witness,
                    r.lods[i].shading.upper, r.lods[i].shading.witness, r.lods[i].flags);
        require(r.lods[i].shape.upper >= r.lods[i].shape.witness, "record U >= W");
    }
    require(r.lods[2].shape.upper >= r.lods[0].shape.witness, "coarser LOD has at least the error of a finer one (witness)");
    const auto bytes = serialise(r);
    require(bytes.size() == recordSerialisedSize && bytes[0] == recordFormatVersion, "serialised size and version byte");
    const auto back = deserialise(bytes);
    require(back.has_value() && serialise(*back) == bytes, "record round trip");
    if (back) {
        require(back->lods[1].triangles == r.lods[1].triangles && back->lods[1].hash == r.lods[1].hash &&
                    back->lods[2].shape.upper == r.lods[2].shape.upper && back->highHash == r.highHash,
                "round-tripped fields");
    }
    auto bad = bytes;
    bad[0] = 2;
    require(!deserialise(bad), "unknown format version is rejected");
    bad = bytes;
    bad[1] = 9;
    require(!deserialise(bad), "unknown metric version is rejected");
    bad = bytes;
    bad.pop_back();
    require(!deserialise(bad), "wrong size is rejected");
    // Little endian: the high triangle count is at bytes 4..7.
    require(bytes[4] == (r.highTriangles & 0xff) && bytes[5] == ((r.highTriangles >> 8) & 0xff), "little-endian layout");
    // The hash of a block is stable and depends on content.
    Mesh changed = m;
    changed.vertices[3].position.y += 1e-3f;
    require(hashMesh(m) == hashMesh(m) && hashMesh(m) != hashMesh(changed), "block hash");
    std::printf("  record size %zu bytes\n", recordSerialisedSize);
}

void testHash() {
    const auto hash = [](const std::string& s) {
        return xxh64({reinterpret_cast<const uint8_t*>(s.data()), s.size()});
    };
    require(hash("") == 0xEF46DB3751D8E999ULL, "XXH64 of the empty input");
    require(hash("abc") == 0x44BC2CF5AD770999ULL, "XXH64 of abc");
    require(hash(std::string(100, 'x')) == hash(std::string(100, 'x')), "XXH64 is deterministic");
}

} // namespace

int main() {
    try {
        std::puts("hash");
        testHash();
        std::puts("soundness against the sampled metrics");
        testSoundnessAgainstSampled();
        testPerAxisScaleSound();
        std::puts("identical LODs");
        testIdentical();
        std::puts("texture");
        testTexture();
        std::puts("shading");
        testShading();
        std::puts("rigged");
        testRigged();
        std::puts("record");
        testRecord();
    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << '\n';
        return 2;
    }
    if (failures) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::puts("ok");
    return 0;
}
