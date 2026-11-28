// Times the certified LOD metric (measureAsset) per mesh. The LODs are meshoptimizer reductions of
// High at 50 %, 20 % and 5 % of the triangles, standing in for creator LODs.
// usage: lodestar_metric_bench [--repeats N] [--synthetic] file...
// SPDX-License-Identifier: MIT
#include <lodestar/lodestar.hpp>
#include <lodestar/metric.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <string>
#include <vector>

namespace {

using namespace lodestar;

Mesh uvSphere(int rings, int segments) {
    Mesh m;
    m.hasUV = true;
    const double pi = std::numbers::pi;
    for (int r = 0; r <= rings; ++r)
        for (int s = 0; s <= segments; ++s) {
            const double theta = pi * r / rings, phi = 2 * pi * s / segments;
            const glm::vec3 p(float(std::sin(theta) * std::cos(phi)), float(std::cos(theta)), float(std::sin(theta) * std::sin(phi)));
            m.vertices.push_back({p, p, glm::vec2(float(double(s) / segments), float(double(r) / rings))});
        }
    for (int r = 0; r < rings; ++r)
        for (int s = 0; s < segments; ++s) {
            const uint32_t a = r * (segments + 1) + s, b = a + 1, c = a + segments + 1, d = c + 1;
            m.indices.insert(m.indices.end(), {a, c, b, b, c, d});
        }
    return m;
}

struct Row {
    std::string name;
    std::size_t triangles;
    double milliseconds;
    double shapeRatio; // U / W of the coarsest LOD
    bool capped;
    unsigned certifiedT, certifiedN;
};

double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    const double index = p * (v.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(index)), hi = static_cast<std::size_t>(std::ceil(index));
    return v[lo] + (v[hi] - v[lo]) * (index - lo);
}

void parts(const Mesh& high) {
    using clock = std::chrono::steady_clock;
    const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    for (const float fraction : {0.5f, 0.05f}) {
        const Mesh lod = simplifyMesh(high, fraction);
        const auto t0 = clock::now();
        const auto shape = metric::shapeBound(high, lod);
        const auto t1 = clock::now();
        const auto texture = metric::textureBound(high, lod);
        const auto t2 = clock::now();
        const auto shading = metric::shadingBound(high, lod, texture);
        const auto t3 = clock::now();
        std::printf("# parts %.2f: shape %.1f ms (U/W %.3f), texture %.1f ms, shading %.1f ms (U=%g W=%g) pairs=%zu covered=%d/%d\n", fraction, ms(t0, t1),
                    shape.witness > 0 ? shape.upper / shape.witness : 0.0, ms(t1, t2), ms(t2, t3), shading.upper, shading.witness, texture.overlay.pairs, int(texture.overlay.covered), int(texture.overlay.targetCovered));
    }
}

Row measure(const std::string& name, const Mesh& high, int repeats) {
    if (getenv("LODESTAR_BENCH_PARTS")) {
        std::printf("# %s\n", name.c_str());
        parts(high);
    }
    std::vector<Mesh> lods;
    for (const float fraction : {0.5f, 0.2f, 0.05f}) lods.push_back(simplifyMesh(high, fraction));
    std::vector<const Mesh*> pointers;
    for (const auto& l : lods) pointers.push_back(&l);
    std::vector<double> times;
    metric::Record record;
    for (int i = 0; i < repeats; ++i) {
        const auto start = std::chrono::steady_clock::now();
        record = metric::measureAsset(high, pointers);
        times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
    Row row{name, high.indices.size() / 3, percentile(times, 0.5), 0.0, false, 0, 0};
    for (std::size_t i = 0; i < lods.size(); ++i) {
        const auto& l = record.lods[i];
        row.capped = row.capped || (l.flags & metric::lodShapeCapped);
        row.certifiedT += (l.flags & metric::lodTextureCertified) != 0;
        row.certifiedN += (l.flags & metric::lodShadingCertified) != 0;
        if (i == lods.size() - 1 && l.shape.witness > 0) row.shapeRatio = l.shape.upper / l.shape.witness;
    }
    std::printf("%s,%zu,%.3f,%.3f,%d,%u,%u\n", name.c_str(), row.triangles, row.milliseconds, row.shapeRatio, row.capped,
                row.certifiedT, row.certifiedN);
    std::fflush(stdout);
    return row;
}

} // namespace

int main(int argc, char** argv) {
    int repeats = 3;
    bool synthetic = false;
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--repeats" && i + 1 < argc) repeats = std::atoi(argv[++i]);
        else if (arg == "--synthetic") synthetic = true;
        else files.push_back(arg);
    }
    std::puts("name,triangles,median_ms,shape_U_over_W_coarsest,capped,texture_certified_lods,shading_certified_lods");
    std::vector<Row> rows;
    for (const auto& file : files) {
        try {
            rows.push_back(measure(file.substr(file.find_last_of("/\\") + 1), loadMesh(file), repeats));
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s: %s\n", file.c_str(), e.what());
        }
    }
    if (synthetic) {
        // A size ladder, to cover the median (about 730 triangles), the mean and the 99th percentile tail.
        for (const auto [rings, segments] : {std::pair{16, 23}, {32, 64}, {60, 100}, {100, 200}, {140, 280}, {200, 400}})
            rows.push_back(measure("sphere-" + std::to_string(rings) + "x" + std::to_string(segments), uvSphere(rings, segments), repeats));
    }
    std::vector<double> ms;
    std::size_t within = 0, withinCount = 0;
    for (const auto& r : rows) {
        ms.push_back(r.milliseconds);
        if (r.shapeRatio > 0) {
            ++withinCount;
            within += r.shapeRatio <= 1.1;
        }
    }
    if (!ms.empty())
        std::printf("# meshes=%zu median=%.3f ms p99=%.3f ms max=%.3f ms; shape U<=1.1W on %zu/%zu\n", ms.size(),
                    percentile(ms, 0.5), percentile(ms, 0.99), *std::max_element(ms.begin(), ms.end()), within, withinCount);
    return 0;
}
