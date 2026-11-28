// Reads a correspondence fixture and prints the C++ correspondenceBound. The Python
// tools/lodestar_surface_fit.py correspondence_bound is the oracle (correspondence_oracle_test.py).
// SPDX-License-Identifier: MIT
#include <lodestar/metric.hpp>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: lodestar_correspondence_oracle fixture\n";
        return 2;
    }
    std::ifstream in(argv[1]);
    std::string tag;
    std::size_t count = 0;
    std::vector<glm::dvec2> uv;
    std::vector<glm::dvec3> positions, normals;
    std::vector<uint32_t> source, target;
    in >> tag >> count;
    if (tag != "V") return 2;
    for (std::size_t i = 0; i < count; ++i) {
        glm::dvec2 t;
        glm::dvec3 p, n;
        in >> t.x >> t.y >> p.x >> p.y >> p.z >> n.x >> n.y >> n.z;
        uv.push_back(t);
        positions.push_back(p);
        normals.push_back(n);
    }
    for (auto* faces : {&source, &target}) {
        in >> tag >> count;
        for (std::size_t i = 0; i < 3 * count; ++i) {
            uint32_t v;
            in >> v;
            faces->push_back(v);
        }
    }
    if (!in) {
        std::cerr << "bad fixture\n";
        return 2;
    }
    const lodestar::metric::Chart a{uv, positions, normals, source, {}}, b{uv, positions, normals, target, {}};
    const auto r = lodestar::metric::correspondenceBound(a, b);
    std::printf("%.17g %.17g %zu %d %d\n", r.surfaceBound, r.normalAngleDegrees, r.pairs, r.covered ? 1 : 0,
                r.limitExceeded ? 1 : 0);
    return 0;
}
