#include <lodestar/lodestar.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace lodestar {
namespace {

using Parameter = glm::dvec3;
using Domain = std::array<Parameter, 3>;
constexpr std::array<std::array<unsigned, 3>, 10> powers{{
    {3, 0, 0}, {0, 3, 0}, {0, 0, 3}, {2, 1, 0}, {1, 2, 0},
    {2, 0, 1}, {1, 0, 2}, {0, 2, 1}, {0, 1, 2}, {1, 1, 1}}};

glm::dvec3 blossom(const BezierTriangle& patch, const std::array<Parameter, 3>& parameters) {
    std::array<std::array<glm::dvec3, 4>, 4> net{};
    for (std::size_t i = 0; i < powers.size(); ++i) net[powers[i][1]][powers[i][2]] = patch.controls[i];
    for (unsigned step = 0; step < 3; ++step) {
        const auto& t = parameters[step];
        const unsigned degree = 2 - step;
        // Ascending indices retain the previous degree's upper neighbors.
        for (unsigned i = 0; i <= degree; ++i) {
            for (unsigned j = 0; j + i <= degree; ++j) {
                net[i][j] = t.x * net[i][j] + t.y * net[i + 1][j] + t.z * net[i][j + 1];
            }
        }
    }
    return net[0][0];
}

double deviationBound(const BezierTriangle& patch, const Domain& domain,
                      const std::array<glm::dvec3, 3>& vertices, double allowance) {
    double maximum = 0.0;
    for (const auto& power : powers) {
        std::array<Parameter, 3> parameters;
        unsigned next = 0;
        glm::dvec3 linear{0.0};
        for (unsigned corner = 0; corner < 3; ++corner) {
            for (unsigned n = 0; n < power[corner]; ++n) parameters[next++] = domain[corner];
            linear += vertices[corner] * (double(power[corner]) / 3.0);
        }
        // Both polynomials use the same domain. Their control-point difference
        // bounds all corresponding points by the Bernstein convex-hull property.
        maximum = std::max(maximum, glm::length(blossom(patch, parameters) - linear));
    }
    return maximum + allowance;
}

SurfaceTessellation build(const BezierTriangle& local, const glm::dvec3& origin,
                          unsigned level, double allowance) {
    const unsigned segments = 1u << level;
    SurfaceTessellation result;
    result.subdivisionLevel = level;
    std::vector<Parameter> parameters;
    std::vector<glm::dvec3> localPositions;
    std::vector<double> rounding;
    const std::size_t vertexCount = std::size_t(segments + 1) * (segments + 2) / 2;
    parameters.reserve(vertexCount);
    localPositions.reserve(vertexCount);
    rounding.reserve(vertexCount);
    result.mesh.vertices.reserve(vertexCount);
    std::vector<std::vector<uint32_t>> rows;
    for (unsigned i = 0; i <= segments; ++i) {
        auto& row = rows.emplace_back();
        for (unsigned j = 0; j + i <= segments; ++j) {
            const double u = double(i) / segments;
            const double v = double(j) / segments;
            const auto d = differentiateSurface(local, u, v);
            const auto geometric = glm::cross(d.du, d.dv);
            const double length = glm::length(geometric);
            if (!(length > 0) || !std::isfinite(length)) {
                throw std::invalid_argument("Surface tessellation needs regular derivatives at its vertices");
            }
            const auto position = glm::vec3(origin + d.position);
            if (!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z)) {
                throw std::invalid_argument("Tessellated positions exceed the float32 mesh range");
            }
            row.push_back(static_cast<uint32_t>(result.mesh.vertices.size()));
            result.mesh.vertices.push_back({position, glm::vec3(geometric / length), {0, 0}});
            parameters.push_back({1 - u - v, u, v});
            localPositions.push_back(d.position);
            rounding.push_back(glm::length((glm::dvec3(position) - origin) - d.position));
        }
    }
    const auto triangle = [&](uint32_t a, uint32_t b, uint32_t c) {
        const double bound = deviationBound(local, {parameters[a], parameters[b], parameters[c]},
            {localPositions[a], localPositions[b], localPositions[c]}, allowance);
        result.maximumDeviationBound = std::max(result.maximumDeviationBound,
                                                bound + std::max({rounding[a], rounding[b], rounding[c]}));
        const auto ab = glm::dvec3(result.mesh.vertices[b].position) - glm::dvec3(result.mesh.vertices[a].position);
        const auto ac = glm::dvec3(result.mesh.vertices[c].position) - glm::dvec3(result.mesh.vertices[a].position);
        if (!(glm::length(glm::cross(ab, ac)) > 0)) {
            throw std::invalid_argument("Float32 surface tessellation contains a collapsed triangle");
        }
        const auto midpoint = (parameters[a] + parameters[b] + parameters[c]) / 3.0;
        const auto d = differentiateSurface(local, midpoint.y, midpoint.z);
        if (!(glm::dot(glm::cross(ab, ac), glm::cross(d.du, d.dv)) > 0)) {
            // Request refinement when a coarse triangle reverses the patch orientation.
            result.maximumDeviationBound = std::numeric_limits<double>::infinity();
        }
        result.mesh.indices.insert(result.mesh.indices.end(), {a, b, c});
    };
    for (unsigned i = 0; i < segments; ++i) {
        for (unsigned j = 0; j + i < segments; ++j) {
            triangle(rows[i][j], rows[i + 1][j], rows[i][j + 1]);
            if (j + i + 1 < segments) triangle(rows[i + 1][j], rows[i + 1][j + 1], rows[i][j + 1]);
        }
    }
    return result;
}

} // namespace

SurfaceTessellation tessellateSurface(const BezierTriangle& patch, double worldUnitsPerPixel,
                                     double maximumPixelError, unsigned maximumDepth) {
    if (maximumDepth > 8) throw std::invalid_argument("Surface tessellation supports a subdivision depth up to 8");
    const std::array patches{patch};
    const auto budget = calculateSurfaceLod(patches, worldUnitsPerPixel, maximumPixelError);
    if (!budget.integral.converged) throw std::invalid_argument("Surface budget integration did not converge");
    auto local = patch;
    double extent = 0.0;
    for (auto& control : local.controls) {
        control -= patch.controls[0];
        extent = std::max(extent, glm::length(control));
    }
    const double allowance = 512 * std::numeric_limits<double>::epsilon() * extent;
    // Start from the curvature integral, then verify the actual generated mesh.
    unsigned level = 0;
    std::size_t triangles = 1;
    while (triangles < budget.suggestedTriangles && level < maximumDepth) {
        triangles *= 4;
        ++level;
    }
    for (; level <= maximumDepth; ++level) {
        auto result = build(local, patch.controls[0], level, allowance);
        if (result.maximumDeviationBound <= budget.worldError) {
            result.budget = budget;
            return result;
        }
    }
    throw std::invalid_argument("Surface tessellation cannot reach the pixel tolerance within its depth limit");
}

} // namespace lodestar
