#include <lodestar/lodestar.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void close(double actual, double expected, double tolerance, const char* message) {
    require(std::abs(actual - expected) <= tolerance * std::max(1.0, std::abs(expected)), message);
}

// Exact degree elevation of the polynomial graph z = a*x*x over a triangle.
lodestar::BezierTriangle parabola(const glm::dvec2& origin, const glm::dvec2& first,
                                  const glm::dvec2& second, double a) {
    constexpr std::array<std::array<int, 2>, 10> degrees{{
        {0, 0}, {3, 0}, {0, 3}, {1, 0}, {2, 0}, {0, 1}, {0, 2}, {2, 1}, {1, 2}, {1, 1}}};
    const auto du = first - origin;
    const auto dv = second - origin;
    lodestar::BezierTriangle result;
    for (std::size_t control = 0; control < degrees.size(); ++control) {
        const double i = degrees[control][0], j = degrees[control][1];
        const auto xy = origin + du * (i / 3) + dv * (j / 3);
        const double z = origin.x * origin.x + 2 * origin.x * (du.x * i + dv.x * j) / 3 +
                         du.x * du.x * i * (i - 1) / 6 + dv.x * dv.x * j * (j - 1) / 6 +
                         du.x * dv.x * i * j / 3;
        result.controls[control] = {xy, a * z};
    }
    return result;
}

void rejects(const auto& operation) {
    try { operation(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("Expected invalid surface input to be rejected");
}

void tests() {
    using namespace lodestar;
    const auto flat = parabola({0, 0}, {1, 0}, {0, 1}, 0);
    const auto plane = integrateSurface(flat);
    close(plane.area, 0.5, 1e-13, "Plane surface area must be exact to roundoff");
    close(plane.absoluteCurvature, 0.0, 1e-12, "A plane has zero curvature");
    require(plane.converged, "A regular planar patch must converge");

    const auto curved = parabola({0, 0}, {1, 0}, {0, 1}, 0.75);
    std::vector<glm::dvec2> uv;
    std::vector<glm::dvec3> points;
    for (unsigned i = 0; i <= 6; ++i) {
        for (unsigned j = 0; j + i <= 6; ++j) {
            const double u = double(i) / 6, v = double(j) / 6;
            uv.push_back({u, v});
            points.push_back({u, v, 0.75 * u * u});
        }
    }
    const auto fitted = fitSurface(uv, points);
    require(fitted.degree == 2 && fitted.maximumVertexResidual < 1e-12,
            "Native fitting must recover a quadratic surface from indexed mesh samples");
    const auto reconstructed = surfacePatch(fitted, {{{0, 0}, {1, 0}, {0, 1}}});
    for (std::size_t i = 0; i < curved.controls.size(); ++i) {
        require(glm::length(reconstructed.controls[i] - curved.controls[i]) < 1e-12,
                "Polynomial restriction must reproduce the equivalent Bezier control net");
    }
    auto cubicPoints = points;
    for (std::size_t i = 0; i < uv.size(); ++i) cubicPoints[i].z = uv[i].x * uv[i].x * uv[i].y;
    const auto cubic = fitSurface(uv, cubicPoints);
    require(cubic.degree == 3 && cubic.maximumVertexResidual < 1e-12,
            "Native fitting must select a cubic when lower degrees cannot reproduce the data");
    const std::array<glm::dvec2, 3> domain{{{0.1, 0.2}, {0.8, 0.1}, {0.15, 0.7}}};
    const auto restricted = surfacePatch(cubic, domain);
    const auto mapped = domain[0] + 0.2 * (domain[1] - domain[0]) + 0.3 * (domain[2] - domain[0]);
    const auto evaluation = differentiateSurface(restricted, 0.2, 0.3);
    require(glm::length(evaluation.position - glm::dvec3(mapped, mapped.x * mapped.x * mapped.y)) < 1e-12,
            "Native polynomial restriction must preserve an arbitrary triangular UV domain");
    rejects([&] { (void)fitSurface(std::vector<glm::dvec2>(points.size(), {0, 0}), points); });
    const auto d = differentiateSurface(curved, 0.2, 0.3);
    require(glm::length(d.position - glm::dvec3(0.2, 0.3, 0.03)) < 1e-13, "Surface evaluation is incorrect");
    require(glm::length(d.du - glm::dvec3(1, 0, 0.3)) < 1e-13, "Analytic first derivative is incorrect");
    require(glm::length(d.dv - glm::dvec3(0, 1, 0)) < 1e-13, "Analytic transverse derivative is incorrect");
    require(glm::length(d.duu - glm::dvec3(0, 0, 1.5)) < 1e-13 && glm::length(d.duv) < 1e-13 &&
                glm::length(d.dvv) < 1e-13, "Analytic second derivatives are incorrect");
    const auto corner = differentiateSurface(curved, 1, 0);
    require(glm::length(corner.position - glm::dvec3(1, 0, 0.75)) < 1e-13 &&
                glm::length(corner.du - glm::dvec3(1, 0, 1.5)) < 1e-13,
            "Analytic derivatives must work at domain boundaries");

    const std::array patches{curved, parabola({1, 1}, {0, 1}, {1, 0}, 0.75)};
    const auto budget = calculateSurfaceLod(patches, 0.01, 0.5, 1e-6);
    const double twiceA = 1.5;
    const double area = 0.5 * (std::sqrt(1 + twiceA * twiceA) + std::asinh(twiceA) / twiceA);
    close(budget.integral.area, area, 2e-6, "Integrated parabolic cylinder area disagrees with its antiderivative");
    close(budget.integral.absoluteCurvature, std::atan(twiceA), 2e-6,
          "Integrated principal curvature disagrees with its antiderivative");
    close(budget.triangleDensityIntegral, 2 * std::atan(twiceA) / (3 * std::sqrt(3.0) * 0.005), 2e-6,
          "LOD density must be integrated over the surface");
    require(budget.integral.converged, "Curved surface quadrature must converge");
    const auto distant = calculateSurfaceLod(patches, 0.02, 0.5, 1e-6);
    close(distant.triangleDensityIntegral, budget.triangleDensityIntegral * 0.5, 1e-12,
          "Doubling pixel footprint must halve the continuous density budget");
    require(distant.suggestedTriangles < budget.suggestedTriangles, "A more distant LOD needs fewer triangles");

    auto scaled = patches;
    for (auto& patch : scaled) for (auto& point : patch.controls) point *= 10;
    const auto scaledBudget = calculateSurfaceLod(scaled, 0.1, 0.5, 1e-6);
    close(scaledBudget.integral.area, budget.integral.area * 100, 2e-6, "Area must scale quadratically");
    close(scaledBudget.integral.absoluteCurvature, budget.integral.absoluteCurvature * 10, 2e-6,
          "Integrated curvature must scale linearly");
    close(scaledBudget.triangleDensityIntegral, budget.triangleDensityIntegral, 2e-6,
          "Changing world units must preserve the LOD budget");
    auto translated = patches;
    for (auto& patch : translated) for (auto& point : patch.controls) point += glm::dvec3(10, -5, 7);
    const auto moved = calculateSurfaceLod(translated, 0.01, 0.5, 1e-6);
    close(moved.triangleDensityIntegral, budget.triangleDensityIntegral, 2e-6, "Translation must preserve density");
    const auto movedDerivative = differentiateSurface(translated[0], 0.2, 0.3);
    require(glm::length(movedDerivative.du - d.du) < 1e-12 &&
                glm::length(movedDerivative.duu - d.duu) < 1e-12,
            "Translations must preserve analytic derivatives");
    require(!integrateSurface(curved, 1e-10, 0).converged, "Exhausted quadrature must report nonconvergence");

    const auto flatMesh = tessellateSurface(flat, 0.01, 0.5);
    require(flatMesh.mesh.vertices.size() == 3 && flatMesh.mesh.indices.size() == 3,
            "A planar patch needs only three vertices");
    const auto nearMesh = tessellateSurface(curved, 0.01, 0.5);
    const auto farMesh = tessellateSurface(curved, 0.04, 0.5);
    require(farMesh.mesh.vertices.size() < nearMesh.mesh.vertices.size(),
            "The surface integral must drive actual vertex reduction at greater distances");
    require(nearMesh.maximumDeviationBound <= 0.005 && farMesh.maximumDeviationBound <= 0.02,
            "Generated LODs must satisfy the control-hull interpolation bound");
    for (const auto& vertex : nearMesh.mesh.vertices) {
        close(vertex.position.z, 0.75 * vertex.position.x * vertex.position.x, 1e-7,
              "Generated vertices must lie on the authoritative polynomial surface");
        const auto normal = glm::normalize(glm::dvec3(-1.5 * vertex.position.x, 0, 1));
        require(glm::length(glm::dvec3(vertex.normal) - normal) < 1e-7,
                "Generated shading normals must follow the analytic derivatives");
    }
    rejects([&] { (void)tessellateSurface(curved, 0.01, 0.5, 0); });

    rejects([&] { (void)calculateSurfaceLod(patches, 0, 1); });
    rejects([&] { (void)calculateSurfaceLod(patches, 1, -1); });
    rejects([&] { (void)calculateSurfaceLod({}, 1, 1); });
    rejects([&] { (void)differentiateSurface(curved, 0.6, 0.6); });
    auto invalid = flat;
    invalid.controls[0].x = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { (void)integrateSurface(invalid); });
    const BezierTriangle singular{};
    rejects([&] { (void)integrateSurface(singular); });
}

} // namespace

int main() {
    try { tests(); } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
