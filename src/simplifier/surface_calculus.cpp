#include <lodestar/surface_calculus.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace lodestar {
namespace {

constexpr std::array<std::array<int, 3>, 10> powers{{
    {3, 0, 0}, {0, 3, 0}, {0, 0, 3}, {2, 1, 0}, {1, 2, 0},
    {2, 0, 1}, {1, 0, 2}, {0, 2, 1}, {0, 1, 2}, {1, 1, 1}}};
constexpr std::array<double, 10> coefficients{1, 1, 1, 3, 3, 3, 3, 3, 3, 6};
constexpr std::array<glm::dvec2, 3> gradients{{{-1, -1}, {1, 0}, {0, 1}}};

bool finite(const glm::dvec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

double monomial(const std::array<double, 3>& values, const std::array<int, 3>& exponents) {
    double result = 1.0;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        for (int degree = 0; degree < exponents[axis]; ++degree) result *= values[axis];
    }
    return result;
}

SurfaceDifferential evaluate(const BezierTriangle& patch, const glm::dvec2& uv) {
    const std::array<double, 3> barycentric{1 - uv.x - uv.y, uv.x, uv.y};
    SurfaceDifferential result{};
    for (std::size_t term = 0; term < powers.size(); ++term) {
        const auto control = patch.controls[term] * coefficients[term];
        result.position += control * monomial(barycentric, powers[term]);
        // Analytic product-rule derivatives of the cubic Bernstein polynomial.
        for (std::size_t first = 0; first < 3; ++first) {
            auto exponents = powers[term];
            const auto degree = exponents[first];
            if (degree == 0) continue;
            --exponents[first];
            const auto derivative = control * double(degree) * monomial(barycentric, exponents);
            result.du += derivative * gradients[first].x;
            result.dv += derivative * gradients[first].y;
            for (std::size_t second = 0; second < 3; ++second) {
                const auto secondDegree = exponents[second];
                if (secondDegree == 0) continue;
                auto secondExponents = exponents;
                --secondExponents[second];
                const auto derivative2 = control * double(degree * secondDegree) *
                                         monomial(barycentric, secondExponents);
                result.duu += derivative2 * gradients[first].x * gradients[second].x;
                result.duv += derivative2 * gradients[first].x * gradients[second].y;
                result.dvv += derivative2 * gradients[first].y * gradients[second].y;
            }
        }
    }
    return result;
}

glm::dvec2 integrand(const BezierTriangle& patch, const glm::dvec2& uv) {
    const auto d = evaluate(patch, uv);
    const auto cross = glm::cross(d.du, d.dv);
    const double jacobian = glm::length(cross);
    const double E = glm::dot(d.du, d.du);
    const double F = glm::dot(d.du, d.dv);
    const double G = glm::dot(d.dv, d.dv);
    // |Su x Sv|^2 avoids cancellation in E*G-F*F for narrow parameter domains.
    const double determinant = glm::dot(cross, cross);
    if (!(determinant > 0.0) || !std::isfinite(determinant)) {
        throw std::invalid_argument("Surface integration encountered a singular parameterization");
    }
    const auto normal = cross / jacobian;
    const double e = glm::dot(normal, d.duu);
    const double f = glm::dot(normal, d.duv);
    const double g = glm::dot(normal, d.dvv);
    const double mean = (e * G - 2.0 * f * F + g * E) / (2.0 * determinant);
    const double gaussian = (e * g - f * f) / determinant;
    const double root = std::sqrt(std::max(0.0, mean * mean - gaussian));
    const double curvature = std::max(std::abs(mean + root), std::abs(mean - root));
    const glm::dvec2 value{jacobian, jacobian * curvature};
    if (!std::isfinite(value.x) || !std::isfinite(value.y)) {
        throw std::invalid_argument("Surface integration exceeded its numerical range");
    }
    return value;
}

using Domain = std::array<glm::dvec2, 3>;

glm::dvec2 quadrature(const BezierTriangle& patch, const Domain& domain, std::size_t& evaluations) {
    // Three-point Gaussian rule on a triangle. Refinement supplies adaptivity.
    constexpr std::array<glm::dvec3, 3> nodes{{{2.0 / 3, 1.0 / 6, 1.0 / 6},
        {1.0 / 6, 2.0 / 3, 1.0 / 6}, {1.0 / 6, 1.0 / 6, 2.0 / 3}}};
    const auto ab = domain[1] - domain[0];
    const auto ac = domain[2] - domain[0];
    const double area = std::abs(ab.x * ac.y - ab.y * ac.x) * 0.5;
    glm::dvec2 result{0.0};
    for (const auto& weights : nodes) {
        const auto uv = weights.x * domain[0] + weights.y * domain[1] + weights.z * domain[2];
        result += integrand(patch, uv) * (area / 3.0);
    }
    evaluations += nodes.size();
    return result;
}

struct IntegralState {
    glm::dvec2 value{0.0};
    glm::dvec2 error{0.0};
    std::size_t evaluations = 0;
    bool converged = true;
};

void integrate(const BezierTriangle& patch, const Domain& domain, const glm::dvec2& coarse,
               const glm::dvec2& tolerance, unsigned depth, IntegralState& result) {
    const auto ab = (domain[0] + domain[1]) * 0.5;
    const auto ac = (domain[0] + domain[2]) * 0.5;
    const auto bc = (domain[1] + domain[2]) * 0.5;
    const std::array<Domain, 4> children{{{domain[0], ab, ac}, {ab, domain[1], bc},
                                        {ac, bc, domain[2]}, {ab, bc, ac}}};
    std::array<glm::dvec2, 4> values;
    glm::dvec2 fine{0.0};
    for (std::size_t i = 0; i < children.size(); ++i) {
        values[i] = quadrature(patch, children[i], result.evaluations);
        fine += values[i];
    }
    const auto error = glm::abs(fine - coarse);
    const bool accepted = glm::all(glm::lessThanEqual(error, tolerance));
    if (accepted || depth == 0) {
        result.value += fine;
        result.error += error;
        result.converged &= accepted;
        return;
    }
    for (std::size_t i = 0; i < children.size(); ++i) {
        integrate(patch, children[i], values[i], tolerance * 0.25, depth - 1, result);
    }
}

} // namespace

SurfaceDifferential differentiateSurface(const BezierTriangle& patch, double u, double v) {
    if (!std::isfinite(u) || !std::isfinite(v) || u < 0 || v < 0 || u + v > 1 ||
        !std::ranges::all_of(patch.controls, finite)) {
        throw std::invalid_argument("Expected finite Bezier controls and triangular domain coordinates");
    }
    auto local = patch;
    for (auto& control : local.controls) control -= patch.controls[0];
    auto result = evaluate(local, {u, v});
    result.position += patch.controls[0];
    if (!finite(result.position) || !finite(result.du) || !finite(result.dv) ||
        !finite(result.duu) || !finite(result.duv) || !finite(result.dvv)) {
        throw std::invalid_argument("Surface derivatives exceeded their numerical range");
    }
    return result;
}

SurfaceIntegral integrateSurface(const BezierTriangle& patch, double relativeTolerance, unsigned maximumDepth) {
    if (!std::ranges::all_of(patch.controls, finite) || !std::isfinite(relativeTolerance) ||
        relativeTolerance < 1e-10 || relativeTolerance > 0.1 || maximumDepth > 8) {
        throw std::invalid_argument("Expected finite controls, tolerance in [1e-10, 0.1], and depth up to 8");
    }
    // Translating controls first improves derivative accuracy for distant objects.
    auto local = patch;
    double extent = 0.0;
    for (auto& point : local.controls) {
        point -= patch.controls[0];
        extent = std::max(extent, glm::length(point));
    }
    const Domain domain{{{0, 0}, {1, 0}, {0, 1}}};
    IntegralState result;
    const auto coarse = quadrature(local, domain, result.evaluations);
    // Scale-aware floors allow planar curvature to converge near roundoff.
    const glm::dvec2 scale{extent * extent, extent};
    const auto tolerance = relativeTolerance * glm::max(glm::abs(coarse), scale * 1e-6);
    integrate(local, domain, coarse, tolerance, maximumDepth, result);
    return {result.value.x, result.value.y, result.error.x, result.error.y,
            result.evaluations, result.converged};
}

SurfaceLodBudget calculateSurfaceLod(std::span<const BezierTriangle> patches, double worldUnitsPerPixel,
                                    double maximumPixelError, double relativeTolerance, unsigned maximumDepth) {
    const double error = worldUnitsPerPixel * maximumPixelError;
    if (patches.empty() || !std::isfinite(worldUnitsPerPixel) || worldUnitsPerPixel <= 0 ||
        !std::isfinite(maximumPixelError) || maximumPixelError <= 0 || !std::isfinite(error) || error <= 0) {
        throw std::invalid_argument("Expected patches, a positive pixel scale, and a positive pixel error");
    }
    SurfaceLodBudget result;
    result.worldError = error;
    double suggested = 0.0;
    for (const auto& patch : patches) {
        const auto part = integrateSurface(patch, relativeTolerance, maximumDepth);
        result.integral.area += part.area;
        result.integral.absoluteCurvature += part.absoluteCurvature;
        result.integral.estimatedAreaError += part.estimatedAreaError;
        result.integral.estimatedCurvatureError += part.estimatedCurvatureError;
        result.integral.evaluations += part.evaluations;
        result.integral.converged &= part.converged;
        const double density = 2.0 * part.absoluteCurvature / (3.0 * std::sqrt(3.0) * error);
        result.triangleDensityIntegral += density;
        suggested += std::max(1.0, std::ceil(density));
    }
    if (!std::isfinite(suggested) || suggested >= double(std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument("Calculated surface budget exceeds the supported size");
    }
    result.interiorVertexDensityIntegral = result.triangleDensityIntegral * 0.5;
    result.suggestedTriangles = static_cast<std::size_t>(suggested);
    return result;
}

} // namespace lodestar
