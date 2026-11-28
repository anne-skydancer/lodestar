#include <lodestar/surface_calculus.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lodestar {
namespace {

constexpr std::array<std::array<unsigned, 2>, 10> powers{{
    {0, 0}, {1, 0}, {0, 1}, {2, 0}, {1, 1}, {0, 2}, {3, 0}, {2, 1}, {1, 2}, {0, 3}}};

double power(double value, unsigned exponent) {
    double result = 1;
    for (unsigned i = 0; i < exponent; ++i) result *= value;
    return result;
}

bool finite(const glm::dvec3& point) {
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

SurfaceDifferential polynomial(const PolynomialSurface& model, const glm::dvec2& uv) {
    SurfaceDifferential result;
    for (std::size_t i = 0; i < powers.size(); ++i) {
        const auto [u, v] = powers[i];
        const auto& coefficient = model.coefficients[i];
        result.position += coefficient * power(uv.x, u) * power(uv.y, v);
        if (u) result.du += coefficient * double(u) * power(uv.x, u - 1) * power(uv.y, v);
        if (v) result.dv += coefficient * double(v) * power(uv.x, u) * power(uv.y, v - 1);
        if (u > 1) result.duu += coefficient * double(u * (u - 1)) * power(uv.x, u - 2) * power(uv.y, v);
        if (u && v) result.duv += coefficient * double(u * v) * power(uv.x, u - 1) * power(uv.y, v - 1);
        if (v > 1) result.dvv += coefficient * double(v * (v - 1)) * power(uv.x, u) * power(uv.y, v - 2);
    }
    return result;
}

std::optional<PolynomialSurface> solve(std::span<const glm::dvec2> uv,
                                      std::span<const glm::dvec3> positions, unsigned degree) {
    const std::size_t width = (degree + 1) * (degree + 2) / 2;
    if (uv.size() < width) return std::nullopt;
    std::vector<std::vector<double>> columns(width, std::vector<double>(uv.size()));
    for (std::size_t column = 0; column < width; ++column) {
        const auto [u, v] = powers[column];
        for (std::size_t row = 0; row < uv.size(); ++row) {
            columns[column][row] = power(uv[row].x, u) * power(uv[row].y, v);
        }
    }
    std::array<std::array<double, 10>, 10> upper{};
    std::array<glm::dvec3, 10> projected;
    std::array<std::size_t, 10> permutation{};
    std::iota(permutation.begin(), permutation.end(), std::size_t{0});
    double largest = 0;
    for (std::size_t column = 0; column < width; ++column) {
        std::size_t pivot = column;
        double squared = -1;
        for (std::size_t candidate = column; candidate < width; ++candidate) {
            const double value = std::inner_product(columns[candidate].begin(), columns[candidate].end(),
                                                     columns[candidate].begin(), 0.0);
            if (value > squared) {
                squared = value;
                pivot = candidate;
            }
        }
        const double length = std::sqrt(squared);
        if (!std::isfinite(length) || length <= largest * 1e-10 || length == 0) return std::nullopt;
        largest = std::max(largest, length);
        std::swap(columns[column], columns[pivot]);
        std::swap(permutation[column], permutation[pivot]);
        for (std::size_t row = 0; row < column; ++row) std::swap(upper[row][column], upper[row][pivot]);
        upper[column][column] = length;
        for (double& value : columns[column]) value /= length;
        projected[column] = glm::dvec3(0.0);
        for (std::size_t row = 0; row < uv.size(); ++row) {
            projected[column] += columns[column][row] * (positions[row] - positions.front());
        }
        for (std::size_t next = column + 1; next < width; ++next) {
            // Reorthogonalization limits loss of orthogonality for near-dependent samples.
            for (unsigned pass = 0; pass < 2; ++pass) {
                const double projection = std::inner_product(columns[column].begin(), columns[column].end(),
                                                              columns[next].begin(), 0.0);
                upper[column][next] += projection;
                for (std::size_t row = 0; row < uv.size(); ++row) {
                    columns[next][row] -= projection * columns[column][row];
                }
            }
        }
    }
    PolynomialSurface result;
    std::array<glm::dvec3, 10> solved;
    for (std::size_t row = width; row-- > 0;) {
        auto value = projected[row];
        for (std::size_t column = row + 1; column < width; ++column) value -= upper[row][column] * solved[column];
        solved[row] = value / upper[row][row];
        if (!finite(solved[row])) return std::nullopt;
        result.coefficients[permutation[row]] = solved[row];
    }
    result.coefficients[0] += positions.front();
    result.degree = degree;
    for (std::size_t i = 0; i < uv.size(); ++i) {
        result.maximumVertexResidual = std::max(result.maximumVertexResidual,
                                                glm::length(polynomial(result, uv[i]).position - positions[i]));
    }
    if (!std::isfinite(result.maximumVertexResidual)) return std::nullopt;
    return result;
}

} // namespace

PolynomialSurface fitSurface(std::span<const glm::dvec2> parameters,
                             std::span<const glm::dvec3> positions) {
    if (parameters.size() != positions.size() || positions.size() < 3 || positions.size() > 1000000 ||
        !std::ranges::all_of(positions, finite) || !std::ranges::all_of(parameters, [](const auto& p) {
            return std::isfinite(p.x) && std::isfinite(p.y);
        })) throw std::invalid_argument("Expected 3 to 1000000 finite position/parameter pairs");
    auto low = positions.front(), high = low;
    for (const auto& point : positions) {
        low = glm::min(low, point);
        high = glm::max(high, point);
    }
    const auto span = high - low;
    const double extent = std::max({span.x, span.y, span.z});
    if (!(extent > 0) || !std::isfinite(extent)) throw std::invalid_argument("Surface fitting needs finite nonzero extent");
    std::optional<PolynomialSurface> best;
    for (unsigned degree = 1; degree <= 3; ++degree) {
        auto candidate = solve(parameters, positions, degree);
        if (!candidate) break;
        best = std::move(candidate);
        if (best->maximumVertexResidual <= extent * 1e-7) break;
    }
    if (!best) throw std::invalid_argument("Surface fitting parameters are rank deficient or ill conditioned");
    return *best;
}

BezierTriangle surfacePatch(const PolynomialSurface& surface, const std::array<glm::dvec2, 3>& domain) {
    if (!std::ranges::all_of(surface.coefficients, finite) || !std::ranges::all_of(domain, [](const auto& p) {
        return std::isfinite(p.x) && std::isfinite(p.y);
    })) throw std::invalid_argument("Surface patch requires finite coefficients and domain points");
    const auto u = domain[1] - domain[0], v = domain[2] - domain[0];
    if (u.x * v.y - u.y * v.x == 0) throw std::invalid_argument("Surface patch parameter triangle is singular");
    const auto a = polynomial(surface, domain[0]);
    const auto b = polynomial(surface, domain[1]);
    const auto c = polynomial(surface, domain[2]);
    const auto direction = [](const auto& d, const glm::dvec2& edge) { return d.du * edge.x + d.dv * edge.y; };
    BezierTriangle result;
    auto& p = result.controls;
    p[0] = a.position;
    p[1] = b.position;
    p[2] = c.position;
    p[3] = a.position + direction(a, u) / 3.0;
    p[4] = b.position - direction(b, u) / 3.0;
    p[5] = a.position + direction(a, v) / 3.0;
    p[6] = c.position - direction(c, v) / 3.0;
    p[7] = b.position + direction(b, domain[2] - domain[1]) / 3.0;
    p[8] = c.position - direction(c, domain[2] - domain[1]) / 3.0;
    const auto mixed = a.duu * u.x * v.x + a.duv * (u.x * v.y + u.y * v.x) + a.dvv * u.y * v.y;
    p[9] = mixed / 6.0 + p[3] + p[5] - p[0];
    if (!std::ranges::all_of(p, finite)) throw std::invalid_argument("Surface patch exceeded its numerical range");
    return result;
}

} // namespace lodestar
