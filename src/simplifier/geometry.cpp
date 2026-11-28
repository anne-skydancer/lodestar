#include <lodestar/lodestar.hpp>

#include <glm/gtc/constants.hpp>
#include <meshoptimizer.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace lodestar {
namespace {

constexpr auto unmappedIndex = std::numeric_limits<uint32_t>::max();
constexpr float vertexErrorLimit = 0.01f;
constexpr double minimumFaceRadius = 0.9;
constexpr double seamQuantization = 1e6;
constexpr float simplificationErrorLimit = 0.02f;

bool finite(const glm::dvec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

bool validSphere(const SphereFit& fit) {
    return fit.valid && finite(fit.center) && std::isfinite(fit.radius) && fit.radius > 0.0f;
}

struct Bounds {
    glm::dvec3 minimum{std::numeric_limits<double>::infinity()};
    glm::dvec3 maximum{-std::numeric_limits<double>::infinity()};

    void include(const glm::dvec3& position) {
        minimum = glm::min(minimum, position);
        maximum = glm::max(maximum, position);
    }

    glm::dvec3 center() const {
        return minimum + (maximum - minimum) * 0.5;
    }

    double halfExtent() const {
        const auto extent = maximum - minimum;
        return std::max({extent.x, extent.y, extent.z}) * 0.5;
    }
};

double distanceSquaredToSegment(const glm::dvec3& a, const glm::dvec3& b) {
    const auto edge = b - a;
    const double lengthSquared = glm::dot(edge, edge);
    if (lengthSquared == 0.0) {
        return glm::dot(a, a);
    }
    const double t = std::clamp(-glm::dot(a, edge) / lengthSquared, 0.0, 1.0);
    const auto closest = a + edge * t;
    return glm::dot(closest, closest);
}

double distanceSquaredToTriangle(const glm::dvec3& a, const glm::dvec3& b, const glm::dvec3& c) {
    const auto normal = glm::cross(b - a, c - a);
    const double normalLengthSquared = glm::dot(normal, normal);
    if (normalLengthSquared > 0.0) {
        const auto projected = normal * (glm::dot(normal, a) / normalLengthSquared);
        const bool inside = glm::dot(glm::cross(b - a, projected - a), normal) >= 0.0 &&
                            glm::dot(glm::cross(c - b, projected - b), normal) >= 0.0 &&
                            glm::dot(glm::cross(a - c, projected - c), normal) >= 0.0;
        if (inside) {
            return glm::dot(projected, projected);
        }
    }
    return std::min({distanceSquaredToSegment(a, b), distanceSquaredToSegment(b, c),
                     distanceSquaredToSegment(c, a)});
}

struct SphereSurface {
    std::optional<float> maxFaceInwardRelativeError;
    bool complete = false;
};

SphereSurface assessSphereSurface(const Mesh& mesh, const SphereFit& fit, bool checkCompleteness) {
    if (mesh.vertices.empty() || mesh.indices.empty() || mesh.indices.size() % 3 != 0) {
        return {};
    }

    std::vector<glm::dvec3> positions;
    positions.reserve(mesh.vertices.size());
    for (const auto& vertex : mesh.vertices) {
        const auto p = (glm::dvec3(vertex.position) - glm::dvec3(fit.center)) / double(fit.radius);
        if (!finite(p) || std::abs(glm::length(p) - 1.0) > vertexErrorLimit) {
            return {};
        }
        positions.push_back(p);
    }

    std::vector<unsigned> welded;
    std::unordered_set<uint64_t> edges;
    if (checkCompleteness) {
        std::vector<std::array<int32_t, 3>> seamPositions;
        seamPositions.reserve(positions.size());
        for (const auto& p : positions) {
            seamPositions.push_back({static_cast<int32_t>(std::round(p.x * seamQuantization)),
                                     static_cast<int32_t>(std::round(p.y * seamQuantization)),
                                     static_cast<int32_t>(std::round(p.z * seamQuantization))});
        }
        welded.resize(positions.size());
        meshopt_generateVertexRemap(welded.data(), nullptr, seamPositions.size(),
                                    seamPositions.data(), seamPositions.size(), sizeof(seamPositions[0]));
        edges.reserve(mesh.indices.size());
    }

    constexpr double minimumDistanceSquared = minimumFaceRadius * minimumFaceRadius;
    double closestDistanceSquared = 1.0;
    double solidAngle = 0.0;
    for (size_t i = 0; i < mesh.indices.size(); i += 3) {
        const std::array<uint32_t, 3> corners{mesh.indices[i], mesh.indices[i + 1], mesh.indices[i + 2]};
        if (std::ranges::any_of(corners, [&](uint32_t index) { return index >= positions.size(); })) {
            return {};
        }
        const auto& a = positions[corners[0]];
        const auto& b = positions[corners[1]];
        const auto& c = positions[corners[2]];
        closestDistanceSquared = std::min(closestDistanceSquared, distanceSquaredToTriangle(a, b, c));
        if (closestDistanceSquared < minimumDistanceSquared) {
            checkCompleteness = false;
        }
        if (!checkCompleteness) {
            continue;
        }
        const double volume = glm::dot(a, glm::cross(b, c));
        if (volume <= 0.0) {
            checkCompleteness = false;
            continue;
        }
        // Sum the outward-facing solid angles to check that the mesh wraps the center once.
        const double la = glm::length(a), lb = glm::length(b), lc = glm::length(c);
        solidAngle += 2.0 * std::atan2(volume, la * lb * lc + glm::dot(a, b) * lc +
                                              glm::dot(b, c) * la + glm::dot(c, a) * lb);

        // Each welded directed edge must appear once and have a reverse counterpart.
        for (size_t corner = 0; corner < corners.size(); ++corner) {
            const uint32_t from = welded[corners[corner]];
            const uint32_t to = welded[corners[(corner + 1) % corners.size()]];
            if (from == to || !edges.insert((uint64_t(from) << 32) | to).second) {
                checkCompleteness = false;
                break;
            }
        }
    }

    const bool complete = checkCompleteness && std::abs(solidAngle - 4.0 * glm::pi<double>()) < 1e-4 &&
        std::ranges::all_of(edges, [&](uint64_t edge) {
            const uint64_t reverse = (edge << 32) | (edge >> 32);
            return edges.contains(reverse);
        });
    return {static_cast<float>(1.0 - std::sqrt(closestDistanceSquared)), complete};
}

SphereEligibility radialEligibility(const SphereFit& fit) {
    if (!validSphere(fit) || !std::isfinite(fit.maxRelativeError)) {
        return SphereEligibility::NoFit;
    }
    return fit.maxRelativeError >= vertexErrorLimit ? SphereEligibility::RadialError
                                                    : SphereEligibility::Eligible;
}

} // namespace

SphereFit fitSphere(const Mesh& mesh) {
    if (mesh.vertices.size() < 4) {
        return {};
    }
    Bounds bounds;
    for (const auto& vertex : mesh.vertices) {
        if (!finite(vertex.position)) {
            return {};
        }
        bounds.include(vertex.position);
    }
    const auto origin = bounds.center();
    const double scale = bounds.halfExtent();
    if (scale <= 0.0) {
        return {};
    }

    // Solve the normal equations of x² + y² + z² = Ax + By + Cz + D.
    // Center and scale first to keep the solve well-conditioned for imported coordinates.
    double system[4][5]{};
    for (const auto& vertex : mesh.vertices) {
        const auto p = (glm::dvec3(vertex.position) - origin) / scale;
        const double row[]{p.x, p.y, p.z, 1.0, glm::dot(p, p)};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 5; ++j) {
                system[i][j] += row[i] * row[j];
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        int pivot = i;
        for (int j = i + 1; j < 4; ++j) {
            if (std::abs(system[j][i]) > std::abs(system[pivot][i])) {
                pivot = j;
            }
        }
        const double divisor = system[pivot][i];
        if (std::abs(divisor) <= 1e-12 * static_cast<double>(mesh.vertices.size())) {
            return {};
        }
        std::swap(system[i], system[pivot]);
        for (int j = i; j < 5; ++j) {
            system[i][j] /= divisor;
        }
        for (int row = 0; row < 4; ++row) {
            if (row != i) {
                const double factor = system[row][i];
                for (int j = i; j < 5; ++j) {
                    system[row][j] -= factor * system[i][j];
                }
            }
        }
    }

    const auto scaledCenter = glm::dvec3(system[0][4], system[1][4], system[2][4]) * 0.5;
    const double radiusSquared = system[3][4] + glm::dot(scaledCenter, scaledCenter);
    if (radiusSquared <= 0.0) {
        return {};
    }
    const auto center = origin + scaledCenter * scale;
    const double radius = std::sqrt(radiusSquared) * scale;
    const double floatMax = std::numeric_limits<float>::max();
    if (!finite(center) || glm::any(glm::greaterThan(glm::abs(center), glm::dvec3(floatMax))) ||
        !std::isfinite(radius) || radius > floatMax || static_cast<float>(radius) <= 0.0f) {
        return {};
    }

    SphereFit result;
    result.center = glm::vec3(center);
    result.radius = static_cast<float>(radius);
    double squaredError = 0.0;
    double maxError = 0.0;
    for (const auto& vertex : mesh.vertices) {
        const auto delta = glm::dvec3(vertex.position) - glm::dvec3(result.center);
        const double error = std::abs(glm::length(delta) / double(result.radius) - 1.0);
        squaredError += error * error;
        maxError = std::max(maxError, error);
    }
    if (maxError > floatMax) {
        return {};
    }
    result.maxRelativeError = static_cast<float>(maxError);
    result.rmsRelativeError = static_cast<float>(std::sqrt(squaredError / mesh.vertices.size()));
    result.valid = true;
    return result;
}

SphereEligibility sphereEligibility(const Mesh& mesh, const SphereFit& fit) {
    if (mesh.hasUV) {
        return SphereEligibility::HasTextureCoordinates;
    }
    const auto radial = radialEligibility(fit);
    if (radial != SphereEligibility::Eligible) {
        return radial;
    }
    return assessSphereSurface(mesh, fit, true).complete ? SphereEligibility::Eligible
                                                         : SphereEligibility::IncompleteSurface;
}

Asset analyze(Mesh mesh) {
    const auto fit = fitSphere(mesh);
    const auto radial = radialEligibility(fit);
    SphereSurface surface;
    if (radial == SphereEligibility::Eligible) {
        surface = assessSphereSurface(mesh, fit, !mesh.hasUV);
    }
    auto eligibility = radial;
    if (mesh.hasUV) {
        eligibility = SphereEligibility::HasTextureCoordinates;
    } else if (radial == SphereEligibility::Eligible && !surface.complete) {
        eligibility = SphereEligibility::IncompleteSurface;
    }
    return {std::move(mesh), fit, eligibility, surface.maxFaceInwardRelativeError};
}

Mesh sphereMesh(const SphereFit& fit, int segments, int rings) {
    if (!validSphere(fit)) {
        throw std::invalid_argument("Sphere requires a valid fit with positive radius");
    }
    const auto furthestCoordinates = glm::abs(glm::dvec3(fit.center)) + double(fit.radius);
    if (glm::any(glm::greaterThan(furthestCoordinates, glm::dvec3(std::numeric_limits<float>::max())))) {
        throw std::invalid_argument("Sphere positions exceed the supported range");
    }

    const auto segmentCount = static_cast<uint32_t>(std::max(8, segments));
    const auto ringCount = static_cast<uint32_t>(std::max(4, rings));
    const uint64_t ringVertices = uint64_t(segmentCount) * (ringCount - 1);
    if (ringVertices > unmappedIndex / 6) {
        throw std::invalid_argument("Sphere resolution exceeds the supported index count");
    }
    Mesh result;
    result.vertices.reserve(static_cast<size_t>(ringVertices + 2));
    result.indices.reserve(static_cast<size_t>(ringVertices * 6));

    const auto vertex = [&](const glm::vec3& normal) {
        return Vertex{fit.center + fit.radius * normal, normal, glm::vec2(0.0f)};
    };
    result.vertices.push_back(vertex({0.0f, 1.0f, 0.0f}));
    for (uint32_t ring = 1; ring < ringCount; ++ring) {
        const float theta = glm::pi<float>() * (float(ring) / float(ringCount));
        const float sinTheta = std::sin(theta);
        const float cosTheta = std::cos(theta);
        for (uint32_t segment = 0; segment < segmentCount; ++segment) {
            const float phi = glm::two_pi<float>() * (float(segment) / float(segmentCount));
            result.vertices.push_back(vertex({sinTheta * std::cos(phi), cosTheta, sinTheta * std::sin(phi)}));
        }
    }
    const auto south = static_cast<uint32_t>(result.vertices.size());
    result.vertices.push_back(vertex({0.0f, -1.0f, 0.0f}));

    for (uint32_t segment = 0; segment < segmentCount; ++segment) {
        const uint32_t next = (segment + 1) % segmentCount;
        result.indices.insert(result.indices.end(), {0, next + 1, segment + 1});
        for (uint32_t ring = 1; ring < ringCount - 1; ++ring) {
            const uint32_t a = (ring - 1) * segmentCount + segment + 1;
            const uint32_t b = (ring - 1) * segmentCount + next + 1;
            result.indices.insert(result.indices.end(), {a, b, a + segmentCount,
                                                         b, b + segmentCount, a + segmentCount});
        }
        const uint32_t lastRing = (ringCount - 2) * segmentCount + 1;
        result.indices.insert(result.indices.end(), {lastRing + segment, lastRing + next, south});
    }
    return result;
}

Mesh simplifyMesh(const Mesh& mesh, float fraction) {
    if (mesh.vertices.empty() || mesh.indices.size() < 6 || !std::isfinite(fraction) || fraction >= 1.0f) {
        return mesh;
    }
    if (mesh.indices.size() % 3 != 0 || std::ranges::any_of(mesh.indices, [&](uint32_t index) {
            return index >= mesh.vertices.size();
        })) {
        throw std::invalid_argument("Simplification requires a valid indexed triangle mesh");
    }
    const size_t triangles = mesh.indices.size() / 3;
    const double requestedTriangles = double(triangles) * double(std::max(fraction, 0.0f));
    const size_t target = std::max<size_t>(1, static_cast<size_t>(requestedTriangles)) * 3;
    Mesh result;
    result.hasUV = mesh.hasUV;
    result.hasAdditionalUV = mesh.hasAdditionalUV;
    result.hasUnrepresentedVertexAttributes = mesh.hasUnrepresentedVertexAttributes;
    result.indices.resize(mesh.indices.size());
    const size_t count = meshopt_simplify(
        result.indices.data(), mesh.indices.data(), mesh.indices.size(), &mesh.vertices.front().position.x,
        mesh.vertices.size(), sizeof(Vertex), target, simplificationErrorLimit, 0, nullptr);
    if (count < 3 || count >= mesh.indices.size()) {
        return mesh;
    }
    result.indices.resize(count);
    result.vertices.resize(mesh.vertices.size());
    const size_t vertexCount = meshopt_optimizeVertexFetch(
        result.vertices.data(), result.indices.data(), result.indices.size(),
        mesh.vertices.data(), mesh.vertices.size(), sizeof(Vertex));
    result.vertices.resize(vertexCount);
    return result;
}

} // namespace lodestar
