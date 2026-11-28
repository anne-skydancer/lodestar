#include <lodestar/mesh_metrics.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace lodestar::metrics {
namespace {

constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr std::size_t leafSize = 8;

struct Bounds {
    glm::dvec3 minimum{infinity};
    glm::dvec3 maximum{-infinity};

    void include(const glm::dvec3& point) {
        minimum = glm::min(minimum, point);
        maximum = glm::max(maximum, point);
    }

    void include(const Bounds& bounds) {
        include(bounds.minimum);
        include(bounds.maximum);
    }

    glm::dvec3 size() const { return maximum - minimum; }

    double distanceSquared(const glm::dvec3& point) const {
        const auto delta = glm::max(glm::dvec3(0.0), glm::max(minimum - point, point - maximum));
        return glm::dot(delta, delta);
    }
};

double squaredLength(const glm::dvec3& vector) {
    return glm::dot(vector, vector);
}

double distanceSquaredToSegment(const glm::dvec3& point, const glm::dvec3& start,
                                const glm::dvec3& end) {
    const auto direction = end - start;
    const auto offset = point - start;
    const double fraction = glm::dot(offset, direction) / squaredLength(direction);
    return squaredLength(offset - direction * std::clamp(fraction, 0.0, 1.0));
}

struct Triangle {
    std::array<glm::dvec3, 3> positions;
    Bounds bounds;
    glm::dvec3 center;
    glm::dvec3 unitNormal;

    double distanceSquared(const glm::dvec3& point) const {
        const auto& [a, b, c] = positions;
        const auto offset = point - a;
        const auto ab = b - a;
        const auto ac = c - a;
        const double altitude = glm::dot(offset, unitNormal);
        const auto projection = offset - altitude * unitNormal;
        const bool inside = glm::dot(glm::cross(ab, projection), unitNormal) >= 0.0 &&
                            glm::dot(glm::cross(c - b, projection - ab), unitNormal) >= 0.0 &&
                            glm::dot(glm::cross(-ac, projection - ac), unitNormal) >= 0.0;
        if (inside) {
            return altitude * altitude;
        }
        return std::min({distanceSquaredToSegment(point, a, b),
                         distanceSquaredToSegment(point, b, c),
                         distanceSquaredToSegment(point, c, a)});
    }
};

struct Node {
    Bounds bounds;
    std::size_t first = 0;
    std::size_t count = 0;
    std::size_t left = 0;
    std::size_t right = 0;
};

glm::length_t longestAxis(const glm::dvec3& extent) {
    const glm::length_t horizontal = extent.x >= extent.y ? 0 : 1;
    return extent[horizontal] >= extent.z ? horizontal : 2;
}

class Surface {
public:
    explicit Surface(const Mesh& mesh) {
        if (mesh.indices.size() % 3 != 0) {
            throw std::invalid_argument("Surface measurement requires an indexed triangle list");
        }

        triangles_.reserve(mesh.indices.size() / 3);
        accumulatedAreas_.reserve(mesh.indices.size() / 3);
        double accumulatedArea = 0.0;
        for (std::size_t i = 0; i < mesh.indices.size(); i += 3) {
            Triangle triangle{};
            for (std::size_t j = 0; j < triangle.positions.size(); ++j) {
                const auto index = mesh.indices[i + j];
                if (index >= mesh.vertices.size()) {
                    throw std::invalid_argument("Surface triangle references a missing vertex");
                }
                const auto& p = mesh.vertices[index].position;
                if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
                    throw std::invalid_argument("Surface triangle references a non-finite position");
                }
                triangle.positions[j] = glm::dvec3(p);
                triangle.bounds.include(triangle.positions[j]);
            }

            const auto& [a, b, c] = triangle.positions;
            const auto normal = glm::cross(b - a, c - a);
            const double twiceArea = glm::length(normal);
            if (twiceArea == 0.0) {
                continue;
            }
            triangle.center = (a + b + c) / 3.0;
            triangle.unitNormal = normal / twiceArea;
            triangles_.push_back(triangle);
            accumulatedArea += 0.5 * twiceArea;
            accumulatedAreas_.push_back(accumulatedArea);
        }
        if (triangles_.empty()) {
            throw std::invalid_argument("Surface measurement requires at least one positive-area triangle");
        }

        order_.resize(triangles_.size());
        std::iota(order_.begin(), order_.end(), std::size_t{0});
        buildNode(0, order_.size());
    }

    double extent() const {
        const auto extent = nodes_.front().bounds.size();
        return extent[longestAxis(extent)];
    }

    glm::dvec3 sample(double areaFraction, double first, double second,
                      std::size_t& triangleIndex) const {
        const double area = areaFraction * accumulatedAreas_.back();
        while (triangleIndex + 1 < accumulatedAreas_.size() && area >= accumulatedAreas_[triangleIndex]) {
            ++triangleIndex;
        }
        const double radius = std::sqrt(first);
        const auto& [a, b, c] = triangles_[triangleIndex].positions;
        return a + (b - a) * (radius * (1.0 - second)) + (c - a) * (radius * second);
    }

    double distanceSquared(const glm::dvec3& point) const {
        double best = infinity;
        findNearest(0, point, best);
        return best;
    }

private:
    std::size_t buildNode(std::size_t first, std::size_t end) {
        const std::size_t index = nodes_.size();
        Node node;
        node.first = first;
        node.count = end - first;
        Bounds centers;
        for (std::size_t i = first; i < end; ++i) {
            const auto& triangle = triangles_[order_[i]];
            node.bounds.include(triangle.bounds);
            centers.include(triangle.center);
        }
        nodes_.push_back(node);
        if (node.count <= leafSize) {
            return index;
        }

        const auto axis = longestAxis(centers.size());
        const auto middle = first + node.count / 2;
        const auto start = order_.begin();
        std::nth_element(start + first, start + middle, start + end, [&](std::size_t a, std::size_t b) {
            const double left = triangles_[a].center[axis];
            const double right = triangles_[b].center[axis];
            return left != right ? left < right : a < b;
        });
        const auto left = buildNode(first, middle);
        const auto right = buildNode(middle, end);
        nodes_[index].left = left;
        nodes_[index].right = right;
        return index;
    }

    void findNearest(std::size_t index, const glm::dvec3& point, double& best) const {
        const auto& node = nodes_[index];
        if (node.count <= leafSize) {
            for (std::size_t i = node.first; i < node.first + node.count; ++i) {
                best = std::min(best, triangles_[order_[i]].distanceSquared(point));
            }
            return;
        }

        auto nearChild = node.left;
        auto farChild = node.right;
        double nearDistance = nodes_[nearChild].bounds.distanceSquared(point);
        double farDistance = nodes_[farChild].bounds.distanceSquared(point);
        if (nearDistance > farDistance) {
            std::swap(nearChild, farChild);
            std::swap(nearDistance, farDistance);
        }
        if (nearDistance < best) {
            findNearest(nearChild, point, best);
        }
        if (farDistance < best) {
            findNearest(farChild, point, best);
        }
    }

    std::vector<Triangle> triangles_;
    std::vector<double> accumulatedAreas_;
    std::vector<std::size_t> order_;
    std::vector<Node> nodes_;
};

double radicalInverse(std::size_t index, std::size_t base) {
    double scale = 1.0 / static_cast<double>(base);
    double result = 0.0;
    while (index != 0) {
        result += static_cast<double>(index % base) * scale;
        index /= base;
        scale /= static_cast<double>(base);
    }
    return result;
}

DirectionalDistance measureDirection(const Surface& from, const Surface& to,
                                     std::size_t samples, double sourceExtent) {
    double squaredSum = 0.0;
    double largestSquared = 0.0;
    std::size_t triangle = 0;
    const double sampleCount = static_cast<double>(samples);
    for (std::size_t i = 0; i < samples; ++i) {
        // Stratify the area axis and use low-discrepancy coordinates within each triangle.
        const double areaFraction = (static_cast<double>(i) + 0.5) / sampleCount;
        const auto position = from.sample(areaFraction, radicalInverse(i + 1, 2),
                                          radicalInverse(i + 1, 3), triangle);
        const double distance = to.distanceSquared(position);
        squaredSum += distance;
        largestSquared = std::max(largestSquared, distance);
    }
    return {std::sqrt(squaredSum / sampleCount) / sourceExtent, std::sqrt(largestSquared) / sourceExtent};
}

} // namespace

SurfaceDistance measureSurfaceDistance(const Mesh& source, const Mesh& candidate,
                                       std::size_t samplesPerDirection) {
    if (samplesPerDirection == 0) {
        throw std::invalid_argument("Surface measurement requires a positive sample count");
    }
    const Surface original(source);
    const Surface comparison(candidate);
    const double extent = original.extent();
    return {measureDirection(original, comparison, samplesPerDirection, extent),
            measureDirection(comparison, original, samplesPerDirection, extent),
            extent,
            samplesPerDirection};
}

} // namespace lodestar::metrics
