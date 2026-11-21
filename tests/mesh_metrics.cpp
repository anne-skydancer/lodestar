#include "mesh_metrics.hpp"

#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using lodestar::Mesh;
using lodestar::metrics::DirectionalDistance;
using lodestar::metrics::measureSurfaceDistance;

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

void requireDistance(const DirectionalDistance& distance, double rms, double maximum,
                     std::string_view message) {
    require(std::abs(distance.rms - rms) < 1e-10 && std::abs(distance.maximum - maximum) < 1e-10,
            message);
}

template <typename Function>
void requireInvalid(Function function, std::string_view message) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(std::string(message));
}

void appendSquare(Mesh& mesh, float x, float y, float z, float width) {
    const auto first = static_cast<uint32_t>(mesh.vertices.size());
    for (const auto& p : std::array<glm::vec3, 4>{{
             {x, y, z}, {x + width, y, z}, {x, y + width, z}, {x + width, y + width, z}}}) {
        mesh.vertices.push_back({p, glm::vec3(0.0f, 0.0f, 1.0f), glm::vec2(0.0f)});
    }
    mesh.indices.insert(mesh.indices.end(), {first, first + 1, first + 2,
                                             first + 1, first + 3, first + 2});
}

void testGridDistance() {
    Mesh flat;
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            appendSquare(flat, static_cast<float>(x), static_cast<float>(y), 0.0f, 1.0f);
        }
    }
    auto translated = flat;
    for (auto& vertex : translated.vertices) {
        vertex.position.z = 2.0f;
    }

    const auto same = measureSurfaceDistance(flat, flat, 1024);
    requireDistance(same.sourceToCandidate, 0.0, 0.0, "Identical meshes have zero distance");
    requireDistance(same.candidateToSource, 0.0, 0.0, "Identical mesh reverse distance");

    const auto offset = measureSurfaceDistance(flat, translated, 1024);
    require(offset.sourceExtent == 8.0 && offset.samplesPerDirection == 1024, "Measurement metadata");
    requireDistance(offset.sourceToCandidate, 0.25, 0.25, "Translated grid forward distance");
    requireDistance(offset.candidateToSource, 0.25, 0.25, "Translated grid reverse distance");
}

void testDirectionalDistance() {
    Mesh smaller, larger;
    appendSquare(smaller, 0.0f, 0.0f, 0.0f, 1.0f);
    appendSquare(larger, 0.0f, 0.0f, 0.0f, 2.0f);
    const auto distance = measureSurfaceDistance(smaller, larger, 4096);
    requireDistance(distance.sourceToCandidate, 0.0, 0.0, "Contained surface has zero forward distance");
    require(distance.candidateToSource.rms > 0.5 && distance.candidateToSource.maximum > 1.3,
            "Containment misses target regions only in reverse");

    Mesh distant;
    appendSquare(distant, 2.0f, 0.0f, 0.0f, 1.0f);
    const auto separated = measureSurfaceDistance(smaller, distant, 8192);
    const double expectedRms = std::sqrt(7.0 / 3.0);
    require(std::abs(separated.sourceToCandidate.rms - expectedRms) < 0.002 &&
                separated.sourceToCandidate.maximum > 1.99 && separated.sourceToCandidate.maximum <= 2.0,
            "Coplanar separated squares agree with distance to the facing edges");
    const auto repeated = measureSurfaceDistance(smaller, distant, 8192);
    require(repeated.sourceToCandidate.rms == separated.sourceToCandidate.rms &&
                repeated.candidateToSource.maximum == separated.candidateToSource.maximum,
            "Surface measurements are deterministic for a fixed sample count");
}

void testAreaWeighting() {
    Mesh source, candidate;
    appendSquare(source, 0.0f, 0.0f, 0.0f, 1.0f);
    appendSquare(source, 10.0f, 0.0f, 0.0f, 3.0f);
    appendSquare(candidate, 0.0f, 0.0f, 2.0f, 1.0f);
    appendSquare(candidate, 10.0f, 0.0f, 0.0f, 3.0f);

    const auto outside = static_cast<uint32_t>(source.vertices.size());
    source.vertices.push_back({glm::vec3(1000.0f), glm::vec3(0.0f), glm::vec2(0.0f)});
    source.indices.insert(source.indices.end(), {outside, outside, outside});
    const auto distance = measureSurfaceDistance(source, candidate, 1000);
    require(distance.sourceExtent == 13.0, "Normalization ignores degenerate faces");
    const double rms = 2.0 * std::sqrt(0.1) / distance.sourceExtent;
    const double maximum = 2.0 / distance.sourceExtent;
    requireDistance(distance.sourceToCandidate, rms, maximum, "Forward RMS is area-weighted");
    requireDistance(distance.candidateToSource, rms, maximum, "Reverse RMS is area-weighted");
}

void testInvalidInput() {
    Mesh mesh;
    appendSquare(mesh, 0.0f, 0.0f, 0.0f, 1.0f);
    requireInvalid([&] { measureSurfaceDistance(mesh, mesh, 0); }, "Zero samples must fail");
    requireInvalid([&] { measureSurfaceDistance(mesh, {}, 256); }, "Empty target must fail");

    auto invalid = mesh;
    invalid.indices.pop_back();
    requireInvalid([&] { measureSurfaceDistance(invalid, mesh, 256); }, "Incomplete face must fail");
    invalid = mesh;
    invalid.indices.front() = 100;
    requireInvalid([&] { measureSurfaceDistance(invalid, mesh, 256); }, "Missing vertex must fail");
    invalid = mesh;
    invalid.vertices.front().position.x = std::numeric_limits<float>::infinity();
    requireInvalid([&] { measureSurfaceDistance(mesh, invalid, 256); }, "Non-finite surface must fail");
}

} // namespace

int main() {
    try {
        testGridDistance();
        testDirectionalDistance();
        testAreaWeighting();
        testInvalidInput();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
