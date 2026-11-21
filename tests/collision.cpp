#include <lodestar/lodestar.hpp>

#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

using namespace lodestar;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void appendQuad(Mesh& mesh, const std::array<glm::vec3, 4>& corners) {
    const auto first = static_cast<uint32_t>(mesh.vertices.size());
    for (const auto& corner : corners) {
        mesh.vertices.push_back({corner, glm::vec3(0.0f), glm::vec2(1.0f)});
    }
    mesh.indices.insert(mesh.indices.end(), {first, first + 1, first + 2, first, first + 2, first + 3});
}

// Segment/triangle intersections for the axis-aligned test fixture.
bool blocksSegment(const CollisionMesh& mesh, const glm::dvec3& start, const glm::dvec3& end) {
    const auto direction = end - start;
    for (std::size_t i = 0; i < mesh.indices.size(); i += 3) {
        std::array<glm::dvec3, 3> triangle;
        for (std::size_t corner = 0; corner < 3; ++corner) {
            const auto& point = mesh.positions.at(mesh.indices[i + corner]);
            triangle[corner] = {point[0], point[1], point[2]};
        }
        const auto edgeA = triangle[1] - triangle[0];
        const auto edgeB = triangle[2] - triangle[0];
        const auto perpendicular = glm::cross(direction, edgeB);
        const auto denominator = glm::dot(edgeA, perpendicular);
        if (denominator == 0.0) continue;
        const auto offset = start - triangle[0];
        const auto across = glm::cross(offset, edgeA);
        const auto u = glm::dot(offset, perpendicular) / denominator;
        const auto v = glm::dot(direction, across) / denominator;
        const auto t = glm::dot(edgeB, across) / denominator;
        if (u >= 0.0 && v >= 0.0 && u + v <= 1.0 && t >= 0.0 && t <= 1.0) return true;
    }
    return false;
}

void testDoorway() {
    Mesh source;
    source.hasUV = true;
    appendQuad(source, {{{-2, 0, 0}, {-1, 0, 0}, {-1, 0, 3}, {-2, 0, 3}}});
    appendQuad(source, {{{1, 0, 0}, {2, 0, 0}, {2, 0, 3}, {1, 0, 3}}});
    appendQuad(source, {{{-1, 0, 2}, {1, 0, 2}, {1, 0, 3}, {-1, 0, 3}}});
    appendQuad(source, {{{-2, -2, 0}, {2, -2, 0}, {2, 2, 0}, {-2, 2, 0}}});
    const auto collider = buildCollisionMesh(source);
    require(collider.indices == source.indices && collider.positions.size() == source.vertices.size(),
            "Collision preserves distinct source vertices, index winding and order");
    for (std::size_t i = 0; i < source.vertices.size(); ++i) {
        const auto& position = source.vertices[i].position;
        require(collider.positions[i] == std::array{position.x, position.y, position.z},
                "Collision follows source coordinates");
    }
    require(!blocksSegment(collider, {0, -1, 1}, {0, 1, 1}), "Doorway must remain open");
    require(blocksSegment(collider, {-1.5, -1, 1}, {-1.5, 1, 1}), "Wall must block passage");
    require(blocksSegment(collider, {0, -1, 2.5}, {0, 1, 2.5}), "Lintel must block passage");
    require(blocksSegment(collider, {0, 0.5, 1}, {0, 0.5, -1}), "Floor must support passage");
}

void testStaircase() {
    Mesh source;
    constexpr float run = 0.32f;
    constexpr float rise = 0.17f;
    for (int step = 0; step < 4; ++step) {
        const float start = step * run;
        const float end = (step + 1) * run;
        const float height = step * rise;
        appendQuad(source, {{{-0.5f, height, start}, {-0.5f, height, end},
                             {0.5f, height, end}, {0.5f, height, start}}});
        if (step > 0) {
            const float lower = (step - 1) * rise;
            appendQuad(source, {{{-0.5f, lower, start}, {-0.5f, height, start},
                                 {0.5f, height, start}, {0.5f, lower, start}}});
        }
    }

    const auto collider = buildCollisionMesh(source);
    for (int step = 0; step < 4; ++step) {
        const double height = step * rise;
        const double center = (step + 0.5) * run;
        require(blocksSegment(collider, {0.1, height + 0.01, center}, {0.1, height - 0.01, center}),
                "Each visible tread must support a contact at its authored height");
        if (step > 0) {
            const double previous = (step - 0.5) * run;
            require(blocksSegment(collider, {0.1, height - 0.06, previous}, {0.1, height - 0.06, center}),
                    "Each visible riser must block movement below its next tread");
            require(!blocksSegment(collider, {0.1, height + 0.02, previous}, {0.1, height + 0.02, center}),
                    "The path just above each tread must stay clear");
        }
    }
}

void requireInvalid(const Mesh& source) {
    try {
        (void)buildCollisionMesh(source);
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("Malformed collision input accepted");
}

void testDegeneratesAndValidation() {
    Mesh source;
    appendQuad(source, {{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}});
    source.vertices.push_back({{2.0f, 0.0f, 0.0f}, glm::vec3(0.0f), glm::vec2(0.0f)});
    source.vertices.push_back({{std::numeric_limits<float>::infinity(), 0.0f, 0.0f},
                               glm::vec3(0.0f), glm::vec2(0.0f)});
    source.indices.insert(source.indices.end(), {0, 1, 4, 0, 0, 1});
    const auto collider = buildCollisionMesh(source);
    require(collider.positions.size() == 4 && collider.indices.size() == 6 && collider.skippedDegenerateTriangles == 2,
            "Zero-area triangles and unused positions stay out of the collider");

    source.indices = {0, 1, 4};
    const auto empty = buildCollisionMesh(source);
    require(empty.positions.empty() && empty.indices.empty() && empty.skippedDegenerateTriangles == 1,
            "A position-only line yields no collision triangles");
    require(buildCollisionMesh({}).indices.empty(), "Empty source yields an empty collider");

    source.indices = {0, 1};
    requireInvalid(source);
    source.indices = {0, 1, 6};
    requireInvalid(source);
    source.indices = {0, 1, 5};
    requireInvalid(source);
}

} // namespace

int main() {
    try {
        testDoorway();
        testStaircase();
        testDegeneratesAndValidation();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
