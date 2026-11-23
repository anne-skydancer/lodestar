#include "collision_collada.hpp"

#include <lodestar/lodestar.hpp>

#include <algorithm>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>

namespace lodestar {
namespace {

void writeGeometry(std::ostream& out, const CollisionMesh& mesh, const std::string& name) {
    out << "    <geometry id=\"" << name << "Geometry\" name=\"" << name << "\">\n"
        << "      <mesh>\n"
        << "        <source id=\"" << name << "Positions\">\n"
        << "          <float_array id=\"" << name << "PositionArray\" count=\"" << mesh.positions.size() * 3 << R"(">
)";
    for (const auto& p : mesh.positions) {
        out << "            " << p[0] << ' ' << p[1] << ' ' << p[2] << '\n';
    }
    out << "          </float_array>\n"
        << "          <technique_common>\n"
        << "            <accessor source=\"#" << name << "PositionArray\" count=\"" << mesh.positions.size() << R"(" stride="3">
              <param name="X" type="float"/>
              <param name="Y" type="float"/>
              <param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
)"
        << "        <vertices id=\"" << name << "Vertices\">\n"
        << "          <input semantic=\"POSITION\" source=\"#" << name << "Positions\"/>\n"
        << "        </vertices>\n"
        << "        <triangles count=\"" << mesh.indices.size() / 3 << "\">\n"
        << "          <input semantic=\"VERTEX\" source=\"#" << name << "Vertices\" offset=\"0\"/>\n"
        << "          <p>\n";
    for (std::size_t i = 0; i < mesh.indices.size(); i += 3) {
        out << "            " << mesh.indices[i] << ' ' << mesh.indices[i + 1] << ' '
            << mesh.indices[i + 2] << '\n';
    }
    out << R"(          </p>
        </triangles>
      </mesh>
    </geometry>
)";
}

void writeScene(std::span<const CollisionMesh> meshes, const std::filesystem::path& path, bool compound) {
    if (meshes.empty() || std::ranges::any_of(meshes, [](const auto& mesh) { return mesh.indices.empty(); })) {
        throw std::invalid_argument("Collision mesh has no triangles to export");
    }

    const std::time_t now = std::time(nullptr);
    const std::tm* utc = std::gmtime(&now);
    if (utc == nullptr) {
        throw std::runtime_error("Could not determine the physics COLLADA export time");
    }
    const std::tm timestamp = *utc;

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Could not open the physics COLLADA output");
    }
    out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<float>::max_digits10);

    out << R"(<?xml version="1.0" encoding="UTF-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset>
)";
    out << "    <created>" << std::put_time(&timestamp, "%Y-%m-%dT%H:%M:%SZ") << "</created>\n"
        << "    <modified>" << std::put_time(&timestamp, "%Y-%m-%dT%H:%M:%SZ") << "</modified>\n";
    out << R"(    <unit name="meter" meter="1"/>
    <up_axis>Y_UP</up_axis>
  </asset>
  <library_geometries>
)";
    const auto name = [=](std::size_t index) {
        return compound ? "Hull" + std::to_string(index + 1) : std::string("Collision");
    };
    for (std::size_t i = 0; i < meshes.size(); ++i) writeGeometry(out, meshes[i], name(i));
    out << R"(  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="CollisionScene">
)";
    for (std::size_t i = 0; i < meshes.size(); ++i) {
        const auto model = name(i);
        out << "      <node id=\"" << model << "Node\" name=\"" << model << "\">\n"
            << "        <instance_geometry url=\"#" << model << "Geometry\"/>\n"
            << "      </node>\n";
    }
    out << R"(    </visual_scene>
  </library_visual_scenes>
  <scene><instance_visual_scene url="#CollisionScene"/></scene>
</COLLADA>
)";
    out.close();
    if (!out) {
        throw std::runtime_error("Could not write the physics COLLADA output");
    }
}

} // namespace

void writeCollisionCollada(const CollisionMesh& mesh, const std::filesystem::path& path) {
    writeScene(std::span(&mesh, 1), path, false);
}

void writeHullCollada(std::span<const CollisionMesh> hulls, const std::filesystem::path& path) {
    writeScene(hulls, path, true);
}

} // namespace lodestar
