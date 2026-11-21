#include "collision_collada.hpp"

#include <lodestar/lodestar.hpp>

#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <stdexcept>

namespace lodestar {

void writeCollisionCollada(const CollisionMesh& mesh, const std::filesystem::path& path) {
    if (mesh.indices.empty()) {
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
    <geometry id="CollisionGeometry" name="Collision">
      <mesh>
        <source id="CollisionPositions">
          <float_array id="CollisionPositionArray" count=")" << mesh.positions.size() * 3 << R"(">
)";
    for (const auto& p : mesh.positions) {
        out << "            " << p[0] << ' ' << p[1] << ' ' << p[2] << '\n';
    }
    out << R"(          </float_array>
          <technique_common>
            <accessor source="#CollisionPositionArray" count=")" << mesh.positions.size() << R"(" stride="3">
              <param name="X" type="float"/>
              <param name="Y" type="float"/>
              <param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <vertices id="CollisionVertices">
          <input semantic="POSITION" source="#CollisionPositions"/>
        </vertices>
        <triangles count=")" << mesh.indices.size() / 3 << R"(">
          <input semantic="VERTEX" source="#CollisionVertices" offset="0"/>
          <p>
)";
    for (std::size_t i = 0; i < mesh.indices.size(); i += 3) {
        out << "            " << mesh.indices[i] << ' ' << mesh.indices[i + 1] << ' '
            << mesh.indices[i + 2] << '\n';
    }
    out << R"(          </p>
        </triangles>
      </mesh>
    </geometry>
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="CollisionScene">
      <node id="CollisionNode" name="Collision">
        <instance_geometry url="#CollisionGeometry"/>
      </node>
    </visual_scene>
  </library_visual_scenes>
  <scene><instance_visual_scene url="#CollisionScene"/></scene>
</COLLADA>
)";
    out.close();
    if (!out) {
        throw std::runtime_error("Could not write the physics COLLADA output");
    }
}

} // namespace lodestar
