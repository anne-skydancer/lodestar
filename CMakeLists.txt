cmake_minimum_required(VERSION 3.24)
project(Lodestar LANGUAGES C CXX)

option(LODESTAR_BUILD_VIEWER "Build the Vulkan mesh viewer" ON)
if(LODESTAR_BUILD_VIEWER)
    find_package(Vulkan REQUIRED COMPONENTS glslc)
endif()

include(FetchContent)
# Link the dependencies into the C ABI library using position-independent objects.
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(ASSIMP_BUILD_TESTS OFF)
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF)
set(ASSIMP_INSTALL OFF)
set(ASSIMP_NO_EXPORT ON)
set(ASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT OFF)
set(ASSIMP_BUILD_COLLADA_IMPORTER ON)
set(ASSIMP_BUILD_GLTF_IMPORTER ON)

FetchContent_Declare(assimp GIT_REPOSITORY https://github.com/assimp/assimp.git GIT_TAG v5.4.3 GIT_SHALLOW TRUE)
FetchContent_Declare(glm GIT_REPOSITORY https://github.com/g-truc/glm.git GIT_TAG 1.0.1 GIT_SHALLOW TRUE)
FetchContent_Declare(meshoptimizer GIT_REPOSITORY https://github.com/zeux/meshoptimizer.git GIT_TAG v0.22 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(assimp glm meshoptimizer)

# The libraries are split by role. Each specialised library depends only on lodestar_common and
# none depends on another one. Public headers stay together in include/lodestar.
option(LODESTAR_COMMON_IMPORT "Include the Assimp mesh importer (loadMesh) in lodestar_common" ON)

function(lodestar_library target)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "ALIAS" "SOURCES")
    add_library(${target} STATIC ${ARG_SOURCES})
    add_library(${ARG_ALIAS} ALIAS ${target})
    target_compile_features(${target} PUBLIC cxx_std_20)
    set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
    target_include_directories(${target} PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>")
endfunction()

# lodestar_common: shared types, topology, the sampled metrics (test oracles) and the certified
# LOD error metric. Dependencies: glm and the standard library only (Assimp only for the optional
# importer). Strict floating point, so records agree across platforms.
lodestar_library(lodestar_common ALIAS Lodestar::common SOURCES
    src/common/mesh_metrics.cpp
    src/common/metric_geometry.cpp
    src/common/metric_rigged.cpp
    src/common/metric_shading.cpp
    src/common/metric_shape.cpp
    src/common/metric_texture.cpp
    src/common/record.cpp
    src/common/silhouette_metrics.cpp
    src/common/topology.cpp
)
if(LODESTAR_COMMON_IMPORT)
    target_sources(lodestar_common PRIVATE src/common/mesh_import.cpp)
    target_link_libraries(lodestar_common PRIVATE assimp::assimp)
endif()
target_link_libraries(lodestar_common PUBLIC glm::glm)
if(MSVC OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    target_compile_options(lodestar_common PRIVATE /fp:strict)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    target_compile_options(lodestar_common PRIVATE -fno-fast-math -ffp-contract=off)
endif()

# lodestar_simplifier: LOD generation and analysis (meshoptimizer, planar, seam, surface fit).
lodestar_library(lodestar_simplifier ALIAS Lodestar::simplifier SOURCES
    src/simplifier/boundary_budget.cpp
    src/simplifier/geometry.cpp
    src/simplifier/lodestar.cpp
    src/simplifier/planar.cpp
    src/simplifier/seam.cpp
    src/simplifier/surface_calculus.cpp
    src/simplifier/surface_fit.cpp
    src/simplifier/surface_tessellation.cpp
)
target_link_libraries(lodestar_simplifier PUBLIC lodestar_common PRIVATE meshoptimizer)
if(MSVC OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    set_property(SOURCE src/simplifier/seam.cpp APPEND PROPERTY COMPILE_OPTIONS /fp:strict)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    set_property(SOURCE src/simplifier/seam.cpp APPEND PROPERTY COMPILE_OPTIONS -fno-fast-math -ffp-contract=off)
endif()

# lodestar_physics: collision meshes and convex hulls.
lodestar_library(lodestar_physics ALIAS Lodestar::physics SOURCES
    src/physics/collision.cpp
    src/physics/convex_hull.cpp
    src/physics/geometry_reduction.cpp
    src/physics/orthogonal_hulls.cpp
)
target_link_libraries(lodestar_physics PUBLIC lodestar_common PRIVATE meshoptimizer)
if(MSVC OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    set_property(SOURCE src/physics/convex_hull.cpp src/physics/orthogonal_hulls.cpp APPEND PROPERTY COMPILE_OPTIONS /fp:strict)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    set_property(SOURCE src/physics/convex_hull.cpp src/physics/orthogonal_hulls.cpp APPEND PROPERTY COMPILE_OPTIONS -fno-fast-math -ffp-contract=off)
endif()

# Umbrella for the existing apps, tests and the C ABI: everything except the preview renderer.
add_library(lodestar_core INTERFACE)
add_library(Lodestar::core ALIAS lodestar_core)
target_link_libraries(lodestar_core INTERFACE lodestar_common lodestar_simplifier lodestar_physics)

add_library(lodestar_c SHARED src/c_api/lodestar_c.cpp)
set_target_properties(lodestar_c PROPERTIES CXX_EXTENSIONS OFF DEFINE_SYMBOL LODESTAR_C_EXPORTS)
target_include_directories(lodestar_c PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(lodestar_c PRIVATE Lodestar::core)

if(LODESTAR_BUILD_VIEWER)
    set(GLFW_BUILD_DOCS OFF)
    set(GLFW_BUILD_TESTS OFF)
    set(GLFW_BUILD_EXAMPLES OFF)
    set(GLFW_INSTALL OFF)
    FetchContent_Declare(glfw GIT_REPOSITORY https://github.com/glfw/glfw.git GIT_TAG 3.4 GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(glfw)

    set(shader_directory "${CMAKE_CURRENT_BINARY_DIR}/generated")
    file(GENERATE OUTPUT "${shader_directory}/lodestar_shaders.hpp" CONTENT [=[
#pragma once

#include <array>
#include <cstdint>

namespace lodestar::shaders {
inline constexpr auto vertex = std::to_array<std::uint32_t>(
#include "vert.spv.inc"
);
inline constexpr auto fragment = std::to_array<std::uint32_t>(
#include "frag.spv.inc"
);
}
]=])

    foreach(stage IN ITEMS vert frag)
        set(source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${stage}.glsl")
        set(output "${shader_directory}/${stage}.spv.inc")
        add_custom_command(
            OUTPUT "${output}"
            COMMAND Vulkan::glslc --target-env=vulkan1.0 -fshader-stage=${stage} -mfmt=c "${source}" -o "${output}"
            DEPENDS "${source}" Vulkan::glslc
            VERBATIM
        )
        list(APPEND shader_outputs "${output}")
    endforeach()

    # lodestar_render: the evaluation and preview renderer. It is never a live viewer renderer.
    lodestar_library(lodestar_render ALIAS Lodestar::render SOURCES src/render/renderer.cpp ${shader_outputs})
    target_include_directories(lodestar_render PUBLIC src/render PRIVATE "${shader_directory}")
    target_link_libraries(lodestar_render PUBLIC lodestar_common PRIVATE Vulkan::Vulkan glfw)

    add_executable(inverseglod src/app/collision_collada.cpp src/app/main.cpp)
    set_target_properties(inverseglod PROPERTIES CXX_EXTENSIONS OFF)
    target_include_directories(inverseglod PRIVATE src/app)
    target_link_libraries(inverseglod PRIVATE Lodestar::core lodestar_render Vulkan::Vulkan glfw)
endif()

if(PROJECT_IS_TOP_LEVEL)
    add_executable(lodestar_benchmark src/tools/benchmark.cpp)
    set_target_properties(lodestar_benchmark PROPERTIES CXX_EXTENSIONS OFF)
    target_link_libraries(lodestar_benchmark PRIVATE Lodestar::core)

    add_executable(lodestar_metric_bench src/tools/metric_bench.cpp)
    set_target_properties(lodestar_metric_bench PROPERTIES CXX_EXTENSIONS OFF)
    target_link_libraries(lodestar_metric_bench PRIVATE Lodestar::core)

    include(CTest)
    if(BUILD_TESTING)
        add_executable(lodestar_surface_calculus_test tests/surface_calculus.cpp)
        target_link_libraries(lodestar_surface_calculus_test PRIVATE lodestar_simplifier)
        add_test(NAME lodestar_surface_calculus COMMAND lodestar_surface_calculus_test)
        if(LODESTAR_BUILD_VIEWER)
            add_test(NAME lodestar_surface_calculus_cli COMMAND inverseglod --surface-lod
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/parabolic-cylinder.patch" --pixel-scale 0.04)
            set_tests_properties(lodestar_surface_calculus_cli PROPERTIES
                PASS_REGULAR_EXPRESSION "Generated surface: 15 vertices, 16 triangles")
        endif()

        if(WIN32)
            add_executable(lodestar_firestorm_decomp_test tests/firestorm_decomp.cpp)
            target_compile_features(lodestar_firestorm_decomp_test PRIVATE cxx_std_20)
            set_target_properties(lodestar_firestorm_decomp_test PROPERTIES CXX_EXTENSIONS OFF)
            target_include_directories(lodestar_firestorm_decomp_test PRIVATE include integrations/firestorm)
            add_dependencies(lodestar_firestorm_decomp_test lodestar_c)
            add_test(NAME lodestar_firestorm_decomp COMMAND lodestar_firestorm_decomp_test $<TARGET_FILE:lodestar_c>)
        endif()

        add_executable(lodestar_core_test tests/core.cpp)
        target_link_libraries(lodestar_core_test PRIVATE Lodestar::core)
        add_test(NAME lodestar_core COMMAND lodestar_core_test
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/planar-scene.dae")

        add_executable(lodestar_collision_test tests/collision.cpp)
        target_link_libraries(lodestar_collision_test PRIVATE lodestar_physics)
        add_test(NAME lodestar_collision COMMAND lodestar_collision_test)

        add_executable(lodestar_planar_test tests/planar.cpp)
        target_link_libraries(lodestar_planar_test PRIVATE lodestar_simplifier)
        add_test(NAME lodestar_planar COMMAND lodestar_planar_test)

        add_executable(lodestar_topology_test tests/topology.cpp)
        target_link_libraries(lodestar_topology_test PRIVATE lodestar_simplifier)
        add_test(NAME lodestar_topology COMMAND lodestar_topology_test)

        add_executable(lodestar_boundary_budget_test tests/boundary_budget.cpp)
        target_link_libraries(lodestar_boundary_budget_test PRIVATE lodestar_simplifier)
        add_test(NAME lodestar_boundary_budget COMMAND lodestar_boundary_budget_test)

        add_executable(lodestar_seam_test tests/seam.cpp)
        target_link_libraries(lodestar_seam_test PRIVATE lodestar_simplifier)
        add_test(NAME lodestar_seam COMMAND lodestar_seam_test)

        add_executable(lodestar_surface_metrics_test tests/mesh_metrics.cpp)
        target_link_libraries(lodestar_surface_metrics_test PRIVATE lodestar_common)
        add_test(NAME lodestar_surface_metrics COMMAND lodestar_surface_metrics_test)

        add_executable(lodestar_silhouette_metrics_test tests/silhouette_metrics.cpp)
        target_link_libraries(lodestar_silhouette_metrics_test PRIVATE lodestar_common)
        add_test(NAME lodestar_silhouette_metrics COMMAND lodestar_silhouette_metrics_test)

        add_executable(lodestar_metric_test tests/metric.cpp)
        target_link_libraries(lodestar_metric_test PRIVATE lodestar_common)
        add_test(NAME lodestar_metric COMMAND lodestar_metric_test)

        add_executable(lodestar_correspondence_oracle tests/correspondence_oracle.cpp)
        target_link_libraries(lodestar_correspondence_oracle PRIVATE lodestar_common)
        find_package(Python3 COMPONENTS Interpreter)
        if(Python3_Interpreter_FOUND)
            # The Python correspondence_bound is the oracle for the C++ port (needs numpy).
            add_test(NAME lodestar_correspondence_oracle
                COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/correspondence_oracle_test.py"
                $<TARGET_FILE:lodestar_correspondence_oracle>)
        endif()

        add_test(NAME lodestar_benchmark_sample
            COMMAND lodestar_benchmark --quality 3 --repeats 1 --samples 128 --resolution 64)
        set_tests_properties(lodestar_benchmark_sample PROPERTIES
            PASS_REGULAR_EXPRESSION ",3,analytic_sphere,ok,eligible,0,[0-9]+,528,[0-9]+,528,matched,")
    endif()
endif()
