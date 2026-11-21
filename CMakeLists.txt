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

add_library(lodestar_core STATIC
    src/boundary_budget.cpp
    src/geometry.cpp
    src/lodestar.cpp
    src/planar.cpp
    src/seam.cpp
    src/topology.cpp
)
# Preserve operation order in the seam edit's exact arithmetic checks.
if(MSVC OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    set_property(SOURCE src/seam.cpp APPEND PROPERTY COMPILE_OPTIONS /fp:strict)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    set_property(SOURCE src/seam.cpp APPEND PROPERTY COMPILE_OPTIONS -fno-fast-math -ffp-contract=off)
endif()
add_library(Lodestar::core ALIAS lodestar_core)
target_compile_features(lodestar_core PUBLIC cxx_std_20)
set_target_properties(lodestar_core PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(lodestar_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(lodestar_core PUBLIC glm::glm PRIVATE assimp::assimp meshoptimizer)

add_library(lodestar_c SHARED src/lodestar_c.cpp)
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

    add_executable(inverseglod src/main.cpp src/renderer.cpp ${shader_outputs})
    set_target_properties(inverseglod PROPERTIES CXX_EXTENSIONS OFF)
    target_include_directories(inverseglod PRIVATE "${shader_directory}")
    target_link_libraries(inverseglod PRIVATE Lodestar::core Vulkan::Vulkan glfw)
endif()

if(PROJECT_IS_TOP_LEVEL)
    add_library(lodestar_metrics STATIC src/mesh_metrics.cpp src/silhouette_metrics.cpp)
    set_target_properties(lodestar_metrics PROPERTIES CXX_EXTENSIONS OFF)
    target_include_directories(lodestar_metrics PUBLIC src)
    target_link_libraries(lodestar_metrics PUBLIC Lodestar::core)

    add_executable(lodestar_benchmark src/benchmark.cpp)
    set_target_properties(lodestar_benchmark PROPERTIES CXX_EXTENSIONS OFF)
    target_link_libraries(lodestar_benchmark PRIVATE lodestar_metrics)

    include(CTest)
    if(BUILD_TESTING)
        add_executable(lodestar_core_test tests/core.cpp)
        target_link_libraries(lodestar_core_test PRIVATE Lodestar::core)
        add_test(NAME lodestar_core COMMAND lodestar_core_test
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/planar-scene.dae")

        add_executable(lodestar_planar_test tests/planar.cpp)
        target_link_libraries(lodestar_planar_test PRIVATE Lodestar::core)
        add_test(NAME lodestar_planar COMMAND lodestar_planar_test)

        add_executable(lodestar_topology_test tests/topology.cpp)
        target_link_libraries(lodestar_topology_test PRIVATE Lodestar::core)
        add_test(NAME lodestar_topology COMMAND lodestar_topology_test)

        add_executable(lodestar_boundary_budget_test tests/boundary_budget.cpp)
        target_link_libraries(lodestar_boundary_budget_test PRIVATE Lodestar::core)
        add_test(NAME lodestar_boundary_budget COMMAND lodestar_boundary_budget_test)

        add_executable(lodestar_seam_test tests/seam.cpp)
        target_link_libraries(lodestar_seam_test PRIVATE Lodestar::core)
        add_test(NAME lodestar_seam COMMAND lodestar_seam_test)

        add_executable(lodestar_surface_metrics_test tests/mesh_metrics.cpp)
        target_link_libraries(lodestar_surface_metrics_test PRIVATE lodestar_metrics)
        add_test(NAME lodestar_surface_metrics COMMAND lodestar_surface_metrics_test)

        add_executable(lodestar_silhouette_metrics_test tests/silhouette_metrics.cpp)
        target_link_libraries(lodestar_silhouette_metrics_test PRIVATE lodestar_metrics)
        add_test(NAME lodestar_silhouette_metrics COMMAND lodestar_silhouette_metrics_test)

        add_test(NAME lodestar_benchmark_sample
            COMMAND lodestar_benchmark --quality 3 --repeats 1 --samples 128 --resolution 64)
        set_tests_properties(lodestar_benchmark_sample PROPERTIES
            PASS_REGULAR_EXPRESSION ",3,analytic_sphere,ok,eligible,0,[0-9]+,528,[0-9]+,528,matched,")
    endif()
endif()
