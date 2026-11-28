# Builds lodestar_common and packs it as an autobuild-style archive (3p-lodestar-common).
#
#   cmake -DPLATFORM=windows64|linux64 [-DOUTPUT_DIR=<dir>] [-DGLM_SOURCE=<glm checkout>]
#         [-DGENERATOR=<cmake generator>] -P packaging/package-common.cmake
#
# The archive holds include/lodestar, lib/release (and lib/debug on Windows), LICENSES/ and
# LODESTAR_COMMON_VERSION.txt. glm is not bundled: the viewer already has its own.
# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.24)

get_filename_component(SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
if(NOT PLATFORM)
    if(CMAKE_HOST_WIN32)
        set(PLATFORM windows64)
    else()
        set(PLATFORM linux64)
    endif()
endif()
if(NOT PLATFORM MATCHES "^(windows64|linux64)$")
    message(FATAL_ERROR "PLATFORM must be windows64 or linux64")
endif()
if(NOT OUTPUT_DIR)
    set(OUTPUT_DIR "${SOURCE_DIR}/build-package")
endif()
get_filename_component(OUTPUT_DIR "${OUTPUT_DIR}" ABSOLUTE)

file(READ "${SOURCE_DIR}/CMakeLists.txt" top)
string(REGEX MATCH "project\\(Lodestar VERSION ([0-9.]+)" _ "${top}")
set(VERSION "${CMAKE_MATCH_1}")
file(READ "${SOURCE_DIR}/include/lodestar/record.hpp" record)
string(REGEX MATCH "recordFormatVersion = ([0-9]+)" _ "${record}")
set(FORMAT "${CMAKE_MATCH_1}")
string(REGEX MATCH "metricVersion = ([0-9]+)" _ "${record}")
set(METRIC "${CMAKE_MATCH_1}")
if(NOT VERSION OR NOT FORMAT OR NOT METRIC)
    message(FATAL_ERROR "Could not read the version from CMakeLists.txt and record.hpp")
endif()

if(PLATFORM STREQUAL windows64)
    set(CONFIGS Release Debug)
else()
    set(CONFIGS Release)
endif()

set(STAGE "${OUTPUT_DIR}/stage")
file(REMOVE_RECURSE "${STAGE}")
foreach(config IN LISTS CONFIGS)
    set(build "${OUTPUT_DIR}/build-${config}")
    set(install "${OUTPUT_DIR}/install-${config}")
    file(REMOVE_RECURSE "${build}" "${install}")
    set(args -S "${SOURCE_DIR}" -B "${build}" -DLODESTAR_COMMON_ONLY=ON -DBUILD_TESTING=OFF
             -DCMAKE_BUILD_TYPE=${config})
    if(GENERATOR)
        list(APPEND args -G "${GENERATOR}")
    endif()
    if(PLATFORM STREQUAL windows64 AND NOT GENERATOR)
        list(APPEND args -A x64)
    endif()
    if(GLM_SOURCE)
        list(APPEND args -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_SOURCE_DIR_GLM=${GLM_SOURCE})
    endif()
    execute_process(COMMAND "${CMAKE_COMMAND}" ${args} RESULT_VARIABLE status)
    if(status)
        message(FATAL_ERROR "Configure failed (${config})")
    endif()
    execute_process(COMMAND "${CMAKE_COMMAND}" --build "${build}" --config ${config} --target lodestar_common
                    RESULT_VARIABLE status)
    if(status)
        message(FATAL_ERROR "Build failed (${config})")
    endif()
    execute_process(COMMAND "${CMAKE_COMMAND}" --install "${build}" --config ${config} --component common
                            --prefix "${install}" RESULT_VARIABLE status)
    if(status)
        message(FATAL_ERROR "Install failed (${config})")
    endif()
    string(TOLOWER "${config}" folder)
    file(GLOB libraries "${install}/lib/*")
    file(COPY ${libraries} DESTINATION "${STAGE}/lib/${folder}")
endforeach()

file(COPY "${OUTPUT_DIR}/install-Release/include" DESTINATION "${STAGE}")
file(COPY "${OUTPUT_DIR}/install-Release/LICENSES" DESTINATION "${STAGE}")
file(WRITE "${STAGE}/LODESTAR_COMMON_VERSION.txt" "${VERSION}.r${FORMAT}m${METRIC}\n")

set(ARCHIVE "${OUTPUT_DIR}/lodestar-common-${VERSION}-${PLATFORM}.tar.bz2")
file(REMOVE "${ARCHIVE}")
execute_process(COMMAND "${CMAKE_COMMAND}" -E tar cjf "${ARCHIVE}" -- include lib LICENSES LODESTAR_COMMON_VERSION.txt
                WORKING_DIRECTORY "${STAGE}" RESULT_VARIABLE status)
if(status)
    message(FATAL_ERROR "Could not create ${ARCHIVE}")
endif()
file(MD5 "${ARCHIVE}" md5)
message(STATUS "Package: ${ARCHIVE}")
message(STATUS "MD5:     ${md5}")
