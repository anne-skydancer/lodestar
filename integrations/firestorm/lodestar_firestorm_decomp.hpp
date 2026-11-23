#pragma once

#include <lodestar/lodestar_c.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace lodestar::firestorm {

struct Hull {
    std::vector<lodestar_collision_vertex> positions;
    std::vector<uint32_t> indices;
};

#if defined(_WIN32)
namespace detail {

inline std::filesystem::path adjacentModule() {
    constexpr std::size_t maxWindowsPath = 32768;
    std::wstring executable(512, L'\0');
    for (;;) {
        const auto count = GetModuleFileNameW(nullptr, executable.data(), static_cast<DWORD>(executable.size()));
        if (count == 0) {
            throw std::system_error(static_cast<int>(GetLastError()), std::system_category(),
                                    "Cannot locate the Firestorm executable");
        }
        if (count < executable.size()) {
            executable.resize(count);
            return std::filesystem::path(executable).parent_path() / L"lodestar_c.dll";
        }
        if (executable.size() == maxWindowsPath) {
            throw std::runtime_error("Firestorm executable path exceeds the Windows path limit");
        }
        executable.resize(executable.size() * 2);
    }
}

class Module {
public:
    explicit Module(const std::filesystem::path& path)
        : handle_(LoadLibraryExW(path.c_str(), nullptr,
                                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)) {
        if (!handle_) {
            throw std::system_error(static_cast<int>(GetLastError()), std::system_category(),
                                    "Cannot load the Lodestar physics module (lodestar_c.dll)");
        }
    }

    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;

    ~Module() { FreeLibrary(handle_); }

    template <typename Function>
    Function resolve(const char* name) const {
        const auto address = GetProcAddress(handle_, name);
        if (!address) throw std::runtime_error(std::string("Lodestar physics module is missing ") + name);
        return std::bit_cast<Function>(address);
    }

private:
    HMODULE handle_;
};

} // namespace detail
#endif

// An empty module path loads the Lodestar DLL installed beside the viewer executable.
// Results own their buffers and use the input coordinate space.
inline std::vector<Hull> decompose(std::span<const lodestar_collision_vertex> positions,
                                   std::span<const uint32_t> indices, std::size_t maxHulls,
                                   std::size_t maxVertices, const std::filesystem::path& modulePath = {},
                                   double maxSurfaceError = 0.0, lodestar_coherent_collision_info* report = nullptr) {
#if defined(_WIN32)
    constexpr std::size_t maxPhysicsHulls = 256;
    constexpr std::size_t maxPhysicsVertices = 256;
    if (maxHulls == 0 || maxHulls > maxPhysicsHulls || maxVertices < 4 || maxVertices > maxPhysicsVertices) {
        throw std::runtime_error("Lodestar physics needs limits of 1 to 256 hulls and 4 to 256 vertices per hull");
    }
    if (positions.empty() || indices.empty() || indices.size() % 3 != 0) {
        throw std::runtime_error("Lodestar physics needs a nonempty indexed triangle mesh");
    }

    const detail::Module module(modulePath.empty() ? detail::adjacentModule() : std::filesystem::absolute(modulePath));
    const auto lastError = module.resolve<decltype(&lodestar_last_error)>("lodestar_last_error");
    const auto convert = module.resolve<decltype(&lodestar_decompose_collision_hulls)>("lodestar_decompose_collision_hulls");
    const auto destroy = module.resolve<decltype(&lodestar_convex_compound_destroy)>("lodestar_convex_compound_destroy");
    const auto getCount = module.resolve<decltype(&lodestar_convex_compound_get_hull_count)>(
        "lodestar_convex_compound_get_hull_count");
    const auto getView = module.resolve<decltype(&lodestar_convex_compound_get_hull_view)>(
        "lodestar_convex_compound_get_hull_view");
    const auto check = [&](lodestar_status status) {
        if (status == LODESTAR_OK) return;
        const char* message = lastError();
        throw std::runtime_error(message && *message ? message : "Lodestar physics decomposition failed");
    };

    lodestar_convex_compound* raw = nullptr;
    lodestar_coherent_collision_info info{};
    lodestar_status status;
    if (maxSurfaceError > 0) {
        const auto coherent = module.resolve<decltype(&lodestar_build_coherent_collision)>("lodestar_build_coherent_collision");
        status = coherent(positions.data(), positions.size(), indices.data(), indices.size(), maxHulls, maxSurfaceError, &raw, &info);
    } else {
        status = convert(positions.data(), positions.size(), indices.data(), indices.size(), maxHulls, &raw);
        info.exact_solid_conversion = status == LODESTAR_OK;
    }
    const std::unique_ptr<lodestar_convex_compound, decltype(destroy)> compound(raw, destroy);
    check(status);
    if (!compound) throw std::runtime_error("Lodestar physics returned a missing compound");

    std::size_t count = 0;
    check(getCount(compound.get(), &count));
    if (count == 0 || count > maxHulls) {
        throw std::runtime_error("Lodestar physics returned an invalid number of hulls");
    }
    std::vector<Hull> hulls;
    hulls.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        lodestar_collision_mesh_view view{};
        check(getView(compound.get(), i, &view));
        if (view.position_count > maxVertices) {
            throw std::runtime_error("Lodestar physics hull requires " + std::to_string(view.position_count) +
                                     " vertices, exceeding the limit of " + std::to_string(maxVertices));
        }
        if (!view.positions || !view.indices || view.position_count < 4 ||
            view.index_count != 3 * (2 * view.position_count - 4)) {
            throw std::runtime_error("Lodestar physics returned an invalid triangle hull");
        }
        const std::span points(view.positions, view.position_count);
        const std::span elements(view.indices, view.index_count);
        if (std::ranges::any_of(points, [](const lodestar_collision_vertex& point) {
                return std::ranges::any_of(point.position, [](float value) { return !std::isfinite(value); });
            }) || std::ranges::any_of(elements, [&](uint32_t index) { return index >= points.size(); })) {
            throw std::runtime_error("Lodestar physics returned invalid hull geometry");
        }
        hulls.push_back({{points.begin(), points.end()}, {elements.begin(), elements.end()}});
    }
    if (report) *report = info;
    return hulls;
#else
    (void)positions;
    (void)indices;
    (void)maxHulls;
    (void)maxVertices;
    (void)modulePath;
    (void)maxSurfaceError;
    (void)report;
    throw std::runtime_error("Lodestar physics module loading requires Windows");
#endif
}

// Choose each hull's vertices from its geometry, subject to the upload format.
inline std::vector<Hull> decompose(std::span<const lodestar_collision_vertex> positions,
                                   std::span<const uint32_t> indices, std::size_t maxHulls,
                                   const std::filesystem::path& modulePath = {}) {
    return decompose(positions, indices, maxHulls, 256, modulePath);
}

} // namespace lodestar::firestorm
