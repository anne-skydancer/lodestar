#include "renderer.hpp"

#include <lodestar/lodestar.hpp>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace lodestar;
using Clock = std::chrono::steady_clock;

constexpr int defaultQuality = 4;

struct Options {
    std::optional<std::string> path;
    int quality = defaultQuality;
    bool inspect = false;
    bool help = false;
};

Options parseOptions(const std::vector<std::string>& args) {
    Options options;
    bool readFlags = true;
    for (size_t i = 0; i < args.size(); ++i) {
        const auto& arg = args[i];
        if (readFlags && arg == "--") {
            readFlags = false;
        } else if (readFlags && (arg == "--help" || arg == "-h")) {
            options.help = true;
        } else if (readFlags && arg == "--inspect") {
            options.inspect = true;
        } else if (readFlags && arg == "--quality") {
            if (++i == args.size()) throw std::invalid_argument("--quality requires a number from 1 to 12");
            const auto& value = args[i];
            const auto* end = value.data() + value.size();
            const auto [next, error] = std::from_chars(value.data(), end, options.quality);
            if (error != std::errc{} || next != end ||
                options.quality < minimumQuality || options.quality > maximumQuality) {
                throw std::invalid_argument("--quality requires a number from 1 to 12");
            }
        } else if (readFlags && arg.starts_with('-')) {
            throw std::invalid_argument("Unknown option " + arg);
        } else if (!arg.empty() && !options.path) {
            options.path = arg;
        } else {
            throw std::invalid_argument("Provide a single mesh path");
        }
    }
    return options;
}

std::string_view eligibilityDescription(SphereEligibility status) {
    switch (status) {
    case SphereEligibility::Eligible: return "eligible";
    case SphereEligibility::HasTextureCoordinates: return "source contains texture coordinates";
    case SphereEligibility::NoFit: return "no valid sphere fit";
    case SphereEligibility::RadialError: return "maximum vertex radial error is 1% or higher";
    case SphereEligibility::IncompleteSurface: return "source triangles do not approximate one complete sphere";
    }
    return "unknown fit result";
}

void printStats(std::string_view name, const Mesh& mesh) {
    std::cout << name << ": " << mesh.indices.size() / 3 << " triangles, " << mesh.vertices.size() << " vertices\n";
}

double millisecondsSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void printTopology(const TopologyReport& topology) {
    if (!topology.valid) {
        std::cout << "Topology: invalid mesh structure\n";
        return;
    }
    std::cout << "Topology: " << topology.geometricEdges << " geometric edges, " << topology.boundaryEdges
              << " geometric boundaries, " << topology.indexedBoundaryEdges << " indexed boundaries, "
              << topology.sharedEdges << " shared indexed edges\n"
              << "Seams: " << topology.uvSeamEdges << " UV, " << topology.normalSeamEdges << " normal, "
              << topology.materialSeamEdges << " material\n"
              << "Split edges with equal attributes: " << topology.compatibleSplitEdges << ", "
              << topology.recoverableSplitEdges << " recoverable, " << topology.crossSurfaceSplitEdges
              << " across source surfaces\n"
              << "Topology issues: " << topology.windingConflictEdges << " winding conflicts, "
              << topology.ambiguousEdges << " ambiguous edges, " << topology.nonManifoldEdges
              << " nonmanifold, " << topology.degenerateTriangles << " degenerate triangles\n";
}

void printBoundaryBudget(const BoundaryBudgetReport& budget, std::size_t target) {
    if (!budget.valid) {
        std::cout << "Fixed-boundary budget: invalid mesh structure\n";
        return;
    }
    std::cout << "Fixed-boundary lower bound: " << budget.lowerBoundTriangles << " triangles"
              << " | " << budget.certifiedCharts << '/' << budget.chartCount << " certified charts"
              << " | " << budget.protectedBoundaryEdges << " protected edge segments"
              << " | " << budget.boundaryLoops << " boundary loops"
              << " | " << budget.interiorVertices << " interior vertices\n";
    if (budget.unsupportedCharts) {
        std::cout << "Lower bound with unsupported charts retained: "
                  << budget.unchangedFallbackLowerBoundTriangles << " triangles\n";
    }
    if (target < budget.lowerBoundTriangles) {
        std::cout << "Planar target is below the fixed-boundary lower bound.\n";
    } else if (target < budget.unchangedFallbackLowerBoundTriangles) {
        std::cout << "Planar target requires changes in unsupported charts.\n";
    }
}

void inspect(const Options& options) {
    const auto importStart = Clock::now();
    const auto asset = options.path ? loadAsset(*options.path) : sampleAsset();
    const double importTime = millisecondsSince(importStart);
    const auto baselineStart = Clock::now();
    const auto baseline = generateMesh(asset, Mode::Baseline, options.quality);
    const double baselineTime = millisecondsSince(baselineStart);
    const auto planarStart = Clock::now();
    const auto planar = analyzePlanarMesh(asset.original);
    const double planarAnalysisTime = millisecondsSince(planarStart);
    const auto planarTarget = planarTriangleTarget(asset.original.indices.size() / 3, options.quality);
    const auto planarGenerationStart = Clock::now();
    const auto planarMesh = generatePlanarMesh(planar, planarTarget);
    const double planarGenerationTime = millisecondsSince(planarGenerationStart);
    const auto seamStart = Clock::now();
    const auto seam = analyzeSeamMesh(asset.original);
    const double seamAnalysisTime = millisecondsSince(seamStart);
    const auto seamGenerationStart = Clock::now();
    const auto seamMesh = generateSeamMesh(seam, planarTarget);
    const double seamGenerationTime = millisecondsSince(seamGenerationStart);

    std::cout << "Lodestar inspection: " << options.path.value_or("built-in sphere") << '\n';
    std::cout << "Quality: " << options.quality << " / " << maximumQuality << '\n';
    std::cout << "Source UVs: " << (asset.original.hasUV ? "present" : "none") << '\n';
    printStats("Original", asset.original);
    printTopology(inspectTopology(asset.original));
    printStats("Baseline", baseline);
    printStats("Planar", planarMesh);
    std::cout << "Planar triangle target: " << planarTarget << ", minimum with validated edits: "
              << planar.minimumTriangles() << '\n';
    printBoundaryBudget(inspectBoundaryBudget(asset.original), planarTarget);
    printStats("Seam", seamMesh);
    std::cout << "Seam edits: " << seam.removablePositions() << " removable boundary positions, minimum "
              << seam.minimumTriangles() << " triangles\n";
    if (asset.original.hasAdditionalUV || asset.original.hasUnrepresentedVertexAttributes) {
        std::cout << "Planar and Seam retained the source because the mesh buffers omit some source vertex attributes.\n";
    }
    if (asset.eligibility == SphereEligibility::Eligible) {
        const auto sphereStart = Clock::now();
        const auto sphere = generateMesh(asset, Mode::Sphere, options.quality);
        const double sphereTime = millisecondsSince(sphereStart);
        printStats("Analytic sphere", sphere);
        std::cout << std::fixed << std::setprecision(3) << "Sphere generation: " << sphereTime << " ms\n";
    } else {
        std::cout << "Analytic sphere: original used (" << eligibilityDescription(asset.eligibility) << ")\n";
    }

    if (asset.fit.valid) {
        std::cout << std::fixed << std::setprecision(6)
                  << "Source vertices relative radial error: RMS " << 100.0 * asset.fit.rmsRelativeError
                  << "%, max " << 100.0 * asset.fit.maxRelativeError << "%\n";
        if (asset.maxFaceInwardRelativeError) {
            std::cout << "Source triangle maximum inward radial deviation: "
                      << 100.0 * *asset.maxFaceInwardRelativeError << "%\n";
        }
    } else {
        std::cout << "Source vertices relative radial error: unavailable\n";
    }
    std::cout << std::fixed << std::setprecision(3)
              << (options.path ? "Import and analysis: " : "Sample and analysis: ") << importTime << " ms\n"
              << "Baseline generation: " << baselineTime << " ms\n"
              << "Planar analysis: " << planarAnalysisTime << " ms\n"
              << "Planar generation: " << planarGenerationTime << " ms\n"
              << "Seam analysis: " << seamAnalysisTime << " ms\n"
              << "Seam generation: " << seamGenerationTime << " ms\n";
}

class Viewer {
public:
    Viewer(GLFWwindow* window, Asset asset, int quality)
        : window_(window), asset_(std::move(asset)), quality_(quality), pendingQuality_(quality) {
        glfwSetWindowUserPointer(window_, this);
        glfwSetKeyCallback(window_, onKey);
        glfwSetDropCallback(window_, onDrop);
        renderer_.initialize(window_);
        renderer_.upload(asset_.original);
        vertexCount_ = asset_.original.vertices.size();
        triangleCount_ = asset_.original.indices.size() / 3;
        showStatus();
    }

    void run() {
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            if (glfwWindowShouldClose(window_)) break;
            applyInput();

            int width = 0;
            int height = 0;
            glfwGetFramebufferSize(window_, &width, &height);
            if (!width || !height) {
                glfwWaitEvents();
                continue;
            }
            renderer_.draw(yaw_, pitch_, distance_);
        }
    }

private:
    static void onDrop(GLFWwindow* window, int count, const char** paths) {
        if (count > 0) {
            auto& viewer = *static_cast<Viewer*>(glfwGetWindowUserPointer(window));
            viewer.pendingPath_ = paths[0];
        }
    }

    static void onKey(GLFWwindow* window, int key, int, int action, int) {
        if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
        auto& viewer = *static_cast<Viewer*>(glfwGetWindowUserPointer(window));
        switch (key) {
        case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(window, GLFW_TRUE); break;
        case GLFW_KEY_1: viewer.pendingMode_ = Mode::Original; break;
        case GLFW_KEY_2: viewer.pendingMode_ = Mode::Baseline; break;
        case GLFW_KEY_3: viewer.pendingMode_ = Mode::Sphere; break;
        case GLFW_KEY_4: viewer.pendingMode_ = Mode::Planar; break;
        case GLFW_KEY_5: viewer.pendingMode_ = Mode::Seam; break;
        case GLFW_KEY_UP:
        case GLFW_KEY_EQUAL:
        case GLFW_KEY_KP_ADD:
            viewer.pendingQuality_ = std::min(maximumQuality, viewer.pendingQuality_ + 1);
            break;
        case GLFW_KEY_DOWN:
        case GLFW_KEY_MINUS:
        case GLFW_KEY_KP_SUBTRACT:
            viewer.pendingQuality_ = std::max(minimumQuality, viewer.pendingQuality_ - 1);
            break;
        case GLFW_KEY_LEFT: viewer.yaw_ -= 0.12f; break;
        case GLFW_KEY_RIGHT: viewer.yaw_ += 0.12f; break;
        case GLFW_KEY_W: viewer.pitch_ = std::min(1.5f, viewer.pitch_ + 0.12f); break;
        case GLFW_KEY_S: viewer.pitch_ = std::max(-1.5f, viewer.pitch_ - 0.12f); break;
        case GLFW_KEY_PAGE_UP: viewer.distance_ = std::max(1.8f, viewer.distance_ - 0.25f); break;
        case GLFW_KEY_PAGE_DOWN: viewer.distance_ = std::min(15.0f, viewer.distance_ + 0.25f); break;
        default: break;
        }
    }

    void applyInput() {
        if (!pendingPath_.empty()) {
            const auto path = std::exchange(pendingPath_, {});
            try {
                auto next = loadAsset(path);
                setMesh(next.original);
                asset_ = std::move(next);
                planar_.reset();
                seam_.reset();
                mode_ = pendingMode_ = renderedMode_ = Mode::Original;
                failedLoad_ = false;
            } catch (const std::exception& error) {
                failedLoad_ = true;
                std::cerr << "Could not open " << path << ": " << error.what() << '\n';
            }
            showStatus();
        }
        if (mode_ == pendingMode_ && quality_ == pendingQuality_) return;

        try {
            const auto nextMode = pendingMode_ == Mode::Sphere && asset_.eligibility != SphereEligibility::Eligible
                ? Mode::Original : pendingMode_;
            if (nextMode != renderedMode_ || (nextMode != Mode::Original && quality_ != pendingQuality_)) {
                if (nextMode == Mode::Original) {
                    setMesh(asset_.original);
                } else if (nextMode == Mode::Planar || nextMode == Mode::Seam) {
                    const auto sourceTriangles = asset_.original.indices.size() / 3;
                    const auto target = planarTriangleTarget(sourceTriangles, pendingQuality_);
                    if (target >= sourceTriangles) {
                        setMesh(asset_.original);
                    } else if (nextMode == Mode::Planar) {
                        if (!planar_) planar_.emplace(analyzePlanarMesh(asset_.original));
                        setMesh(generatePlanarMesh(*planar_, target));
                    } else {
                        if (!seam_) seam_.emplace(analyzeSeamMesh(asset_.original));
                        setMesh(generateSeamMesh(*seam_, target));
                    }
                } else {
                    setMesh(generateMesh(asset_, nextMode, pendingQuality_));
                }
            }
            renderedMode_ = nextMode;
            mode_ = pendingMode_;
            quality_ = pendingQuality_;
            showStatus();
        } catch (const std::exception& error) {
            std::cerr << "Could not update mesh: " << error.what() << '\n';
            pendingMode_ = mode_;
            pendingQuality_ = quality_;
        }
    }

    void setMesh(const Mesh& mesh) {
        renderer_.upload(mesh);
        vertexCount_ = mesh.vertices.size();
        triangleCount_ = mesh.indices.size() / 3;
    }

    void showStatus() const {
        std::ostringstream title;
        title << "Lodestar | ";
        if (failedLoad_) title << "Import failed | ";
        if (mode_ == Mode::Baseline) {
            title << "Baseline";
        } else if (mode_ == Mode::Planar) {
            title << "Planar";
        } else if (mode_ == Mode::Seam) {
            title << "Seam";
        } else if (mode_ == Mode::Sphere && renderedMode_ == Mode::Sphere) {
            title << "Analytic sphere";
        } else {
            title << "Original";
            if (mode_ == Mode::Sphere) title << " (sphere unavailable)";
        }
        title << " | " << triangleCount_ << '/' << asset_.original.indices.size() / 3 << " triangles"
              << " | " << vertexCount_ << " vertices | Q " << quality_ << '/' << maximumQuality;
        if (asset_.fit.valid) {
            title << std::fixed << std::setprecision(3) << " | sphere RMS "
                  << 100.0 * asset_.fit.rmsRelativeError << '%';
        }
        glfwSetWindowTitle(window_, title.str().c_str());
        std::cout << title.str() << std::endl;
        if (mode_ == Mode::Sphere && renderedMode_ == Mode::Original) {
            std::cout << "Keeping the source mesh: " << eligibilityDescription(asset_.eligibility) << std::endl;
        }
    }

    GLFWwindow* window_;
    Renderer renderer_;
    Asset asset_;
    std::optional<PlanarAnalysis> planar_;
    std::optional<SeamAnalysis> seam_;
    size_t vertexCount_{};
    size_t triangleCount_{};
    Mode mode_ = Mode::Original;
    Mode pendingMode_ = Mode::Original;
    Mode renderedMode_ = Mode::Original;
    int quality_;
    int pendingQuality_;
    bool failedLoad_ = false;
    float yaw_ = 0.45f;
    float pitch_ = -0.2f;
    float distance_ = 3.7f;
    std::string pendingPath_;
};

void view(const Options& options) {
    auto asset = options.path ? loadAsset(*options.path) : sampleAsset();

    glfwSetErrorCallback([](int, const char* description) { std::cerr << "GLFW: " << description << '\n'; });
    if (!glfwInit()) throw std::runtime_error("Could not initialize GLFW");
    struct GlfwLifetime {
        ~GlfwLifetime() { glfwTerminate(); }
    } glfwLifetime;

    if (!glfwVulkanSupported()) throw std::runtime_error("GLFW could not find a Vulkan loader and driver");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)> window(
        glfwCreateWindow(1120, 760, "Lodestar", nullptr, nullptr), glfwDestroyWindow);
    if (!window) throw std::runtime_error("Could not open the Lodestar window");

    std::cout << "Drop a model to import | 1 original, 2 baseline, 3 sphere, 4 planar, 5 seam | Up/Down quality\n"
              << "Left/Right and W/S orbit | Page Up/Down zoom | Esc exits\n";
    Viewer viewer(window.get(), std::move(asset), options.quality);
    viewer.run();
}

int run(const std::vector<std::string>& args) {
    try {
        const auto options = parseOptions(args);
        if (options.help) {
            std::cout << "Usage: inverseglod [--inspect] [--quality 1-12] [mesh.dae|mesh.gltf|mesh.glb]\n"
                      << "Without a mesh path, Lodestar uses a built-in sphere.\n"
                      << "--inspect compares representations and reports the fixed-boundary budget, fit error, and CPU timings.\n";
        } else if (options.inspect) {
            inspect(options);
        } else {
            view(options);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Lodestar: " << error.what() << '\n';
        return 1;
    }
}

} // namespace

#ifdef _MSC_VER
int wmain(int argc, wchar_t** argv) {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        const auto utf8 = std::filesystem::path(argv[i]).u8string();
        args.emplace_back(utf8.begin(), utf8.end());
    }
    return run(args);
}
#else
int main(int argc, char** argv) {
    return run(std::vector<std::string>(argv + 1, argv + argc));
}
#endif
