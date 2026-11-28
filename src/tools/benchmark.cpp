#include <lodestar/mesh_metrics.hpp>
#include <lodestar/silhouette_metrics.hpp>

#include <lodestar/lodestar.hpp>

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace lodestar;
using Clock = std::chrono::steady_clock;

constexpr double budgetTolerance = 0.05;
constexpr int csvPrecision = 9;

struct Options {
    std::vector<std::string> paths;
    std::vector<int> qualities;
    std::optional<std::string> authoredLod;
    std::optional<glm::dvec3> authoredTranslation;
    int repeats = 5;
    int samples = 4096;
    int resolution = 192;
    bool sample = false;
    bool help = false;
};

int parseInteger(const std::vector<std::string>& args, std::size_t& index, int min, int max) {
    const auto& flag = args[index];
    if (++index < args.size()) {
        const auto& text = args[index];
        const char* end = text.data() + text.size();
        int value = 0;
        const auto [next, error] = std::from_chars(text.data(), end, value);
        if (error == std::errc{} && next == end && value >= min && value <= max) return value;
    }
    throw std::invalid_argument(flag + " requires an integer from " + std::to_string(min) +
                                " to " + std::to_string(max));
}

double parseTranslation(const std::vector<std::string>& args, std::size_t& index) {
    if (++index < args.size()) {
        std::string_view text = args[index];
        if (text.starts_with('+') && !text.starts_with("+-")) text.remove_prefix(1);
        const char* end = text.data() + text.size();
        double value = 0.0;
        const auto [next, error] = std::from_chars(text.data(), end, value);
        if (error == std::errc{} && next == end && std::isfinite(value)) return value;
    }
    throw std::invalid_argument("--authored-translate requires three finite numbers in scene coordinates");
}

Options parseOptions(const std::vector<std::string>& args) {
    Options options;
    bool readFlags = true;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const auto& arg = args[i];
        if (readFlags && arg == "--") {
            readFlags = false;
        } else if (readFlags && (arg == "--help" || arg == "-h")) {
            options.help = true;
        } else if (readFlags && arg == "--sample") {
            options.sample = true;
        } else if (readFlags && arg == "--authored-lod") {
            if (options.authoredLod || ++i == args.size() || args[i].empty() || args[i].starts_with('-')) {
                throw std::invalid_argument("--authored-lod requires one authored mesh path");
            }
            options.authoredLod = args[i];
        } else if (readFlags && arg == "--authored-translate") {
            if (options.authoredTranslation) throw std::invalid_argument("Provide one authored translation");
            options.authoredTranslation = glm::dvec3{
                parseTranslation(args, i), parseTranslation(args, i), parseTranslation(args, i)
            };
        } else if (readFlags && arg == "--repeats") {
            options.repeats = parseInteger(args, i, 1, 100);
        } else if (readFlags && arg == "--quality") {
            options.qualities.push_back(parseInteger(args, i, minimumQuality, maximumQuality));
        } else if (readFlags && arg == "--samples") {
            options.samples = parseInteger(args, i, 1, 1'000'000);
        } else if (readFlags && arg == "--resolution") {
            options.resolution = parseInteger(args, i, 16, 1024);
        } else if (readFlags && arg.starts_with('-')) {
            throw std::invalid_argument("Unknown option " + arg);
        } else if (!arg.empty()) {
            options.paths.push_back(arg);
        } else {
            throw std::invalid_argument("Mesh paths cannot be empty");
        }
    }
    if (options.authoredLod) {
        if (options.paths.size() != 1 || options.sample || !options.qualities.empty()) {
            throw std::invalid_argument("--authored-lod requires one source mesh and uses its authored triangle budget");
        }
        return options;
    }
    if (options.authoredTranslation) {
        throw std::invalid_argument("--authored-translate requires --authored-lod");
    }
    if (options.qualities.empty()) options.qualities = {1, 4, 8, 12};
    std::sort(options.qualities.begin(), options.qualities.end());
    options.qualities.erase(std::unique(options.qualities.begin(), options.qualities.end()), options.qualities.end());
    options.sample = options.sample || options.paths.empty();
    return options;
}

std::string_view eligibilityName(SphereEligibility eligibility) {
    switch (eligibility) {
    case SphereEligibility::Eligible: return "eligible";
    case SphereEligibility::HasTextureCoordinates: return "has_uv";
    case SphereEligibility::NoFit: return "no_fit";
    case SphereEligibility::RadialError: return "radial_error";
    case SphereEligibility::IncompleteSurface: return "incomplete_surface";
    }
    throw std::logic_error("Unknown sphere eligibility");
}

template <typename Operation>
auto measureMedian(int repeats, Operation operation) {
    operation();
    std::vector<double> durations;
    durations.reserve(static_cast<std::size_t>(repeats));
    std::optional<decltype(operation())> result;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        const auto start = Clock::now();
        auto output = operation();
        durations.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
        if (repeat + 1 == repeats) result.emplace(std::move(output));
    }
    std::sort(durations.begin(), durations.end());
    const auto middle = durations.size() / 2;
    const double median = (durations[middle] + durations[(durations.size() - 1) / 2]) / 2.0;
    return std::pair(std::move(*result), median);
}

struct Fidelity {
    metrics::SurfaceDistance surface;
    metrics::SilhouetteDistance silhouette;
};

Fidelity compare(const Mesh& source, const Mesh& candidate, const Options& options) {
    return {
        metrics::measureSurfaceDistance(source, candidate, static_cast<std::size_t>(options.samples)),
        metrics::measureSilhouetteDistance(source, candidate, static_cast<unsigned>(options.resolution))
    };
}

class CsvRow {
public:
    explicit CsvRow(std::string_view assetName) { writeText(assetName); }

    CsvRow& text(std::string_view value) {
        std::cout << ',';
        writeText(value);
        return *this;
    }

    template <typename Number>
    CsvRow& number(Number value) {
        std::cout << ',' << value;
        return *this;
    }

    template <typename Number>
    CsvRow& number(std::optional<Number> value) {
        return value ? number(*value) : text("");
    }

    void end() { std::cout << '\n'; }

private:
    static void writeText(std::string_view value) {
        if (value.find_first_of(",\"\r\n") == std::string_view::npos) {
            std::cout << value;
            return;
        }
        std::cout << '"';
        for (const char character : value) {
            if (character == '"') std::cout << '"';
            std::cout << character;
        }
        std::cout << '"';
    }
};

struct Input {
    const Options& options;
    std::string_view name;
    const Asset* asset = nullptr;
    std::optional<double> importMs{};
    std::optional<double> authoredImportMs{};
    const TopologyReport* topology = nullptr;
    const BoundaryBudgetReport* boundaryBudget = nullptr;
    std::optional<double> boundaryAnalysisMs{};
    std::optional<std::size_t> authoredTarget{};
};

template <typename Report>
struct ReportColumn {
    const char* name;
    std::size_t Report::*count;
};

constexpr ReportColumn<TopologyReport> topologyColumns[]{
    {"source_geometric_edges", &TopologyReport::geometricEdges},
    {"source_boundary_edges", &TopologyReport::boundaryEdges},
    {"source_indexed_boundary_edges", &TopologyReport::indexedBoundaryEdges},
    {"source_shared_edges", &TopologyReport::sharedEdges},
    {"source_compatible_split_edges", &TopologyReport::compatibleSplitEdges},
    {"source_recoverable_split_edges", &TopologyReport::recoverableSplitEdges},
    {"source_cross_surface_split_edges", &TopologyReport::crossSurfaceSplitEdges},
    {"source_uv_seam_edges", &TopologyReport::uvSeamEdges},
    {"source_normal_seam_edges", &TopologyReport::normalSeamEdges},
    {"source_material_seam_edges", &TopologyReport::materialSeamEdges},
    {"source_winding_conflict_edges", &TopologyReport::windingConflictEdges},
    {"source_non_manifold_edges", &TopologyReport::nonManifoldEdges},
    {"source_ambiguous_edges", &TopologyReport::ambiguousEdges},
    {"source_degenerate_triangles", &TopologyReport::degenerateTriangles},
};

constexpr ReportColumn<BoundaryBudgetReport> boundaryColumns[]{
    {"boundary_source_triangles", &BoundaryBudgetReport::sourceTriangles},
    {"boundary_chart_count", &BoundaryBudgetReport::chartCount},
    {"boundary_certified_charts", &BoundaryBudgetReport::certifiedCharts},
    {"boundary_unsupported_charts", &BoundaryBudgetReport::unsupportedCharts},
    {"boundary_unsupported_triangles", &BoundaryBudgetReport::unsupportedTriangles},
    {"boundary_protected_edges", &BoundaryBudgetReport::protectedBoundaryEdges},
    {"boundary_loops", &BoundaryBudgetReport::boundaryLoops},
    {"boundary_interior_vertices", &BoundaryBudgetReport::interiorVertices},
    {"boundary_closed_charts", &BoundaryBudgetReport::closedCharts},
    {"boundary_lower_bound_triangles", &BoundaryBudgetReport::lowerBoundTriangles},
    {"boundary_fallback_lower_bound_triangles", &BoundaryBudgetReport::unchangedFallbackLowerBoundTriangles},
    {"boundary_upper_bound_triangles", &BoundaryBudgetReport::upperBoundTriangles},
};

std::string_view boundaryBudgetStatus(const Input& input) {
    const auto* report = input.boundaryBudget;
    if (!report || !input.authoredTarget) return "";
    if (!report->valid) return "invalid";
    const auto target = *input.authoredTarget;
    if (target < report->lowerBoundTriangles) return "below_bound";
    if (target < report->unchangedFallbackLowerBoundTriangles) return "below_fallback_bound";
    return "not_ruled_out";
}

struct Representation {
    std::string_view name;
    std::string_view status = "ok";
    std::optional<int> quality{};
    const Mesh* mesh = nullptr;
    std::optional<std::size_t> target{};
    std::string_view budgetStatus{};
    std::optional<double> generateMs{};
    const Fidelity* fidelity = nullptr;
    std::string_view error{};
    std::optional<double> planarAnalysisMs{};
    std::optional<std::size_t> planarMinimum{};
    std::optional<double> seamAnalysisMs{};
    std::optional<std::size_t> seamMinimum{};
    std::optional<std::size_t> seamRemovablePositions{};
};

void writeRow(const Input& input, const Representation& representation) {
    const auto* asset = input.asset;
    const auto* fit = asset && asset->fit.valid ? &asset->fit : nullptr;
    const auto* mesh = representation.mesh;
    const auto& target = representation.target;
    const auto* fidelity = representation.fidelity;
    const auto triangles = mesh ? std::optional(mesh->indices.size() / 3) : std::nullopt;
    const auto budgetDelta = triangles && target
        ? std::optional((static_cast<double>(*triangles) - static_cast<double>(*target)) / static_cast<double>(*target))
        : std::nullopt;

    CsvRow row(input.name);
    row.number(representation.quality).text(representation.name).text(representation.status)
        .text(asset ? eligibilityName(asset->eligibility) : "")
        .number(asset ? std::optional(asset->original.hasUV) : std::nullopt)
        .number(asset ? std::optional(asset->original.indices.size() / 3) : std::nullopt)
        .number(triangles).number(mesh ? std::optional(mesh->vertices.size()) : std::nullopt)
        .number(target).text(representation.budgetStatus).number(budgetDelta)
        .number(input.options.repeats).number(input.importMs).number(representation.generateMs)
        .number(fit ? std::optional(fit->rmsRelativeError) : std::nullopt)
        .number(fit ? std::optional(fit->maxRelativeError) : std::nullopt)
        .number(asset ? asset->maxFaceInwardRelativeError : std::nullopt);

    if (fidelity) {
        const auto& surface = fidelity->surface;
        const auto& a = surface.sourceToCandidate;
        const auto& b = surface.candidateToSource;
        const auto& silhouette = fidelity->silhouette;
        row.number(surface.samplesPerDirection).number(surface.sourceExtent)
            .number(std::hypot(a.rms, b.rms) / std::sqrt(2.0)).number(std::max(a.maximum, b.maximum))
            .number(a.rms).number(a.maximum).number(b.rms).number(b.maximum)
            .number(silhouette.viewCount).number(silhouette.resolution)
            .number(silhouette.meanIoUError).number(silhouette.maximumIoUError);
    } else {
        for (int column = 0; column < 12; ++column) row.text("");
    }
    row.text(representation.error);
    if (input.options.authoredLod) {
        row.text(*input.options.authoredLod).number(input.authoredImportMs)
            .number(representation.planarAnalysisMs).number(representation.planarMinimum);
        const auto translation = input.options.authoredTranslation.value_or(glm::dvec3{0.0});
        std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
        row.number(translation.x).number(translation.y).number(translation.z);
        std::cout << std::setprecision(csvPrecision);
        const auto* topology = input.topology;
        row.number(asset ? std::optional(asset->original.hasAdditionalUV) : std::nullopt)
            .number(asset ? std::optional(asset->original.hasUnrepresentedVertexAttributes) : std::nullopt)
            .number(topology ? std::optional(topology->valid) : std::nullopt);
        for (const auto& column : topologyColumns) {
            row.number(topology ? std::optional(topology->*column.count) : std::nullopt);
        }
        const auto* boundaryBudget = input.boundaryBudget;
        row.number(boundaryBudget ? std::optional(boundaryBudget->valid) : std::nullopt)
            .text(boundaryBudgetStatus(input)).number(input.boundaryAnalysisMs);
        for (const auto& column : boundaryColumns) {
            row.number(boundaryBudget ? std::optional(boundaryBudget->*column.count) : std::nullopt);
        }
        row.number(representation.seamAnalysisMs).number(representation.seamMinimum)
            .number(representation.seamRemovablePositions);
    }
    row.end();
}

float fractionForBudget(std::size_t target, std::size_t source) {
    if (target >= source) return 1.0f;
    const auto fraction = static_cast<float>(static_cast<double>(target) / static_cast<double>(source));
    // The simplifier floors the fraction after multiplying by the source triangle count.
    return static_cast<double>(fraction) * static_cast<double>(source) < static_cast<double>(target)
        ? std::nextafter(fraction, 1.0f)
        : fraction;
}

std::string_view budgetStatus(std::size_t actual, std::size_t target, std::size_t source) {
    if (target > source) return "above_source";
    const double difference = std::abs(static_cast<double>(actual) - static_cast<double>(target));
    return difference <= budgetTolerance * static_cast<double>(target) ? "matched" : "unmatched";
}

template <glm::length_t Size, glm::qualifier Qualifier>
bool sameComponents(const glm::vec<Size, float, Qualifier>& a, const glm::vec<Size, float, Qualifier>& b) {
    for (glm::length_t i = 0; i < Size; ++i) {
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i])) return false;
    }
    return true;
}

bool sameMesh(const Mesh& source, const Mesh& candidate) {
    return source.hasUV == candidate.hasUV && source.hasAdditionalUV == candidate.hasAdditionalUV &&
           source.indices == candidate.indices && source.triangleMaterials == candidate.triangleMaterials &&
           source.triangleSurfaceIds == candidate.triangleSurfaceIds &&
           source.vertexPositionIds == candidate.vertexPositionIds &&
           source.hasUnrepresentedVertexAttributes == candidate.hasUnrepresentedVertexAttributes &&
           std::ranges::equal(source.vertices, candidate.vertices, [](const Vertex& a, const Vertex& b) {
               return sameComponents(a.position, b.position) && sameComponents(a.normal, b.normal) &&
                      sameComponents(a.uv, b.uv);
           });
}

bool verifyFallback(std::string_view mode, const Mesh& source, const Mesh& result, std::size_t target) {
    const auto sourceTriangles = source.indices.size() / 3;
    const bool fallback = target < sourceTriangles && result.indices.size() / 3 == sourceTriangles;
    if (fallback && !sameMesh(source, result)) {
        throw std::runtime_error(std::string(mode) + " fallback changed the source mesh");
    }
    return fallback;
}

void compareAtQuality(const Input& input, int quality) {
    const auto& asset = *input.asset;
    const auto& source = asset.original;
    const auto sourceTriangles = source.indices.size() / 3;
    const auto repeats = input.options.repeats;
    if (asset.eligibility != SphereEligibility::Eligible) {
        const auto fallback = generateMesh(asset, Mode::Sphere, quality);
        if (!sameMesh(source, fallback)) {
            throw std::runtime_error("Ineligible sphere generation changed the source mesh");
        }
        const auto [baseline, baselineMs] = measureMedian(repeats, [&] {
            return generateMesh(asset, Mode::Baseline, quality);
        });
        const auto baselineTarget = std::max<std::size_t>(
            1, static_cast<std::size_t>(static_cast<double>(sourceTriangles) * baselineFraction(quality)));
        const auto baselineFidelity = compare(source, baseline, input.options);
        writeRow(input, {.name = "baseline", .quality = quality, .mesh = &baseline, .target = baselineTarget,
                         .budgetStatus = "unpaired", .generateMs = baselineMs, .fidelity = &baselineFidelity});
        writeRow(input, {.name = "analytic_sphere", .status = "original_fallback", .quality = quality,
                         .mesh = &fallback, .budgetStatus = "unpaired"});
        return;
    }

    const auto [sphere, sphereMs] = measureMedian(repeats, [&] {
        return generateMesh(asset, Mode::Sphere, quality);
    });
    const auto target = sphere.indices.size() / 3;
    const float fraction = fractionForBudget(target, sourceTriangles);
    const auto [baseline, baselineMs] = measureMedian(repeats, [&] { return simplifyMesh(source, fraction); });
    const auto status = budgetStatus(baseline.indices.size() / 3, target, sourceTriangles);
    const auto baselineFidelity = compare(source, baseline, input.options);
    const auto sphereFidelity = compare(source, sphere, input.options);
    writeRow(input, {.name = "baseline", .quality = quality, .mesh = &baseline, .target = target,
                     .budgetStatus = status, .generateMs = baselineMs, .fidelity = &baselineFidelity});
    writeRow(input, {.name = "analytic_sphere", .quality = quality, .mesh = &sphere, .target = target,
                     .budgetStatus = status, .generateMs = sphereMs, .fidelity = &sphereFidelity});
}

bool benchmark(const Options& options, const std::string* path) {
    const std::string_view name = path ? std::string_view(*path) : "built-in sphere";
    try {
        const auto [asset, importMs] = measureMedian(options.repeats, [&] {
            return path ? loadAsset(*path) : sampleAsset();
        });
        const Input input{options, name, &asset, importMs};
        writeRow(input, {.name = "original", .status = "reference", .mesh = &asset.original});
        for (const int quality : options.qualities) compareAtQuality(input, quality);
        return true;
    } catch (const std::exception& error) {
        writeRow({options, name}, {.name = "", .status = "error", .error = error.what()});
        std::cerr << "Could not benchmark " << name << ": " << error.what() << '\n';
        return false;
    }
}

void translateAuthored(Mesh& mesh, const glm::dvec3& translation) {
    for (auto& vertex : mesh.vertices) {
        const auto position = glm::dvec3(vertex.position) + translation;
        if (glm::any(glm::greaterThan(glm::abs(position), glm::dvec3(std::numeric_limits<float>::max())))) {
            throw std::invalid_argument("Authored translation puts a vertex outside the supported coordinate range");
        }
        vertex.position = glm::vec3(position);
    }
}

bool benchmarkAuthored(const Options& options) {
    const auto& path = options.paths.front();
    const auto& authoredPath = *options.authoredLod;
    try {
        const auto [asset, importMs] = measureMedian(options.repeats, [&] {
            return loadAsset(path, ImportSpace::Scene);
        });
        auto [authored, authoredMs] = measureMedian(options.repeats, [&] {
            return loadMesh(authoredPath, ImportSpace::Scene);
        });
        if (options.authoredTranslation && *options.authoredTranslation != glm::dvec3{0.0}) {
            translateAuthored(authored, *options.authoredTranslation);
        }
        const auto& source = asset.original;
        const auto topology = inspectTopology(source);
        const auto sourceTriangles = source.indices.size() / 3;
        const auto target = authored.indices.size() / 3;
        const auto [boundaryBudget, boundaryMs] = measureMedian(options.repeats, [&] {
            return inspectBoundaryBudget(source);
        });
        const Input input{.options = options, .name = path, .asset = &asset, .importMs = importMs,
                          .authoredImportMs = authoredMs, .topology = &topology, .boundaryBudget = &boundaryBudget,
                          .boundaryAnalysisMs = boundaryMs, .authoredTarget = target};

        writeRow(input, {.name = "original", .status = "reference", .mesh = &source});
        const auto authoredFidelity = compare(source, authored, options);
        writeRow(input, {.name = "authored_lod", .mesh = &authored, .target = target,
                         .budgetStatus = "authored_target", .fidelity = &authoredFidelity});

        const float fraction = fractionForBudget(target, sourceTriangles);
        const auto [baseline, baselineMs] = measureMedian(options.repeats, [&] {
            return simplifyMesh(source, fraction);
        });
        const auto baselineFidelity = compare(source, baseline, options);
        writeRow(input, {.name = "baseline", .mesh = &baseline, .target = target,
                         .budgetStatus = budgetStatus(baseline.indices.size() / 3, target, sourceTriangles),
                         .generateMs = baselineMs, .fidelity = &baselineFidelity});

        const auto [analysis, analysisMs] = measureMedian(options.repeats, [&] {
            return analyzePlanarMesh(source);
        });
        const auto [planar, planarMs] = measureMedian(options.repeats, [&] {
            return generatePlanarMesh(analysis, target);
        });
        const auto planarTriangles = planar.indices.size() / 3;
        const bool fallback = verifyFallback("Planar", source, planar, target);
        const auto planarFidelity = compare(source, planar, options);
        writeRow(input, {.name = "planar", .status = fallback ? "original_fallback" : "ok", .mesh = &planar,
                         .target = target, .budgetStatus = budgetStatus(planarTriangles, target, sourceTriangles),
                         .generateMs = planarMs, .fidelity = &planarFidelity, .planarAnalysisMs = analysisMs,
                         .planarMinimum = analysis.minimumTriangles()});

        const auto [seamAnalysis, seamAnalysisMs] = measureMedian(options.repeats, [&] {
            return analyzeSeamMesh(source);
        });
        const auto [seam, seamMs] = measureMedian(options.repeats, [&] {
            return generateSeamMesh(seamAnalysis, target);
        });
        const bool seamFallback = verifyFallback("Seam", source, seam, target);
        const auto seamFidelity = compare(source, seam, options);
        writeRow(input, {.name = "seam", .status = seamFallback ? "original_fallback" : "ok", .mesh = &seam,
                         .target = target, .budgetStatus = budgetStatus(seam.indices.size() / 3, target, sourceTriangles),
                         .generateMs = seamMs, .fidelity = &seamFidelity, .seamAnalysisMs = seamAnalysisMs,
                         .seamMinimum = seamAnalysis.minimumTriangles(),
                         .seamRemovablePositions = seamAnalysis.removablePositions()});
        return true;
    } catch (const std::exception& error) {
        writeRow({options, path}, {.name = "", .status = "error", .error = error.what()});
        std::cerr << "Could not benchmark " << path << " against " << authoredPath << ": " << error.what() << '\n';
        return false;
    }
}

int run(const std::vector<std::string>& args) {
    try {
        const auto options = parseOptions(args);
        if (options.help) {
            std::cout << "Usage: lodestar_benchmark [--sample] [--quality 1-12] [--repeats N]\n"
                         "                          [--samples N] [--resolution N] [mesh ...]\n"
                         "       lodestar_benchmark --authored-lod LOD SOURCE [--repeats N]\n"
                         "                          [--samples N] [--resolution N] [--authored-translate X Y Z]\n"
                         "Writes CSV to stdout. With no paths, benchmarks the built-in sphere.\n"
                         "Repeat --quality to choose levels (default 1, 4, 8, 12).\n"
                         "Authored LOD comparison preserves shared scene coordinates and uses the LOD triangle count.\n"
                         "The fixed-boundary report applies to original indexed segments.\n"
                         "Defaults: 5 timed repetitions after one warmup, 4096 surface samples per direction,\n"
                         "192 pixels per silhouette axis. Budgets match within +/-5% of the target triangle count.\n";
            return 0;
        }

        std::cout.imbue(std::locale::classic());
        std::cout << std::setprecision(csvPrecision)
                  << "asset,quality,representation,status,sphere_eligibility,source_has_uv,source_triangles,"
                     "triangles,vertices,target_triangles,budget_status,budget_delta_relative,repeats,"
                     "import_analysis_median_ms,generation_median_ms,source_fit_rms_relative,source_fit_max_relative,"
                     "source_face_inward_max_relative,surface_samples_per_direction,source_extent,"
                     "surface_rms_relative,surface_max_relative,source_to_candidate_rms_relative,"
                     "source_to_candidate_max_relative,candidate_to_source_rms_relative,candidate_to_source_max_relative,"
                     "silhouette_views,silhouette_resolution,silhouette_mean_iou_error,silhouette_max_iou_error,error";
        if (options.authoredLod) {
            std::cout << ",authored_lod_path,authored_import_median_ms,planar_analysis_median_ms,planar_minimum_triangles,"
                         "authored_translate_x,authored_translate_y,authored_translate_z,source_has_additional_uv,"
                         "source_has_unrepresented_vertex_attributes,source_topology_valid";
            for (const auto& column : topologyColumns) std::cout << ',' << column.name;
            std::cout << ",boundary_budget_valid,boundary_budget_status,boundary_analysis_median_ms";
            for (const auto& column : boundaryColumns) std::cout << ',' << column.name;
            std::cout << ",seam_analysis_median_ms,seam_minimum_triangles,seam_removable_positions";
        }
        std::cout << '\n';
        bool okay = true;
        if (options.authoredLod) {
            okay = benchmarkAuthored(options);
        } else {
            if (options.sample) okay = benchmark(options, nullptr);
            for (const auto& path : options.paths) {
                if (!benchmark(options, &path)) okay = false;
            }
        }
        if (!std::cout) throw std::runtime_error("Could not write benchmark CSV");
        return okay ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Lodestar benchmark: " << error.what() << '\n';
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
