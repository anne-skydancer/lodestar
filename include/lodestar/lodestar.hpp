#pragma once

#include <glm/glm.hpp>
#include <lodestar/surface_calculus.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace lodestar {

inline constexpr int minimumQuality = 1;
inline constexpr int maximumQuality = 12;

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    bool hasUV = false;
    // True when the imported asset contains UV components outside the stored UV0.xy.
    bool hasAdditionalUV = false;
    // Imported material IDs in triangle order, when available. IDs are local to the source asset.
    std::vector<uint32_t> triangleMaterials;
    // Optional original geometric vertex identities in vertex order. Equal nonzero IDs
    // assert one authored geometric vertex. Zero means its identity is unknown.
    std::vector<uint64_t> vertexPositionIds;
    // Optional surface/instance IDs in triangle order. Recovery stays within these scopes.
    std::vector<uint32_t> triangleSurfaceIds;
    // Includes source color, skin, morph, and tangent data absent from Vertex.
    bool hasUnrepresentedVertexAttributes = false;
    // Referenced source bounds before zero-area faces are removed, in scene
    // metres. Legacy bounds carriers can matter even when they draw no surface.
    std::optional<std::array<glm::dvec3, 2>> declaredSceneBounds;
};

struct CollisionMesh {
    std::vector<std::array<float, 3>> positions;
    std::vector<uint32_t> indices;
    std::size_t skippedDegenerateTriangles = 0;
};

struct SurfaceTessellation {
    Mesh mesh;
    SurfaceLodBudget budget;
    double maximumDeviationBound = 0.0;
    unsigned subdivisionLevel = 0;
};

// Tessellate one explicitly supplied regular patch. Derivatives provide normals.
// Integral density proposes the resolution. Bezier control hulls then bound the
// interpolation error, with an arithmetic allowance and measured float32 rounding.
// Throws if the requested tolerance cannot be reached within maximumDepth.
SurfaceTessellation tessellateSurface(const BezierTriangle& patch, double worldUnitsPerPixel,
                                     double maximumPixelError, unsigned maximumDepth = 8);

// Build static triangle collision in the source coordinate space. Every nondegenerate
// source triangle keeps its position and winding. Malformed indices or referenced
// nonfinite positions throw std::invalid_argument.
CollisionMesh buildCollisionMesh(const Mesh& source);

struct ConvexHullOptions {
    std::size_t targetVertices = 256;
    double maxDeviation = 0.0;
    // Retain the full footprints on the lowest and highest Y planes.
    bool preserveVerticalSupport = true;
};

struct ConvexHullResult {
    CollisionMesh mesh;
    std::size_t inputVertices = 0;
    double maxDeviation = 0.0;
    bool metTarget = false;
};

// Simplify one closed, outward-wound convex collision solid in its input units.
// The result uses source positions and stays inside the source. The vertex target
// is best effort within the deviation limit. Invalid or concave hulls throw
// std::invalid_argument. Simplify separate collision pieces independently.
ConvexHullResult simplifyConvexHull(const CollisionMesh& source, const ConvexHullOptions& options = {});

// Construct the outward convex envelope of finite 3D points. Uses input corners
// and robust orientation predicates; rejects coplanar/collinear inputs. This
// constructs an envelope, not a certificate of mesh conformity or hole safety.
CollisionMesh buildConvexHull(std::span<const std::array<float, 3>> points);

struct CoherentCollisionOptions {
    double maxSurfaceError = 0.0;
    std::size_t maxHulls = 256;
};

struct CoherentCollisionResult {
    std::vector<CollisionMesh> hulls;
    double surfaceErrorBound = 0.0;
    bool exactSolidConversion = false;
};

struct MeshSizePlan {
    glm::dvec3 visibleMinimum{};
    glm::dvec3 visibleMaximum{};
    glm::dvec3 containerMinimum{};
    glm::dvec3 containerMaximum{};
    glm::dvec3 center{};
    glm::dvec3 visibleSize{};
    glm::dvec3 containerSize{};
    glm::dvec3 visibleFraction{};
    bool requiresContainer = false;
    bool requiresHullPhysics = false;
};

// Propose a shared visual/container frame without changing source vertices.
// Container-normalized p = (source p - center) / containerSize; invert that
// transform to retain authored physical scale. A physics box alone does not
// change the visual asset's normalization. Limits use input metre coordinates.
MeshSizePlan planMeshSize(std::span<const std::array<float, 3>> points,
                         double minimumDimension = 0.01, double trianglePhysicsMinimum = 0.5);

// Construct a source-following compound, including from open surface elements.
// Every output piece is a closed convex volume. When exact solid conversion
// is unavailable, surface coverage and free-space halfspaces are checked within
// maxSurfaceError. That surface cover is not an exact occupied-volume claim.
CoherentCollisionResult buildCoherentCollisionShape(const CollisionMesh& source,
    const CoherentCollisionOptions& options);

// Convert supported closed solids to separate convex pieces with the same volume.
// Supports convex shells and axis-aligned orthogonal concave solids in source space.
// Throws std::invalid_argument for unsupported geometry and std::length_error when
// the conversion exceeds the piece budget or the orthogonal geometry work limits.
std::vector<CollisionMesh> decomposeCollisionHulls(const CollisionMesh& source, std::size_t maxHulls = 256);

struct TopologyReport {
    bool valid = false;
    std::size_t geometricEdges = 0;
    std::size_t boundaryEdges = 0;
    std::size_t indexedBoundaryEdges = 0;
    std::size_t sharedEdges = 0;
    std::size_t compatibleSplitEdges = 0;
    std::size_t recoverableSplitEdges = 0;
    std::size_t crossSurfaceSplitEdges = 0;
    std::size_t uvSeamEdges = 0;
    std::size_t normalSeamEdges = 0;
    std::size_t materialSeamEdges = 0;
    std::size_t windingConflictEdges = 0;
    std::size_t ambiguousEdges = 0;
    std::size_t nonManifoldEdges = 0;
    std::size_t degenerateTriangles = 0;
};

// Compare indexed edges with exact position matches. Seam counts can overlap.
TopologyReport inspectTopology(const Mesh& mesh);

struct BoundaryBudgetReport {
    bool valid = false;
    std::size_t sourceTriangles = 0;
    std::size_t chartCount = 0;
    std::size_t certifiedCharts = 0;
    std::size_t unsupportedCharts = 0;
    std::size_t unsupportedTriangles = 0;
    // The following boundary and vertex counts describe certified charts only.
    std::size_t protectedBoundaryEdges = 0;
    std::size_t boundaryLoops = 0;
    std::size_t interiorVertices = 0;
    std::size_t closedCharts = 0;
    // Necessary triangle count under fixed boundaries and unchanged chart topology.
    std::size_t lowerBoundTriangles = 0;
    // Adds the source triangle count of each unsupported chart if kept unchanged.
    std::size_t unchangedFallbackLowerBoundTriangles = 0;
    std::size_t upperBoundTriangles = 0;
};

// Analyze the current indexed charts with material and surface boundaries fixed.
// This leaves split indices intact, including those with authored position IDs.
BoundaryBudgetReport inspectBoundaryBudget(const Mesh& mesh);

struct SphereFit {
    glm::vec3 center{0.0f};
    float radius = 0.0f;
    float maxRelativeError = 0.0f;
    float rmsRelativeError = 0.0f;
    bool valid = false;
};

enum class SphereEligibility {
    Eligible,
    HasTextureCoordinates,
    NoFit,
    RadialError,
    IncompleteSurface
};

enum class Mode { Original, Baseline, Sphere, Planar, Seam };
enum class ImportSpace { Normalized, Scene };

struct Asset {
    Mesh original;
    SphereFit fit;
    SphereEligibility eligibility = SphereEligibility::NoFit;
    // The largest fraction of the fitted radius by which a point on a source triangle
    // falls inside the sphere, including triangle interiors and edges. Available for
    // valid indexed triangles when the maximum vertex residual is below 1%.
    std::optional<float> maxFaceInwardRelativeError;
};

// Analyze the supplied triangles and retain them for exact fallback.
Asset analyze(Mesh mesh);
// Import a mesh from a UTF-8 path, with scene transforms applied and positions normalized.
Asset loadAsset(const std::string& utf8Path);
Asset loadAsset(const std::string& utf8Path, ImportSpace space);
// Return a built-in UV-free sphere for testing the full LOD workflow.
Asset sampleAsset();
// Quality is clamped to [minimumQuality, maximumQuality].
// Sphere mode returns the source mesh if the analysis found it ineligible.
// Planar and Seam target planarTriangleTarget() and preserve unsupported triangles.
Mesh generateMesh(const Asset& asset, Mode mode, int quality);
float baselineFraction(int quality);
std::size_t planarTriangleTarget(std::size_t sourceTriangles, int quality);

Mesh loadMesh(const std::string& path);
// Scene space retains the transforms and units provided by the importer for aligned mesh comparisons.
Mesh loadMesh(const std::string& path, ImportSpace space);
SphereFit fitSphere(const Mesh& mesh);
SphereEligibility sphereEligibility(const Mesh& mesh, const SphereFit& fit);
Mesh sphereMesh(const SphereFit& fit, int segments, int rings);
Mesh simplifyMesh(const Mesh& mesh, float fraction);
struct GeometryReduction {
    std::vector<uint32_t> indices;
    float estimatedError = 0.0f;
};
// Simplify an indexed position buffer using original vertices and an absolute
// meshoptimizer error budget. Callers validate application-specific shape and topology.
GeometryReduction simplifyGeometry(std::span<const float> positions, std::span<const uint32_t> indices,
                                   std::span<const uint8_t> lockedVertices, std::size_t targetTriangles,
                                   float maxError, bool lockBoundary);
void normalizeMesh(Mesh& mesh);

class PlanarAnalysis {
public:
    const Mesh& original() const;
    std::size_t minimumTriangles() const;

private:
    struct Data;
    std::shared_ptr<const Data> data_;
    explicit PlanarAnalysis(std::shared_ptr<const Data> data);
    friend PlanarAnalysis analyzePlanarMesh(Mesh source);
    friend Mesh generatePlanarMesh(const PlanarAnalysis&, std::size_t targetTriangles);
};

// Own the source and prepare conservative, reusable retriangulation edits.
PlanarAnalysis analyzePlanarMesh(Mesh source);
// Apply enough edits to meet the requested budget when possible, retaining unsupported regions.
// The actual count can remain above the target.
Mesh generatePlanarMesh(const PlanarAnalysis& analysis, std::size_t targetTriangles);
Mesh simplifyPlanarMesh(const Mesh& mesh, std::size_t targetTriangles);

class SeamAnalysis {
public:
    const Mesh& original() const;
    std::size_t minimumTriangles() const;
    // Counts geometric positions whose incident seam sides can be edited together.
    std::size_t removablePositions() const;

private:
    struct Data;
    std::shared_ptr<const Data> data_;
    explicit SeamAnalysis(std::shared_ptr<const Data> data);
    friend SeamAnalysis analyzeSeamMesh(Mesh source);
    friend Mesh generateSeamMesh(const SeamAnalysis&, std::size_t targetTriangles);
};

// Prepare coupled edits that shorten straight boundary chains while retaining each side's attributes.
SeamAnalysis analyzeSeamMesh(Mesh source);
// Apply the prepared edits toward the budget. Unsupported regions retain their source triangles.
Mesh generateSeamMesh(const SeamAnalysis& analysis, std::size_t targetTriangles);
Mesh simplifySeamMesh(const Mesh& mesh, std::size_t targetTriangles);

} // namespace lodestar
