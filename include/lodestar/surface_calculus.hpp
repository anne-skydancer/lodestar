#pragma once

#include <glm/glm.hpp>

#include <array>
#include <cstddef>
#include <span>

namespace lodestar {

// Cubic triangular Bezier control points in the order
// (300), (030), (003), (210), (120), (201), (102), (021), (012), (111).
// The barycentric coordinates are (1-u-v, u, v).
struct BezierTriangle {
    std::array<glm::dvec3, 10> controls;
};

struct PolynomialSurface {
    // Basis 1, u, v, u^2, uv, v^2, u^3, u^2v, uv^2, v^3.
    std::array<glm::dvec3, 10> coefficients;
    double maximumVertexResidual = 0.0;
    unsigned degree = 0;

    PolynomialSurface() { coefficients.fill(glm::dvec3(0.0)); }
};

// Fit the lowest adequate degree up to three using column-pivoted QR.
// Vertex residuals do not bound the fit between samples. Callers retain their
// High reference and check patch-to-triangle correspondence independently.
PolynomialSurface fitSurface(std::span<const glm::dvec2> parameters,
                             std::span<const glm::dvec3> positions);
BezierTriangle surfacePatch(const PolynomialSurface& surface,
                            const std::array<glm::dvec2, 3>& domain);

struct SurfaceDifferential {
    glm::dvec3 position{0.0};
    glm::dvec3 du{0.0};
    glm::dvec3 dv{0.0};
    glm::dvec3 duu{0.0};
    glm::dvec3 duv{0.0};
    glm::dvec3 dvv{0.0};
};

SurfaceDifferential differentiateSurface(const BezierTriangle& patch, double u, double v);

struct SurfaceIntegral {
    double area = 0.0;
    // Integral of max(abs(k1), abs(k2)) dA, where k1 and k2 are principal curvatures.
    double absoluteCurvature = 0.0;
    double estimatedAreaError = 0.0;
    double estimatedCurvatureError = 0.0;
    std::size_t evaluations = 0;
    bool converged = true;
};

// Adaptive numerical integration over the complete triangular parameter domain.
// Error estimates compare successive quadrature refinements and are not proofs.
// Singular sampled parameterizations and nonfinite data throw invalid_argument.
SurfaceIntegral integrateSurface(const BezierTriangle& patch, double relativeTolerance = 1e-5,
                                 unsigned maximumDepth = 8);

struct SurfaceLodBudget {
    SurfaceIntegral integral;
    double worldError = 0.0;
    // Continuous interior density estimates. Boundaries, seams and topology can
    // require additional vertices. These are planning estimates, not certificates.
    double triangleDensityIntegral = 0.0;
    double interiorVertexDensityIntegral = 0.0;
    std::size_t suggestedTriangles = 0;
};

// Uses the local equilateral interpolation model h^2 ~= 6*error/abs(k_max)
// and triangle area sqrt(3)*h^2/4. Actual tessellation checks control-hull bounds.
// Each supplied patch retains at least one triangle in the suggested budget.
// All patches must be in the same coordinate system, with disjoint interiors.
// The caller establishes that the patch model preserves its authoritative input.
SurfaceLodBudget calculateSurfaceLod(std::span<const BezierTriangle> patches,
                                    double worldUnitsPerPixel, double maximumPixelError,
                                    double relativeTolerance = 1e-5, unsigned maximumDepth = 8);

} // namespace lodestar
