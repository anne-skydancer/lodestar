#include <lodestar/lodestar.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lodestar {

Asset loadAsset(const std::string& utf8Path) {
    return analyze(loadMesh(utf8Path));
}

Asset loadAsset(const std::string& utf8Path, ImportSpace space) {
    return analyze(loadMesh(utf8Path, space));
}

Asset sampleAsset() {
    SphereFit sphere;
    sphere.radius = 1.0f;
    sphere.valid = true;
    return analyze(sphereMesh(sphere, 64, 32));
}

float baselineFraction(int quality) {
    const int level = std::clamp(quality, minimumQuality, maximumQuality);
    const float interval = static_cast<float>(maximumQuality - minimumQuality);
    const float exponent = 5.0f * (level - maximumQuality) / interval;
    return std::exp2(exponent);
}

std::size_t planarTriangleTarget(std::size_t sourceTriangles, int quality) {
    const float fraction = baselineFraction(quality);
    if (sourceTriangles <= 1 || fraction >= 1.0f) return sourceTriangles;
    return std::max<std::size_t>(1, static_cast<std::size_t>(static_cast<double>(sourceTriangles) * fraction));
}

Mesh generateMesh(const Asset& asset, Mode mode, int quality) {
    switch (mode) {
    case Mode::Original:
        return asset.original;
    case Mode::Baseline:
        return simplifyMesh(asset.original, baselineFraction(quality));
    case Mode::Planar:
        return simplifyPlanarMesh(asset.original, planarTriangleTarget(asset.original.indices.size() / 3, quality));
    case Mode::Seam:
        return simplifySeamMesh(asset.original, planarTriangleTarget(asset.original.indices.size() / 3, quality));
    case Mode::Sphere: {
        if (asset.eligibility != SphereEligibility::Eligible) {
            return asset.original;
        }
        const int level = std::clamp(quality, minimumQuality, maximumQuality);
        return sphereMesh(asset.fit, level * 8, level * 4);
    }
    }
    throw std::invalid_argument("Unknown geometry mode");
}

} // namespace lodestar
