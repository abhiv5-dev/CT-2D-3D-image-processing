#pragma once

#include "ct/image2d.hpp"
#include "ct/volume3d.hpp"

namespace ct {

struct WienerParams2D {
    int radius = 1;                 // Window radius: size = 2 * radius + 1
    double noise_variance = -1.0;   // If < 0, defaults to mean of local variances

    WienerParams2D() = default;
    WienerParams2D(int r, double nv = -1.0) : radius(r), noise_variance(nv) {}
};

struct WienerParams3D {
    int radius = 1;                 // Window radius: size = 2 * radius + 1 per axis
    double noise_variance = -1.0;   // If < 0, defaults to mean of local variances

    WienerParams3D() = default;
    WienerParams3D(int r, double nv = -1.0) : radius(r), noise_variance(nv) {}
};

// 2D Adaptive Wiener filter with reflect boundary and separable running sums
Image2D wiener_filter(const Image2D& input, const WienerParams2D& params);
Image2D wiener_filter(const Image2D& input, int radius = 1, double noise_variance = -1.0);

// 3D Adaptive Wiener filter with reflect boundary and separable running sums
Volume3D wiener_filter(const Volume3D& input, const WienerParams3D& params);
Volume3D wiener_filter(const Volume3D& input, int radius = 1, double noise_variance = -1.0);

// Slice-by-slice 2D Adaptive Wiener filter applied along z-axis of a 3D volume
Volume3D wiener_filter_slice_by_slice(const Volume3D& input, int radius = 1, double noise_variance = -1.0);

}  // namespace ct

using ct::wiener_filter;
using ct::wiener_filter_slice_by_slice;
