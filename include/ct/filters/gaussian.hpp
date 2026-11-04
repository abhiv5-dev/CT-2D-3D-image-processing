#pragma once

#include "ct/image2d.hpp"
#include "ct/volume3d.hpp"
#include "ct/parallel.hpp"
#include <array>
#include <vector>
#include <functional>

namespace ct {

// Reflects an out-of-bounds index back into [0, size-1].
// Edge pixel is repeated: index -1 maps to 0, -2 maps to 1,
// index size maps to size-1, size+1 maps to size-2, etc.
// Matches scipy.ndimage's default mode='reflect' (d c b a | a b c d | d c b a).
int reflect_index(int index, int size) noexcept;

// Builds a normalized 1D Gaussian kernel with (2*radius + 1) taps centered at radius.
// Sum of weights is normalized to 1.0.
std::vector<double> make_gaussian_kernel(double sigma, int radius);

struct GaussianParams2D {
    double sigma_x = 1.0;
    double sigma_y = 1.0;
    int radius_x = -1;  // if <= 0, auto-picked as ceil(3 * sigma)
    int radius_y = -1;

    GaussianParams2D() = default;
    GaussianParams2D(double sigma, int radius = -1)
        : sigma_x(sigma), sigma_y(sigma), radius_x(radius), radius_y(radius) {}
    GaussianParams2D(double sx, double sy, int rx, int ry)
        : sigma_x(sx), sigma_y(sy), radius_x(rx), radius_y(ry) {}
};

struct GaussianParams3D {
    std::array<double, 3> sigma{1.0, 1.0, 1.0};   // (x, y, z)
    std::array<int, 3> radius{-1, -1, -1};        // (x, y, z); if <= 0, ceil(3 * sigma)

    GaussianParams3D() = default;
    GaussianParams3D(double s, int r = -1)
        : sigma{s, s, s}, radius{r, r, r} {}
    GaussianParams3D(double sx, double sy, double sz, int rx = -1, int ry = -1, int rz = -1)
        : sigma{sx, sy, sz}, radius{rx, ry, rz} {}
    GaussianParams3D(std::array<double, 3> s, std::array<int, 3> r)
        : sigma(s), radius(r) {}
};

// 2D Gaussian blur.
//
// NOTE: Default radius ceil(3 * sigma) differs from scipy.ndimage's default
// (int(4 * sigma + 0.5)). When validating numerical equivalence against scipy,
// provide an explicit radius matching scipy's kernel size.
Image2D gaussian_blur(const Image2D& input, const GaussianParams2D& params);
Image2D gaussian_blur(const Image2D& input, double sigma, int radius = -1);

// 3D Gaussian blur (separable along x, y, and z with cache-friendly sequential passes).
Volume3D gaussian_blur(const Volume3D& input, const GaussianParams3D& params);
Volume3D gaussian_blur(const Volume3D& input, double sigma, int radius = -1);

// Naive reference implementation based on initial commit 20d0f07.
// Uses reflect_index on every tap and strided column/slice access without interior/border split.
// Used for benchmark baselines and numerical equivalence testing.
Image2D gaussian_blur_naive(const Image2D& input, const GaussianParams2D& params);
Image2D gaussian_blur_naive(const Image2D& input, double sigma, int radius = -1);
Volume3D gaussian_blur_naive(const Volume3D& input, const GaussianParams3D& params);
Volume3D gaussian_blur_naive(const Volume3D& input, double sigma, int radius = -1);

// Slice-wise wrapper: applies a 2D filter to each slice of a 3D volume independently.
template <typename FilterFunc>
Volume3D apply_slice_by_slice(const Volume3D& volume, FilterFunc&& filter_2d) {
    if (volume.empty()) {
        throw std::invalid_argument("apply_slice_by_slice: input volume is empty");
    }
    const int W = volume.width();
    const int H = volume.height();
    const int D = volume.depth();
    const size_t slice_size = static_cast<size_t>(W) * H;
    Volume3D result(W, H, D);
    result.metadata() = volume.metadata();

    parallel_for(0, D, [&](int z) {
        Image2D s(W, H);
        const float* src = volume.data().data() + static_cast<size_t>(z) * slice_size;
        std::copy(src, src + slice_size, s.data().data());
        Image2D filtered_s = filter_2d(s);
        float* dst = result.data().data() + static_cast<size_t>(z) * slice_size;
        std::copy(filtered_s.data().data(), filtered_s.data().data() + slice_size, dst);
    }, slice_size * 5);

    return result;
}

}  // namespace ct

using ct::gaussian_blur;
using ct::gaussian_blur_naive;
