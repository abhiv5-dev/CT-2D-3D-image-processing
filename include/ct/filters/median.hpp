#pragma once

#include "ct/image2d.hpp"
#include "ct/volume3d.hpp"
#include <array>

namespace ct {

struct MedianParams2D {
    int radius_x = 1;
    int radius_y = 1;

    MedianParams2D() = default;
    MedianParams2D(int radius) : radius_x(radius), radius_y(radius) {}
    MedianParams2D(int rx, int ry) : radius_x(rx), radius_y(ry) {}
};

struct MedianParams3D {
    std::array<int, 3> radius{1, 1, 1};  // (x, y, z)

    MedianParams3D() = default;
    MedianParams3D(int radius) : radius{radius, radius, radius} {}
    MedianParams3D(int rx, int ry, int rz) : radius{rx, ry, rz} {}
};

// 2D and 3D median filter.
// NaN Handling:
// If the local filter window contains NaN values, NaNs are sorted to the end using a strict
// weak ordering predicate, preventing undefined behavior in std::nth_element. If at least half
// the window consists of NaNs, the output is NaN; otherwise, the median of the finite values is returned.
Image2D median_filter(const Image2D& input, const MedianParams2D& params);
Image2D median_filter(const Image2D& input, int radius = 1);

Volume3D median_filter(const Volume3D& input, const MedianParams3D& params);
Volume3D median_filter(const Volume3D& input, int radius = 1);

}  // namespace ct

using ct::median_filter;
