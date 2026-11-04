#pragma once

#include "ct/image2d.hpp"
#include "ct/volume3d.hpp"
#include <array>

namespace ct {

struct BilateralParams2D {
    double sigma_spatial = 2.0;
    double sigma_range = 0.1;
    int radius = -1;  // if <= 0, ceil(3 * sigma_spatial)
    bool exact = true; // if false, use fast range weight lookup table

    BilateralParams2D() = default;
    BilateralParams2D(double spatial, double range, int r = -1, bool exact_w = true)
        : sigma_spatial(spatial), sigma_range(range), radius(r), exact(exact_w) {}
};

struct BilateralParams3D {
    std::array<double, 3> sigma_spatial{2.0, 2.0, 2.0}; // (x, y, z)
    double sigma_range = 0.1;
    std::array<int, 3> radius{-1, -1, -1};
    bool exact = true; // if false, use fast range weight lookup table

    BilateralParams3D() = default;
    BilateralParams3D(double spatial, double range, int r = -1, bool exact_w = true)
        : sigma_spatial{spatial, spatial, spatial}, sigma_range(range), radius{r, r, r}, exact(exact_w) {}
    BilateralParams3D(std::array<double, 3> spatial, double range, std::array<int, 3> r = {-1, -1, -1}, bool exact_w = true)
        : sigma_spatial(spatial), sigma_range(range), radius(r), exact(exact_w) {}
};

Image2D bilateral_filter(const Image2D& input, const BilateralParams2D& params);
Image2D bilateral_filter(const Image2D& input, double sigma_spatial, double sigma_range, int radius = -1, bool exact = true);

Volume3D bilateral_filter(const Volume3D& input, const BilateralParams3D& params);
Volume3D bilateral_filter(const Volume3D& input, double sigma_spatial, double sigma_range, int radius = -1, bool exact = true);

}  // namespace ct

using ct::bilateral_filter;
