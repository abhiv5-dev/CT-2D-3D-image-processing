#pragma once

#include "ct/image2d.hpp"
#include "ct/volume3d.hpp"

namespace ct {

struct NLMParams2D {
    int patch_radius = 2;   // 5x5 patch
    int search_radius = 5;  // 11x11 search window
    double h = 0.05;        // Filtering parameter

    NLMParams2D() = default;
    NLMParams2D(int patch_r, int search_r, double h_param)
        : patch_radius(patch_r), search_radius(search_r), h(h_param) {}
};

struct NLMParams3D {
    int patch_radius = 1;   // 3x3x3 patch
    int search_radius = 3;  // 7x7x7 search window
    double h = 0.05;

    NLMParams3D() = default;
    NLMParams3D(int patch_r, int search_r, double h_param)
        : patch_radius(patch_r), search_radius(search_r), h(h_param) {}
};

Image2D nlm_filter(const Image2D& input, const NLMParams2D& params);
Image2D nlm_filter(const Image2D& input, int patch_radius = 2, int search_radius = 5, double h = 0.05);

Volume3D nlm_filter(const Volume3D& input, const NLMParams3D& params);
Volume3D nlm_filter(const Volume3D& input, int patch_radius = 1, int search_radius = 3, double h = 0.05);

}  // namespace ct

using ct::nlm_filter;
