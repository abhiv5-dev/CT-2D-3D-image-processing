#pragma once

#include "ct/image2d.hpp"
#include "ct/volume3d.hpp"
#include <limits>

namespace ct {

// Windowing (level / width) for CT data:
// Linearly maps [center - width/2, center + width/2] to [out_min, out_max],
// clamping values outside the window.
Image2D apply_windowing(const Image2D& input, double window_center, double window_width,
                        float out_min = 0.0f, float out_max = 1.0f);
Volume3D apply_windowing(const Volume3D& input, double window_center, double window_width,
                         float out_min = 0.0f, float out_max = 1.0f);

// Linear contrast stretch: maps [in_min, in_max] to [out_min, out_max].
// If in_min or in_max are NaN, they default to input.min() and input.max().
Image2D linear_stretch(const Image2D& input,
                       double in_min = std::numeric_limits<double>::quiet_NaN(),
                       double in_max = std::numeric_limits<double>::quiet_NaN(),
                       float out_min = 0.0f, float out_max = 1.0f);
Volume3D linear_stretch(const Volume3D& input,
                        double in_min = std::numeric_limits<double>::quiet_NaN(),
                        double in_max = std::numeric_limits<double>::quiet_NaN(),
                        float out_min = 0.0f, float out_max = 1.0f);

// Contrast-Limited Adaptive Histogram Equalization (CLAHE) for 2D slices.
// Enhances local contrast while limiting noise amplification.
Image2D apply_clahe(const Image2D& input, int tile_grid_size = 8, double clip_limit = 2.0);
Volume3D apply_clahe(const Volume3D& input, int tile_grid_size = 8, double clip_limit = 2.0);

// Computes 1st and 99th percentile intensity values (robust display window).
std::pair<double, double> compute_percentiles_1_99(const Image2D& input);
std::pair<double, double> compute_percentiles_1_99(const Volume3D& input);

}  // namespace ct

using ct::apply_windowing;
using ct::linear_stretch;
using ct::apply_clahe;
using ct::compute_percentiles_1_99;
