#pragma once

#include "ct/image2d.hpp"
#include <limits>
#include <string>

// Writes `image` out as an 8-bit grayscale TIFF that standard image viewers can display.
//
// Normalises physical float values to [0, 255].
// Values outside [display_min, display_max] are clipped.
// NaN values are mapped to 0 before casting.
// For a constant image (where display_max <= display_min), outputs uniform mid-gray (128).
// Guarded against empty images.
void write_preview_tiff(const Image2D& image,
                        const std::string& path,
                        double display_min = std::numeric_limits<double>::quiet_NaN(),
                        double display_max = std::numeric_limits<double>::quiet_NaN());

namespace ct {
    using ::write_preview_tiff;
}
