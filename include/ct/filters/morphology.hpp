#pragma once

#include "ct/image2d.hpp"
#include "ct/volume3d.hpp"
#include <string>

namespace ct {

enum class MorphOp {
    Erosion,
    Dilation,
    Opening,
    Closing,
    WhiteTopHat,
    BlackTopHat
};

enum class MorphElement {
    Box,
    Ball
};

struct MorphologyParams2D {
    MorphOp op = MorphOp::Opening;
    MorphElement element = MorphElement::Box;
    int radius_x = 1;
    int radius_y = 1;

    MorphologyParams2D() = default;
    MorphologyParams2D(MorphOp o, MorphElement elem = MorphElement::Box, int rx = 1, int ry = 1)
        : op(o), element(elem), radius_x(rx), radius_y(ry) {}
};

struct MorphologyParams3D {
    MorphOp op = MorphOp::Opening;
    MorphElement element = MorphElement::Box;
    int radius_x = 1;
    int radius_y = 1;
    int radius_z = 1;

    MorphologyParams3D() = default;
    MorphologyParams3D(MorphOp o, MorphElement elem = MorphElement::Box, int rx = 1, int ry = 1, int rz = 1)
        : op(o), element(elem), radius_x(rx), radius_y(ry), radius_z(rz) {}
};

// Generic morphology runners
Image2D morphology_filter(const Image2D& input, const MorphologyParams2D& params);
Volume3D morphology_filter(const Volume3D& input, const MorphologyParams3D& params);

// 2D Convenience functions
Image2D grayscale_erosion(const Image2D& input, int radius = 1, const std::string& element = "box");
Image2D grayscale_dilation(const Image2D& input, int radius = 1, const std::string& element = "box");
Image2D grayscale_opening(const Image2D& input, int radius = 1, const std::string& element = "box");
Image2D grayscale_closing(const Image2D& input, int radius = 1, const std::string& element = "box");
Image2D white_tophat(const Image2D& input, int radius = 1, const std::string& element = "box");
Image2D black_tophat(const Image2D& input, int radius = 1, const std::string& element = "box");

// 3D Convenience functions
Volume3D grayscale_erosion(const Volume3D& input, int radius = 1, const std::string& element = "box");
Volume3D grayscale_dilation(const Volume3D& input, int radius = 1, const std::string& element = "box");
Volume3D grayscale_opening(const Volume3D& input, int radius = 1, const std::string& element = "box");
Volume3D grayscale_closing(const Volume3D& input, int radius = 1, const std::string& element = "box");
Volume3D white_tophat(const Volume3D& input, int radius = 1, const std::string& element = "box");
Volume3D black_tophat(const Volume3D& input, int radius = 1, const std::string& element = "box");

// Slice-by-slice wrappers
Volume3D grayscale_erosion_slice_by_slice(const Volume3D& input, int radius = 1, const std::string& element = "box");
Volume3D grayscale_dilation_slice_by_slice(const Volume3D& input, int radius = 1, const std::string& element = "box");
Volume3D grayscale_opening_slice_by_slice(const Volume3D& input, int radius = 1, const std::string& element = "box");
Volume3D grayscale_closing_slice_by_slice(const Volume3D& input, int radius = 1, const std::string& element = "box");
Volume3D white_tophat_slice_by_slice(const Volume3D& input, int radius = 1, const std::string& element = "box");
Volume3D black_tophat_slice_by_slice(const Volume3D& input, int radius = 1, const std::string& element = "box");

}  // namespace ct

using ct::grayscale_erosion;
using ct::grayscale_dilation;
using ct::grayscale_opening;
using ct::grayscale_closing;
using ct::white_tophat;
using ct::black_tophat;
