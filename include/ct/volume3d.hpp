#pragma once

#include "ct/image2d.hpp"
#include <array>
#include <string>
#include <vector>
#include <stdexcept>

// Metadata for a 3D CT volume.
struct VolumeMetadata {
    std::array<double, 3> spacing{1.0, 1.0, 1.0};  // physical units per voxel (x, y, z)
    bool spacing_is_known = false;
    std::array<double, 3> origin{0.0, 0.0, 0.0};
    std::string raw_dtype = "float32";
    double rescale_slope = 1.0;
    double rescale_offset = 0.0;
};

// A 3D CT volume stored in x-fastest row-major order: index = (z * height + y) * width + x.
class Volume3D {
public:
    Volume3D() = default;
    Volume3D(int width, int height, int depth);
    Volume3D(int width, int height, int depth, float fill_value);

    // Voxel access: z = slice, y = row, x = col
    float& at(int z, int y, int x);
    float at(int z, int y, int x) const;

    int width() const noexcept { return width_; }
    int height() const noexcept { return height_; }
    int depth() const noexcept { return depth_; }
    bool empty() const noexcept { return data_.empty() || width_ <= 0 || height_ <= 0 || depth_ <= 0; }
    size_t size() const noexcept { return data_.size(); }

    const VolumeMetadata& metadata() const noexcept { return meta_; }
    VolumeMetadata& metadata() noexcept { return meta_; }

    std::vector<float>& data() noexcept { return data_; }
    const std::vector<float>& data() const noexcept { return data_; }

    // Slice access
    Image2D slice(int z) const;
    void set_slice(int z, const Image2D& image);

    float min() const;
    float max() const;
    double mean() const;

private:
    int width_ = 0;
    int height_ = 0;
    int depth_ = 0;
    VolumeMetadata meta_;
    std::vector<float> data_;
};
