#pragma once

#include <string>
#include <vector>
#include <array>
#include <stdexcept>

// Metadata associated with a 2D CT slice.
// Does NOT include width/height so callers cannot desynchronise dimensions
// from the underlying pixel buffer.
struct ImageMetadata {
    std::array<double, 2> spacing{1.0, 1.0};   // physical units per pixel (x, y)
    bool spacing_is_known = false;             // true if physical spacing was reliably provided
    std::array<double, 2> origin{0.0, 0.0};
    std::string raw_dtype = "float32";         // dtype as stored in the source file
    double rescale_slope = 1.0;                // true_value = raw * rescale_slope + rescale_offset
    double rescale_offset = 0.0;
};

// A single 2D image, stored as row-major float32 in physical/rescaled units.
class Image2D {
public:
    Image2D() = default;
    Image2D(int width, int height);
    Image2D(int width, int height, float fill_value);

    float& at(int row, int col);
    float at(int row, int col) const;

    int width() const noexcept { return width_; }
    int height() const noexcept { return height_; }
    bool empty() const noexcept { return data_.empty() || width_ <= 0 || height_ <= 0; }
    size_t size() const noexcept { return data_.size(); }

    const ImageMetadata& metadata() const noexcept { return meta_; }
    ImageMetadata& metadata() noexcept { return meta_; }

    std::vector<float>& data() noexcept { return data_; }
    const std::vector<float>& data() const noexcept { return data_; }

    float min() const;
    float max() const;
    double mean() const;

private:
    int width_ = 0;
    int height_ = 0;
    ImageMetadata meta_;
    std::vector<float> data_;
};
