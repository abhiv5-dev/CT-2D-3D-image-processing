#include "ct/volume3d.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

Volume3D::Volume3D(int width, int height, int depth)
    : width_(width), height_(height), depth_(depth) {
    if (width_ > 0 && height_ > 0 && depth_ > 0) {
        data_.assign(static_cast<size_t>(width_) * static_cast<size_t>(height_) * static_cast<size_t>(depth_), 0.0f);
    }
}

Volume3D::Volume3D(int width, int height, int depth, float fill_value)
    : width_(width), height_(height), depth_(depth) {
    if (width_ > 0 && height_ > 0 && depth_ > 0) {
        data_.assign(static_cast<size_t>(width_) * static_cast<size_t>(height_) * static_cast<size_t>(depth_), fill_value);
    }
}

float& Volume3D::at(int z, int y, int x) {
    size_t idx = (static_cast<size_t>(z) * static_cast<size_t>(height_) + static_cast<size_t>(y)) *
                 static_cast<size_t>(width_) + static_cast<size_t>(x);
    return data_[idx];
}

float Volume3D::at(int z, int y, int x) const {
    size_t idx = (static_cast<size_t>(z) * static_cast<size_t>(height_) + static_cast<size_t>(y)) *
                 static_cast<size_t>(width_) + static_cast<size_t>(x);
    return data_[idx];
}

Image2D Volume3D::slice(int z) const {
    if (z < 0 || z >= depth_) {
        throw std::out_of_range("Volume3D::slice: z index out of range (" + std::to_string(z) +
                                ", depth=" + std::to_string(depth_) + ")");
    }
    Image2D img(width_, height_);
    size_t slice_voxels = static_cast<size_t>(width_) * static_cast<size_t>(height_);
    const float* src = &data_[static_cast<size_t>(z) * slice_voxels];
    std::copy(src, src + slice_voxels, img.data().begin());

    img.metadata().spacing = {meta_.spacing[0], meta_.spacing[1]};
    img.metadata().spacing_is_known = meta_.spacing_is_known;
    img.metadata().origin = {meta_.origin[0], meta_.origin[1]};
    img.metadata().raw_dtype = meta_.raw_dtype;
    img.metadata().rescale_slope = meta_.rescale_slope;
    img.metadata().rescale_offset = meta_.rescale_offset;
    return img;
}

void Volume3D::set_slice(int z, const Image2D& image) {
    if (z < 0 || z >= depth_) {
        throw std::out_of_range("Volume3D::set_slice: z index out of range (" + std::to_string(z) +
                                ", depth=" + std::to_string(depth_) + ")");
    }
    if (image.width() != width_ || image.height() != height_) {
        throw std::invalid_argument("Volume3D::set_slice: slice dimensions mismatch (expected " +
                                    std::to_string(width_) + "x" + std::to_string(height_) +
                                    ", got " + std::to_string(image.width()) + "x" +
                                    std::to_string(image.height()) + ")");
    }
    size_t slice_voxels = static_cast<size_t>(width_) * static_cast<size_t>(height_);
    float* dst = &data_[static_cast<size_t>(z) * slice_voxels];
    std::copy(image.data().begin(), image.data().end(), dst);
}

float Volume3D::min() const {
    if (empty()) {
        throw std::runtime_error("min() called on empty Volume3D");
    }
    return *std::min_element(data_.begin(), data_.end());
}

float Volume3D::max() const {
    if (empty()) {
        throw std::runtime_error("max() called on empty Volume3D");
    }
    return *std::max_element(data_.begin(), data_.end());
}

double Volume3D::mean() const {
    if (empty()) {
        throw std::runtime_error("mean() called on empty Volume3D");
    }
    double sum = 0.0;
    for (float v : data_) {
        sum += static_cast<double>(v);
    }
    return sum / static_cast<double>(data_.size());
}
