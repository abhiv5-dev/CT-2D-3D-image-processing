#include "ct/image2d.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

Image2D::Image2D(int width, int height)
    : width_(width), height_(height) {
    if (width_ > 0 && height_ > 0) {
        data_.assign(static_cast<size_t>(width_) * static_cast<size_t>(height_), 0.0f);
    }
}

Image2D::Image2D(int width, int height, float fill_value)
    : width_(width), height_(height) {
    if (width_ > 0 && height_ > 0) {
        data_.assign(static_cast<size_t>(width_) * static_cast<size_t>(height_), fill_value);
    }
}

float& Image2D::at(int row, int col) {
    return data_[static_cast<size_t>(row) * static_cast<size_t>(width_) + static_cast<size_t>(col)];
}

float Image2D::at(int row, int col) const {
    return data_[static_cast<size_t>(row) * static_cast<size_t>(width_) + static_cast<size_t>(col)];
}

float Image2D::min() const {
    if (empty()) {
        throw std::runtime_error("min() called on empty Image2D");
    }
    return *std::min_element(data_.begin(), data_.end());
}

float Image2D::max() const {
    if (empty()) {
        throw std::runtime_error("max() called on empty Image2D");
    }
    return *std::max_element(data_.begin(), data_.end());
}

double Image2D::mean() const {
    if (empty()) {
        throw std::runtime_error("mean() called on empty Image2D");
    }
    double sum = 0.0;
    for (float v : data_) {
        sum += static_cast<double>(v);
    }
    return sum / static_cast<double>(data_.size());
}
