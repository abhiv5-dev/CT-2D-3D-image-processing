#include "ct/intensity.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace ct {

Image2D apply_windowing(const Image2D& input, double window_center, double window_width,
                        float out_min, float out_max) {
    if (input.empty()) {
        throw std::invalid_argument("apply_windowing: input Image2D is empty");
    }
    if (!(window_width > 0.0)) {
        throw std::invalid_argument("apply_windowing: window_width must be positive");
    }

    const double low = window_center - 0.5 * window_width;
    const double out_range = static_cast<double>(out_max - out_min);

    Image2D output(input.width(), input.height());
    output.metadata() = input.metadata();

    for (size_t i = 0; i < input.size(); ++i) {
        double v = static_cast<double>(input.data()[i]);
        double norm = std::clamp((v - low) / window_width, 0.0, 1.0);
        output.data()[i] = static_cast<float>(out_min + norm * out_range);
    }
    return output;
}

Volume3D apply_windowing(const Volume3D& input, double window_center, double window_width,
                         float out_min, float out_max) {
    if (input.empty()) {
        throw std::invalid_argument("apply_windowing: input Volume3D is empty");
    }
    if (!(window_width > 0.0)) {
        throw std::invalid_argument("apply_windowing: window_width must be positive");
    }

    const double low = window_center - 0.5 * window_width;
    const double out_range = static_cast<double>(out_max - out_min);

    Volume3D output(input.width(), input.height(), input.depth());
    output.metadata() = input.metadata();

    for (size_t i = 0; i < input.size(); ++i) {
        double v = static_cast<double>(input.data()[i]);
        double norm = std::clamp((v - low) / window_width, 0.0, 1.0);
        output.data()[i] = static_cast<float>(out_min + norm * out_range);
    }
    return output;
}

Image2D linear_stretch(const Image2D& input, double in_min, double in_max,
                       float out_min, float out_max) {
    if (input.empty()) {
        throw std::invalid_argument("linear_stretch: input Image2D is empty");
    }
    if (std::isnan(in_min)) in_min = input.min();
    if (std::isnan(in_max)) in_max = input.max();

    Image2D output(input.width(), input.height());
    output.metadata() = input.metadata();

    if (in_max <= in_min) {
        float mid_val = 0.5f * (out_min + out_max);
        std::fill(output.data().begin(), output.data().end(), mid_val);
        return output;
    }

    const double in_range = in_max - in_min;
    const double out_range = static_cast<double>(out_max - out_min);

    for (size_t i = 0; i < input.size(); ++i) {
        double v = static_cast<double>(input.data()[i]);
        double norm = std::clamp((v - in_min) / in_range, 0.0, 1.0);
        output.data()[i] = static_cast<float>(out_min + norm * out_range);
    }
    return output;
}

Volume3D linear_stretch(const Volume3D& input, double in_min, double in_max,
                        float out_min, float out_max) {
    if (input.empty()) {
        throw std::invalid_argument("linear_stretch: input Volume3D is empty");
    }
    if (std::isnan(in_min)) in_min = input.min();
    if (std::isnan(in_max)) in_max = input.max();

    Volume3D output(input.width(), input.height(), input.depth());
    output.metadata() = input.metadata();

    if (in_max <= in_min) {
        float mid_val = 0.5f * (out_min + out_max);
        std::fill(output.data().begin(), output.data().end(), mid_val);
        return output;
    }

    const double in_range = in_max - in_min;
    const double out_range = static_cast<double>(out_max - out_min);

    for (size_t i = 0; i < input.size(); ++i) {
        double v = static_cast<double>(input.data()[i]);
        double norm = std::clamp((v - in_min) / in_range, 0.0, 1.0);
        output.data()[i] = static_cast<float>(out_min + norm * out_range);
    }
    return output;
}

Image2D apply_clahe(const Image2D& input, int tile_grid_size, double clip_limit) {
    if (input.empty()) {
        throw std::invalid_argument("apply_clahe: input Image2D is empty");
    }
    if (tile_grid_size <= 0) {
        tile_grid_size = 8;
    }
    if (!(clip_limit >= 1.0)) {
        clip_limit = 2.0;
    }

    const int W = input.width();
    const int H = input.height();
    const float min_v = input.min();
    const float max_v = input.max();
    if (max_v <= min_v) {
        return input;
    }

    const double in_range = max_v - min_v;
    const int num_bins = 256;

    // Grid dimensions
    int num_tiles_x = std::min(tile_grid_size, W);
    int num_tiles_y = std::min(tile_grid_size, H);

    double tile_w = static_cast<double>(W) / num_tiles_x;
    double tile_h = static_cast<double>(H) / num_tiles_y;

    // Compute CDF for each tile
    std::vector<std::vector<double>> cdfs(static_cast<size_t>(num_tiles_y * num_tiles_x),
                                         std::vector<double>(num_bins, 0.0));

    for (int ty = 0; ty < num_tiles_y; ++ty) {
        int y_start = static_cast<int>(ty * tile_h);
        int y_end = static_cast<int>((ty + 1) * tile_h);
        if (ty == num_tiles_y - 1) y_end = H;

        for (int tx = 0; tx < num_tiles_x; ++tx) {
            int x_start = static_cast<int>(tx * tile_w);
            int x_end = static_cast<int>((tx + 1) * tile_w);
            if (tx == num_tiles_x - 1) x_end = W;

            int tile_area = (y_end - y_start) * (x_end - x_start);
            if (tile_area <= 0) continue;

            std::vector<int> hist(num_bins, 0);
            for (int y = y_start; y < y_end; ++y) {
                for (int x = x_start; x < x_end; ++x) {
                    double norm = (input.at(y, x) - min_v) / in_range;
                    int bin = std::clamp(static_cast<int>(norm * (num_bins - 1) + 0.5), 0, num_bins - 1);
                    hist[static_cast<size_t>(bin)]++;
                }
            }

            // Clip histogram
            int actual_clip = std::max(1, static_cast<int>(clip_limit * tile_area / num_bins));
            int excess = 0;
            for (int b = 0; b < num_bins; ++b) {
                if (hist[static_cast<size_t>(b)] > actual_clip) {
                    excess += (hist[static_cast<size_t>(b)] - actual_clip);
                    hist[static_cast<size_t>(b)] = actual_clip;
                }
            }

            int bonus = excess / num_bins;
            int rem = excess % num_bins;
            for (int b = 0; b < num_bins; ++b) {
                hist[static_cast<size_t>(b)] += bonus + (b < rem ? 1 : 0);
            }

            // Compute normalized CDF
            auto& cdf = cdfs[static_cast<size_t>(ty * num_tiles_x + tx)];
            double running_sum = 0.0;
            for (int b = 0; b < num_bins; ++b) {
                running_sum += hist[static_cast<size_t>(b)];
                cdf[static_cast<size_t>(b)] = running_sum / tile_area;
            }
        }
    }

    Image2D output(W, H);
    output.metadata() = input.metadata();

    for (int y = 0; y < H; ++y) {
        double norm_y = (y + 0.5) / tile_h - 0.5;
        int y1 = std::clamp(static_cast<int>(std::floor(norm_y)), 0, num_tiles_y - 1);
        int y2 = std::clamp(y1 + 1, 0, num_tiles_y - 1);
        double dy = std::clamp(norm_y - y1, 0.0, 1.0);

        for (int x = 0; x < W; ++x) {
            double norm_x = (x + 0.5) / tile_w - 0.5;
            int x1 = std::clamp(static_cast<int>(std::floor(norm_x)), 0, num_tiles_x - 1);
            int x2 = std::clamp(x1 + 1, 0, num_tiles_x - 1);
            double dx = std::clamp(norm_x - x1, 0.0, 1.0);

            double norm_val = (input.at(y, x) - min_v) / in_range;
            int bin = std::clamp(static_cast<int>(norm_val * (num_bins - 1) + 0.5), 0, num_bins - 1);

            double c11 = cdfs[static_cast<size_t>(y1 * num_tiles_x + x1)][static_cast<size_t>(bin)];
            double c12 = cdfs[static_cast<size_t>(y1 * num_tiles_x + x2)][static_cast<size_t>(bin)];
            double c21 = cdfs[static_cast<size_t>(y2 * num_tiles_x + x1)][static_cast<size_t>(bin)];
            double c22 = cdfs[static_cast<size_t>(y2 * num_tiles_x + x2)][static_cast<size_t>(bin)];

            // Bilinear interpolation
            double eq = (1.0 - dy) * ((1.0 - dx) * c11 + dx * c12) +
                        dy * ((1.0 - dx) * c21 + dx * c22);

            output.at(y, x) = static_cast<float>(min_v + eq * in_range);
        }
    }

    return output;
}

Volume3D apply_clahe(const Volume3D& input, int tile_grid_size, double clip_limit) {
    if (input.empty()) {
        throw std::invalid_argument("apply_clahe: input Volume3D is empty");
    }
    Volume3D output(input.width(), input.height(), input.depth());
    output.metadata() = input.metadata();
    for (int z = 0; z < input.depth(); ++z) {
        output.set_slice(z, apply_clahe(input.slice(z), tile_grid_size, clip_limit));
    }
    return output;
}

std::pair<double, double> compute_percentiles_1_99(const Image2D& input) {
    if (input.empty()) {
        throw std::invalid_argument("compute_percentiles_1_99: input Image2D is empty");
    }
    std::vector<float> vals = input.data();
    size_t n = vals.size();
    size_t idx1 = static_cast<size_t>(0.01 * (n - 1));
    size_t idx99 = static_cast<size_t>(0.99 * (n - 1));
    std::nth_element(vals.begin(), vals.begin() + idx1, vals.end());
    float p1 = vals[idx1];
    std::nth_element(vals.begin() + idx1 + 1, vals.begin() + idx99, vals.end());
    float p99 = vals[idx99];
    if (p99 <= p1) {
        p99 = p1 + 1e-6f;
    }
    return {static_cast<double>(p1), static_cast<double>(p99)};
}

std::pair<double, double> compute_percentiles_1_99(const Volume3D& input) {
    if (input.empty()) {
        throw std::invalid_argument("compute_percentiles_1_99: input Volume3D is empty");
    }
    std::vector<float> vals = input.data();
    size_t n = vals.size();
    size_t idx1 = static_cast<size_t>(0.01 * (n - 1));
    size_t idx99 = static_cast<size_t>(0.99 * (n - 1));
    std::nth_element(vals.begin(), vals.begin() + idx1, vals.end());
    float p1 = vals[idx1];
    std::nth_element(vals.begin() + idx1 + 1, vals.begin() + idx99, vals.end());
    float p99 = vals[idx99];
    if (p99 <= p1) {
        p99 = p1 + 1e-6f;
    }
    return {static_cast<double>(p1), static_cast<double>(p99)};
}

}  // namespace ct
