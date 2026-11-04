#include "ct/filters/bilateral.hpp"
#include "ct/filters/gaussian.hpp"  // for ct::reflect_index
#include "ct/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace ct {

Image2D bilateral_filter(const Image2D& input, const BilateralParams2D& params) {
    if (input.empty()) {
        throw std::invalid_argument("bilateral_filter: input Image2D is empty");
    }
    if (!(params.sigma_spatial > 0.0) || !(params.sigma_range > 0.0)) {
        throw std::invalid_argument("bilateral_filter: sigma_spatial and sigma_range must be positive and non-NaN");
    }

    const int W = input.width();
    const int H = input.height();
    const int r = (params.radius > 0) ? params.radius : static_cast<int>(std::ceil(3.0 * params.sigma_spatial));

    const double two_spatial_sq = 2.0 * params.sigma_spatial * params.sigma_spatial;
    const double two_range_sq = 2.0 * params.sigma_range * params.sigma_range;

    // Precompute spatial weights (strictly read-only)
    const int k_dim = 2 * r + 1;
    std::vector<double> spatial_weights(static_cast<size_t>(k_dim * k_dim));
    for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
            double d_sq = static_cast<double>(dx * dx + dy * dy);
            spatial_weights[static_cast<size_t>((dy + r) * k_dim + (dx + r))] = std::exp(-d_sq / two_spatial_sq);
        }
    }

    // Precompute range weight lookup table if exact is false
    const size_t lut_size = 16384;
    const double cutoff_d = 6.0 * params.sigma_range;
    const double scale_d = static_cast<double>(lut_size - 1) / cutoff_d;
    std::vector<double> range_table;
    if (!params.exact) {
        range_table.resize(lut_size);
        for (size_t i = 0; i < lut_size; ++i) {
            double d = (static_cast<double>(i) / static_cast<double>(lut_size - 1)) * cutoff_d;
            range_table[i] = std::exp(-(d * d) / two_range_sq);
        }
    }

    auto get_range_weight = [&](double diff) noexcept -> double {
        if (params.exact) {
            return std::exp(-(diff * diff) / two_range_sq);
        }
        double abs_d = std::abs(diff);
        if (abs_d >= cutoff_d) {
            return 0.0;
        }
        double pos = abs_d * scale_d;
        size_t idx = static_cast<size_t>(pos);
        double frac = pos - static_cast<double>(idx);
        return range_table[idx] + frac * (range_table[idx + 1] - range_table[idx]);
    };

    Image2D output(W, H);
    output.metadata() = input.metadata();

    const bool can_use_interior = (W > 2 * r) && (H > 2 * r);

    parallel_for(0, H, [&](int y) {
        float* out_row = output.data().data() + static_cast<size_t>(y) * W;

        if (can_use_interior && y >= r && y < H - r) {
            // Left border: x in [0, r)
            for (int x = 0; x < r; ++x) {
                const float center_val = input.at(y, x);
                double sum_weights = 0.0;
                double sum_values = 0.0;

                for (int dy = -r; dy <= r; ++dy) {
                    const float* in_row = input.data().data() + static_cast<size_t>(y + dy) * W;
                    const size_t w_row_offset = static_cast<size_t>(dy + r) * k_dim;

                    for (int dx = -r; dx <= r; ++dx) {
                        int sx = reflect_index(x + dx, W);
                        float val = in_row[sx];
                        double diff = static_cast<double>(val - center_val);
                        double total_w = spatial_weights[w_row_offset + static_cast<size_t>(dx + r)] * get_range_weight(diff);
                        sum_weights += total_w;
                        sum_values += total_w * static_cast<double>(val);
                    }
                }
                out_row[x] = (sum_weights > 0.0) ? static_cast<float>(sum_values / sum_weights) : center_val;
            }

            // Interior fast-path: x in [r, W - r) - direct indexing, no reflect_index calls
            for (int x = r; x < W - r; ++x) {
                const float center_val = input.at(y, x);
                double sum_weights = 0.0;
                double sum_values = 0.0;

                for (int dy = -r; dy <= r; ++dy) {
                    const float* in_row = input.data().data() + static_cast<size_t>(y + dy) * W;
                    const size_t w_row_offset = static_cast<size_t>(dy + r) * k_dim;

                    for (int dx = -r; dx <= r; ++dx) {
                        float val = in_row[x + dx];
                        double diff = static_cast<double>(val - center_val);
                        double total_w = spatial_weights[w_row_offset + static_cast<size_t>(dx + r)] * get_range_weight(diff);
                        sum_weights += total_w;
                        sum_values += total_w * static_cast<double>(val);
                    }
                }
                out_row[x] = (sum_weights > 0.0) ? static_cast<float>(sum_values / sum_weights) : center_val;
            }

            // Right border: x in [W - r, W)
            for (int x = W - r; x < W; ++x) {
                const float center_val = input.at(y, x);
                double sum_weights = 0.0;
                double sum_values = 0.0;

                for (int dy = -r; dy <= r; ++dy) {
                    const float* in_row = input.data().data() + static_cast<size_t>(y + dy) * W;
                    const size_t w_row_offset = static_cast<size_t>(dy + r) * k_dim;

                    for (int dx = -r; dx <= r; ++dx) {
                        int sx = reflect_index(x + dx, W);
                        float val = in_row[sx];
                        double diff = static_cast<double>(val - center_val);
                        double total_w = spatial_weights[w_row_offset + static_cast<size_t>(dx + r)] * get_range_weight(diff);
                        sum_weights += total_w;
                        sum_values += total_w * static_cast<double>(val);
                    }
                }
                out_row[x] = (sum_weights > 0.0) ? static_cast<float>(sum_values / sum_weights) : center_val;
            }
        } else {
            // Border row or narrow image: full reflection checks
            for (int x = 0; x < W; ++x) {
                const float center_val = input.at(y, x);
                double sum_weights = 0.0;
                double sum_values = 0.0;

                for (int dy = -r; dy <= r; ++dy) {
                    int sy = reflect_index(y + dy, H);
                    const float* in_row = input.data().data() + static_cast<size_t>(sy) * W;
                    const size_t w_row_offset = static_cast<size_t>(dy + r) * k_dim;

                    for (int dx = -r; dx <= r; ++dx) {
                        int sx = reflect_index(x + dx, W);
                        float val = in_row[sx];
                        double diff = static_cast<double>(val - center_val);
                        double total_w = spatial_weights[w_row_offset + static_cast<size_t>(dx + r)] * get_range_weight(diff);
                        sum_weights += total_w;
                        sum_values += total_w * static_cast<double>(val);
                    }
                }
                out_row[x] = (sum_weights > 0.0) ? static_cast<float>(sum_values / sum_weights) : center_val;
            }
        }
    });

    return output;
}

Image2D bilateral_filter(const Image2D& input, double sigma_spatial, double sigma_range, int radius, bool exact) {
    return bilateral_filter(input, BilateralParams2D(sigma_spatial, sigma_range, radius, exact));
}

Volume3D bilateral_filter(const Volume3D& input, const BilateralParams3D& params) {
    if (input.empty()) {
        throw std::invalid_argument("bilateral_filter: input Volume3D is empty");
    }
    if (!(params.sigma_spatial[0] > 0.0) || !(params.sigma_spatial[1] > 0.0) ||
        !(params.sigma_spatial[2] > 0.0) || !(params.sigma_range > 0.0)) {
        throw std::invalid_argument("bilateral_filter: sigmas must be positive and non-NaN");
    }

    const int W = input.width();
    const int H = input.height();
    const int D = input.depth();

    const int rx = (params.radius[0] > 0) ? params.radius[0] : static_cast<int>(std::ceil(3.0 * params.sigma_spatial[0]));
    const int ry = (params.radius[1] > 0) ? params.radius[1] : static_cast<int>(std::ceil(3.0 * params.sigma_spatial[1]));
    const int rz = (params.radius[2] > 0) ? params.radius[2] : static_cast<int>(std::ceil(3.0 * params.sigma_spatial[2]));

    const double two_sx_sq = 2.0 * params.sigma_spatial[0] * params.sigma_spatial[0];
    const double two_sy_sq = 2.0 * params.sigma_spatial[1] * params.sigma_spatial[1];
    const double two_sz_sq = 2.0 * params.sigma_spatial[2] * params.sigma_spatial[2];
    const double two_range_sq = 2.0 * params.sigma_range * params.sigma_range;

    // Precompute 3D spatial weights (strictly read-only)
    const int kx_dim = 2 * rx + 1;
    const int ky_dim = 2 * ry + 1;
    const int kz_dim = 2 * rz + 1;
    std::vector<double> spatial_weights(static_cast<size_t>(kz_dim * ky_dim * kx_dim));

    for (int dz = -rz; dz <= rz; ++dz) {
        double wz = static_cast<double>(dz * dz) / two_sz_sq;
        for (int dy = -ry; dy <= ry; ++dy) {
            double wy = static_cast<double>(dy * dy) / two_sy_sq;
            for (int dx = -rx; dx <= rx; ++dx) {
                double wx = static_cast<double>(dx * dx) / two_sx_sq;
                size_t idx = (static_cast<size_t>(dz + rz) * ky_dim + static_cast<size_t>(dy + ry)) * kx_dim +
                             static_cast<size_t>(dx + rx);
                spatial_weights[idx] = std::exp(-(wx + wy + wz));
            }
        }
    }

    // Precompute range weight lookup table if exact is false
    const size_t lut_size = 16384;
    const double cutoff_d = 6.0 * params.sigma_range;
    const double scale_d = static_cast<double>(lut_size - 1) / cutoff_d;
    std::vector<double> range_table;
    if (!params.exact) {
        range_table.resize(lut_size);
        for (size_t i = 0; i < lut_size; ++i) {
            double d = (static_cast<double>(i) / static_cast<double>(lut_size - 1)) * cutoff_d;
            range_table[i] = std::exp(-(d * d) / two_range_sq);
        }
    }

    auto get_range_weight = [&](double diff) noexcept -> double {
        if (params.exact) {
            return std::exp(-(diff * diff) / two_range_sq);
        }
        double abs_d = std::abs(diff);
        if (abs_d >= cutoff_d) {
            return 0.0;
        }
        double pos = abs_d * scale_d;
        size_t idx = static_cast<size_t>(pos);
        double frac = pos - static_cast<double>(idx);
        return range_table[idx] + frac * (range_table[idx + 1] - range_table[idx]);
    };

    Volume3D output(W, H, D);
    output.metadata() = input.metadata();

    const bool can_use_interior = (W > 2 * rx) && (H > 2 * ry) && (D > 2 * rz);

    parallel_for(0, D, [&](int z) {
        const bool z_interior = can_use_interior && (z >= rz && z < D - rz);

        for (int y = 0; y < H; ++y) {
            float* out_row = output.data().data() + (static_cast<size_t>(z) * H + y) * W;

            if (z_interior && y >= ry && y < H - ry) {
                // Left border: x in [0, rx)
                for (int x = 0; x < rx; ++x) {
                    const float center_val = input.at(z, y, x);
                    double sum_weights = 0.0;
                    double sum_values = 0.0;

                    for (int dz = -rz; dz <= rz; ++dz) {
                        size_t z_offset = static_cast<size_t>(z + dz) * H * W;
                        size_t wz_offset = static_cast<size_t>(dz + rz) * ky_dim * kx_dim;

                        for (int dy = -ry; dy <= ry; ++dy) {
                            const float* in_row = input.data().data() + z_offset + static_cast<size_t>(y + dy) * W;
                            size_t w_row_offset = wz_offset + static_cast<size_t>(dy + ry) * kx_dim;

                            for (int dx = -rx; dx <= rx; ++dx) {
                                int sx = reflect_index(x + dx, W);
                                float val = in_row[sx];
                                double diff = static_cast<double>(val - center_val);
                                double total_w = spatial_weights[w_row_offset + static_cast<size_t>(dx + rx)] * get_range_weight(diff);
                                sum_weights += total_w;
                                sum_values += total_w * static_cast<double>(val);
                            }
                        }
                    }
                    out_row[x] = (sum_weights > 0.0) ? static_cast<float>(sum_values / sum_weights) : center_val;
                }

                // Interior fast-path: x in [rx, W - rx) - direct indexing across all 3 dimensions
                for (int x = rx; x < W - rx; ++x) {
                    const float center_val = input.at(z, y, x);
                    double sum_weights = 0.0;
                    double sum_values = 0.0;

                    for (int dz = -rz; dz <= rz; ++dz) {
                        size_t z_offset = static_cast<size_t>(z + dz) * H * W;
                        size_t wz_offset = static_cast<size_t>(dz + rz) * ky_dim * kx_dim;

                        for (int dy = -ry; dy <= ry; ++dy) {
                            const float* in_row = input.data().data() + z_offset + static_cast<size_t>(y + dy) * W;
                            size_t w_row_offset = wz_offset + static_cast<size_t>(dy + ry) * kx_dim;

                            for (int dx = -rx; dx <= rx; ++dx) {
                                float val = in_row[x + dx];
                                double diff = static_cast<double>(val - center_val);
                                double total_w = spatial_weights[w_row_offset + static_cast<size_t>(dx + rx)] * get_range_weight(diff);
                                sum_weights += total_w;
                                sum_values += total_w * static_cast<double>(val);
                            }
                        }
                    }
                    out_row[x] = (sum_weights > 0.0) ? static_cast<float>(sum_values / sum_weights) : center_val;
                }

                // Right border: x in [W - rx, W)
                for (int x = W - rx; x < W; ++x) {
                    const float center_val = input.at(z, y, x);
                    double sum_weights = 0.0;
                    double sum_values = 0.0;

                    for (int dz = -rz; dz <= rz; ++dz) {
                        size_t z_offset = static_cast<size_t>(z + dz) * H * W;
                        size_t wz_offset = static_cast<size_t>(dz + rz) * ky_dim * kx_dim;

                        for (int dy = -ry; dy <= ry; ++dy) {
                            const float* in_row = input.data().data() + z_offset + static_cast<size_t>(y + dy) * W;
                            size_t w_row_offset = wz_offset + static_cast<size_t>(dy + ry) * kx_dim;

                            for (int dx = -rx; dx <= rx; ++dx) {
                                int sx = reflect_index(x + dx, W);
                                float val = in_row[sx];
                                double diff = static_cast<double>(val - center_val);
                                double total_w = spatial_weights[w_row_offset + static_cast<size_t>(dx + rx)] * get_range_weight(diff);
                                sum_weights += total_w;
                                sum_values += total_w * static_cast<double>(val);
                            }
                        }
                    }
                    out_row[x] = (sum_weights > 0.0) ? static_cast<float>(sum_values / sum_weights) : center_val;
                }
            } else {
                // Border region: full boundary reflection checks
                for (int x = 0; x < W; ++x) {
                    const float center_val = input.at(z, y, x);
                    double sum_weights = 0.0;
                    double sum_values = 0.0;

                    for (int dz = -rz; dz <= rz; ++dz) {
                        int sz = reflect_index(z + dz, D);
                        size_t z_offset = static_cast<size_t>(sz) * H * W;
                        size_t wz_offset = static_cast<size_t>(dz + rz) * ky_dim * kx_dim;

                        for (int dy = -ry; dy <= ry; ++dy) {
                            int sy = reflect_index(y + dy, H);
                            const float* in_row = input.data().data() + z_offset + static_cast<size_t>(sy) * W;
                            size_t w_row_offset = wz_offset + static_cast<size_t>(dy + ry) * kx_dim;

                            for (int dx = -rx; dx <= rx; ++dx) {
                                int sx = reflect_index(x + dx, W);
                                float val = in_row[sx];
                                double diff = static_cast<double>(val - center_val);
                                double total_w = spatial_weights[w_row_offset + static_cast<size_t>(dx + rx)] * get_range_weight(diff);
                                sum_weights += total_w;
                                sum_values += total_w * static_cast<double>(val);
                            }
                        }
                    }
                    out_row[x] = (sum_weights > 0.0) ? static_cast<float>(sum_values / sum_weights) : center_val;
                }
            }
        }
    });

    return output;
}

Volume3D bilateral_filter(const Volume3D& input, double sigma_spatial, double sigma_range, int radius, bool exact) {
    return bilateral_filter(input, BilateralParams3D(sigma_spatial, sigma_range, radius, exact));
}

}  // namespace ct
