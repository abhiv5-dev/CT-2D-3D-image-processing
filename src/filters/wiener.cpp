#include "ct/filters/wiener.hpp"
#include "ct/filters/gaussian.hpp" // for reflect_index
#include "ct/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace ct {

namespace {

// 1D box running sum along rows (X direction) with reflect boundary
void box_running_sum_1d_x(const double* in, double* out, int width, int height, int radius) {
    parallel_for(0, height, [&](int y) {
        const double* in_row = in + static_cast<size_t>(y) * width;
        double* out_row = out + static_cast<size_t>(y) * width;

        double sum = 0.0;
        for (int k = -radius; k <= radius; ++k) {
            int sx = reflect_index(k, width);
            sum += in_row[sx];
        }
        out_row[0] = sum;

        for (int x = 1; x < width; ++x) {
            int add_x = reflect_index(x + radius, width);
            int sub_x = reflect_index(x - 1 - radius, width);
            sum += in_row[add_x] - in_row[sub_x];
            out_row[x] = sum;
        }
    });
}

// 1D box running sum along columns (Y direction) with reflect boundary
void box_running_sum_1d_y(const double* in, double* out, int width, int height, int radius) {
    // Process each column or row-streaming
    parallel_for(0, width, [&](int x) {
        double sum = 0.0;
        for (int k = -radius; k <= radius; ++k) {
            int sy = reflect_index(k, height);
            sum += in[static_cast<size_t>(sy) * width + x];
        }
        out[x] = sum;

        for (int y = 1; y < height; ++y) {
            int add_y = reflect_index(y + radius, height);
            int sub_y = reflect_index(y - 1 - radius, height);
            sum += in[static_cast<size_t>(add_y) * width + x] - in[static_cast<size_t>(sub_y) * width + x];
            out[static_cast<size_t>(y) * width + x] = sum;
        }
    });
}

// 1D box running sum along depth (Z direction) with reflect boundary
void box_running_sum_1d_z(const double* in, double* out, int width, int height, int depth, int radius) {
    const size_t slice_size = static_cast<size_t>(width) * height;

    parallel_for(0, height, [&](int y) {
        for (int x = 0; x < width; ++x) {
            size_t col_offset = static_cast<size_t>(y) * width + x;
            double sum = 0.0;
            for (int k = -radius; k <= radius; ++k) {
                int sz = reflect_index(k, depth);
                sum += in[static_cast<size_t>(sz) * slice_size + col_offset];
            }
            out[col_offset] = sum;

            for (int z = 1; z < depth; ++z) {
                int add_z = reflect_index(z + radius, depth);
                int sub_z = reflect_index(z - 1 - radius, depth);
                sum += in[static_cast<size_t>(add_z) * slice_size + col_offset] -
                       in[static_cast<size_t>(sub_z) * slice_size + col_offset];
                out[static_cast<size_t>(z) * slice_size + col_offset] = sum;
            }
        }
    });
}

}  // namespace

Image2D wiener_filter(const Image2D& input, const WienerParams2D& params) {
    if (input.empty()) {
        throw std::invalid_argument("wiener_filter: input Image2D is empty");
    }
    if (params.radius < 1) {
        throw std::invalid_argument("wiener_filter: radius must be >= 1");
    }

    const int W = input.width();
    const int H = input.height();
    const size_t N = static_cast<size_t>(W) * H;
    const int radius = params.radius;
    const double K = static_cast<double>(2 * radius + 1) * (2 * radius + 1);

    std::vector<double> in_d(N);
    std::vector<double> in_sq_d(N);
    const float* in_ptr = input.data().data();
    for (size_t i = 0; i < N; ++i) {
        double v = static_cast<double>(in_ptr[i]);
        in_d[i] = v;
        in_sq_d[i] = v * v;
    }

    // 1. Box running sum for I: X-pass then Y-pass
    std::vector<double> tmp_sum1(N);
    std::vector<double> sum1(N);
    box_running_sum_1d_x(in_d.data(), tmp_sum1.data(), W, H, radius);
    box_running_sum_1d_y(tmp_sum1.data(), sum1.data(), W, H, radius);

    // 2. Box running sum for I^2: X-pass then Y-pass
    std::vector<double> tmp_sum2(N);
    std::vector<double> sum2(N);
    box_running_sum_1d_x(in_sq_d.data(), tmp_sum2.data(), W, H, radius);
    box_running_sum_1d_y(tmp_sum2.data(), sum2.data(), W, H, radius);

    // 3. Compute local mean and local variance
    std::vector<double> mean_buf(N);
    std::vector<double> var_buf(N);
    double total_var = 0.0;

    for (size_t i = 0; i < N; ++i) {
        double m = sum1[i] / K;
        double v = std::max(0.0, (sum2[i] / K) - (m * m));
        mean_buf[i] = m;
        var_buf[i] = v;
        total_var += v;
    }

    // 4. Noise variance
    double noise = (params.noise_variance >= 0.0) ? params.noise_variance : (total_var / static_cast<double>(N));

    // 5. Output filtered image
    Image2D output(W, H);
    output.metadata() = input.metadata();
    float* out_ptr = output.data().data();

    parallel_for(0, H, [&](int y) {
        size_t row_start = static_cast<size_t>(y) * W;
        for (int x = 0; x < W; ++x) {
            size_t idx = row_start + x;
            double m = mean_buf[idx];
            double v = var_buf[idx];
            double val = in_d[idx];

            if (v <= noise || v <= 0.0) {
                out_ptr[idx] = static_cast<float>(m);
            } else {
                double ratio = (v - noise) / v;
                out_ptr[idx] = static_cast<float>(m + ratio * (val - m));
            }
        }
    });

    return output;
}

Image2D wiener_filter(const Image2D& input, int radius, double noise_variance) {
    return wiener_filter(input, WienerParams2D(radius, noise_variance));
}

Volume3D wiener_filter(const Volume3D& input, const WienerParams3D& params) {
    if (input.empty()) {
        throw std::invalid_argument("wiener_filter: input Volume3D is empty");
    }
    if (params.radius < 1) {
        throw std::invalid_argument("wiener_filter: radius must be >= 1");
    }

    const int W = input.width();
    const int H = input.height();
    const int D = input.depth();
    const size_t N = static_cast<size_t>(W) * H * D;
    const int radius = params.radius;
    const double K = std::pow(2.0 * radius + 1.0, 3.0);

    std::vector<double> in_d(N);
    std::vector<double> in_sq_d(N);
    const float* in_ptr = input.data().data();
    for (size_t i = 0; i < N; ++i) {
        double v = static_cast<double>(in_ptr[i]);
        in_d[i] = v;
        in_sq_d[i] = v * v;
    }

    // 1. Box running sum for I: X-pass -> Y-pass -> Z-pass
    std::vector<double> buf_a(N);
    std::vector<double> buf_b(N);
    std::vector<double> sum1(N);

    // Apply X-pass to each slice
    const size_t slice_size = static_cast<size_t>(W) * H;
    parallel_for(0, D, [&](int z) {
        const double* in_slice = in_d.data() + static_cast<size_t>(z) * slice_size;
        double* out_slice = buf_a.data() + static_cast<size_t>(z) * slice_size;
        for (int y = 0; y < H; ++y) {
            const double* r_in = in_slice + static_cast<size_t>(y) * W;
            double* r_out = out_slice + static_cast<size_t>(y) * W;
            double s = 0.0;
            for (int k = -radius; k <= radius; ++k) s += r_in[reflect_index(k, W)];
            r_out[0] = s;
            for (int x = 1; x < W; ++x) {
                s += r_in[reflect_index(x + radius, W)] - r_in[reflect_index(x - 1 - radius, W)];
                r_out[x] = s;
            }
        }
    });

    // Apply Y-pass to each slice
    parallel_for(0, D, [&](int z) {
        const double* in_slice = buf_a.data() + static_cast<size_t>(z) * slice_size;
        double* out_slice = buf_b.data() + static_cast<size_t>(z) * slice_size;
        for (int x = 0; x < W; ++x) {
            double s = 0.0;
            for (int k = -radius; k <= radius; ++k) s += in_slice[static_cast<size_t>(reflect_index(k, H)) * W + x];
            out_slice[x] = s;
            for (int y = 1; y < H; ++y) {
                s += in_slice[static_cast<size_t>(reflect_index(y + radius, H)) * W + x] -
                     in_slice[static_cast<size_t>(reflect_index(y - 1 - radius, H)) * W + x];
                out_slice[static_cast<size_t>(y) * W + x] = s;
            }
        }
    });

    // Apply Z-pass
    box_running_sum_1d_z(buf_b.data(), sum1.data(), W, H, D, radius);

    // 2. Box running sum for I^2
    parallel_for(0, D, [&](int z) {
        const double* in_slice = in_sq_d.data() + static_cast<size_t>(z) * slice_size;
        double* out_slice = buf_a.data() + static_cast<size_t>(z) * slice_size;
        for (int y = 0; y < H; ++y) {
            const double* r_in = in_slice + static_cast<size_t>(y) * W;
            double* r_out = out_slice + static_cast<size_t>(y) * W;
            double s = 0.0;
            for (int k = -radius; k <= radius; ++k) s += r_in[reflect_index(k, W)];
            r_out[0] = s;
            for (int x = 1; x < W; ++x) {
                s += r_in[reflect_index(x + radius, W)] - r_in[reflect_index(x - 1 - radius, W)];
                r_out[x] = s;
            }
        }
    });

    parallel_for(0, D, [&](int z) {
        const double* in_slice = buf_a.data() + static_cast<size_t>(z) * slice_size;
        double* out_slice = buf_b.data() + static_cast<size_t>(z) * slice_size;
        for (int x = 0; x < W; ++x) {
            double s = 0.0;
            for (int k = -radius; k <= radius; ++k) s += in_slice[static_cast<size_t>(reflect_index(k, H)) * W + x];
            out_slice[x] = s;
            for (int y = 1; y < H; ++y) {
                s += in_slice[static_cast<size_t>(reflect_index(y + radius, H)) * W + x] -
                     in_slice[static_cast<size_t>(reflect_index(y - 1 - radius, H)) * W + x];
                out_slice[static_cast<size_t>(y) * W + x] = s;
            }
        }
    });

    std::vector<double> sum2(N);
    box_running_sum_1d_z(buf_b.data(), sum2.data(), W, H, D, radius);

    // 3. Compute local mean, local variance, and noise
    std::vector<double> mean_buf(N);
    std::vector<double> var_buf(N);
    double total_var = 0.0;

    for (size_t i = 0; i < N; ++i) {
        double m = sum1[i] / K;
        double v = std::max(0.0, (sum2[i] / K) - (m * m));
        mean_buf[i] = m;
        var_buf[i] = v;
        total_var += v;
    }

    double noise = (params.noise_variance >= 0.0) ? params.noise_variance : (total_var / static_cast<double>(N));

    // 4. Output filtered volume
    Volume3D output(W, H, D);
    output.metadata() = input.metadata();
    float* out_ptr = output.data().data();

    parallel_for(0, D, [&](int z) {
        size_t z_start = static_cast<size_t>(z) * slice_size;
        for (size_t i = 0; i < slice_size; ++i) {
            size_t idx = z_start + i;
            double m = mean_buf[idx];
            double v = var_buf[idx];
            double val = in_d[idx];

            if (v <= noise || v <= 0.0) {
                out_ptr[idx] = static_cast<float>(m);
            } else {
                double ratio = (v - noise) / v;
                out_ptr[idx] = static_cast<float>(m + ratio * (val - m));
            }
        }
    });

    return output;
}

Volume3D wiener_filter(const Volume3D& input, int radius, double noise_variance) {
    return wiener_filter(input, WienerParams3D(radius, noise_variance));
}

Volume3D wiener_filter_slice_by_slice(const Volume3D& input, int radius, double noise_variance) {
    return apply_slice_by_slice(input, [radius, noise_variance](const Image2D& s) {
        return wiener_filter(s, radius, noise_variance);
    });
}

}  // namespace ct
