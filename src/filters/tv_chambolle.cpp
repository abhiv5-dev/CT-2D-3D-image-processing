#include "ct/filters/tv_chambolle.hpp"
#include "ct/filters/gaussian.hpp"
#include "ct/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace ct {

Image2D tv_chambolle(const Image2D& input, const TVParams2D& params) {
    if (input.empty()) {
        throw std::invalid_argument("tv_chambolle: input Image2D is empty");
    }
    if (!(params.weight > 0.0)) {
        throw std::invalid_argument("tv_chambolle: weight must be positive and non-NaN");
    }
    if (params.max_iterations < 1) {
        throw std::invalid_argument("tv_chambolle: max_iterations must be >= 1");
    }
    if (!(params.eps > 0.0)) {
        throw std::invalid_argument("tv_chambolle: eps must be positive and non-NaN");
    }

    const int W = input.width();
    const int H = input.height();
    const size_t N = static_cast<size_t>(W) * H;
    const double weight = params.weight;
    const double eps = params.eps;
    const int max_iter = params.max_iterations;

    // Stability step size in 2D is tau <= 1 / (2 * ndim) = 0.25
    const double tau = 0.25;

    // Dual variables: 2 components (py, px)
    // Extra memory: 2 * N * sizeof(float)
    std::vector<float> py(N, 0.0f);
    std::vector<float> px(N, 0.0f);

    std::vector<float> d(N, 0.0f);
    std::vector<float> u(N);
    const float* in_ptr = input.data().data();

    // Initial state: u = input
    std::copy(in_ptr, in_ptr + N, u.begin());

    double E_init = 0.0;
    double E_prev = 0.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        if (iter > 0) {
            // Compute d = (negative) divergence of p
            // d[y, x] = -(py[y, x] - py[y-1, x]) - (px[y, x] - px[y, x-1])
            parallel_for(0, H, [&](int y) {
                size_t row_start = static_cast<size_t>(y) * W;
                size_t prev_row = (y > 0) ? (static_cast<size_t>(y - 1) * W) : 0;
                for (int x = 0; x < W; ++x) {
                    size_t idx = row_start + x;
                    float div_y = py[idx] - ((y > 0) ? py[prev_row + x] : 0.0f);
                    float div_x = px[idx] - ((x > 0) ? px[idx - 1] : 0.0f);
                    float d_val = -(div_y + div_x);
                    d[idx] = d_val;
                    u[idx] = in_ptr[idx] + d_val;
                }
            });
        }

        // Compute gradients, norm, and update dual variables
        std::vector<double> row_d_sq(static_cast<size_t>(H), 0.0);
        std::vector<double> row_norm_sum(static_cast<size_t>(H), 0.0);

        parallel_for(0, H, [&](int y) {
            size_t row_start = static_cast<size_t>(y) * W;
            size_t next_row = (y < H - 1) ? (static_cast<size_t>(y + 1) * W) : 0;
            double local_d_sq = 0.0;
            double local_norm = 0.0;

            for (int x = 0; x < W; ++x) {
                size_t idx = row_start + x;
                float d_val = d[idx];
                local_d_sq += static_cast<double>(d_val) * d_val;

                float gy = (y < H - 1) ? (u[next_row + x] - u[idx]) : 0.0f;
                float gx = (x < W - 1) ? (u[idx + 1] - u[idx]) : 0.0f;

                double g_norm = std::sqrt(static_cast<double>(gy) * gy + static_cast<double>(gx) * gx);
                local_norm += g_norm;

                double denom = 1.0 + (tau / weight) * g_norm;
                py[idx] = static_cast<float>((static_cast<double>(py[idx]) - tau * gy) / denom);
                px[idx] = static_cast<float>((static_cast<double>(px[idx]) - tau * gx) / denom);
            }
            row_d_sq[static_cast<size_t>(y)] = local_d_sq;
            row_norm_sum[static_cast<size_t>(y)] = local_norm;
        });

        // Deterministic scalar reduction
        double sum_d_sq = 0.0;
        double sum_norm = 0.0;
        for (int y = 0; y < H; ++y) {
            sum_d_sq += row_d_sq[static_cast<size_t>(y)];
            sum_norm += row_norm_sum[static_cast<size_t>(y)];
        }

        double E = (sum_d_sq + weight * sum_norm) / static_cast<double>(N);

        if (iter == 0) {
            E_init = E;
            E_prev = E;
        } else {
            if (std::abs(E_prev - E) < eps * E_init) {
                break;
            }
            E_prev = E;
        }
    }

    Image2D output(W, H);
    output.metadata() = input.metadata();
    std::copy(u.begin(), u.end(), output.data().begin());
    return output;
}

Image2D tv_chambolle(const Image2D& input, double weight, double eps, int max_iterations) {
    return tv_chambolle(input, TVParams2D(weight, eps, max_iterations));
}

Volume3D tv_chambolle(const Volume3D& input, const TVParams3D& params) {
    if (input.empty()) {
        throw std::invalid_argument("tv_chambolle: input Volume3D is empty");
    }
    if (!(params.weight > 0.0)) {
        throw std::invalid_argument("tv_chambolle: weight must be positive and non-NaN");
    }
    if (params.max_iterations < 1) {
        throw std::invalid_argument("tv_chambolle: max_iterations must be >= 1");
    }
    if (!(params.eps > 0.0)) {
        throw std::invalid_argument("tv_chambolle: eps must be positive and non-NaN");
    }

    const int W = input.width();
    const int H = input.height();
    const int D = input.depth();
    const size_t slice_size = static_cast<size_t>(W) * H;
    const size_t N = slice_size * D;
    const double weight = params.weight;
    const double eps = params.eps;
    const int max_iter = params.max_iterations;

    // Stability step size in 3D is tau <= 1 / (2 * ndim) = 1/6
    const double tau = 1.0 / 6.0;

    // Dual variables: 3 components (pz, py, px)
    // Extra memory: 3 * N * sizeof(float)
    std::vector<float> pz(N, 0.0f);
    std::vector<float> py(N, 0.0f);
    std::vector<float> px(N, 0.0f);

    std::vector<float> d(N, 0.0f);
    std::vector<float> u(N);
    const float* in_ptr = input.data().data();

    std::copy(in_ptr, in_ptr + N, u.begin());

    double E_init = 0.0;
    double E_prev = 0.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        if (iter > 0) {
            parallel_for(0, D, [&](int z) {
                size_t z_start = static_cast<size_t>(z) * slice_size;
                size_t prev_z = (z > 0) ? (static_cast<size_t>(z - 1) * slice_size) : 0;
                for (int y = 0; y < H; ++y) {
                    size_t row_start = z_start + static_cast<size_t>(y) * W;
                    size_t prev_y = (y > 0) ? (z_start + static_cast<size_t>(y - 1) * W) : 0;
                    for (int x = 0; x < W; ++x) {
                        size_t idx = row_start + x;
                        float div_z = pz[idx] - ((z > 0) ? pz[prev_z + static_cast<size_t>(y) * W + x] : 0.0f);
                        float div_y = py[idx] - ((y > 0) ? py[prev_y + x] : 0.0f);
                        float div_x = px[idx] - ((x > 0) ? px[idx - 1] : 0.0f);
                        float d_val = -(div_z + div_y + div_x);
                        d[idx] = d_val;
                        u[idx] = in_ptr[idx] + d_val;
                    }
                }
            });
        }

        std::vector<double> slice_d_sq(static_cast<size_t>(D), 0.0);
        std::vector<double> slice_norm_sum(static_cast<size_t>(D), 0.0);

        parallel_for(0, D, [&](int z) {
            size_t z_start = static_cast<size_t>(z) * slice_size;
            size_t next_z = (z < D - 1) ? (static_cast<size_t>(z + 1) * slice_size) : 0;
            double local_d_sq = 0.0;
            double local_norm = 0.0;

            for (int y = 0; y < H; ++y) {
                size_t row_start = z_start + static_cast<size_t>(y) * W;
                size_t next_y = (y < H - 1) ? (z_start + static_cast<size_t>(y + 1) * W) : 0;
                for (int x = 0; x < W; ++x) {
                    size_t idx = row_start + x;
                    float d_val = d[idx];
                    local_d_sq += static_cast<double>(d_val) * d_val;

                    float gz = (z < D - 1) ? (u[next_z + static_cast<size_t>(y) * W + x] - u[idx]) : 0.0f;
                    float gy = (y < H - 1) ? (u[next_y + x] - u[idx]) : 0.0f;
                    float gx = (x < W - 1) ? (u[idx + 1] - u[idx]) : 0.0f;

                    double g_norm = std::sqrt(static_cast<double>(gz) * gz +
                                              static_cast<double>(gy) * gy +
                                              static_cast<double>(gx) * gx);
                    local_norm += g_norm;

                    double denom = 1.0 + (tau / weight) * g_norm;
                    pz[idx] = static_cast<float>((static_cast<double>(pz[idx]) - tau * gz) / denom);
                    py[idx] = static_cast<float>((static_cast<double>(py[idx]) - tau * gy) / denom);
                    px[idx] = static_cast<float>((static_cast<double>(px[idx]) - tau * gx) / denom);
                }
            }
            slice_d_sq[static_cast<size_t>(z)] = local_d_sq;
            slice_norm_sum[static_cast<size_t>(z)] = local_norm;
        });

        double sum_d_sq = 0.0;
        double sum_norm = 0.0;
        for (int z = 0; z < D; ++z) {
            sum_d_sq += slice_d_sq[static_cast<size_t>(z)];
            sum_norm += slice_norm_sum[static_cast<size_t>(z)];
        }

        double E = (sum_d_sq + weight * sum_norm) / static_cast<double>(N);

        if (iter == 0) {
            E_init = E;
            E_prev = E;
        } else {
            if (std::abs(E_prev - E) < eps * E_init) {
                break;
            }
            E_prev = E;
        }
    }

    Volume3D output(W, H, D);
    output.metadata() = input.metadata();
    std::copy(u.begin(), u.end(), output.data().begin());
    return output;
}

Volume3D tv_chambolle(const Volume3D& input, double weight, double eps, int max_iterations) {
    return tv_chambolle(input, TVParams3D(weight, eps, max_iterations));
}

Volume3D tv_chambolle_slice_by_slice(const Volume3D& input, double weight, double eps, int max_iterations) {
    return apply_slice_by_slice(input, [weight, eps, max_iterations](const Image2D& s) {
        return tv_chambolle(s, weight, eps, max_iterations);
    });
}

}  // namespace ct
