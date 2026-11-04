#pragma once

#include "ct/image2d.hpp"
#include "ct/volume3d.hpp"
#include "ct/parallel.hpp"
#include "ct/filters/gaussian.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace ct {

/**
 * Robust median of float buffer using nth_element (O(N) expected time).
 * For even N, returns the average of the two middle elements (matching numpy.median).
 */
inline double compute_median(std::vector<float>& vals) {
    if (vals.empty()) return 0.0;
    size_t n = vals.size();
    size_t mid = n / 2;
    std::nth_element(vals.begin(), vals.begin() + mid, vals.end());
    if (n % 2 != 0) {
        return static_cast<double>(vals[mid]);
    } else {
        float m2 = vals[mid];
        auto it_m1 = std::max_element(vals.begin(), vals.begin() + mid);
        float m1 = *it_m1;
        return 0.5 * (static_cast<double>(m1) + static_cast<double>(m2));
    }
}

/**
 * Immerkaer robust noise estimator on 2D slice:
 * Convolves with the 3x3 Laplacian-of-Laplacian kernel:
 *   [  1, -2,  1 ]
 *   [ -2,  4, -2 ]
 *   [  1, -2,  1 ]
 * using reflect boundary conditions, and computes:
 *   sigma_hat = median(|conv|) / (0.67448975 * 6.0)
 */
inline double estimate_noise_sigma(const Image2D& img) {
    if (img.empty()) return 0.0;
    const int W = img.width();
    const int H = img.height();
    std::vector<float> abs_res(static_cast<size_t>(W) * H);

    static constexpr float K[3][3] = {
        { 1.0f, -2.0f,  1.0f},
        {-2.0f,  4.0f, -2.0f},
        { 1.0f, -2.0f,  1.0f}
    };

    parallel_for(0, H, [&](int r) {
        for (int c = 0; c < W; ++c) {
            float sum = 0.0f;
            for (int dr = -1; dr <= 1; ++dr) {
                int rr = reflect_index(r + dr, H);
                for (int dc = -1; dc <= 1; ++dc) {
                    int cc = reflect_index(c + dc, W);
                    sum += K[dr + 1][dc + 1] * img.at(rr, cc);
                }
            }
            abs_res[static_cast<size_t>(r) * W + c] = std::abs(sum);
        }
    });

    double med = compute_median(abs_res);
    return med / (0.67448975 * 6.0);
}

/**
 * Immerkaer robust noise estimator on 3D volume:
 * Convolves slice-by-slice with the Immerkaer 2D kernel with reflect boundaries,
 * and takes the median of absolute residuals across the entire volume:
 *   sigma_hat = median(|conv|) / (0.67448975 * 6.0)
 */
inline double estimate_noise_sigma(const Volume3D& vol) {
    if (vol.empty()) return 0.0;
    const int D = vol.depth();
    const int H = vol.height();
    const int W = vol.width();
    const size_t total_voxels = static_cast<size_t>(D) * H * W;
    std::vector<float> abs_res(total_voxels);

    static constexpr float K[3][3] = {
        { 1.0f, -2.0f,  1.0f},
        {-2.0f,  4.0f, -2.0f},
        { 1.0f, -2.0f,  1.0f}
    };

    parallel_for(0, D, [&](int z) {
        size_t slice_offset = static_cast<size_t>(z) * H * W;
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < W; ++c) {
                float sum = 0.0f;
                for (int dr = -1; dr <= 1; ++dr) {
                    int rr = reflect_index(r + dr, H);
                    for (int dc = -1; dc <= 1; ++dc) {
                        int cc = reflect_index(c + dc, W);
                        sum += K[dr + 1][dc + 1] * vol.at(z, rr, cc);
                    }
                }
                abs_res[slice_offset + static_cast<size_t>(r) * W + c] = std::abs(sum);
            }
        }
    });

    double med = compute_median(abs_res);
    return med / (0.67448975 * 6.0);
}

} // namespace ct
