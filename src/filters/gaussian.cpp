#include "ct/filters/gaussian.hpp"
#include "ct/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace ct {

int reflect_index(int index, int size) noexcept {
    if (size <= 1) {
        return 0;
    }
    while (index < 0 || index >= size) {
        if (index < 0) {
            index = -index - 1;
        } else if (index >= size) {
            index = 2 * size - index - 1;
        }
    }
    return index;
}

std::vector<double> make_gaussian_kernel(double sigma, int radius) {
    if (!(sigma > 0.0)) {
        throw std::invalid_argument("make_gaussian_kernel: sigma must be positive and non-NaN");
    }
    if (radius < 0) {
        radius = static_cast<int>(std::ceil(3.0 * sigma));
    }

    std::vector<double> kernel(static_cast<size_t>(2 * radius + 1));
    double sum = 0.0;
    const double two_sigma_sq = 2.0 * sigma * sigma;
    for (int i = -radius; i <= radius; ++i) {
        double weight = std::exp(-(static_cast<double>(i) * i) / two_sigma_sq);
        kernel[static_cast<size_t>(i + radius)] = weight;
        sum += weight;
    }
    for (double& w : kernel) {
        w /= sum;
    }
    return kernel;
}

namespace {

// 1D horizontal pass along X with interior/border split (parallelised over rows)
void convolve_1d_x(const float* in, float* out, int width, int height,
                   const std::vector<double>& kernel, int radius) {
    const int k_size = 2 * radius + 1;
    parallel_for(0, height, [&](int y) {
        const float* in_row = in + static_cast<size_t>(y) * width;
        float* out_row = out + static_cast<size_t>(y) * width;

        if (width > 2 * radius) {
            // Left border: x in [0, radius)
            for (int x = 0; x < radius; ++x) {
                double acc = 0.0;
                for (int k = -radius; k <= radius; ++k) {
                    int sx = reflect_index(x + k, width);
                    acc += static_cast<double>(in_row[sx]) * kernel[static_cast<size_t>(k + radius)];
                }
                out_row[x] = static_cast<float>(acc);
            }

            // Interior hot loop: no boundary reflection checks
            for (int x = radius; x < width - radius; ++x) {
                double acc = 0.0;
                const float* p = in_row + (x - radius);
                for (int k = 0; k < k_size; ++k) {
                    acc += static_cast<double>(p[k]) * kernel[static_cast<size_t>(k)];
                }
                out_row[x] = static_cast<float>(acc);
            }

            // Right border: x in [width - radius, width)
            for (int x = width - radius; x < width; ++x) {
                double acc = 0.0;
                for (int k = -radius; k <= radius; ++k) {
                    int sx = reflect_index(x + k, width);
                    acc += static_cast<double>(in_row[sx]) * kernel[static_cast<size_t>(k + radius)];
                }
                out_row[x] = static_cast<float>(acc);
            }
        } else {
            // Narrow image fallback
            for (int x = 0; x < width; ++x) {
                double acc = 0.0;
                for (int k = -radius; k <= radius; ++k) {
                    int sx = reflect_index(x + k, width);
                    acc += static_cast<double>(in_row[sx]) * kernel[static_cast<size_t>(k + radius)];
                }
                out_row[x] = static_cast<float>(acc);
            }
        }
    }, width, 16384);
}

// 1D vertical pass along Y: sequential whole-row streaming (parallelised over rows)
void convolve_1d_y(const float* in, float* out, int width, int height,
                   const std::vector<double>& kernel, int radius) {
    parallel_for(0, height, [&](int y) {
        std::vector<double> acc(static_cast<size_t>(width), 0.0);
        for (int k = -radius; k <= radius; ++k) {
            int src_y = (y + k >= 0 && y + k < height) ? (y + k) : reflect_index(y + k, height);
            double w = kernel[static_cast<size_t>(k + radius)];
            const float* in_row = in + static_cast<size_t>(src_y) * width;
            for (int x = 0; x < width; ++x) {
                acc[static_cast<size_t>(x)] += w * static_cast<double>(in_row[x]);
            }
        }
        float* out_row = out + static_cast<size_t>(y) * width;
        for (int x = 0; x < width; ++x) {
            out_row[x] = static_cast<float>(acc[static_cast<size_t>(x)]);
        }
    }, width, 16384);
}

// 1D depth pass along Z: sequential whole-row streaming across slices (parallelised over slices)
void convolve_1d_z(const float* in, float* out, int width, int height, int depth,
                   const std::vector<double>& kernel, int radius) {
    const size_t slice_voxels = static_cast<size_t>(width) * height;

    parallel_for(0, depth, [&](int z) {
        float* out_slice = out + static_cast<size_t>(z) * slice_voxels;
        std::vector<double> acc(static_cast<size_t>(width));

        for (int y = 0; y < height; ++y) {
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int k = -radius; k <= radius; ++k) {
                int src_z = (z + k >= 0 && z + k < depth) ? (z + k) : reflect_index(z + k, depth);
                double w = kernel[static_cast<size_t>(k + radius)];
                const float* in_row = in + (static_cast<size_t>(src_z) * height + y) * width;
                for (int x = 0; x < width; ++x) {
                    acc[static_cast<size_t>(x)] += w * static_cast<double>(in_row[x]);
                }
            }
            float* out_row = out_slice + static_cast<size_t>(y) * width;
            for (int x = 0; x < width; ++x) {
                out_row[x] = static_cast<float>(acc[static_cast<size_t>(x)]);
            }
        }
    }, static_cast<int64_t>(slice_voxels), 16384);
}

}  // namespace

Image2D gaussian_blur(const Image2D& input, const GaussianParams2D& params) {
    if (input.empty()) {
        throw std::invalid_argument("gaussian_blur: input Image2D is empty");
    }
    if (!(params.sigma_x > 0.0) || !(params.sigma_y > 0.0)) {
        throw std::invalid_argument("gaussian_blur: sigma must be positive and non-NaN");
    }

    int rx = (params.radius_x > 0) ? params.radius_x : static_cast<int>(std::ceil(3.0 * params.sigma_x));
    int ry = (params.radius_y > 0) ? params.radius_y : static_cast<int>(std::ceil(3.0 * params.sigma_y));

    auto kx = make_gaussian_kernel(params.sigma_x, rx);
    auto ky = make_gaussian_kernel(params.sigma_y, ry);

    Image2D temp(input.width(), input.height());
    convolve_1d_x(input.data().data(), temp.data().data(), input.width(), input.height(), kx, rx);

    Image2D output(input.width(), input.height());
    output.metadata() = input.metadata();
    convolve_1d_y(temp.data().data(), output.data().data(), input.width(), input.height(), ky, ry);

    return output;
}

Image2D gaussian_blur(const Image2D& input, double sigma, int radius) {
    GaussianParams2D p(sigma, radius);
    return gaussian_blur(input, p);
}

Volume3D gaussian_blur(const Volume3D& input, const GaussianParams3D& params) {
    if (input.empty()) {
        throw std::invalid_argument("gaussian_blur: input Volume3D is empty");
    }
    if (!(params.sigma[0] > 0.0) || !(params.sigma[1] > 0.0) || !(params.sigma[2] > 0.0)) {
        throw std::invalid_argument("gaussian_blur: 3D sigma values must be positive and non-NaN");
    }

    int rx = (params.radius[0] > 0) ? params.radius[0] : static_cast<int>(std::ceil(3.0 * params.sigma[0]));
    int ry = (params.radius[1] > 0) ? params.radius[1] : static_cast<int>(std::ceil(3.0 * params.sigma[1]));
    int rz = (params.radius[2] > 0) ? params.radius[2] : static_cast<int>(std::ceil(3.0 * params.sigma[2]));

    auto kx = make_gaussian_kernel(params.sigma[0], rx);
    auto ky = make_gaussian_kernel(params.sigma[1], ry);
    auto kz = make_gaussian_kernel(params.sigma[2], rz);

    int W = input.width();
    int H = input.height();
    int D = input.depth();

    // Ping-pong between output and temp buffers to keep peak memory <= 3 volumes
    Volume3D output(W, H, D);
    output.metadata() = input.metadata();
    Volume3D temp(W, H, D);

    // Pass 1: X convolution on each slice (input -> output)
    parallel_for(0, D, [&](int z) {
        size_t offset = static_cast<size_t>(z) * W * H;
        convolve_1d_x(input.data().data() + offset, output.data().data() + offset, W, H, kx, rx);
    }, static_cast<int64_t>(W) * H, 16384);

    // Pass 2: Y convolution on each slice (output -> temp)
    parallel_for(0, D, [&](int z) {
        size_t offset = static_cast<size_t>(z) * W * H;
        convolve_1d_y(output.data().data() + offset, temp.data().data() + offset, W, H, ky, ry);
    }, static_cast<int64_t>(W) * H, 16384);

    // Pass 3: Z convolution across slices (temp -> output)
    convolve_1d_z(temp.data().data(), output.data().data(), W, H, D, kz, rz);

    return output;
}

Volume3D gaussian_blur(const Volume3D& input, double sigma, int radius) {
    GaussianParams3D p(sigma, radius);
    return gaussian_blur(input, p);
}

}  // namespace ct
