#include "ct/filters/gaussian.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace ct {

namespace {

// Naive 1D horizontal convolution pass: reflect_index on every tap
Image2D convolve_horizontal_naive(const Image2D& input, const std::vector<double>& kernel, int radius) {
    Image2D output(input.width(), input.height());
    output.metadata() = input.metadata();

    for (int row = 0; row < input.height(); ++row) {
        for (int col = 0; col < input.width(); ++col) {
            double acc = 0.0;
            for (int k = -radius; k <= radius; ++k) {
                int src_col = reflect_index(col + k, input.width());
                acc += input.at(row, src_col) * kernel[static_cast<size_t>(k + radius)];
            }
            output.at(row, col) = static_cast<float>(acc);
        }
    }
    return output;
}

// Naive 1D vertical convolution pass: reflect_index on every tap and strided column access
Image2D convolve_vertical_naive(const Image2D& input, const std::vector<double>& kernel, int radius) {
    Image2D output(input.width(), input.height());
    output.metadata() = input.metadata();

    for (int row = 0; row < input.height(); ++row) {
        for (int col = 0; col < input.width(); ++col) {
            double acc = 0.0;
            for (int k = -radius; k <= radius; ++k) {
                int src_row = reflect_index(row + k, input.height());
                acc += input.at(src_row, col) * kernel[static_cast<size_t>(k + radius)];
            }
            output.at(row, col) = static_cast<float>(acc);
        }
    }
    return output;
}

}  // namespace

Image2D gaussian_blur_naive(const Image2D& input, const GaussianParams2D& params) {
    if (input.empty()) {
        throw std::invalid_argument("gaussian_blur_naive: input Image2D is empty");
    }
    if (!(params.sigma_x > 0.0) || !(params.sigma_y > 0.0)) {
        throw std::invalid_argument("gaussian_blur_naive: sigma must be positive and non-NaN");
    }

    int rx = (params.radius_x > 0) ? params.radius_x : static_cast<int>(std::ceil(3.0 * params.sigma_x));
    int ry = (params.radius_y > 0) ? params.radius_y : static_cast<int>(std::ceil(3.0 * params.sigma_y));

    std::vector<double> kx = make_gaussian_kernel(params.sigma_x, rx);
    std::vector<double> ky = make_gaussian_kernel(params.sigma_y, ry);

    Image2D horizontal = convolve_horizontal_naive(input, kx, rx);
    return convolve_vertical_naive(horizontal, ky, ry);
}

Image2D gaussian_blur_naive(const Image2D& input, double sigma, int radius) {
    GaussianParams2D p(sigma, radius);
    return gaussian_blur_naive(input, p);
}

Volume3D gaussian_blur_naive(const Volume3D& input, const GaussianParams3D& params) {
    if (input.empty()) {
        throw std::invalid_argument("gaussian_blur_naive: input Volume3D is empty");
    }
    if (!(params.sigma[0] > 0.0) || !(params.sigma[1] > 0.0) || !(params.sigma[2] > 0.0)) {
        throw std::invalid_argument("gaussian_blur_naive: 3D sigma values must be positive and non-NaN");
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

    // Pass 1: X pass with reflect_index on every tap
    Volume3D buf1(W, H, D);
    buf1.metadata() = input.metadata();
    for (int z = 0; z < D; ++z) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                double acc = 0.0;
                for (int k = -rx; k <= rx; ++k) {
                    int src_x = reflect_index(x + k, W);
                    acc += input.at(z, y, src_x) * kx[static_cast<size_t>(k + rx)];
                }
                buf1.at(z, y, x) = static_cast<float>(acc);
            }
        }
    }

    // Pass 2: Y pass with reflect_index on every tap and strided column access
    Volume3D buf2(W, H, D);
    buf2.metadata() = input.metadata();
    for (int z = 0; z < D; ++z) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                double acc = 0.0;
                for (int k = -ry; k <= ry; ++k) {
                    int src_y = reflect_index(y + k, H);
                    acc += buf1.at(z, src_y, x) * ky[static_cast<size_t>(k + ry)];
                }
                buf2.at(z, y, x) = static_cast<float>(acc);
            }
        }
    }

    // Pass 3: Z pass with reflect_index on every tap and strided depth access
    Volume3D output(W, H, D);
    output.metadata() = input.metadata();
    for (int z = 0; z < D; ++z) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                double acc = 0.0;
                for (int k = -rz; k <= rz; ++k) {
                    int src_z = reflect_index(z + k, D);
                    acc += buf2.at(src_z, y, x) * kz[static_cast<size_t>(k + rz)];
                }
                output.at(z, y, x) = static_cast<float>(acc);
            }
        }
    }

    return output;
}

Volume3D gaussian_blur_naive(const Volume3D& input, double sigma, int radius) {
    GaussianParams3D p(sigma, radius);
    return gaussian_blur_naive(input, p);
}

}  // namespace ct
