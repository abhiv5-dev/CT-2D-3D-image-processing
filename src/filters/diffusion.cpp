#include "ct/filters/diffusion.hpp"
#include "ct/parallel.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace ct {

namespace {

inline double conductance_val(double grad, double kappa_sq, ConductanceFunction cond) {
    double s = (grad * grad) / kappa_sq;
    if (cond == ConductanceFunction::Exponential) {
        return std::exp(-s);
    } else {
        return 1.0 / (1.0 + s);
    }
}

}  // namespace

Image2D anisotropic_diffusion(const Image2D& input, const DiffusionParams2D& params) {
    if (input.empty()) {
        throw std::invalid_argument("anisotropic_diffusion: input Image2D is empty");
    }
    if (!(params.kappa > 0.0)) {
        throw std::invalid_argument("anisotropic_diffusion: kappa must be positive and non-NaN");
    }
    if (params.iterations < 0) {
        throw std::invalid_argument("anisotropic_diffusion: iterations must be non-negative");
    }
    if (!(params.dt > 0.0) || params.dt > 0.25) {
        throw std::invalid_argument("anisotropic_diffusion: stability requires 0 < dt <= 0.25 in 2D");
    }

    if (params.iterations == 0) {
        return input;
    }

    const int W = input.width();
    const int H = input.height();
    const double kappa_sq = params.kappa * params.kappa;
    const double dt = params.dt;

    Image2D cur = input;
    Image2D next(W, H);
    next.metadata() = input.metadata();

    for (int iter = 0; iter < params.iterations; ++iter) {
        parallel_for(0, H, [&](int y) {
            const float* cur_row = cur.data().data() + static_cast<size_t>(y) * W;
            const float* prev_row = (y > 0) ? (cur.data().data() + static_cast<size_t>(y - 1) * W) : nullptr;
            const float* next_row = (y < H - 1) ? (cur.data().data() + static_cast<size_t>(y + 1) * W) : nullptr;
            float* out_row = next.data().data() + static_cast<size_t>(y) * W;

            for (int x = 0; x < W; ++x) {
                float center = cur_row[x];

                double dN = prev_row ? static_cast<double>(prev_row[x] - center) : 0.0;
                double dS = next_row ? static_cast<double>(next_row[x] - center) : 0.0;
                double dW = (x > 0) ? static_cast<double>(cur_row[x - 1] - center) : 0.0;
                double dE = (x < W - 1) ? static_cast<double>(cur_row[x + 1] - center) : 0.0;

                double cN = conductance_val(dN, kappa_sq, params.conductance);
                double cS = conductance_val(dS, kappa_sq, params.conductance);
                double cW = conductance_val(dW, kappa_sq, params.conductance);
                double cE = conductance_val(dE, kappa_sq, params.conductance);

                double flux = cN * dN + cS * dS + cW * dW + cE * dE;
                out_row[x] = static_cast<float>(center + dt * flux);
            }
        });
        std::swap(cur, next);
    }

    return cur;
}

Image2D anisotropic_diffusion(const Image2D& input, double kappa, int iterations, double dt) {
    return anisotropic_diffusion(input, DiffusionParams2D(kappa, iterations, dt));
}

Volume3D anisotropic_diffusion(const Volume3D& input, const DiffusionParams3D& params) {
    if (input.empty()) {
        throw std::invalid_argument("anisotropic_diffusion: input Volume3D is empty");
    }
    if (!(params.kappa > 0.0)) {
        throw std::invalid_argument("anisotropic_diffusion: kappa must be positive and non-NaN");
    }
    if (params.iterations < 0) {
        throw std::invalid_argument("anisotropic_diffusion: iterations must be non-negative");
    }
    if (!(params.dt > 0.0) || params.dt > (1.0 / 6.0 + 1e-6)) {
        throw std::invalid_argument("anisotropic_diffusion: stability requires 0 < dt <= 1/6 in 3D");
    }

    if (params.iterations == 0) {
        return input;
    }

    const int W = input.width();
    const int H = input.height();
    const int D = input.depth();
    const size_t slice_size = static_cast<size_t>(W) * H;
    const double kappa_sq = params.kappa * params.kappa;
    const double dt = params.dt;

    Volume3D cur = input;
    Volume3D next(W, H, D);
    next.metadata() = input.metadata();

    for (int iter = 0; iter < params.iterations; ++iter) {
        parallel_for(0, D, [&](int z) {
            size_t z_off = static_cast<size_t>(z) * slice_size;
            const float* cur_slice = cur.data().data() + z_off;
            const float* prev_slice = (z > 0) ? (cur.data().data() + (static_cast<size_t>(z - 1) * slice_size)) : nullptr;
            const float* next_slice = (z < D - 1) ? (cur.data().data() + (static_cast<size_t>(z + 1) * slice_size)) : nullptr;
            float* out_slice = next.data().data() + z_off;

            for (int y = 0; y < H; ++y) {
                size_t y_off = static_cast<size_t>(y) * W;
                const float* cur_row = cur_slice + y_off;
                const float* prev_row = (y > 0) ? (cur_slice + (static_cast<size_t>(y - 1) * W)) : nullptr;
                const float* next_row = (y < H - 1) ? (cur_slice + (static_cast<size_t>(y + 1) * W)) : nullptr;
                float* out_row = out_slice + y_off;

                for (int x = 0; x < W; ++x) {
                    float center = cur_row[x];

                    double dN = prev_row ? static_cast<double>(prev_row[x] - center) : 0.0;
                    double dS = next_row ? static_cast<double>(next_row[x] - center) : 0.0;
                    double dW = (x > 0) ? static_cast<double>(cur_row[x - 1] - center) : 0.0;
                    double dE = (x < W - 1) ? static_cast<double>(cur_row[x + 1] - center) : 0.0;
                    double dUp = prev_slice ? static_cast<double>(prev_slice[y_off + static_cast<size_t>(x)] - center) : 0.0;
                    double dDown = next_slice ? static_cast<double>(next_slice[y_off + static_cast<size_t>(x)] - center) : 0.0;

                    double cN = conductance_val(dN, kappa_sq, params.conductance);
                    double cS = conductance_val(dS, kappa_sq, params.conductance);
                    double cW = conductance_val(dW, kappa_sq, params.conductance);
                    double cE = conductance_val(dE, kappa_sq, params.conductance);
                    double cUp = conductance_val(dUp, kappa_sq, params.conductance);
                    double cDown = conductance_val(dDown, kappa_sq, params.conductance);

                    double flux = cN * dN + cS * dS + cW * dW + cE * dE + cUp * dUp + cDown * dDown;
                    out_row[x] = static_cast<float>(center + dt * flux);
                }
            }
        });
        std::swap(cur, next);
    }

    return cur;
}

Volume3D anisotropic_diffusion(const Volume3D& input, double kappa, int iterations, double dt) {
    return anisotropic_diffusion(input, DiffusionParams3D(kappa, iterations, dt));
}

}  // namespace ct
