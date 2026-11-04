#pragma once

#include "ct/image2d.hpp"
#include "ct/volume3d.hpp"

namespace ct {

enum class ConductanceFunction {
    Exponential,  // c(grad) = exp(-(grad / kappa)^2)
    Reciprocal    // c(grad) = 1 / (1 + (grad / kappa)^2)
};

struct DiffusionParams2D {
    double kappa = 0.05;
    int iterations = 10;
    double dt = 0.20;  // Numerical stability requires dt <= 0.25
    ConductanceFunction conductance = ConductanceFunction::Exponential;

    DiffusionParams2D() = default;
    DiffusionParams2D(double k, int iters, double time_step = 0.20,
                      ConductanceFunction cond = ConductanceFunction::Exponential)
        : kappa(k), iterations(iters), dt(time_step), conductance(cond) {}
};

struct DiffusionParams3D {
    double kappa = 0.05;
    int iterations = 10;
    double dt = 0.15;  // Numerical stability requires dt <= 1/6 (~0.166667)
    ConductanceFunction conductance = ConductanceFunction::Exponential;

    DiffusionParams3D() = default;
    DiffusionParams3D(double k, int iters, double time_step = 0.15,
                      ConductanceFunction cond = ConductanceFunction::Exponential)
        : kappa(k), iterations(iters), dt(time_step), conductance(cond) {}
};

// 2D Perona-Malik anisotropic diffusion
Image2D anisotropic_diffusion(const Image2D& input, const DiffusionParams2D& params);
Image2D anisotropic_diffusion(const Image2D& input, double kappa = 0.05, int iterations = 10, double dt = 0.20);

// 3D Perona-Malik anisotropic diffusion
Volume3D anisotropic_diffusion(const Volume3D& input, const DiffusionParams3D& params);
Volume3D anisotropic_diffusion(const Volume3D& input, double kappa = 0.05, int iterations = 10, double dt = 0.15);

}  // namespace ct

using ct::anisotropic_diffusion;
