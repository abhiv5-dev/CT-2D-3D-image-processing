#pragma once

#include "ct/image2d.hpp"
#include "ct/volume3d.hpp"

namespace ct {

struct TVParams2D {
    double weight = 0.1;        // Denoising regularization weight (lambda^-1)
    double eps = 2.0e-4;        // Convergence stopping tolerance
    int max_iterations = 200;   // Maximum optimization iterations

    TVParams2D() = default;
    TVParams2D(double w, double e = 2.0e-4, int iters = 200)
        : weight(w), eps(e), max_iterations(iters) {}
};

struct TVParams3D {
    double weight = 0.1;        // Denoising regularization weight
    double eps = 2.0e-4;        // Convergence stopping tolerance
    int max_iterations = 200;   // Maximum optimization iterations

    TVParams3D() = default;
    TVParams3D(double w, double e = 2.0e-4, int iters = 200)
        : weight(w), eps(e), max_iterations(iters) {}
};

// 2D Total Variation denoising via Chambolle's projection algorithm (step size tau = 0.25)
Image2D tv_chambolle(const Image2D& input, const TVParams2D& params);
Image2D tv_chambolle(const Image2D& input, double weight = 0.1, double eps = 2.0e-4, int max_iterations = 200);

// 3D Total Variation denoising via Chambolle's projection algorithm (step size tau = 1/6)
Volume3D tv_chambolle(const Volume3D& input, const TVParams3D& params);
Volume3D tv_chambolle(const Volume3D& input, double weight = 0.1, double eps = 2.0e-4, int max_iterations = 200);

// Slice-by-slice 2D Total Variation denoising applied along z-axis of a 3D volume
Volume3D tv_chambolle_slice_by_slice(const Volume3D& input, double weight = 0.1, double eps = 2.0e-4, int max_iterations = 200);

}  // namespace ct

using ct::tv_chambolle;
using ct::tv_chambolle_slice_by_slice;
