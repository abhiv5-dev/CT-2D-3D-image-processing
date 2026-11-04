#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "ct/image2d.hpp"
#include "ct/volume3d.hpp"
#include "ct/tiff_io.hpp"
#include "ct/preview.hpp"
#include "ct/filters/gaussian.hpp"
#include "ct/filters/median.hpp"
#include "ct/filters/bilateral.hpp"
#include "ct/filters/diffusion.hpp"
#include "ct/filters/nlm.hpp"
#include "ct/filters/wiener.hpp"
#include "ct/filters/tv_chambolle.hpp"
#include "ct/filters/morphology.hpp"
#include "ct/backend.hpp"
#include "ct/intensity.hpp"
#include "ct/parallel.hpp"
#include "ct/noise.hpp"

#include <cmath>
#include <string>
#include <variant>

namespace py = pybind11;

namespace {

// Helper: Convert 2D NumPy array to C++ Image2D.
// Memory is deep-copied from Python NumPy buffer to internal C++ std::vector.
Image2D numpy_to_image2d(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
    auto buf = arr.request();
    if (buf.ndim != 2) {
        throw std::invalid_argument("Expected 2D array (H, W), got " + std::to_string(buf.ndim) + "D");
    }
    int H = static_cast<int>(buf.shape[0]);
    int W = static_cast<int>(buf.shape[1]);
    if (H <= 0 || W <= 0) {
        throw std::invalid_argument("Input array cannot be empty");
    }
    Image2D img(W, H);
    const float* ptr = static_cast<const float*>(buf.ptr);
    std::copy(ptr, ptr + (static_cast<size_t>(W) * H), img.data().begin());
    return img;
}

// Helper: Convert C++ Image2D to 2D NumPy array.
// Allocates a new NumPy array and copies data from Image2D.
py::array_t<float> image2d_to_numpy(const Image2D& img) {
    py::array_t<float> result({img.height(), img.width()});
    auto buf = result.request();
    float* ptr = static_cast<float*>(buf.ptr);
    std::copy(img.data().begin(), img.data().end(), ptr);
    return result;
}

// Helper: Convert 3D NumPy array (D, H, W) to C++ Volume3D.
// Memory is deep-copied from Python NumPy buffer to internal C++ std::vector.
Volume3D numpy_to_volume3d(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
    auto buf = arr.request();
    if (buf.ndim != 3) {
        throw std::invalid_argument("Expected 3D array (D, H, W) [z, y, x], got " + std::to_string(buf.ndim) + "D");
    }
    int D = static_cast<int>(buf.shape[0]);
    int H = static_cast<int>(buf.shape[1]);
    int W = static_cast<int>(buf.shape[2]);
    if (D <= 0 || H <= 0 || W <= 0) {
        throw std::invalid_argument("Input volume cannot be empty");
    }
    Volume3D vol(W, H, D);
    const float* ptr = static_cast<const float*>(buf.ptr);
    std::copy(ptr, ptr + (static_cast<size_t>(W) * H * D), vol.data().begin());
    return vol;
}

// Helper: Convert C++ Volume3D to 3D NumPy array (D, H, W).
// Allocates a new NumPy array and copies data from Volume3D.
py::array_t<float> volume3d_to_numpy(const Volume3D& vol) {
    py::array_t<float> result({vol.depth(), vol.height(), vol.width()});
    auto buf = result.request();
    float* ptr = static_cast<float*>(buf.ptr);
    std::copy(vol.data().begin(), vol.data().end(), ptr);
    return result;
}

}  // namespace

PYBIND11_MODULE(ct, m) {
    m.doc() = "ct: High-performance 2D and 3D CT image processing and filtering module.\n"
              "All filter functions make explicit copies between NumPy buffers and C++ memory buffers.";

    // 2D Gaussian filter
    m.def("gaussian_filter_2d", [](py::array_t<float> input,
                                   std::variant<double, std::pair<double, double>> sigma,
                                   std::variant<int, std::pair<int, int>> radius) {
        Image2D img = numpy_to_image2d(input);
        ct::GaussianParams2D params;
        if (std::holds_alternative<double>(sigma)) {
            double s = std::get<double>(sigma);
            params.sigma_x = s;
            params.sigma_y = s;
        } else {
            auto [sy, sx] = std::get<std::pair<double, double>>(sigma);
            params.sigma_x = sx;
            params.sigma_y = sy;
        }

        if (std::holds_alternative<int>(radius)) {
            int r = std::get<int>(radius);
            params.radius_x = r;
            params.radius_y = r;
        } else {
            auto [ry, rx] = std::get<std::pair<int, int>>(radius);
            params.radius_x = rx;
            params.radius_y = ry;
        }

        if (!(params.sigma_x > 0.0) || !(params.sigma_y > 0.0)) {
            throw std::invalid_argument("gaussian_filter_2d: sigma must be positive and non-NaN");
        }

        Image2D out;
        {
            py::gil_scoped_release release;
            out = ct::gaussian_blur(img, params);
        }
        return image2d_to_numpy(out);
    }, py::arg("input"), py::arg("sigma") = 1.0, py::arg("radius") = -1,
       "Applies 2D separable Gaussian blur with reflect boundary matching scipy.\nCopies input and output.");

    // 3D Gaussian filter
    m.def("gaussian_filter_3d", [](py::array_t<float> input,
                                   std::variant<double, std::vector<double>> sigma,
                                   std::variant<int, std::vector<int>> radius) {
        Volume3D vol = numpy_to_volume3d(input);
        ct::GaussianParams3D params;

        if (std::holds_alternative<double>(sigma)) {
            double s = std::get<double>(sigma);
            params.sigma = {s, s, s};
        } else {
            auto vec = std::get<std::vector<double>>(sigma);
            if (vec.size() != 3) throw std::invalid_argument("sigma vector must have 3 elements (sz, sy, sx)");
            params.sigma = {vec[2], vec[1], vec[0]};
        }

        if (std::holds_alternative<int>(radius)) {
            int r = std::get<int>(radius);
            params.radius = {r, r, r};
        } else {
            auto vec = std::get<std::vector<int>>(radius);
            if (vec.size() != 3) throw std::invalid_argument("radius vector must have 3 elements (rz, ry, rx)");
            params.radius = {vec[2], vec[1], vec[0]};
        }

        for (double s : params.sigma) {
            if (!(s > 0.0)) {
                throw std::invalid_argument("gaussian_filter_3d: sigma must be positive and non-NaN");
            }
        }

        Volume3D out;
        {
            py::gil_scoped_release release;
            out = ct::gaussian_blur(vol, params);
        }
        return volume3d_to_numpy(out);
    }, py::arg("input"), py::arg("sigma") = 1.0, py::arg("radius") = -1,
       "Applies 3D separable Gaussian blur (axis order: z, y, x).\nCopies input and output.");

    // Naive reference Gaussian filters
    m.def("gaussian_filter_2d_naive", [](py::array_t<float> input, double sigma, int radius) {
        Image2D img = numpy_to_image2d(input);
        if (!(sigma > 0.0)) {
            throw std::invalid_argument("gaussian_filter_2d_naive: sigma must be positive and non-NaN");
        }
        Image2D out;
        {
            py::gil_scoped_release release;
            out = ct::gaussian_blur_naive(img, sigma, radius);
        }
        return image2d_to_numpy(out);
    }, py::arg("input"), py::arg("sigma") = 1.0, py::arg("radius") = -1,
       "Applies naive 2D Gaussian blur reference.\nCopies input and output.");

    m.def("gaussian_filter_3d_naive", [](py::array_t<float> input, double sigma, int radius) {
        Volume3D vol = numpy_to_volume3d(input);
        if (!(sigma > 0.0)) {
            throw std::invalid_argument("gaussian_filter_3d_naive: sigma must be positive and non-NaN");
        }
        Volume3D out;
        {
            py::gil_scoped_release release;
            out = ct::gaussian_blur_naive(vol, sigma, radius);
        }
        return volume3d_to_numpy(out);
    }, py::arg("input"), py::arg("sigma") = 1.0, py::arg("radius") = -1,
       "Applies naive 3D Gaussian blur reference.\nCopies input and output.");

    // Slice-by-slice 2D Gaussian applied to 3D volume
    m.def("gaussian_filter_slice_by_slice", [](py::array_t<float> input, double sigma, int radius) {
        Volume3D vol = numpy_to_volume3d(input);
        if (!(sigma > 0.0)) {
            throw std::invalid_argument("gaussian_filter_slice_by_slice: sigma must be positive and non-NaN");
        }
        Volume3D out;
        {
            py::gil_scoped_release release;
            out = ct::apply_slice_by_slice(vol, [sigma, radius](const Image2D& s) {
                return ct::gaussian_blur(s, sigma, radius);
            });
        }
        return volume3d_to_numpy(out);
    }, py::arg("input"), py::arg("sigma") = 1.0, py::arg("radius") = -1,
       "Applies 2D Gaussian blur slice-by-slice along the z-axis.\nCopies input and output.");

    // 2D Median filter
    m.def("median_filter_2d", [](py::array_t<float> input, int radius) {
        Image2D img = numpy_to_image2d(input);
        if (radius < 1) {
            throw std::invalid_argument("median_filter_2d: radius must be >= 1");
        }
        Image2D out;
        {
            py::gil_scoped_release release;
            out = ct::median_filter(img, radius);
        }
        return image2d_to_numpy(out);
    }, py::arg("input"), py::arg("radius") = 1,
       "Applies 2D median filter with given window radius.\nCopies input and output.");

    // 3D Median filter
    m.def("median_filter_3d", [](py::array_t<float> input, int radius) {
        Volume3D vol = numpy_to_volume3d(input);
        if (radius < 1) {
            throw std::invalid_argument("median_filter_3d: radius must be >= 1");
        }
        Volume3D out;
        {
            py::gil_scoped_release release;
            out = ct::median_filter(vol, radius);
        }
        return volume3d_to_numpy(out);
    }, py::arg("input"), py::arg("radius") = 1,
       "Applies 3D median filter with given window radius (axis order: z, y, x).\nCopies input and output.");

    // Slice-by-slice 2D Median applied to 3D volume
    m.def("median_filter_slice_by_slice", [](py::array_t<float> input, int radius) {
        Volume3D vol = numpy_to_volume3d(input);
        if (radius < 1) {
            throw std::invalid_argument("median_filter_slice_by_slice: radius must be >= 1");
        }
        Volume3D out;
        {
            py::gil_scoped_release release;
            out = ct::apply_slice_by_slice(vol, [radius](const Image2D& s) {
                return ct::median_filter(s, radius);
            });
        }
        return volume3d_to_numpy(out);
    }, py::arg("input"), py::arg("radius") = 1,
       "Applies 2D median filter slice-by-slice along the z-axis.\nCopies input and output.");

    // 2D Bilateral filter
    m.def("bilateral_filter_2d", [](py::array_t<float> input, double sigma_spatial, double sigma_range, int radius, bool exact) {
        Image2D img = numpy_to_image2d(input);
        if (!(sigma_spatial > 0.0) || !(sigma_range > 0.0)) {
            throw std::invalid_argument("bilateral_filter_2d: sigmas must be positive and non-NaN");
        }
        Image2D out;
        {
            py::gil_scoped_release release;
            out = ct::bilateral_filter(img, sigma_spatial, sigma_range, radius, exact);
        }
        return image2d_to_numpy(out);
    }, py::arg("input"), py::arg("sigma_spatial") = 2.0, py::arg("sigma_range") = 0.1, py::arg("radius") = -1, py::arg("exact") = true,
       "Applies 2D edge-preserving bilateral filter.\nCopies input and output.");

    // 3D Bilateral filter
    m.def("bilateral_filter_3d", [](py::array_t<float> input, double sigma_spatial, double sigma_range, int radius, bool exact) {
        Volume3D vol = numpy_to_volume3d(input);
        if (!(sigma_spatial > 0.0) || !(sigma_range > 0.0)) {
            throw std::invalid_argument("bilateral_filter_3d: sigmas must be positive and non-NaN");
        }
        Volume3D out;
        {
            py::gil_scoped_release release;
            out = ct::bilateral_filter(vol, sigma_spatial, sigma_range, radius, exact);
        }
        return volume3d_to_numpy(out);
    }, py::arg("input"), py::arg("sigma_spatial") = 2.0, py::arg("sigma_range") = 0.1, py::arg("radius") = -1, py::arg("exact") = true,
       "Applies 3D edge-preserving bilateral filter (axis order: z, y, x).\nCopies input and output.");

    // Slice-by-slice 2D Bilateral applied to 3D volume
    m.def("bilateral_filter_slice_by_slice", [](py::array_t<float> input, double sigma_spatial, double sigma_range, int radius) {
        Volume3D vol = numpy_to_volume3d(input);
        if (!(sigma_spatial > 0.0) || !(sigma_range > 0.0)) {
            throw std::invalid_argument("bilateral_filter_slice_by_slice: sigmas must be positive and non-NaN");
        }
        Volume3D out;
        {
            py::gil_scoped_release release;
            out = ct::apply_slice_by_slice(vol, [sigma_spatial, sigma_range, radius](const Image2D& s) {
                return ct::bilateral_filter(s, sigma_spatial, sigma_range, radius);
            });
        }
        return volume3d_to_numpy(out);
    }, py::arg("input"), py::arg("sigma_spatial") = 2.0, py::arg("sigma_range") = 0.1, py::arg("radius") = -1,
       "Applies 2D bilateral filter slice-by-slice along the z-axis.\nCopies input and output.");

    // 2D Anisotropic Diffusion (Perona-Malik)
    m.def("anisotropic_diffusion_2d", [](py::array_t<float> input, double kappa, int iterations, double dt,
                                        const std::string& conductance) {
        Image2D img = numpy_to_image2d(input);
        if (!(kappa > 0.0) || iterations < 1 || !(dt > 0.0)) {
            throw std::invalid_argument("anisotropic_diffusion_2d: parameters must be positive and non-NaN");
        }
        ct::DiffusionParams2D params;
        params.kappa = kappa;
        params.iterations = iterations;
        params.dt = dt;
        params.conductance = (conductance == "reciprocal") ? ct::ConductanceFunction::Reciprocal : ct::ConductanceFunction::Exponential;
        Image2D out;
        {
            py::gil_scoped_release release;
            out = ct::anisotropic_diffusion(img, params);
        }
        return image2d_to_numpy(out);
    }, py::arg("input"), py::arg("kappa") = 0.05, py::arg("iterations") = 10, py::arg("dt") = 0.20,
       py::arg("conductance") = "exponential",
       "Applies 2D Perona-Malik anisotropic diffusion.\nCopies input and output.");

    // 3D Anisotropic Diffusion (Perona-Malik)
    m.def("anisotropic_diffusion_3d", [](py::array_t<float> input, double kappa, int iterations, double dt,
                                        const std::string& conductance) {
        Volume3D vol = numpy_to_volume3d(input);
        if (!(kappa > 0.0) || iterations < 1 || !(dt > 0.0)) {
            throw std::invalid_argument("anisotropic_diffusion_3d: parameters must be positive and non-NaN");
        }
        ct::DiffusionParams3D params;
        params.kappa = kappa;
        params.iterations = iterations;
        params.dt = dt;
        params.conductance = (conductance == "reciprocal") ? ct::ConductanceFunction::Reciprocal : ct::ConductanceFunction::Exponential;
        Volume3D out;
        {
            py::gil_scoped_release release;
            out = ct::anisotropic_diffusion(vol, params);
        }
        return volume3d_to_numpy(out);
    }, py::arg("input"), py::arg("kappa") = 0.05, py::arg("iterations") = 10, py::arg("dt") = 0.15,
       py::arg("conductance") = "exponential",
       "Applies 3D Perona-Malik anisotropic diffusion (axis order: z, y, x).\nCopies input and output.");

    // Slice-by-slice 2D Anisotropic Diffusion
    m.def("anisotropic_diffusion_slice_by_slice", [](py::array_t<float> input, double kappa, int iterations, double dt,
                                                     const std::string& conductance) {
        Volume3D vol = numpy_to_volume3d(input);
        if (!(kappa > 0.0) || iterations < 1 || !(dt > 0.0)) {
            throw std::invalid_argument("anisotropic_diffusion_slice_by_slice: parameters must be positive and non-NaN");
        }
        ct::ConductanceFunction cond = (conductance == "reciprocal") ? ct::ConductanceFunction::Reciprocal : ct::ConductanceFunction::Exponential;
        Volume3D out;
        {
            py::gil_scoped_release release;
            out = ct::apply_slice_by_slice(vol, [kappa, iterations, dt, cond](const Image2D& s) {
                ct::DiffusionParams2D params(kappa, iterations, dt, cond);
                return ct::anisotropic_diffusion(s, params);
            });
        }
        return volume3d_to_numpy(out);
    }, py::arg("input"), py::arg("kappa") = 0.05, py::arg("iterations") = 10, py::arg("dt") = 0.20,
       py::arg("conductance") = "exponential",
       "Applies 2D anisotropic diffusion slice-by-slice along the z-axis.\nCopies input and output.");

    // 2D Wiener filter
    m.def("wiener_filter_2d", [](py::array_t<float> input, int radius, double noise_variance) {
        Image2D img = numpy_to_image2d(input);
        if (radius < 1) throw std::invalid_argument("wiener_filter_2d: radius must be >= 1");
        Image2D out;
        {
            py::gil_scoped_release release;
            out = ct::wiener_filter(img, radius, noise_variance);
        }
        return image2d_to_numpy(out);
    }, py::arg("input"), py::arg("radius") = 1, py::arg("noise_variance") = -1.0,
       "Applies 2D Adaptive Wiener filter with separable running sums and reflect boundary.\nCopies input and output.");

    // 3D Wiener filter
    m.def("wiener_filter_3d", [](py::array_t<float> input, int radius, double noise_variance) {
        Volume3D vol = numpy_to_volume3d(input);
        if (radius < 1) throw std::invalid_argument("wiener_filter_3d: radius must be >= 1");
        Volume3D out;
        {
            py::gil_scoped_release release;
            out = ct::wiener_filter(vol, radius, noise_variance);
        }
        return volume3d_to_numpy(out);
    }, py::arg("input"), py::arg("radius") = 1, py::arg("noise_variance") = -1.0,
       "Applies 3D Adaptive Wiener filter with separable running sums and reflect boundary.\nCopies input and output.");

    // Slice-by-slice 2D Wiener filter applied to 3D volume
    m.def("wiener_filter_slice_by_slice", [](py::array_t<float> input, int radius, double noise_variance) {
        Volume3D vol = numpy_to_volume3d(input);
        if (radius < 1) throw std::invalid_argument("wiener_filter_slice_by_slice: radius must be >= 1");
        Volume3D out;
        {
            py::gil_scoped_release release;
            out = ct::wiener_filter_slice_by_slice(vol, radius, noise_variance);
        }
        return volume3d_to_numpy(out);
    }, py::arg("input"), py::arg("radius") = 1, py::arg("noise_variance") = -1.0,
       "Applies 2D Adaptive Wiener filter slice-by-slice along z-axis.\nCopies input and output.");

    // 2D Total Variation Denoising (ROF Chambolle)
    m.def("tv_chambolle_2d", [](py::array_t<float> input, double weight, double eps, int max_iterations) {
        Image2D img = numpy_to_image2d(input);
        if (!(weight > 0.0) || !(eps > 0.0) || max_iterations < 1) {
            throw std::invalid_argument("tv_chambolle_2d: parameters must be positive");
        }
        Image2D out;
        {
            py::gil_scoped_release release;
            out = ct::tv_chambolle(img, weight, eps, max_iterations);
        }
        return image2d_to_numpy(out);
    }, py::arg("input"), py::arg("weight") = 0.1, py::arg("eps") = 2.0e-4, py::arg("max_iterations") = 200,
       "Applies 2D Total Variation denoising via Chambolle's projection algorithm (step size tau = 0.25).\nCopies input and output.");

    // 3D Total Variation Denoising (ROF Chambolle)
    m.def("tv_chambolle_3d", [](py::array_t<float> input, double weight, double eps, int max_iterations) {
        Volume3D vol = numpy_to_volume3d(input);
        if (!(weight > 0.0) || !(eps > 0.0) || max_iterations < 1) {
            throw std::invalid_argument("tv_chambolle_3d: parameters must be positive");
        }
        Volume3D out;
        {
            py::gil_scoped_release release;
            out = ct::tv_chambolle(vol, weight, eps, max_iterations);
        }
        return volume3d_to_numpy(out);
    }, py::arg("input"), py::arg("weight") = 0.1, py::arg("eps") = 2.0e-4, py::arg("max_iterations") = 200,
       "Applies 3D Total Variation denoising via Chambolle's projection algorithm (step size tau = 1/6).\nCopies input and output.");

    // Slice-by-slice 2D Total Variation Denoising applied to 3D volume
    m.def("tv_chambolle_slice_by_slice", [](py::array_t<float> input, double weight, double eps, int max_iterations) {
        Volume3D vol = numpy_to_volume3d(input);
        if (!(weight > 0.0) || !(eps > 0.0) || max_iterations < 1) {
            throw std::invalid_argument("tv_chambolle_slice_by_slice: parameters must be positive");
        }
        Volume3D out;
        {
            py::gil_scoped_release release;
            out = ct::tv_chambolle_slice_by_slice(vol, weight, eps, max_iterations);
        }
        return volume3d_to_numpy(out);
    }, py::arg("input"), py::arg("weight") = 0.1, py::arg("eps") = 2.0e-4, py::arg("max_iterations") = 200,
       "Applies 2D Total Variation denoising slice-by-slice along z-axis.\nCopies input and output.");

    // Grayscale Morphology 2D
    m.def("grayscale_erosion_2d", [](py::array_t<float> input, int radius, const std::string& element) {
        Image2D img = numpy_to_image2d(input);
        Image2D out;
        { py::gil_scoped_release r; out = ct::grayscale_erosion(img, radius, element); }
        return image2d_to_numpy(out);
    }, py::arg("input"), py::arg("radius") = 1, py::arg("element") = "box", "Applies 2D grayscale erosion");

    m.def("grayscale_dilation_2d", [](py::array_t<float> input, int radius, const std::string& element) {
        Image2D img = numpy_to_image2d(input);
        Image2D out;
        { py::gil_scoped_release r; out = ct::grayscale_dilation(img, radius, element); }
        return image2d_to_numpy(out);
    }, py::arg("input"), py::arg("radius") = 1, py::arg("element") = "box", "Applies 2D grayscale dilation");

    m.def("grayscale_opening_2d", [](py::array_t<float> input, int radius, const std::string& element) {
        Image2D img = numpy_to_image2d(input);
        Image2D out;
        { py::gil_scoped_release r; out = ct::grayscale_opening(img, radius, element); }
        return image2d_to_numpy(out);
    }, py::arg("input"), py::arg("radius") = 1, py::arg("element") = "box", "Applies 2D grayscale opening");

    m.def("grayscale_closing_2d", [](py::array_t<float> input, int radius, const std::string& element) {
        Image2D img = numpy_to_image2d(input);
        Image2D out;
        { py::gil_scoped_release r; out = ct::grayscale_closing(img, radius, element); }
        return image2d_to_numpy(out);
    }, py::arg("input"), py::arg("radius") = 1, py::arg("element") = "box", "Applies 2D grayscale closing");

    m.def("white_tophat_2d", [](py::array_t<float> input, int radius, const std::string& element) {
        Image2D img = numpy_to_image2d(input);
        Image2D out;
        { py::gil_scoped_release r; out = ct::white_tophat(img, radius, element); }
        return image2d_to_numpy(out);
    }, py::arg("input"), py::arg("radius") = 1, py::arg("element") = "box", "Applies 2D white top-hat");

    m.def("black_tophat_2d", [](py::array_t<float> input, int radius, const std::string& element) {
        Image2D img = numpy_to_image2d(input);
        Image2D out;
        { py::gil_scoped_release r; out = ct::black_tophat(img, radius, element); }
        return image2d_to_numpy(out);
    }, py::arg("input"), py::arg("radius") = 1, py::arg("element") = "box", "Applies 2D black top-hat");

    // Grayscale Morphology 3D
    m.def("grayscale_erosion_3d", [](py::array_t<float> input, int radius, const std::string& element) {
        Volume3D vol = numpy_to_volume3d(input);
        Volume3D out;
        { py::gil_scoped_release r; out = ct::grayscale_erosion(vol, radius, element); }
        return volume3d_to_numpy(out);
    }, py::arg("input"), py::arg("radius") = 1, py::arg("element") = "box", "Applies 3D grayscale erosion");

    m.def("grayscale_dilation_3d", [](py::array_t<float> input, int radius, const std::string& element) {
        Volume3D vol = numpy_to_volume3d(input);
        Volume3D out;
        { py::gil_scoped_release r; out = ct::grayscale_dilation(vol, radius, element); }
        return volume3d_to_numpy(out);
    }, py::arg("input"), py::arg("radius") = 1, py::arg("element") = "box", "Applies 3D grayscale dilation");

    m.def("grayscale_opening_3d", [](py::array_t<float> input, int radius, const std::string& element) {
        Volume3D vol = numpy_to_volume3d(input);
        Volume3D out;
        { py::gil_scoped_release r; out = ct::grayscale_opening(vol, radius, element); }
        return volume3d_to_numpy(out);
    }, py::arg("input"), py::arg("radius") = 1, py::arg("element") = "box", "Applies 3D grayscale opening");

    m.def("grayscale_closing_3d", [](py::array_t<float> input, int radius, const std::string& element) {
        Volume3D vol = numpy_to_volume3d(input);
        Volume3D out;
        { py::gil_scoped_release r; out = ct::grayscale_closing(vol, radius, element); }
        return volume3d_to_numpy(out);
    }, py::arg("input"), py::arg("radius") = 1, py::arg("element") = "box", "Applies 3D grayscale closing");

    m.def("white_tophat_3d", [](py::array_t<float> input, int radius, const std::string& element) {
        Volume3D vol = numpy_to_volume3d(input);
        Volume3D out;
        { py::gil_scoped_release r; out = ct::white_tophat(vol, radius, element); }
        return volume3d_to_numpy(out);
    }, py::arg("input"), py::arg("radius") = 1, py::arg("element") = "box", "Applies 3D white top-hat");

    m.def("black_tophat_3d", [](py::array_t<float> input, int radius, const std::string& element) {
        Volume3D vol = numpy_to_volume3d(input);
        Volume3D out;
        { py::gil_scoped_release r; out = ct::black_tophat(vol, radius, element); }
        return volume3d_to_numpy(out);
    }, py::arg("input"), py::arg("radius") = 1, py::arg("element") = "box", "Applies 3D black top-hat");

    // Slice-by-slice morphology
    m.def("grayscale_opening_slice_by_slice", [](py::array_t<float> input, int radius, const std::string& element) {
        Volume3D vol = numpy_to_volume3d(input);
        Volume3D out;
        { py::gil_scoped_release r; out = ct::grayscale_opening_slice_by_slice(vol, radius, element); }
        return volume3d_to_numpy(out);
    }, py::arg("input"), py::arg("radius") = 1, py::arg("element") = "box", "Applies 2D grayscale opening slice-by-slice");

    m.def("grayscale_closing_slice_by_slice", [](py::array_t<float> input, int radius, const std::string& element) {
        Volume3D vol = numpy_to_volume3d(input);
        Volume3D out;
        { py::gil_scoped_release r; out = ct::grayscale_closing_slice_by_slice(vol, radius, element); }
        return volume3d_to_numpy(out);
    }, py::arg("input"), py::arg("radius") = 1, py::arg("element") = "box", "Applies 2D grayscale closing slice-by-slice");

    // Threading control
    m.def("set_num_threads", &ct::set_num_threads, py::arg("n"), "Sets thread count for parallel operations");
    m.def("get_num_threads", &ct::get_num_threads, "Gets thread count for parallel operations");

    // 2D Non-Local Means (NLM)
    m.def("nlm_filter_2d", [](py::array_t<float> input, int patch_radius, int search_radius, double h) {
        Image2D img = numpy_to_image2d(input);
        Image2D out;
        {
            py::gil_scoped_release release;
            out = ct::nlm_filter(img, patch_radius, search_radius, h);
        }
        return image2d_to_numpy(out);
    }, py::arg("input"), py::arg("patch_radius") = 2, py::arg("search_radius") = 5, py::arg("h") = 0.05,
       "Applies 2D Non-Local Means filter.\nCopies input and output.");

    // 3D Non-Local Means (NLM)
    m.def("nlm_filter_3d", [](py::array_t<float> input, int patch_radius, int search_radius, double h) {
        Volume3D vol = numpy_to_volume3d(input);
        Volume3D out;
        {
            py::gil_scoped_release release;
            out = ct::nlm_filter(vol, patch_radius, search_radius, h);
        }
        return volume3d_to_numpy(out);
    }, py::arg("input"), py::arg("patch_radius") = 1, py::arg("search_radius") = 3, py::arg("h") = 0.05,
       "Applies 3D Non-Local Means filter (axis order: z, y, x).\nCopies input and output.");

    // Slice-by-slice Non-Local Means (NLM)
    m.def("nlm_filter_slice_by_slice", [](py::array_t<float> input, int patch_radius, int search_radius, double h) {
        Volume3D vol = numpy_to_volume3d(input);
        Volume3D out;
        {
            py::gil_scoped_release release;
            out = ct::apply_slice_by_slice(vol, [&](const Image2D& s) {
                return ct::nlm_filter(s, patch_radius, search_radius, h);
            });
        }
        return volume3d_to_numpy(out);
    }, py::arg("input"), py::arg("patch_radius") = 2, py::arg("search_radius") = 5, py::arg("h") = 0.05,
       "Applies 2D Non-Local Means filter slice-by-slice across 3D volume.\nCopies input and output.");

    // Backend device info
    m.def("get_device_info", []() {
        auto info = ct::get_active_device_info();
        py::dict d;
        d["name"] = info.name;
        d["unified_memory"] = info.unified_memory;
        d["memory_bytes"] = info.memory_bytes;
        return d;
    }, "Queries compute device capabilities and memory architecture");

    // Intensity processing
    m.def("apply_windowing", [](py::array_t<float> input, double window_center, double window_width,
                                float out_min, float out_max) -> py::object {
        auto buf = input.request();
        if (buf.ndim == 2) {
            Image2D img = numpy_to_image2d(input);
            Image2D out;
            {
                py::gil_scoped_release release;
                out = ct::apply_windowing(img, window_center, window_width, out_min, out_max);
            }
            return image2d_to_numpy(out);
        } else if (buf.ndim == 3) {
            Volume3D vol = numpy_to_volume3d(input);
            Volume3D out;
            {
                py::gil_scoped_release release;
                out = ct::apply_windowing(vol, window_center, window_width, out_min, out_max);
            }
            return volume3d_to_numpy(out);
        } else {
            throw std::invalid_argument("Expected 2D or 3D array for apply_windowing");
        }
    }, py::arg("input"), py::arg("window_center"), py::arg("window_width"),
       py::arg("out_min") = 0.0f, py::arg("out_max") = 1.0f,
       "Applies CT level/width windowing and clamping.\nCopies input and output.");

    m.def("linear_stretch", [](py::array_t<float> input, double in_min, double in_max,
                                float out_min, float out_max) -> py::object {
        auto buf = input.request();
        if (buf.ndim == 2) {
            Image2D img = numpy_to_image2d(input);
            Image2D out;
            {
                py::gil_scoped_release release;
                out = ct::linear_stretch(img, in_min, in_max, out_min, out_max);
            }
            return image2d_to_numpy(out);
        } else if (buf.ndim == 3) {
            Volume3D vol = numpy_to_volume3d(input);
            Volume3D out;
            {
                py::gil_scoped_release release;
                out = ct::linear_stretch(vol, in_min, in_max, out_min, out_max);
            }
            return volume3d_to_numpy(out);
        } else {
            throw std::invalid_argument("Expected 2D or 3D array for linear_stretch");
        }
    }, py::arg("input"),
       py::arg("in_min") = std::numeric_limits<double>::quiet_NaN(),
       py::arg("in_max") = std::numeric_limits<double>::quiet_NaN(),
       py::arg("out_min") = 0.0f, py::arg("out_max") = 1.0f,
       "Applies linear contrast stretching.\nCopies input and output.");

    m.def("apply_clahe", [](py::array_t<float> input, int tile_grid_size, double clip_limit) -> py::object {
        auto buf = input.request();
        if (buf.ndim == 2) {
            Image2D img = numpy_to_image2d(input);
            Image2D out;
            {
                py::gil_scoped_release release;
                out = ct::apply_clahe(img, tile_grid_size, clip_limit);
            }
            return image2d_to_numpy(out);
        } else if (buf.ndim == 3) {
            Volume3D vol = numpy_to_volume3d(input);
            Volume3D out;
            {
                py::gil_scoped_release release;
                out = ct::apply_clahe(vol, tile_grid_size, clip_limit);
            }
            return volume3d_to_numpy(out);
        } else {
            throw std::invalid_argument("Expected 2D or 3D array for apply_clahe");
        }
    }, py::arg("input"), py::arg("tile_grid_size") = 8, py::arg("clip_limit") = 2.0,
       "Applies Contrast-Limited Adaptive Histogram Equalization (CLAHE).\nCopies input and output.");

    // File I/O
    m.def("load_tiff", [](const std::string& path) {
        Image2D img = load_tiff(path);
        return image2d_to_numpy(img);
    }, py::arg("path"), "Loads 2D TIFF into float32 NumPy array");

    m.def("save_tiff", [](py::array_t<float> arr, const std::string& path) {
        Image2D img = numpy_to_image2d(arr);
        save_tiff(img, path);
    }, py::arg("array"), py::arg("path"), "Saves 2D float32 NumPy array as TIFF");

    m.def("load_volume_directory", [](const std::string& dir_path, const std::string& pattern) {
        Volume3D vol = load_volume_from_directory(dir_path, pattern);
        return volume3d_to_numpy(vol);
    }, py::arg("dir_path"), py::arg("pattern") = "", "Loads 3D volume from directory of TIFF slices");

    m.def("load_volume_multipage", [](const std::string& path) {
        Volume3D vol = load_volume_multipage(path);
        return volume3d_to_numpy(vol);
    }, py::arg("path"), "Loads 3D volume from multi-page TIFF");

    m.def("save_volume_directory", [](py::array_t<float> arr, const std::string& dir_path, const std::string& prefix) {
        Volume3D vol = numpy_to_volume3d(arr);
        save_volume_directory(vol, dir_path, prefix);
    }, py::arg("array"), py::arg("dir_path"), py::arg("prefix") = "slice_",
       "Saves 3D array as directory of TIFF slices");

    m.def("save_volume_multipage", [](py::array_t<float> arr, const std::string& path) {
        Volume3D vol = numpy_to_volume3d(arr);
        save_volume_multipage(vol, path);
    }, py::arg("array"), py::arg("path"), "Saves 3D array as multi-page TIFF");

    m.def("write_preview", [](py::array_t<float> arr, const std::string& path,
                              double display_min, double display_max) {
        Image2D img = numpy_to_image2d(arr);
        write_preview_tiff(img, path, display_min, display_max);
    }, py::arg("array"), py::arg("path"),
       py::arg("display_min") = std::numeric_limits<double>::quiet_NaN(),
       py::arg("display_max") = std::numeric_limits<double>::quiet_NaN(),
       "Writes 8-bit viewable grayscale TIFF preview with optional display window");

    // Noise estimation
    m.def("estimate_noise_sigma", [](py::handle input) -> double {
        if (py::isinstance<py::array>(input)) {
            py::array arr = py::reinterpret_borrow<py::array>(input);
            if (arr.ndim() == 2) {
                Image2D img = numpy_to_image2d(arr.cast<py::array_t<float>>());
                py::gil_scoped_release release;
                return ct::estimate_noise_sigma(img);
            } else if (arr.ndim() == 3) {
                Volume3D vol = numpy_to_volume3d(arr.cast<py::array_t<float>>());
                py::gil_scoped_release release;
                return ct::estimate_noise_sigma(vol);
            }
        }
        throw std::invalid_argument("Expected 2D or 3D float32 array for estimate_noise_sigma");
    }, py::arg("input"), "Estimates noise standard deviation using Immerkaer 3x3 kernel and Laplacian MAD");
}
