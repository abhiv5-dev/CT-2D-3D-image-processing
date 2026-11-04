#include "ct/image2d.hpp"
#include "ct/volume3d.hpp"
#include "ct/tiff_io.hpp"
#include "ct/preview.hpp"
#include "ct/intensity.hpp"
#include "ct/filters/gaussian.hpp"
#include "ct/filters/median.hpp"
#include "ct/filters/bilateral.hpp"
#include "ct/filters/diffusion.hpp"
#include "ct/filters/wiener.hpp"
#include "ct/filters/tv_chambolle.hpp"
#include "ct/filters/morphology.hpp"
#include "ct/filters/nlm.hpp"
#include "ct/path_utils.hpp"
#include "ct/noise.hpp"
#include "ct/config.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Natural numeric sort comparator for filenames (e.g. slice_2.tif before slice_10.tif)
bool natural_numeric_compare(const fs::path& a, const fs::path& b) {
    std::string sa = a.filename().string();
    std::string sb = b.filename().string();
    size_t ia = 0, ib = 0;
    while (ia < sa.size() && ib < sb.size()) {
        if (std::isdigit(static_cast<unsigned char>(sa[ia])) &&
            std::isdigit(static_cast<unsigned char>(sb[ib]))) {
            size_t end_a = ia, end_b = ib;
            while (end_a < sa.size() && std::isdigit(static_cast<unsigned char>(sa[end_a]))) ++end_a;
            while (end_b < sb.size() && std::isdigit(static_cast<unsigned char>(sb[end_b]))) ++end_b;
            unsigned long num_a = std::stoul(sa.substr(ia, end_a - ia));
            unsigned long num_b = std::stoul(sb.substr(ib, end_b - ib));
            if (num_a != num_b) return num_a < num_b;
            ia = end_a;
            ib = end_b;
        } else {
            if (sa[ia] != sb[ib]) return sa[ia] < sb[ib];
            ++ia;
            ++ib;
        }
    }
    return sa.size() < sb.size();
}

Image2D apply_2d_filter(const Image2D& img, const std::string& filter_name,
                        double sigma, int radius, double sigma_spatial, double sigma_range,
                        double kappa, int iters, double dt, double noise_var,
                        double weight, double eps, const std::string& element,
                        int patch_radius = 2, int search_radius = 5, double h = 0.05) {
    if (filter_name == "gaussian") {
        return ct::gaussian_blur(img, sigma, radius);
    } else if (filter_name == "median") {
        return ct::median_filter(img, radius > 0 ? radius : 1);
    } else if (filter_name == "bilateral") {
        return ct::bilateral_filter(img, sigma_spatial, sigma_range, radius);
    } else if (filter_name == "diffusion") {
        double time_step = (dt > 0.0) ? dt : 0.20;
        return ct::anisotropic_diffusion(img, kappa, iters, time_step);
    } else if (filter_name == "nlm") {
        return ct::nlm_filter(img, patch_radius > 0 ? patch_radius : 2,
                              search_radius > 0 ? search_radius : 5, h);
    } else if (filter_name == "wiener") {
        return ct::wiener_filter(img, radius > 0 ? radius : 1, noise_var);
    } else if (filter_name == "tv" || filter_name == "tv_chambolle") {
        return ct::tv_chambolle(img, weight, eps, iters);
    } else if (filter_name == "erosion") {
        return ct::grayscale_erosion(img, radius > 0 ? radius : 1, element);
    } else if (filter_name == "dilation") {
        return ct::grayscale_dilation(img, radius > 0 ? radius : 1, element);
    } else if (filter_name == "opening" || filter_name == "morphology") {
        return ct::grayscale_opening(img, radius > 0 ? radius : 1, element);
    } else if (filter_name == "closing") {
        return ct::grayscale_closing(img, radius > 0 ? radius : 1, element);
    } else if (filter_name == "white_tophat") {
        return ct::white_tophat(img, radius > 0 ? radius : 1, element);
    } else if (filter_name == "black_tophat") {
        return ct::black_tophat(img, radius > 0 ? radius : 1, element);
    } else {
        throw std::invalid_argument("Unknown filter: " + filter_name);
    }
}

Volume3D apply_3d_filter(const Volume3D& vol, const std::string& filter_name,
                        double sigma, int radius, double sigma_spatial, double sigma_range,
                        double kappa, int iters, double dt, double noise_var,
                        double weight, double eps, const std::string& element,
                        int patch_radius = 1, int search_radius = 3, double h = 0.05) {
    if (filter_name == "gaussian") {
        return ct::gaussian_blur(vol, sigma, radius);
    } else if (filter_name == "median") {
        return ct::median_filter(vol, radius > 0 ? radius : 1);
    } else if (filter_name == "bilateral") {
        return ct::bilateral_filter(vol, sigma_spatial, sigma_range, radius);
    } else if (filter_name == "diffusion") {
        double time_step = (dt > 0.0) ? dt : 0.15;
        return ct::anisotropic_diffusion(vol, kappa, iters, time_step);
    } else if (filter_name == "nlm") {
        return ct::nlm_filter(vol, patch_radius > 0 ? patch_radius : 1,
                              search_radius > 0 ? search_radius : 3, h);
    } else if (filter_name == "wiener") {
        return ct::wiener_filter(vol, radius > 0 ? radius : 1, noise_var);
    } else if (filter_name == "tv" || filter_name == "tv_chambolle") {
        return ct::tv_chambolle(vol, weight, eps, iters);
    } else if (filter_name == "erosion") {
        return ct::grayscale_erosion(vol, radius > 0 ? radius : 1, element);
    } else if (filter_name == "dilation") {
        return ct::grayscale_dilation(vol, radius > 0 ? radius : 1, element);
    } else if (filter_name == "opening" || filter_name == "morphology") {
        return ct::grayscale_opening(vol, radius > 0 ? radius : 1, element);
    } else if (filter_name == "closing") {
        return ct::grayscale_closing(vol, radius > 0 ? radius : 1, element);
    } else if (filter_name == "white_tophat") {
        return ct::white_tophat(vol, radius > 0 ? radius : 1, element);
    } else if (filter_name == "black_tophat") {
        return ct::black_tophat(vol, radius > 0 ? radius : 1, element);
    } else {
        throw std::invalid_argument("Unknown filter: " + filter_name);
    }
}

void print_help(const char* prog, std::ostream& os = std::cout) {
    os << "Usage: " << prog << " [options]\n\n"
       << "Single file mode:\n"
       << "  " << prog << " --in <file> [--out <file>] --filter <name|all> [params]\n\n"
       << "Folder-of-slices mode:\n"
       << "  " << prog << " --in-dir <dir> [--out-dir <dir>] --filter <name|all> --mode slice|3d [params]\n\n"
       << "Options:\n"
       << "  --in <path> / --in-dir <path>     Input TIFF file or directory of slices\n"
       << "  --out <path> / --out-dir <path>   Explicit output file or directory (single filter only;\n"
       << "                                    previews are written next to it as *_preview.tif and\n"
       << "                                    *_original_preview.tif; not allowed with --filter all)\n"
       << "  --output-root <dir>               Base directory for output hierarchy (default: ct_output)\n"
       << "  --filter <name>                   Filter name: gaussian, median, bilateral, diffusion,\n"
       << "                                    wiener, tv, opening, etc., or 'all'\n"
       << "  --mode slice|3d                   Folder processing mode (slice: slice-by-slice 2D; 3d: true volumetric 3D)\n"
       << "  --preview <path>                  Generate viewable 8-bit TIFF preview\n"
       << "  --range <min> <max>               Display range for preview normalization\n"
       << "  --quiet, -q                       Suppress informational console output (errors and warnings kept)\n"
       << "  --yes, -y                         Confirm any execution prompts automatically\n"
       << "  --sigma <float>                   Gaussian sigma (default: 1.0, uncalibrated)\n"
       << "  --radius <int>                    Filter window radius (default: calibrated per filter)\n"
       << "  --sigma-spatial <float>           Bilateral spatial sigma (default: 2.0)\n"
       << "  --sigma-range <float>             Bilateral range sigma (default: 3.0 * sigma_hat)\n"
       << "  --kappa <float>                   Diffusion kappa (default: 1.5 * sigma_hat)\n"
       << "  --iters <int>                     Iterations for diffusion (15) / TV (30)\n"
       << "  --dt <float>                      Time step for diffusion (default: 0.20)\n"
       << "  --noise-var <float>               Noise variance for Wiener filter (default: estimated)\n"
       << "  --weight <float>                  Regularization weight for TV Chambolle (default: 1.0 * sigma_hat)\n"
       << "  --eps <float>                     Convergence tolerance for TV Chambolle (default: 2e-4)\n"
       << "  --element box|ball                Structuring element for morphology (default: box)\n"
       << "  --patch-radius <int>              NLM patch radius (default: 2 for 2D, 1 for 3D)\n"
       << "  --search-radius <int>             NLM search window radius (default: 5 for 2D, 3 for 3D)\n"
       << "  --h <float>                       NLM filtering strength (default: 0.75 * sigma_hat)\n"
       << "  --help, -h                        Display this help message and exit\n\n"
       << "Note: If filter parameters are omitted, calibrated defaults are automatically\n"
       << "      computed from estimated image noise (sigma_hat) and reported on stdout.\n\n"
       << "Caution: The noise estimate measures remaining high-frequency content, not image quality.\n"
       << "         Smoother is not better: compare the *_preview.tif files for lost detail.\n";
}

std::string prompt_str(const std::string& msg, const std::string& def_val = "") {
    if (!def_val.empty()) {
        std::cout << msg << " [" << def_val << "]: " << std::flush;
    } else {
        std::cout << msg << ": " << std::flush;
    }
    std::string s;
    if (!std::getline(std::cin, s)) {
        return def_val;
    }
    size_t first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return def_val;
    size_t last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

double prompt_double(const std::string& msg, double def_val) {
    std::ostringstream ss;
    ss << def_val;
    std::string s = prompt_str(msg, ss.str());
    try { return std::stod(s); } catch (...) { return def_val; }
}

int prompt_int(const std::string& msg, int def_val) {
    std::string s = prompt_str(msg, std::to_string(def_val));
    try { return std::stoi(s); } catch (...) { return def_val; }
}

struct OutputTriple {
    fs::path result_file;
    fs::path preview_file;
    fs::path orig_preview_file;
};

// Generates unique, non-overwriting output paths for result, preview, and original preview
OutputTriple get_unique_output_triple(const fs::path& dir, const std::string& stem) {
    fs::create_directories(dir);
    fs::path r = dir / (stem + "_result_float32.tif");
    fs::path p = dir / (stem + "_preview.tif");
    fs::path op = dir / (stem + "_original_preview.tif");
    if (!fs::exists(r) && !fs::exists(p) && !fs::exists(op)) {
        return {r, p, op};
    }
    for (int i = 1; ; ++i) {
        fs::path ri = dir / (stem + "_result_float32_" + std::to_string(i) + ".tif");
        fs::path pi = dir / (stem + "_preview_" + std::to_string(i) + ".tif");
        fs::path opi = dir / (stem + "_original_preview_" + std::to_string(i) + ".tif");
        if (!fs::exists(ri) && !fs::exists(pi) && !fs::exists(opi)) {
            return {ri, pi, opi};
        }
    }
}

// Generates unique, non-overwriting output paths when an explicit --out FILE is provided
OutputTriple get_unique_output_triple_for_explicit_file(const fs::path& explicit_file) {
    fs::path dir = explicit_file.parent_path();
    if (dir.empty()) dir = ".";
    fs::create_directories(dir);
    std::string stem = explicit_file.stem().string();
    std::string ext = explicit_file.extension().string();
    if (ext.empty()) ext = ".tif";

    fs::path r = dir / (stem + ext);
    fs::path p = dir / (stem + "_preview" + ext);
    fs::path op = dir / (stem + "_original_preview" + ext);
    if (!fs::exists(r) && !fs::exists(p) && !fs::exists(op)) {
        return {r, p, op};
    }
    for (int i = 1; ; ++i) {
        fs::path ri = dir / (stem + "_" + std::to_string(i) + ext);
        fs::path pi = dir / (stem + "_preview_" + std::to_string(i) + ext);
        fs::path opi = dir / (stem + "_original_preview_" + std::to_string(i) + ext);
        if (!fs::exists(ri) && !fs::exists(pi) && !fs::exists(opi)) {
            return {ri, pi, opi};
        }
    }
}

std::string get_input_stem(const fs::path& p) {
    fs::path clean = p;
    while (!clean.empty() && clean.filename().empty()) {
        clean = clean.parent_path();
    }
    if (fs::is_directory(clean)) {
        return clean.filename().string();
    }
    std::string s = clean.stem().string();
    return s.empty() ? "ct_data" : s;
}

struct BenchmarkRow {
    std::string filter_name;
    double time_ms = 0.0;
    double noise_before = 0.0;
    double noise_after = 0.0;
    double mean_before = 0.0;
    double mean_after = 0.0;
    double mean_change = 0.0;
};

void run_all_filters_2d(const Image2D& img_in, const fs::path& out_dir, bool quiet) {
    fs::create_directories(out_dir);

    struct FilterSpec {
        std::string name;
        std::string dirname;
        std::string display_name;
    };
    std::vector<FilterSpec> specs = {
        {"gaussian", "gaussian", "Gaussian Blur"},
        {"median", "median", "Median Filter"},
        {"bilateral", "bilateral", "Bilateral Filter"},
        {"diffusion", "diffusion", "Anisotropic Diffusion"},
        {"nlm", "nlm", "Non-Local Means (NLM)"},
        {"wiener", "wiener", "Adaptive Wiener Filter"},
        {"tv", "tv_chambolle", "Total Variation (TV Chambolle)"},
        {"opening", "morphology", "Grayscale Morphology (Opening)"}
    };

    double sigma_hat = ct::estimate_noise_sigma(img_in);
    auto [p1, p99] = ct::compute_percentiles_1_99(img_in);
    double mean_before = img_in.mean();

    ct::FilterConfig cfg = ct::FilterConfig::load();
    ct::UserFilterFlags flags;
    std::vector<BenchmarkRow> rows;

    for (size_t i = 0; i < specs.size(); ++i) {
        const auto& spec = specs[i];
        if (!quiet) {
            std::cout << "[" << (i + 1) << "/" << specs.size() << "] Running " << spec.display_name << "...\n" << std::flush;
        }

        ct::ResolvedParams p = ct::resolve_filter_parameters(spec.name, sigma_hat, flags, cfg, false);

        auto t0 = std::chrono::high_resolution_clock::now();
        Image2D img_out = apply_2d_filter(img_in, p.filter_name, p.sigma, p.radius, p.sigma_spatial, p.sigma_range,
                                          p.kappa, p.iters, p.dt, p.noise_var, p.weight, p.eps, p.element,
                                          p.patch_radius, p.search_radius, p.h);
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        double noise_after = ct::estimate_noise_sigma(img_out);
        double mean_after = img_out.mean();
        double mean_diff = mean_after - mean_before;

        rows.push_back({spec.display_name, elapsed_ms, sigma_hat, noise_after, mean_before, mean_after, mean_diff});

        // Write outputs into ct_output/X/<filter>/
        fs::path f_dir = out_dir / spec.dirname;
        OutputTriple triple = get_unique_output_triple(f_dir, spec.dirname);
        ct::save_tiff(img_out, triple.result_file.string());
        ct::write_preview_tiff(img_out, triple.preview_file.string(), p1, p99);
        ct::write_preview_tiff(img_in, triple.orig_preview_file.string(), p1, p99);
    }

    // Write summary.csv
    fs::path csv_path = out_dir / "summary.csv";
    std::ofstream csv(csv_path);
    if (csv.is_open()) {
        csv << "filter,time_ms,noise_estimate_before,noise_estimate_after,mean_before,mean_after,mean_change\n";
        for (const auto& r : rows) {
            csv << r.filter_name << ","
                << std::fixed << std::setprecision(2) << r.time_ms << ","
                << std::setprecision(6) << r.noise_before << ","
                << r.noise_after << ","
                << r.mean_before << ","
                << r.mean_after << ","
                << r.mean_change << "\n";
        }
    }

    // Print summary table
    if (!quiet) {
        std::cout << "\n" << std::string(98, '=') << "\n";
        std::cout << std::left << std::setw(32) << "Filter"
                  << " | " << std::right << std::setw(10) << "Time (ms)"
                  << " | " << std::left << std::setw(28) << "Noise (sigma_hat)"
                  << " | " << std::right << std::setw(14) << "Mean Change" << "\n";
        std::cout << std::string(98, '-') << "\n";
        for (const auto& r : rows) {
            std::ostringstream ss_noise;
            ss_noise << std::fixed << std::setprecision(6) << r.noise_before
                     << " -> " << r.noise_after;
            std::cout << std::left << std::setw(32) << r.filter_name
                      << " | " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << r.time_ms
                      << " | " << std::left << std::setw(28) << ss_noise.str()
                      << " | " << std::right << std::setw(14) << std::showpos << std::fixed << std::setprecision(6) << r.mean_change << std::noshowpos << "\n";
        }
        std::cout << std::string(98, '=') << "\n";
        std::cout << "All results saved under: " << out_dir.string() << "\n";
        std::cout << "Summary table written to: " << csv_path.string() << "\n";
        std::cout << "Caution: The noise estimate measures remaining high-frequency content, not image quality. Smoother is not better: compare the *_preview.tif files for lost detail.\n";
        std::cout << "Open the *_preview.tif files to look at the result; *_result_float32.tif holds the raw values.\n";
    }
}

void run_all_filters_folder(const std::vector<fs::path>& slice_files, const fs::path& out_dir,
                            const std::string& folder_stem, bool is_3d, bool quiet) {
    fs::create_directories(out_dir);

    struct FilterSpec {
        std::string name;
        std::string dirname;
        std::string display_name;
    };
    std::vector<FilterSpec> specs = {
        {"gaussian", "gaussian", "Gaussian Blur"},
        {"median", "median", "Median Filter"},
        {"bilateral", "bilateral", "Bilateral Filter"},
        {"diffusion", "diffusion", "Anisotropic Diffusion"},
        {"nlm", "nlm", "Non-Local Means (NLM)"},
        {"wiener", "wiener", "Adaptive Wiener Filter"},
        {"tv", "tv_chambolle", "Total Variation (TV Chambolle)"},
        {"opening", "morphology", "Grayscale Morphology (Opening)"}
    };

    Image2D first_slice = ct::load_tiff(slice_files[0].string());
    const int W0 = first_slice.width();
    const int H0 = first_slice.height();
    const size_t N_slices = slice_files.size();
    size_t mid_idx = N_slices / 2;
    Image2D mid_orig = ct::load_tiff(slice_files[mid_idx].string());
    auto [p1, p99] = ct::compute_percentiles_1_99(mid_orig);
    double sigma_hat = ct::estimate_noise_sigma(first_slice);

    ct::FilterConfig cfg = ct::FilterConfig::load();
    ct::UserFilterFlags flags;
    std::vector<BenchmarkRow> rows;

    for (size_t i = 0; i < specs.size(); ++i) {
        const auto& spec = specs[i];
        if (!quiet) {
            std::cout << "[" << (i + 1) << "/" << specs.size() << "] Running " << spec.display_name << "...\n" << std::flush;
        }

        ct::ResolvedParams p = ct::resolve_filter_parameters(spec.name, sigma_hat, flags, cfg, is_3d);

        fs::path f_dir = out_dir / spec.dirname;
        fs::create_directories(f_dir);

        double total_sum_before = 0.0;
        double total_sum_after = 0.0;
        size_t total_pix = 0;
        Image2D mid_preview;

        auto t0 = std::chrono::high_resolution_clock::now();

        if (is_3d) {
            Volume3D vol(W0, H0, static_cast<int>(N_slices));
            vol.set_slice(0, first_slice);
            total_sum_before += first_slice.mean() * first_slice.size();
            total_pix += first_slice.size();

            for (size_t idx = 1; idx < N_slices; ++idx) {
                Image2D s = ct::load_tiff(slice_files[idx].string());
                vol.set_slice(static_cast<int>(idx), s);
                total_sum_before += s.mean() * s.size();
                total_pix += s.size();
            }

            Volume3D vol_out = apply_3d_filter(vol, p.filter_name, p.sigma, p.radius, p.sigma_spatial, p.sigma_range,
                                               p.kappa, p.iters, p.dt, p.noise_var, p.weight, p.eps, p.element,
                                               p.patch_radius, p.search_radius, p.h);
            total_sum_after = vol_out.mean() * total_pix;

            for (size_t idx = 0; idx < N_slices; ++idx) {
                Image2D s_out = vol_out.slice(static_cast<int>(idx));
                fs::path out_file = f_dir / slice_files[idx].filename();
                ct::save_tiff(s_out, out_file.string());
                if (idx == mid_idx) mid_preview = s_out;
            }
        } else {
            for (size_t idx = 0; idx < N_slices; ++idx) {
                Image2D s = (idx == 0) ? first_slice : ct::load_tiff(slice_files[idx].string());
                total_sum_before += s.mean() * s.size();
                total_pix += s.size();

                Image2D s_out = apply_2d_filter(s, p.filter_name, p.sigma, p.radius, p.sigma_spatial, p.sigma_range,
                                                p.kappa, p.iters, p.dt, p.noise_var, p.weight, p.eps, p.element,
                                                p.patch_radius, p.search_radius, p.h);
                total_sum_after += s_out.mean() * s_out.size();

                fs::path out_file = f_dir / slice_files[idx].filename();
                ct::save_tiff(s_out, out_file.string());
                if (idx == mid_idx) mid_preview = s_out;
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        double mean_before = total_sum_before / total_pix;
        double mean_after = total_sum_after / total_pix;
        double noise_after = ct::estimate_noise_sigma(mid_preview);
        double mean_diff = mean_after - mean_before;

        rows.push_back({spec.display_name, elapsed_ms, sigma_hat, noise_after, mean_before, mean_after, mean_diff});

        OutputTriple triple = get_unique_output_triple(f_dir, folder_stem);
        ct::write_preview_tiff(mid_preview, triple.preview_file.string(), p1, p99);
        ct::write_preview_tiff(mid_orig, triple.orig_preview_file.string(), p1, p99);
    }

    // Write summary.csv
    fs::path csv_path = out_dir / "summary.csv";
    std::ofstream csv(csv_path);
    if (csv.is_open()) {
        csv << "filter,time_ms,noise_estimate_before,noise_estimate_after,mean_before,mean_after,mean_change\n";
        for (const auto& r : rows) {
            csv << r.filter_name << ","
                << std::fixed << std::setprecision(2) << r.time_ms << ","
                << std::setprecision(6) << r.noise_before << ","
                << r.noise_after << ","
                << r.mean_before << ","
                << r.mean_after << ","
                << r.mean_change << "\n";
        }
    }

    if (!quiet) {
        std::cout << "\n" << std::string(98, '=') << "\n";
        std::cout << std::left << std::setw(32) << "Filter"
                  << " | " << std::right << std::setw(10) << "Time (ms)"
                  << " | " << std::left << std::setw(28) << "Noise (sigma_hat)"
                  << " | " << std::right << std::setw(14) << "Mean Change" << "\n";
        std::cout << std::string(98, '-') << "\n";
        for (const auto& r : rows) {
            std::ostringstream ss_noise;
            ss_noise << std::fixed << std::setprecision(6) << r.noise_before
                     << " -> " << r.noise_after;
            std::cout << std::left << std::setw(32) << r.filter_name
                      << " | " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << r.time_ms
                      << " | " << std::left << std::setw(28) << ss_noise.str()
                      << " | " << std::right << std::setw(14) << std::showpos << std::fixed << std::setprecision(6) << r.mean_change << std::noshowpos << "\n";
        }
        std::cout << std::string(98, '=') << "\n";
        std::cout << "All results saved under: " << out_dir.string() << "\n";
        std::cout << "Summary table written to: " << csv_path.string() << "\n";
        std::cout << "Caution: The noise estimate measures remaining high-frequency content, not image quality. Smoother is not better: compare the *_preview.tif files for lost detail.\n";
        std::cout << "Open the *_preview.tif files to look at the result; *_result_float32.tif holds the raw values.\n";
    }
}

int run_interactive() {
    std::cout << "=== CT Filter Interactive CLI ===\n";
    std::string in_path;
    bool path_ok = false;

    for (int attempt = 1; attempt <= 5; ++attempt) {
        std::cout << "Enter path to TIFF image or folder of slices (or 'q' to quit): " << std::flush;
        std::string raw_line;
        if (!std::getline(std::cin, raw_line)) {
            std::cerr << "\nEnd of input reached.\n";
            return 1;
        }

        std::string trimmed = raw_line;
        size_t f = trimmed.find_first_not_of(" \t\r\n");
        if (f != std::string::npos) {
            size_t l = trimmed.find_last_not_of(" \t\r\n");
            trimmed = trimmed.substr(f, l - f + 1);
        }
        if (trimmed == "q" || trimmed == "Q") {
            std::cout << "Operation cancelled by user.\n";
            return 0;
        }

        std::string norm = ct::normalize_input_path(raw_line);
        if (norm.empty()) {
            std::cerr << "Invalid path: Path cannot be empty. Please try again (attempt "
                      << attempt << "/5).\n";
            continue;
        }

        std::error_code ec;
        if (!fs::exists(norm, ec)) {
            std::cerr << "Invalid path: Path does not exist (\"" << norm
                      << "\"). Please try again (attempt " << attempt << "/5).\n";
            continue;
        }

        if (fs::is_directory(norm, ec)) {
            bool has_tiff = false;
            for (const auto& entry : fs::directory_iterator(norm, ec)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".tif" || ext == ".tiff") {
                        has_tiff = true;
                        break;
                    }
                }
            }
            if (!has_tiff) {
                std::cerr << "Invalid path: Directory contains no .tif or .tiff slice files (\""
                          << norm << "\"). Please try again (attempt " << attempt << "/5).\n";
                continue;
            }
        } else if (!fs::is_regular_file(norm, ec)) {
            std::cerr << "Invalid path: Path is neither a regular file nor a directory (\""
                      << norm << "\"). Please try again (attempt " << attempt << "/5).\n";
            continue;
        }

        in_path = norm;
        path_ok = true;
        break;
    }

    if (!path_ok) {
        std::cerr << "Error: Maximum attempts (5) reached without a valid path.\n";
        return 1;
    }

    bool is_dir = fs::is_directory(in_path);
    std::string stem = is_dir ? fs::path(in_path).filename().string() : fs::path(in_path).stem().string();
    if (stem.empty()) stem = "ct_data";
    fs::path out_base_dir = fs::path("./ct_output") / stem;
    fs::create_directories(out_base_dir);

    ct::FilterConfig cfg = ct::FilterConfig::load();

    while (true) {
        std::cout << "\nAvailable Filters:\n"
                  << "  1. Gaussian Blur\n"
                  << "  2. Median Filter\n"
                  << "  3. Bilateral Filter\n"
                  << "  4. Anisotropic Diffusion\n"
                  << "  5. Non-Local Means (NLM)\n"
                  << "  6. Adaptive Wiener Filter\n"
                  << "  7. Total Variation (TV Chambolle)\n"
                  << "  8. Grayscale Morphology\n"
                  << "  9. All Filters (benchmark sweep)\n";
        int choice = prompt_int("Select filter (1-9)", 1);
        if (choice < 1 || choice > 9) {
            std::cerr << "Error: Invalid filter choice.\n";
            return 1;
        }

        if (choice == 9) {
            if (!is_dir) {
                Image2D img_in = ct::load_tiff(in_path);
                run_all_filters_2d(img_in, out_base_dir, false);
            } else {
                std::cout << "All Filters mode on folder currently runs per-slice benchmark on middle slice.\n";
                // Gather slices
                std::vector<fs::path> slice_files;
                for (const auto& entry : fs::directory_iterator(in_path)) {
                    if (entry.is_regular_file()) {
                        std::string ext = entry.path().extension().string();
                        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                        if (ext == ".tif" || ext == ".tiff") slice_files.push_back(entry.path());
                    }
                }
                std::sort(slice_files.begin(), slice_files.end(), natural_numeric_compare);
                if (slice_files.size() > 20) {
                    std::cout << "Notice: Folder contains " << slice_files.size() << " slices (>20). Estimated NLM runtime is > 1 minute.\n";
                    std::string conf = prompt_str("Proceed with All Filters on the entire folder? [y/N]", "N");
                    if (conf.empty() || (conf[0] != 'y' && conf[0] != 'Y')) {
                        std::cout << "Operation cancelled.\n";
                        continue;
                    }
                }
                Image2D mid_s = ct::load_tiff(slice_files[slice_files.size() / 2].string());
                run_all_filters_2d(mid_s, out_base_dir, false);
            }

            std::string again = prompt_str("Try another filter? [y/N]", "N");
            if (again.empty() || (again[0] != 'y' && again[0] != 'Y')) {
                break;
            }
            continue;
        }

        std::string filter_name;
        switch (choice) {
            case 1: filter_name = "gaussian"; break;
            case 2: filter_name = "median"; break;
            case 3: filter_name = "bilateral"; break;
            case 4: filter_name = "diffusion"; break;
            case 5: filter_name = "nlm"; break;
            case 6: filter_name = "wiener"; break;
            case 7: filter_name = "tv"; break;
            case 8: filter_name = "opening"; break;
        }

        // Estimate noise on input
        double sigma_hat = 0.0;
        Image2D img_in;
        if (!is_dir) {
            img_in = ct::load_tiff(in_path);
            sigma_hat = ct::estimate_noise_sigma(img_in);
        } else {
            for (const auto& entry : fs::directory_iterator(in_path)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".tif" || ext == ".tiff") {
                        Image2D tmp = ct::load_tiff(entry.path().string());
                        sigma_hat = ct::estimate_noise_sigma(tmp);
                        break;
                    }
                }
            }
        }

        std::string use_calib_str = prompt_str("Use calibrated parameters? [Y/n]", "Y");
        bool use_calib = (use_calib_str.empty() || use_calib_str[0] == 'y' || use_calib_str[0] == 'Y');

        ct::UserFilterFlags flags;
        ct::ResolvedParams p = ct::resolve_filter_parameters(filter_name, sigma_hat, flags, cfg, is_dir);

        if (use_calib) {
            std::cout << "Using calibrated parameters (sigma_hat = " << std::fixed << std::setprecision(6) << sigma_hat << "):\n";
            p.print_notes(std::cout);
        } else {
            switch (choice) {
                case 1:
                    p.sigma = prompt_double("Gaussian sigma", p.sigma);
                    p.radius = prompt_int("Filter radius (0 for auto)", p.radius);
                    break;
                case 2:
                    p.radius = prompt_int("Median radius", p.radius);
                    break;
                case 3:
                    p.sigma_spatial = prompt_double("Spatial sigma", p.sigma_spatial);
                    p.sigma_range = prompt_double("Range sigma", p.sigma_range);
                    p.radius = prompt_int("Filter radius (0 for auto)", p.radius);
                    break;
                case 4:
                    p.kappa = prompt_double("Diffusion kappa", p.kappa);
                    p.iters = prompt_int("Iterations", p.iters);
                    p.dt = prompt_double("Time step dt (<=0 for auto)", p.dt);
                    break;
                case 5:
                    p.h = prompt_double("Filter parameter h", p.h);
                    p.patch_radius = prompt_int("Patch radius", p.patch_radius);
                    p.search_radius = prompt_int("Search radius", p.search_radius);
                    break;
                case 6:
                    p.radius = prompt_int("Wiener radius", p.radius);
                    p.noise_var = prompt_double("Noise variance (-1 for auto)", p.noise_var);
                    break;
                case 7:
                    p.weight = prompt_double("Regularization weight", p.weight);
                    p.iters = prompt_int("Iterations", p.iters);
                    break;
                case 8:
                    p.filter_name = prompt_str("Morphology operation (opening/closing/erosion/dilation)", "opening");
                    p.radius = prompt_int("Radius", p.radius);
                    p.element = prompt_str("Structuring element (box/ball)", p.element);
                    break;
            }
        }

        std::cout << std::fixed << std::setprecision(6);
        if (!is_dir) {
            auto [p1, p99] = ct::compute_percentiles_1_99(img_in);
            double mean_before = img_in.mean();

            auto t0 = std::chrono::high_resolution_clock::now();
            Image2D img_out = apply_2d_filter(img_in, p.filter_name, p.sigma, p.radius, p.sigma_spatial, p.sigma_range,
                                              p.kappa, p.iters, p.dt, p.noise_var, p.weight, p.eps, p.element,
                                              p.patch_radius, p.search_radius, p.h);
            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            double mean_after = img_out.mean();
            double noise_after = ct::estimate_noise_sigma(img_out);

            // Single line summary
            std::cout << "Mean: " << mean_before << " -> " << mean_after
                      << " | Noise: " << sigma_hat << " -> " << noise_after
                      << " | Time: " << std::setprecision(2) << elapsed_ms << " ms\n";

            OutputTriple triple = get_unique_output_triple(out_base_dir, stem);
            ct::save_tiff(img_out, triple.result_file.string());
            ct::write_preview_tiff(img_out, triple.preview_file.string(), p1, p99);
            ct::write_preview_tiff(img_in, triple.orig_preview_file.string(), p1, p99);

            std::cout << "\nOutput saved:\n"
                      << "  Raw result:       " << triple.result_file.string() << "\n"
                      << "  Filtered preview: " << triple.preview_file.string() << "\n"
                      << "  Original preview: " << triple.orig_preview_file.string() << "\n"
                      << "Open the *_preview.tif files to look at the result; *_result_float32.tif holds the raw values.\n";
        } else {
            std::vector<fs::path> slice_files;
            for (const auto& entry : fs::directory_iterator(in_path)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".tif" || ext == ".tiff") slice_files.push_back(entry.path());
                }
            }
            if (slice_files.empty()) {
                std::cerr << "Error: No TIFF slices found in directory: " << in_path << "\n";
                return 1;
            }
            std::sort(slice_files.begin(), slice_files.end(), natural_numeric_compare);
            std::cout << "Discovered " << slice_files.size() << " slices in sorted numeric order.\n";

            fs::path out_sub = out_base_dir / filter_name;
            fs::create_directories(out_sub);

            double total_sum_before = 0.0;
            double total_sum_after = 0.0;
            size_t total_pix = 0;

            Image2D mid_orig = ct::load_tiff(slice_files[slice_files.size() / 2].string());
            Image2D mid_preview;
            size_t mid_idx = slice_files.size() / 2;

            auto t0 = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < slice_files.size(); ++i) {
                Image2D s = ct::load_tiff(slice_files[i].string());
                total_sum_before += s.mean() * s.size();
                total_pix += s.size();

                Image2D s_out = apply_2d_filter(s, p.filter_name, p.sigma, p.radius, p.sigma_spatial, p.sigma_range,
                                                p.kappa, p.iters, p.dt, p.noise_var, p.weight, p.eps, p.element,
                                                p.patch_radius, p.search_radius, p.h);
                total_sum_after += s_out.mean() * s_out.size();

                fs::path out_slice = out_sub / slice_files[i].filename();
                ct::save_tiff(s_out, out_slice.string());
                if (i == mid_idx) mid_preview = s_out;
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            double mean_before = total_sum_before / total_pix;
            double mean_after = total_sum_after / total_pix;
            double noise_after = ct::estimate_noise_sigma(mid_preview);

            std::cout << "Mean: " << mean_before << " -> " << mean_after
                      << " | Noise: " << sigma_hat << " -> " << noise_after
                      << " | Time: " << std::setprecision(2) << elapsed_ms << " ms\n";

            auto [p1, p99] = ct::compute_percentiles_1_99(mid_orig);
            OutputTriple triple = get_unique_output_triple(out_sub, stem);
            ct::write_preview_tiff(mid_preview, triple.preview_file.string(), p1, p99);
            ct::write_preview_tiff(mid_orig, triple.orig_preview_file.string(), p1, p99);

            std::cout << "\nOutput saved:\n"
                      << "  Filtered slices:  " << out_sub.string() << "\n"
                      << "  Filtered preview: " << triple.preview_file.string() << "\n"
                      << "  Original preview: " << triple.orig_preview_file.string() << "\n"
                      << "Open the *_preview.tif files to look at the result; *_result_float32.tif holds the raw values.\n";
        }

        std::string again = prompt_str("Try another filter? [y/N]", "N");
        if (again.empty() || (again[0] != 'y' && again[0] != 'Y')) {
            break;
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 1) {
        return run_interactive();
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help(argv[0], std::cout);
            return 0;
        }
    }

    std::string in_path;
    std::string out_path;
    std::string output_root = "ct_output";
    std::string preview_path;
    std::string filter_name;
    std::string mode = "slice"; // default for folder mode
    bool folder_flag = false;
    bool quiet = false;
    bool auto_yes = false;

    double range_min = std::numeric_limits<double>::quiet_NaN();
    double range_max = std::numeric_limits<double>::quiet_NaN();

    // User-specified options and tracking flags
    ct::UserFilterFlags flags;
    double user_sigma = 1.0;
    int user_radius = 0;
    double user_sigma_spatial = 2.0;
    double user_sigma_range = 0.1;
    double user_kappa = 0.05;
    int user_iters = 10;
    double user_dt = -1.0;
    double user_noise_var = -1.0;
    double user_weight = 0.1;
    double user_eps = 2.0e-4;
    std::string user_element = "box";
    int user_patch_radius = 2;
    int user_search_radius = 5;
    double user_h = 0.05;

    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            try {
                if (arg == "--in" && i + 1 < argc) in_path = argv[++i];
                else if (arg == "--in-dir" && i + 1 < argc) { in_path = argv[++i]; folder_flag = true; }
                else if (arg == "--out" && i + 1 < argc) out_path = argv[++i];
                else if (arg == "--out-dir" && i + 1 < argc) { out_path = argv[++i]; folder_flag = true; }
                else if (arg == "--output-root" && i + 1 < argc) output_root = argv[++i];
                else if (arg == "--filter" && i + 1 < argc) filter_name = argv[++i];
                else if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
                else if (arg == "--preview" && i + 1 < argc) preview_path = argv[++i];
                else if (arg == "--quiet" || arg == "-q") quiet = true;
                else if (arg == "--yes" || arg == "-y") auto_yes = true;
                else if (arg == "--range" && i + 2 < argc) {
                    range_min = std::stod(argv[++i]);
                    range_max = std::stod(argv[++i]);
                }
                else if (arg == "--sigma" && i + 1 < argc) { user_sigma = std::stod(argv[++i]); flags.has_sigma = true; }
                else if (arg == "--radius" && i + 1 < argc) { user_radius = std::stoi(argv[++i]); flags.has_radius = true; }
                else if (arg == "--sigma-spatial" && i + 1 < argc) { user_sigma_spatial = std::stod(argv[++i]); flags.has_sigma_spatial = true; }
                else if (arg == "--sigma-range" && i + 1 < argc) { user_sigma_range = std::stod(argv[++i]); flags.has_sigma_range = true; }
                else if (arg == "--kappa" && i + 1 < argc) { user_kappa = std::stod(argv[++i]); flags.has_kappa = true; }
                else if (arg == "--iters" && i + 1 < argc) { user_iters = std::stoi(argv[++i]); flags.has_iters = true; }
                else if (arg == "--dt" && i + 1 < argc) { user_dt = std::stod(argv[++i]); flags.has_dt = true; }
                else if (arg == "--noise-var" && i + 1 < argc) { user_noise_var = std::stod(argv[++i]); flags.has_noise_var = true; }
                else if (arg == "--weight" && i + 1 < argc) { user_weight = std::stod(argv[++i]); flags.has_weight = true; }
                else if (arg == "--eps" && i + 1 < argc) { user_eps = std::stod(argv[++i]); flags.has_eps = true; }
                else if (arg == "--element" && i + 1 < argc) { user_element = argv[++i]; flags.has_element = true; }
                else if (arg == "--patch-radius" && i + 1 < argc) { user_patch_radius = std::stoi(argv[++i]); flags.has_patch_radius = true; }
                else if (arg == "--search-radius" && i + 1 < argc) { user_search_radius = std::stoi(argv[++i]); flags.has_search_radius = true; }
                else if (arg == "--h" && i + 1 < argc) { user_h = std::stod(argv[++i]); flags.has_h = true; }
                else {
                    std::cerr << "Unknown or invalid argument: " << arg << "\n";
                    print_help(argv[0], std::cerr);
                    return 2;
                }
            } catch (const std::exception& e) {
                std::cerr << "Invalid value for argument " << arg << ": " << e.what() << "\n";
                print_help(argv[0], std::cerr);
                return 2;
            }
        }

        if (in_path.empty() || filter_name.empty()) {
            std::cerr << "Error: Input and filter are required.\n";
            print_help(argv[0], std::cerr);
            return 2;
        }

        bool is_dir = folder_flag || fs::is_directory(in_path);
        std::string stem = get_input_stem(in_path);

        std::string filter_lower = filter_name;
        std::transform(filter_lower.begin(), filter_lower.end(), filter_lower.begin(), ::tolower);

        // ==============================================================================
        // --filter all Mode
        // ==============================================================================
        if (filter_lower == "all") {
            if (!out_path.empty()) {
                std::cerr << "Error: --out is not used with --filter all; results go to "
                          << (fs::path(output_root) / stem).string() << "/\n";
                return 2;
            }
            fs::path ct_dir = fs::path(output_root) / stem;
            fs::create_directories(ct_dir);

            if (!is_dir) {
                Image2D img_in = ct::load_tiff(in_path);
                run_all_filters_2d(img_in, ct_dir, quiet);
                return 0;
            } else {
                std::vector<fs::path> slice_files;
                for (const auto& entry : fs::directory_iterator(in_path)) {
                    if (entry.is_regular_file()) {
                        std::string ext = entry.path().extension().string();
                        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                        if (ext == ".tif" || ext == ".tiff") slice_files.push_back(entry.path());
                    }
                }
                std::sort(slice_files.begin(), slice_files.end(), natural_numeric_compare);
                if (slice_files.empty()) {
                    throw std::runtime_error("No TIFF slice files found in directory: " + in_path);
                }
                if (slice_files.size() > 20 && !auto_yes) {
                    std::cout << "Warning: Input contains " << slice_files.size()
                              << " slices (>20). Estimated NLM run time exceeds 1 minute.\n"
                              << "Proceed with All Filters? [y/N]: " << std::flush;
                    std::string resp;
                    if (!std::getline(std::cin, resp) || (resp != "y" && resp != "Y")) {
                        std::cout << "Operation cancelled by user.\n";
                        return 0;
                    }
                }
                run_all_filters_folder(slice_files, ct_dir, stem, mode == "3d", quiet);
                return 0;
            }
        }

        ct::FilterConfig cfg = ct::FilterConfig::load();

        // ==============================================================================
        // Folder-of-Slices Mode (B4)
        // ==============================================================================
        if (is_dir) {
            std::vector<fs::path> slice_files;
            for (const auto& entry : fs::directory_iterator(in_path)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".tif" || ext == ".tiff") {
                        slice_files.push_back(entry.path());
                    }
                }
            }

            if (slice_files.empty()) {
                throw std::runtime_error("No TIFF slice files found in directory: " + in_path);
            }

            std::sort(slice_files.begin(), slice_files.end(), natural_numeric_compare);
            if (!quiet) {
                std::cout << "Discovered " << slice_files.size() << " slices in sorted numeric order.\n";
            }

            // Load first slice to determine expected dimensions and estimate noise
            Image2D first_slice = ct::load_tiff(slice_files[0].string());
            const int W0 = first_slice.width();
            const int H0 = first_slice.height();
            double sigma_hat = ct::estimate_noise_sigma(first_slice);

            bool is_3d_mode = (mode == "3d");
            ct::ResolvedParams p = ct::resolve_filter_parameters(filter_name, sigma_hat, flags, cfg, is_3d_mode);

            // Apply explicit CLI overrides
            if (flags.has_sigma) p.sigma = user_sigma;
            if (flags.has_radius) p.radius = user_radius;
            if (flags.has_sigma_spatial) p.sigma_spatial = user_sigma_spatial;
            if (flags.has_sigma_range) p.sigma_range = user_sigma_range;
            if (flags.has_kappa) p.kappa = user_kappa;
            if (flags.has_iters) p.iters = user_iters;
            if (flags.has_dt) p.dt = user_dt;
            if (flags.has_noise_var) p.noise_var = user_noise_var;
            if (flags.has_weight) p.weight = user_weight;
            if (flags.has_eps) p.eps = user_eps;
            if (flags.has_element) p.element = user_element;
            if (flags.has_patch_radius) p.patch_radius = user_patch_radius;
            if (flags.has_search_radius) p.search_radius = user_search_radius;
            if (flags.has_h) p.h = user_h;

            if (!quiet) {
                p.print_notes(std::cout);
            }

            fs::path target_out_dir = out_path.empty() ? (fs::path(output_root) / stem / filter_lower) : fs::path(out_path);
            fs::create_directories(target_out_dir);

            double total_sum_before = 0.0;
            double total_sum_after = 0.0;
            size_t total_pix = 0;
            Image2D mid_preview;
            size_t mid_idx = slice_files.size() / 2;
            Image2D mid_orig = ct::load_tiff(slice_files[mid_idx].string());

            auto t0 = std::chrono::high_resolution_clock::now();

            if (mode == "slice") {
                for (size_t idx = 0; idx < slice_files.size(); ++idx) {
                    const auto& item_path = slice_files[idx];
                    Image2D s = (idx == 0) ? first_slice : ct::load_tiff(item_path.string());

                    if (s.width() != W0 || s.height() != H0) {
                        throw std::runtime_error("Size mismatch in directory slices: " + item_path.filename().string() +
                                                 " is (" + std::to_string(s.width()) + "x" + std::to_string(s.height()) +
                                                 ") but expected (" + std::to_string(W0) + "x" + std::to_string(H0) + ")");
                    }
                    total_sum_before += s.mean() * s.size();
                    total_pix += s.size();

                    Image2D s_out = apply_2d_filter(s, p.filter_name, p.sigma, p.radius, p.sigma_spatial, p.sigma_range,
                                                    p.kappa, p.iters, p.dt, p.noise_var, p.weight, p.eps, p.element,
                                                    p.patch_radius, p.search_radius, p.h);
                    total_sum_after += s_out.mean() * s_out.size();

                    fs::path out_file = target_out_dir / item_path.filename();
                    ct::save_tiff(s_out, out_file.string());
                    if (idx == mid_idx) mid_preview = s_out;
                }

            } else if (mode == "3d") {
                if (!quiet) {
                    std::cout << "Assembling " << slice_files.size() << " slices into 3D volume ("
                              << W0 << "x" << H0 << "x" << slice_files.size() << ")...\n";
                }
                Volume3D vol(W0, H0, static_cast<int>(slice_files.size()));
                vol.set_slice(0, first_slice);
                total_sum_before += first_slice.mean() * first_slice.size();
                total_pix += first_slice.size();

                for (size_t idx = 1; idx < slice_files.size(); ++idx) {
                    const auto& item_path = slice_files[idx];
                    Image2D s = ct::load_tiff(item_path.string());
                    if (s.width() != W0 || s.height() != H0) {
                        throw std::runtime_error("Size mismatch in directory slices: " + item_path.filename().string() +
                                                 " is (" + std::to_string(s.width()) + "x" + std::to_string(s.height()) +
                                                 ") but expected (" + std::to_string(W0) + "x" + std::to_string(H0) + ")");
                    }
                    vol.set_slice(static_cast<int>(idx), s);
                    total_sum_before += s.mean() * s.size();
                    total_pix += s.size();
                }

                Volume3D vol_out = apply_3d_filter(vol, p.filter_name, p.sigma, p.radius, p.sigma_spatial, p.sigma_range,
                                                   p.kappa, p.iters, p.dt, p.noise_var, p.weight, p.eps, p.element,
                                                   p.patch_radius, p.search_radius, p.h);
                total_sum_after = vol_out.mean() * total_pix;

                for (size_t idx = 0; idx < slice_files.size(); ++idx) {
                    Image2D s_out = vol_out.slice(static_cast<int>(idx));
                    fs::path out_file = target_out_dir / slice_files[idx].filename();
                    ct::save_tiff(s_out, out_file.string());
                    if (idx == mid_idx) mid_preview = s_out;
                }
            } else {
                throw std::invalid_argument("Unknown --mode: " + mode + " (must be 'slice' or '3d')");
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            double mean_before = total_sum_before / total_pix;
            double mean_after = total_sum_after / total_pix;
            double noise_after = ct::estimate_noise_sigma(mid_preview);

            if (!quiet) {
                std::cout << std::fixed << std::setprecision(6)
                          << "Mean: " << mean_before << " -> " << mean_after
                          << " | Noise: " << sigma_hat << " -> " << noise_after
                          << " | Time: " << std::setprecision(2) << elapsed_ms << " ms\n";
            }

            auto [p1, p99] = ct::compute_percentiles_1_99(mid_orig);
            OutputTriple triple = get_unique_output_triple(target_out_dir, stem);
            ct::write_preview_tiff(mid_preview, triple.preview_file.string(), p1, p99);
            ct::write_preview_tiff(mid_orig, triple.orig_preview_file.string(), p1, p99);

            if (!preview_path.empty()) {
                double p_min = std::isnan(range_min) ? p1 : range_min;
                double p_max = std::isnan(range_max) ? p99 : range_max;
                ct::write_preview_tiff(mid_preview, preview_path, p_min, p_max);
            }

            if (!quiet) {
                std::cout << "Open the *_preview.tif files to look at the result; *_result_float32.tif holds the raw values.\n";
            }

        // ==============================================================================
        // Single File Mode (2D Slice or Multi-page 3D Volume)
        // ==============================================================================
        } else {
            bool is_multipage_vol = false;
            Volume3D vol_in;
            Image2D img_in;

            try {
                vol_in = ct::load_volume_multipage(in_path);
                if (vol_in.depth() > 1) {
                    is_multipage_vol = true;
                } else {
                    img_in = vol_in.slice(0);
                }
            } catch (...) {
                img_in = ct::load_tiff(in_path);
            }

            double sigma_hat = is_multipage_vol ? ct::estimate_noise_sigma(vol_in) : ct::estimate_noise_sigma(img_in);
            ct::ResolvedParams p = ct::resolve_filter_parameters(filter_name, sigma_hat, flags, cfg, is_multipage_vol);

            // Apply explicit CLI overrides
            if (flags.has_sigma) p.sigma = user_sigma;
            if (flags.has_radius) p.radius = user_radius;
            if (flags.has_sigma_spatial) p.sigma_spatial = user_sigma_spatial;
            if (flags.has_sigma_range) p.sigma_range = user_sigma_range;
            if (flags.has_kappa) p.kappa = user_kappa;
            if (flags.has_iters) p.iters = user_iters;
            if (flags.has_dt) p.dt = user_dt;
            if (flags.has_noise_var) p.noise_var = user_noise_var;
            if (flags.has_weight) p.weight = user_weight;
            if (flags.has_eps) p.eps = user_eps;
            if (flags.has_element) p.element = user_element;
            if (flags.has_patch_radius) p.patch_radius = user_patch_radius;
            if (flags.has_search_radius) p.search_radius = user_search_radius;
            if (flags.has_h) p.h = user_h;

            if (!quiet) {
                p.print_notes(std::cout);
            }

            auto [p1, p99] = is_multipage_vol ? ct::compute_percentiles_1_99(vol_in.slice(vol_in.depth() / 2))
                                              : ct::compute_percentiles_1_99(img_in);

            fs::path ct_dir = fs::path(output_root) / stem;
            OutputTriple triple = out_path.empty() ? get_unique_output_triple(ct_dir, stem)
                                                   : get_unique_output_triple_for_explicit_file(fs::path(out_path));

            if (is_multipage_vol) {
                double mean_before = vol_in.mean();
                auto t0 = std::chrono::high_resolution_clock::now();
                Volume3D vol_out = apply_3d_filter(vol_in, p.filter_name, p.sigma, p.radius, p.sigma_spatial, p.sigma_range,
                                                   p.kappa, p.iters, p.dt, p.noise_var, p.weight, p.eps, p.element,
                                                   p.patch_radius, p.search_radius, p.h);
                auto t1 = std::chrono::high_resolution_clock::now();
                double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

                double mean_after = vol_out.mean();
                double noise_after = ct::estimate_noise_sigma(vol_out);

                if (!quiet) {
                    std::cout << std::fixed << std::setprecision(6)
                              << "Mean: " << mean_before << " -> " << mean_after
                              << " | Noise: " << sigma_hat << " -> " << noise_after
                              << " | Time: " << std::setprecision(2) << elapsed_ms << " ms\n";
                }

                ct::save_volume_multipage(vol_out, triple.result_file.string());

                Image2D mid_out = vol_out.slice(vol_out.depth() / 2);
                Image2D mid_in = vol_in.slice(vol_in.depth() / 2);
                ct::write_preview_tiff(mid_out, triple.preview_file.string(), p1, p99);
                ct::write_preview_tiff(mid_in, triple.orig_preview_file.string(), p1, p99);

                if (!preview_path.empty()) {
                    double p_min = std::isnan(range_min) ? p1 : range_min;
                    double p_max = std::isnan(range_max) ? p99 : range_max;
                    ct::write_preview_tiff(mid_out, preview_path, p_min, p_max);
                }

            } else {
                double mean_before = img_in.mean();
                auto t0 = std::chrono::high_resolution_clock::now();
                Image2D img_out = apply_2d_filter(img_in, p.filter_name, p.sigma, p.radius, p.sigma_spatial, p.sigma_range,
                                                  p.kappa, p.iters, p.dt, p.noise_var, p.weight, p.eps, p.element,
                                                  p.patch_radius, p.search_radius, p.h);
                auto t1 = std::chrono::high_resolution_clock::now();
                double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

                double mean_after = img_out.mean();
                double noise_after = ct::estimate_noise_sigma(img_out);

                if (!quiet) {
                    std::cout << std::fixed << std::setprecision(6)
                              << "Mean: " << mean_before << " -> " << mean_after
                              << " | Noise: " << sigma_hat << " -> " << noise_after
                              << " | Time: " << std::setprecision(2) << elapsed_ms << " ms\n";
                }

                ct::save_tiff(img_out, triple.result_file.string());
                ct::write_preview_tiff(img_out, triple.preview_file.string(), p1, p99);
                ct::write_preview_tiff(img_in, triple.orig_preview_file.string(), p1, p99);

                if (!preview_path.empty()) {
                    double p_min = std::isnan(range_min) ? p1 : range_min;
                    double p_max = std::isnan(range_max) ? p99 : range_max;
                    ct::write_preview_tiff(img_out, preview_path, p_min, p_max);
                }
            }

            if (!quiet) {
                std::cout << "Open the *_preview.tif files to look at the result; *_result_float32.tif holds the raw values.\n";
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
