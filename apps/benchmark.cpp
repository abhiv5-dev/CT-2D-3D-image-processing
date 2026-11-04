#include "ct/image2d.hpp"
#include "ct/volume3d.hpp"
#include "ct/parallel.hpp"
#include "ct/filters/gaussian.hpp"
#include "ct/filters/median.hpp"
#include "ct/filters/bilateral.hpp"
#include "ct/filters/diffusion.hpp"
#include "ct/filters/wiener.hpp"
#include "ct/filters/tv_chambolle.hpp"
#include "ct/filters/nlm.hpp"
#include "ct/filters/morphology.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <algorithm>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace {

double get_peak_rss_mb() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<double>(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0);
    }
    return 0.0;
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
#if defined(__APPLE__) && defined(__MACH__)
        // On macOS, ru_maxrss is in bytes
        return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
#else
        // On Linux, ru_maxrss is in kilobytes
        return static_cast<double>(usage.ru_maxrss) / 1024.0;
#endif
    }
    return 0.0;
#endif
}

struct Stats {
    double min_s;
    double median_s;
    double max_s;
    double mean_s;
    double std_s;
};

template <typename Func>
Stats time_operation(Func&& fn, int warmup, int runs) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    std::vector<double> times;
    times.reserve(static_cast<size_t>(runs));
    for (int i = 0; i < runs; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = t1 - t0;
        times.push_back(diff.count());
    }
    std::sort(times.begin(), times.end());

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / runs;
    double sq_diff = 0.0;
    for (double t : times) {
        sq_diff += (t - mean) * (t - mean);
    }
    double std_dev = std::sqrt(sq_diff / runs);

    return Stats{
        times.front(),
        times[static_cast<size_t>(runs / 2)],
        times.back(),
        mean,
        std_dev
    };
}

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " --filter <gaussian|median|bilateral|diffusion|wiener|tv|nlm|morphology> "
              << "--dim <2D|3D> --size <N> --threads <T> --variant <variant> "
              << "[--mode <3d|slice>] [--warmup <W>] [--runs <R>]\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string filter;
    std::string dim = "2D";
    int size = 512;
    int threads = 1;
    std::string variant = "optimized";
    std::string mode = "3d";
    int warmup = 1;
    int runs = 5;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--filter" && i + 1 < argc) filter = argv[++i];
        else if (arg == "--dim" && i + 1 < argc) dim = argv[++i];
        else if (arg == "--size" && i + 1 < argc) size = std::stoi(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
        else if (arg == "--variant" && i + 1 < argc) variant = argv[++i];
        else if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (arg == "--warmup" && i + 1 < argc) warmup = std::stoi(argv[++i]);
        else if (arg == "--runs" && i + 1 < argc) runs = std::stoi(argv[++i]);
        else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (filter.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    ct::set_num_threads(threads);

    double total_voxels = 0.0;
    Stats stats{};

    if (dim == "2D") {
        total_voxels = static_cast<double>(size) * size;
        Image2D img(size, size, 0.5f);
        // Add a slight gradient to prevent compiler from optimizing away memory loads
        for (size_t i = 0; i < img.data().size(); ++i) {
            img.data()[i] = static_cast<float>(i % 256) / 255.0f;
        }

        if (filter == "gaussian") {
            if (variant == "naive_reference") {
                stats = time_operation([&]() {
                    volatile auto res = ct::gaussian_blur_naive(img, 1.5, 4);
                    (void)res;
                }, warmup, runs);
            } else {
                stats = time_operation([&]() {
                    volatile auto res = ct::gaussian_blur(img, 1.5, 4);
                    (void)res;
                }, warmup, runs);
            }
        } else if (filter == "median") {
            stats = time_operation([&]() {
                volatile auto res = ct::median_filter(img, 1);
                (void)res;
            }, warmup, runs);
        } else if (filter == "bilateral") {
            bool exact = (variant == "baseline");
            stats = time_operation([&]() {
                volatile auto res = ct::bilateral_filter(img, 1.5, 0.08, 3, exact);
                (void)res;
            }, warmup, runs);
        } else if (filter == "diffusion") {
            stats = time_operation([&]() {
                volatile auto res = ct::anisotropic_diffusion(img, 0.05, 5, 0.20);
                (void)res;
            }, warmup, runs);
        } else if (filter == "wiener") {
            stats = time_operation([&]() {
                volatile auto res = ct::wiener_filter(img, 1, 0.05);
                (void)res;
            }, warmup, runs);
        } else if (filter == "tv" || filter == "tv_chambolle") {
            stats = time_operation([&]() {
                volatile auto res = ct::tv_chambolle(img, 0.05, 1e-4, 20);
                (void)res;
            }, warmup, runs);
        } else if (filter == "nlm") {
            stats = time_operation([&]() {
                volatile auto res = ct::nlm_filter(img, 1, 3, 0.05);
                (void)res;
            }, warmup, runs);
        } else if (filter == "morphology") {
            stats = time_operation([&]() {
                volatile auto res = ct::grayscale_opening(img, 1, "box");
                (void)res;
            }, warmup, runs);
        } else {
            std::cerr << "Unknown filter: " << filter << "\n";
            return 1;
        }

    } else if (dim == "3D") {
        total_voxels = static_cast<double>(size) * size * size;
        Volume3D vol(size, size, size, 0.5f);
        for (size_t i = 0; i < std::min<size_t>(vol.data().size(), 1000000ULL); ++i) {
            vol.data()[i] = static_cast<float>(i % 256) / 255.0f;
        }

        if (mode == "slice") {
            variant = "slicewise";
            if (filter == "gaussian") {
                stats = time_operation([&]() {
                    volatile auto res = ct::apply_slice_by_slice(vol, [](const Image2D& s) {
                        return ct::gaussian_blur(s, 0.8, 3);
                    });
                    (void)res;
                }, warmup, runs);
            } else if (filter == "median") {
                stats = time_operation([&]() {
                    volatile auto res = ct::apply_slice_by_slice(vol, [](const Image2D& s) {
                        return ct::median_filter(s, 1);
                    });
                    (void)res;
                }, warmup, runs);
            } else if (filter == "bilateral") {
                stats = time_operation([&]() {
                    volatile auto res = ct::apply_slice_by_slice(vol, [](const Image2D& s) {
                        return ct::bilateral_filter(s, 1.5, 0.12, 3, false);
                    });
                    (void)res;
                }, warmup, runs);
            } else if (filter == "diffusion") {
                stats = time_operation([&]() {
                    volatile auto res = ct::apply_slice_by_slice(vol, [](const Image2D& s) {
                        return ct::anisotropic_diffusion(s, 0.08, 6, 0.20);
                    });
                    (void)res;
                }, warmup, runs);
            } else if (filter == "wiener") {
                stats = time_operation([&]() {
                    volatile auto res = ct::wiener_filter_slice_by_slice(vol, 1, -1.0);
                    (void)res;
                }, warmup, runs);
            } else if (filter == "tv" || filter == "tv_chambolle") {
                stats = time_operation([&]() {
                    volatile auto res = ct::tv_chambolle_slice_by_slice(vol, 0.05, 1e-4, 40);
                    (void)res;
                }, warmup, runs);
            } else if (filter == "nlm") {
                stats = time_operation([&]() {
                    volatile auto res = ct::apply_slice_by_slice(vol, [](const Image2D& s) {
                        return ct::nlm_filter(s, 1, 2, 0.05);
                    });
                    (void)res;
                }, warmup, runs);
            } else if (filter == "morphology") {
                stats = time_operation([&]() {
                    volatile auto res = ct::grayscale_closing_slice_by_slice(vol, 1, "ball");
                    (void)res;
                }, warmup, runs);
            } else {
                std::cerr << "Unknown filter: " << filter << "\n";
                return 1;
            }
        } else if (variant == "eval") {
            if (filter == "gaussian") {
                stats = time_operation([&]() {
                    volatile auto res = ct::gaussian_blur(vol, 0.8, 3);
                    (void)res;
                }, warmup, runs);
            } else if (filter == "median") {
                stats = time_operation([&]() {
                    volatile auto res = ct::median_filter(vol, 1);
                    (void)res;
                }, warmup, runs);
            } else if (filter == "bilateral") {
                stats = time_operation([&]() {
                    volatile auto res = ct::bilateral_filter(vol, 1.5, 0.12, 3, false);
                    (void)res;
                }, warmup, runs);
            } else if (filter == "diffusion") {
                stats = time_operation([&]() {
                    volatile auto res = ct::anisotropic_diffusion(vol, 0.08, 8, 0.15);
                    (void)res;
                }, warmup, runs);
            } else if (filter == "wiener") {
                stats = time_operation([&]() {
                    volatile auto res = ct::wiener_filter(vol, 1, -1.0);
                    (void)res;
                }, warmup, runs);
            } else if (filter == "tv" || filter == "tv_chambolle") {
                stats = time_operation([&]() {
                    volatile auto res = ct::tv_chambolle(vol, 0.05, 1e-4, 40);
                    (void)res;
                }, warmup, runs);
            } else if (filter == "nlm") {
                stats = time_operation([&]() {
                    volatile auto res = ct::nlm_filter(vol, 1, 2, 0.05);
                    (void)res;
                }, warmup, runs);
            } else if (filter == "morphology") {
                stats = time_operation([&]() {
                    volatile auto res = ct::grayscale_closing(vol, 1, "ball");
                    (void)res;
                }, warmup, runs);
            } else {
                std::cerr << "Unknown filter: " << filter << "\n";
                return 1;
            }
        } else if (filter == "gaussian") {
            if (variant == "naive_reference") {
                stats = time_operation([&]() {
                    volatile auto res = ct::gaussian_blur_naive(vol, 1.5, 4);
                    (void)res;
                }, warmup, runs);
            } else {
                stats = time_operation([&]() {
                    volatile auto res = ct::gaussian_blur(vol, 1.5, 4);
                    (void)res;
                }, warmup, runs);
            }
        } else if (filter == "median") {
            stats = time_operation([&]() {
                volatile auto res = ct::median_filter(vol, 1);
                (void)res;
            }, warmup, runs);
        } else if (filter == "bilateral") {
            bool exact = (variant == "baseline");
            stats = time_operation([&]() {
                volatile auto res = ct::bilateral_filter(vol, 1.5, 0.08, 3, exact);
                (void)res;
            }, warmup, runs);
        } else if (filter == "diffusion") {
            stats = time_operation([&]() {
                volatile auto res = ct::anisotropic_diffusion(vol, 0.05, 5, 0.15);
                (void)res;
            }, warmup, runs);
        } else if (filter == "wiener") {
            stats = time_operation([&]() {
                volatile auto res = ct::wiener_filter(vol, 1, 0.05);
                (void)res;
            }, warmup, runs);
        } else if (filter == "tv" || filter == "tv_chambolle") {
            stats = time_operation([&]() {
                volatile auto res = ct::tv_chambolle(vol, 0.05, 1e-4, 20);
                (void)res;
            }, warmup, runs);
        } else if (filter == "nlm") {
            stats = time_operation([&]() {
                volatile auto res = ct::nlm_filter(vol, 1, 2, 0.05);
                (void)res;
            }, warmup, runs);
        } else if (filter == "morphology") {
            stats = time_operation([&]() {
                volatile auto res = ct::grayscale_opening(vol, 1, "box");
                (void)res;
            }, warmup, runs);
        } else {
            std::cerr << "Unknown filter: " << filter << "\n";
            return 1;
        }
    } else {
        std::cerr << "Unknown dim: " << dim << "\n";
        return 1;
    }

    double peak_rss_mb = get_peak_rss_mb();
    double mv_s = (total_voxels / 1e6) / stats.median_s;

    // Output strictly formatted CSV record to stdout
    std::cout << filter << "," << dim << "," << size << "," << threads << "," << variant << ","
              << stats.median_s << "," << mv_s << "," << peak_rss_mb << ","
              << stats.min_s << "," << stats.max_s << "," << stats.std_s << "\n";

    return 0;
}
