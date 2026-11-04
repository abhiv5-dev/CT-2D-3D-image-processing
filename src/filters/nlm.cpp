#include "ct/filters/nlm.hpp"
#include "ct/filters/gaussian.hpp"  // for ct::reflect_index
#include "ct/parallel.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace ct {

Image2D nlm_filter(const Image2D& input, const NLMParams2D& params) {
    if (input.empty()) {
        throw std::invalid_argument("nlm_filter: input Image2D is empty");
    }
    if (params.patch_radius < 0 || params.search_radius < 0 || !(params.h > 0.0)) {
        throw std::invalid_argument("nlm_filter: invalid parameters");
    }

    const int W = input.width();
    const int H = input.height();
    const int rp = params.patch_radius;
    const int rs = params.search_radius;
    const double h_sq = params.h * params.h;
    const double patch_area = static_cast<double>((2 * rp + 1) * (2 * rp + 1));
    const double norm_const = h_sq * patch_area;

    Image2D output(W, H);
    output.metadata() = input.metadata();

    parallel_for(0, H, [&](int y) {
        for (int x = 0; x < W; ++x) {
            double sum_weights = 0.0;
            double sum_values = 0.0;
            double max_weight = 0.0;

            for (int dy = -rs; dy <= rs; ++dy) {
                int qy = reflect_index(y + dy, H);
                for (int dx = -rs; dx <= rs; ++dx) {
                    if (dy == 0 && dx == 0) continue;
                    int qx = reflect_index(x + dx, W);

                    double dist_sq = 0.0;
                    for (int py = -rp; py <= rp; ++py) {
                        int y1 = reflect_index(y + py, H);
                        int y2 = reflect_index(qy + py, H);
                        const float* r1 = input.data().data() + static_cast<size_t>(y1) * W;
                        const float* r2 = input.data().data() + static_cast<size_t>(y2) * W;
                        for (int px = -rp; px <= rp; ++px) {
                            int x1 = reflect_index(x + px, W);
                            int x2 = reflect_index(qx + px, W);
                            double diff = static_cast<double>(r1[x1] - r2[x2]);
                            dist_sq += diff * diff;
                        }
                    }

                    double w = std::exp(-dist_sq / norm_const);
                    sum_weights += w;
                    sum_values += w * static_cast<double>(input.at(qy, qx));
                    if (w > max_weight) max_weight = w;
                }
            }

            // Central pixel receives max weight
            if (max_weight == 0.0) max_weight = 1.0;
            sum_weights += max_weight;
            sum_values += max_weight * static_cast<double>(input.at(y, x));

            output.at(y, x) = static_cast<float>(sum_values / sum_weights);
        }
    });

    return output;
}

Image2D nlm_filter(const Image2D& input, int patch_radius, int search_radius, double h) {
    return nlm_filter(input, NLMParams2D(patch_radius, search_radius, h));
}

Volume3D nlm_filter(const Volume3D& input, const NLMParams3D& params) {
    if (input.empty()) {
        throw std::invalid_argument("nlm_filter: input Volume3D is empty");
    }
    if (params.patch_radius < 0 || params.search_radius < 0 || !(params.h > 0.0)) {
        throw std::invalid_argument("nlm_filter: invalid parameters");
    }

    const int W = input.width();
    const int H = input.height();
    const int D = input.depth();
    const int rp = params.patch_radius;
    const int rs = params.search_radius;
    const double h_sq = params.h * params.h;
    const double patch_vol = static_cast<double>((2 * rp + 1) * (2 * rp + 1) * (2 * rp + 1));
    const double norm_const = h_sq * patch_vol;

    Volume3D output(W, H, D);
    output.metadata() = input.metadata();

    parallel_for(0, D, [&](int z) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                double sum_weights = 0.0;
                double sum_values = 0.0;
                double max_weight = 0.0;

                for (int dz = -rs; dz <= rs; ++dz) {
                    int qz = reflect_index(z + dz, D);
                    for (int dy = -rs; dy <= rs; ++dy) {
                        int qy = reflect_index(y + dy, H);
                        for (int dx = -rs; dx <= rs; ++dx) {
                            if (dz == 0 && dy == 0 && dx == 0) continue;
                            int qx = reflect_index(x + dx, W);

                            double dist_sq = 0.0;
                            for (int pz = -rp; pz <= rp; ++pz) {
                                int z1 = reflect_index(z + pz, D);
                                int z2 = reflect_index(qz + pz, D);
                                for (int py = -rp; py <= rp; ++py) {
                                    int y1 = reflect_index(y + py, H);
                                    int y2 = reflect_index(qy + py, H);
                                    for (int px = -rp; px <= rp; ++px) {
                                        int x1 = reflect_index(x + px, W);
                                        int x2 = reflect_index(qx + px, W);
                                        double diff = static_cast<double>(input.at(z1, y1, x1) - input.at(z2, y2, x2));
                                        dist_sq += diff * diff;
                                    }
                                }
                            }

                            double w = std::exp(-dist_sq / norm_const);
                            sum_weights += w;
                            sum_values += w * static_cast<double>(input.at(qz, qy, qx));
                            if (w > max_weight) max_weight = w;
                        }
                    }
                }

                if (max_weight == 0.0) max_weight = 1.0;
                sum_weights += max_weight;
                sum_values += max_weight * static_cast<double>(input.at(z, y, x));

                output.at(z, y, x) = static_cast<float>(sum_values / sum_weights);
            }
        }
    });

    return output;
}

Volume3D nlm_filter(const Volume3D& input, int patch_radius, int search_radius, double h) {
    return nlm_filter(input, NLMParams3D(patch_radius, search_radius, h));
}

}  // namespace ct
