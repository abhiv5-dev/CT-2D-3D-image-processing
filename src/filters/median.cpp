#include "ct/filters/median.hpp"
#include "ct/filters/gaussian.hpp"  // for ct::reflect_index
#include "ct/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace ct {

namespace {

// Strict weak ordering for float that sorts NaNs deterministically to the end.
// Guarantees no undefined behavior or infinite loops in std::nth_element when NaNs are present.
struct NanAwareLess {
    bool operator()(float a, float b) const noexcept {
        bool a_nan = std::isnan(a);
        bool b_nan = std::isnan(b);
        if (a_nan && b_nan) return false;
        if (a_nan) return false;  // NaNs placed at the upper end
        if (b_nan) return true;
        return a < b;
    }
};

}  // namespace

Image2D median_filter(const Image2D& input, const MedianParams2D& params) {
    if (input.empty()) {
        throw std::invalid_argument("median_filter: input Image2D is empty");
    }
    if (params.radius_x <= 0 || params.radius_y <= 0) {
        throw std::invalid_argument("median_filter: radii must be positive");
    }

    const int W = input.width();
    const int H = input.height();
    const int rx = params.radius_x;
    const int ry = params.radius_y;
    const size_t win_size = static_cast<size_t>(2 * rx + 1) * static_cast<size_t>(2 * ry + 1);
    const size_t mid = win_size / 2;

    Image2D output(W, H);
    output.metadata() = input.metadata();

    const bool can_use_interior = (W > 2 * rx) && (H > 2 * ry);

    parallel_for(0, H, [&](int y) {
        std::vector<float> window(win_size);
        float* out_row = output.data().data() + static_cast<size_t>(y) * W;

        if (can_use_interior && y >= ry && y < H - ry) {
            // Left border: x in [0, rx)
            for (int x = 0; x < rx; ++x) {
                size_t idx = 0;
                bool has_nan = false;
                for (int dy = -ry; dy <= ry; ++dy) {
                    const float* in_row = input.data().data() + static_cast<size_t>(y + dy) * W;
                    for (int dx = -rx; dx <= rx; ++dx) {
                        int sx = reflect_index(x + dx, W);
                        float val = in_row[sx];
                        window[idx++] = val;
                        if (std::isnan(val)) has_nan = true;
                    }
                }
                if (!has_nan) {
                    std::nth_element(window.begin(), window.begin() + mid, window.end());
                } else {
                    std::nth_element(window.begin(), window.begin() + mid, window.end(), NanAwareLess{});
                }
                out_row[x] = window[mid];
            }

            // Interior fast-path: x in [rx, W - rx) - direct indexing, no reflect_index calls
            for (int x = rx; x < W - rx; ++x) {
                size_t idx = 0;
                bool has_nan = false;
                for (int dy = -ry; dy <= ry; ++dy) {
                    const float* in_row = input.data().data() + static_cast<size_t>(y + dy) * W;
                    for (int dx = -rx; dx <= rx; ++dx) {
                        float val = in_row[x + dx];
                        window[idx++] = val;
                        if (std::isnan(val)) has_nan = true;
                    }
                }
                if (!has_nan) {
                    std::nth_element(window.begin(), window.begin() + mid, window.end());
                } else {
                    std::nth_element(window.begin(), window.begin() + mid, window.end(), NanAwareLess{});
                }
                out_row[x] = window[mid];
            }

            // Right border: x in [W - rx, W)
            for (int x = W - rx; x < W; ++x) {
                size_t idx = 0;
                bool has_nan = false;
                for (int dy = -ry; dy <= ry; ++dy) {
                    const float* in_row = input.data().data() + static_cast<size_t>(y + dy) * W;
                    for (int dx = -rx; dx <= rx; ++dx) {
                        int sx = reflect_index(x + dx, W);
                        float val = in_row[sx];
                        window[idx++] = val;
                        if (std::isnan(val)) has_nan = true;
                    }
                }
                if (!has_nan) {
                    std::nth_element(window.begin(), window.begin() + mid, window.end());
                } else {
                    std::nth_element(window.begin(), window.begin() + mid, window.end(), NanAwareLess{});
                }
                out_row[x] = window[mid];
            }
        } else {
            // Border row or narrow image: full reflection checks
            for (int x = 0; x < W; ++x) {
                size_t idx = 0;
                bool has_nan = false;
                for (int dy = -ry; dy <= ry; ++dy) {
                    int sy = reflect_index(y + dy, H);
                    const float* in_row = input.data().data() + static_cast<size_t>(sy) * W;
                    for (int dx = -rx; dx <= rx; ++dx) {
                        int sx = reflect_index(x + dx, W);
                        float val = in_row[sx];
                        window[idx++] = val;
                        if (std::isnan(val)) has_nan = true;
                    }
                }
                if (!has_nan) {
                    std::nth_element(window.begin(), window.begin() + mid, window.end());
                } else {
                    std::nth_element(window.begin(), window.begin() + mid, window.end(), NanAwareLess{});
                }
                out_row[x] = window[mid];
            }
        }
    });

    return output;
}

Image2D median_filter(const Image2D& input, int radius) {
    return median_filter(input, MedianParams2D(radius));
}

Volume3D median_filter(const Volume3D& input, const MedianParams3D& params) {
    if (input.empty()) {
        throw std::invalid_argument("median_filter: input Volume3D is empty");
    }
    if (params.radius[0] <= 0 || params.radius[1] <= 0 || params.radius[2] <= 0) {
        throw std::invalid_argument("median_filter: 3D radii must be positive");
    }

    const int W = input.width();
    const int H = input.height();
    const int D = input.depth();
    const int rx = params.radius[0];
    const int ry = params.radius[1];
    const int rz = params.radius[2];
    const size_t win_size = static_cast<size_t>(2 * rx + 1) *
                            static_cast<size_t>(2 * ry + 1) *
                            static_cast<size_t>(2 * rz + 1);
    const size_t mid = win_size / 2;

    Volume3D output(W, H, D);
    output.metadata() = input.metadata();

    const bool can_use_interior = (W > 2 * rx) && (H > 2 * ry) && (D > 2 * rz);

    parallel_for(0, D, [&](int z) {
        std::vector<float> window(win_size);
        const bool z_interior = can_use_interior && (z >= rz && z < D - rz);

        for (int y = 0; y < H; ++y) {
            float* out_row = output.data().data() + (static_cast<size_t>(z) * H + y) * W;

            if (z_interior && y >= ry && y < H - ry) {
                // Left border: x in [0, rx)
                for (int x = 0; x < rx; ++x) {
                    size_t idx = 0;
                    bool has_nan = false;
                    for (int dz = -rz; dz <= rz; ++dz) {
                        size_t z_offset = static_cast<size_t>(z + dz) * H * W;
                        for (int dy = -ry; dy <= ry; ++dy) {
                            const float* in_row = input.data().data() + z_offset + static_cast<size_t>(y + dy) * W;
                            for (int dx = -rx; dx <= rx; ++dx) {
                                int sx = reflect_index(x + dx, W);
                                float val = in_row[sx];
                                window[idx++] = val;
                                if (std::isnan(val)) has_nan = true;
                            }
                        }
                    }
                    if (!has_nan) {
                        std::nth_element(window.begin(), window.begin() + mid, window.end());
                    } else {
                        std::nth_element(window.begin(), window.begin() + mid, window.end(), NanAwareLess{});
                    }
                    out_row[x] = window[mid];
                }

                // Interior fast-path: x in [rx, W - rx) - direct indexing across all 3 axes
                for (int x = rx; x < W - rx; ++x) {
                    size_t idx = 0;
                    bool has_nan = false;
                    for (int dz = -rz; dz <= rz; ++dz) {
                        size_t z_offset = static_cast<size_t>(z + dz) * H * W;
                        for (int dy = -ry; dy <= ry; ++dy) {
                            const float* in_row = input.data().data() + z_offset + static_cast<size_t>(y + dy) * W;
                            for (int dx = -rx; dx <= rx; ++dx) {
                                float val = in_row[x + dx];
                                window[idx++] = val;
                                if (std::isnan(val)) has_nan = true;
                            }
                        }
                    }
                    if (!has_nan) {
                        std::nth_element(window.begin(), window.begin() + mid, window.end());
                    } else {
                        std::nth_element(window.begin(), window.begin() + mid, window.end(), NanAwareLess{});
                    }
                    out_row[x] = window[mid];
                }

                // Right border: x in [W - rx, W)
                for (int x = W - rx; x < W; ++x) {
                    size_t idx = 0;
                    bool has_nan = false;
                    for (int dz = -rz; dz <= rz; ++dz) {
                        size_t z_offset = static_cast<size_t>(z + dz) * H * W;
                        for (int dy = -ry; dy <= ry; ++dy) {
                            const float* in_row = input.data().data() + z_offset + static_cast<size_t>(y + dy) * W;
                            for (int dx = -rx; dx <= rx; ++dx) {
                                int sx = reflect_index(x + dx, W);
                                float val = in_row[sx];
                                window[idx++] = val;
                                if (std::isnan(val)) has_nan = true;
                            }
                        }
                    }
                    if (!has_nan) {
                        std::nth_element(window.begin(), window.begin() + mid, window.end());
                    } else {
                        std::nth_element(window.begin(), window.begin() + mid, window.end(), NanAwareLess{});
                    }
                    out_row[x] = window[mid];
                }
            } else {
                // Border region: full boundary reflection checks
                for (int x = 0; x < W; ++x) {
                    size_t idx = 0;
                    bool has_nan = false;
                    for (int dz = -rz; dz <= rz; ++dz) {
                        int sz = reflect_index(z + dz, D);
                        size_t z_offset = static_cast<size_t>(sz) * H * W;
                        for (int dy = -ry; dy <= ry; ++dy) {
                            int sy = reflect_index(y + dy, H);
                            const float* in_row = input.data().data() + z_offset + static_cast<size_t>(sy) * W;
                            for (int dx = -rx; dx <= rx; ++dx) {
                                int sx = reflect_index(x + dx, W);
                                float val = in_row[sx];
                                window[idx++] = val;
                                if (std::isnan(val)) has_nan = true;
                            }
                        }
                    }
                    if (!has_nan) {
                        std::nth_element(window.begin(), window.begin() + mid, window.end());
                    } else {
                        std::nth_element(window.begin(), window.begin() + mid, window.end(), NanAwareLess{});
                    }
                    out_row[x] = window[mid];
                }
            }
        }
    });

    return output;
}

Volume3D median_filter(const Volume3D& input, int radius) {
    return median_filter(input, MedianParams3D(radius));
}

}  // namespace ct
