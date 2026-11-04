#include "ct/filters/morphology.hpp"
#include "ct/filters/gaussian.hpp" // for reflect_index, apply_slice_by_slice
#include "ct/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ct {

namespace {

// ==============================================================================
// 1D Min / Max Kernels for Separable Box Morphology
// ==============================================================================

void box_min_1d_x(const float* in, float* out, int width, int height, int radius) {
    parallel_for(0, height, [&](int y) {
        const float* in_row = in + static_cast<size_t>(y) * width;
        float* out_row = out + static_cast<size_t>(y) * width;

        for (int x = 0; x < width; ++x) {
            float min_v = in_row[reflect_index(x - radius, width)];
            for (int k = -radius + 1; k <= radius; ++k) {
                float v = in_row[reflect_index(x + k, width)];
                if (v < min_v) min_v = v;
            }
            out_row[x] = min_v;
        }
    });
}

void box_max_1d_x(const float* in, float* out, int width, int height, int radius) {
    parallel_for(0, height, [&](int y) {
        const float* in_row = in + static_cast<size_t>(y) * width;
        float* out_row = out + static_cast<size_t>(y) * width;

        for (int x = 0; x < width; ++x) {
            float max_v = in_row[reflect_index(x - radius, width)];
            for (int k = -radius + 1; k <= radius; ++k) {
                float v = in_row[reflect_index(x + k, width)];
                if (v > max_v) max_v = v;
            }
            out_row[x] = max_v;
        }
    });
}

void box_min_1d_y(const float* in, float* out, int width, int height, int radius) {
    parallel_for(0, height, [&](int y) {
        float* out_row = out + static_cast<size_t>(y) * width;
        for (int x = 0; x < width; ++x) {
            float min_v = in[static_cast<size_t>(reflect_index(y - radius, height)) * width + x];
            for (int k = -radius + 1; k <= radius; ++k) {
                float v = in[static_cast<size_t>(reflect_index(y + k, height)) * width + x];
                if (v < min_v) min_v = v;
            }
            out_row[x] = min_v;
        }
    });
}

void box_max_1d_y(const float* in, float* out, int width, int height, int radius) {
    parallel_for(0, height, [&](int y) {
        float* out_row = out + static_cast<size_t>(y) * width;
        for (int x = 0; x < width; ++x) {
            float max_v = in[static_cast<size_t>(reflect_index(y - radius, height)) * width + x];
            for (int k = -radius + 1; k <= radius; ++k) {
                float v = in[static_cast<size_t>(reflect_index(y + k, height)) * width + x];
                if (v > max_v) max_v = v;
            }
            out_row[x] = max_v;
        }
    });
}

void box_min_1d_z(const float* in, float* out, int width, int height, int depth, int radius) {
    const size_t slice_size = static_cast<size_t>(width) * height;
    parallel_for(0, depth, [&](int z) {
        float* out_slice = out + static_cast<size_t>(z) * slice_size;
        for (size_t i = 0; i < slice_size; ++i) {
            float min_v = in[static_cast<size_t>(reflect_index(z - radius, depth)) * slice_size + i];
            for (int k = -radius + 1; k <= radius; ++k) {
                float v = in[static_cast<size_t>(reflect_index(z + k, depth)) * slice_size + i];
                if (v < min_v) min_v = v;
            }
            out_slice[i] = min_v;
        }
    });
}

void box_max_1d_z(const float* in, float* out, int width, int height, int depth, int radius) {
    const size_t slice_size = static_cast<size_t>(width) * height;
    parallel_for(0, depth, [&](int z) {
        float* out_slice = out + static_cast<size_t>(z) * slice_size;
        for (size_t i = 0; i < slice_size; ++i) {
            float max_v = in[static_cast<size_t>(reflect_index(z - radius, depth)) * slice_size + i];
            for (int k = -radius + 1; k <= radius; ++k) {
                float v = in[static_cast<size_t>(reflect_index(z + k, depth)) * slice_size + i];
                if (v > max_v) max_v = v;
            }
            out_slice[i] = max_v;
        }
    });
}

// ==============================================================================
// 2D Box and Ball Erosion / Dilation
// ==============================================================================

Image2D erode_2d_box(const Image2D& input, int rx, int ry) {
    int W = input.width();
    int H = input.height();
    Image2D tmp(W, H);
    Image2D out(W, H);
    out.metadata() = input.metadata();

    box_min_1d_x(input.data().data(), tmp.data().data(), W, H, rx);
    box_min_1d_y(tmp.data().data(), out.data().data(), W, H, ry);
    return out;
}

Image2D dilate_2d_box(const Image2D& input, int rx, int ry) {
    int W = input.width();
    int H = input.height();
    Image2D tmp(W, H);
    Image2D out(W, H);
    out.metadata() = input.metadata();

    box_max_1d_x(input.data().data(), tmp.data().data(), W, H, rx);
    box_max_1d_y(tmp.data().data(), out.data().data(), W, H, ry);
    return out;
}

Image2D erode_2d_ball(const Image2D& input, int rx, int ry) {
    int W = input.width();
    int H = input.height();
    Image2D out(W, H);
    out.metadata() = input.metadata();

    // Structuring element offsets
    std::vector<std::pair<int, int>> offsets;
    for (int dy = -ry; dy <= ry; ++dy) {
        for (int dx = -rx; dx <= rx; ++dx) {
            double dist_sq = (static_cast<double>(dx * dx) / (rx * rx)) +
                             (static_cast<double>(dy * dy) / (ry * ry));
            if (dist_sq <= 1.0) {
                offsets.emplace_back(dy, dx);
            }
        }
    }

    const float* in_ptr = input.data().data();
    float* out_ptr = out.data().data();

    parallel_for(0, H, [&](int y) {
        for (int x = 0; x < W; ++x) {
            float min_v = std::numeric_limits<float>::infinity();
            for (const auto& [dy, dx] : offsets) {
                int sy = reflect_index(y + dy, H);
                int sx = reflect_index(x + dx, W);
                float v = in_ptr[static_cast<size_t>(sy) * W + sx];
                if (v < min_v) min_v = v;
            }
            out_ptr[static_cast<size_t>(y) * W + x] = min_v;
        }
    });
    return out;
}

Image2D dilate_2d_ball(const Image2D& input, int rx, int ry) {
    int W = input.width();
    int H = input.height();
    Image2D out(W, H);
    out.metadata() = input.metadata();

    std::vector<std::pair<int, int>> offsets;
    for (int dy = -ry; dy <= ry; ++dy) {
        for (int dx = -rx; dx <= rx; ++dx) {
            double dist_sq = (static_cast<double>(dx * dx) / (rx * rx)) +
                             (static_cast<double>(dy * dy) / (ry * ry));
            if (dist_sq <= 1.0) {
                offsets.emplace_back(dy, dx);
            }
        }
    }

    const float* in_ptr = input.data().data();
    float* out_ptr = out.data().data();

    parallel_for(0, H, [&](int y) {
        for (int x = 0; x < W; ++x) {
            float max_v = -std::numeric_limits<float>::infinity();
            for (const auto& [dy, dx] : offsets) {
                int sy = reflect_index(y + dy, H);
                int sx = reflect_index(x + dx, W);
                float v = in_ptr[static_cast<size_t>(sy) * W + sx];
                if (v > max_v) max_v = v;
            }
            out_ptr[static_cast<size_t>(y) * W + x] = max_v;
        }
    });
    return out;
}

// ==============================================================================
// 3D Box and Ball Erosion / Dilation
// ==============================================================================

Volume3D erode_3d_box(const Volume3D& input, int rx, int ry, int rz) {
    int W = input.width();
    int H = input.height();
    int D = input.depth();
    size_t slice_size = static_cast<size_t>(W) * H;

    Volume3D buf1(W, H, D);
    Volume3D buf2(W, H, D);
    Volume3D out(W, H, D);
    out.metadata() = input.metadata();

    // X-pass
    parallel_for(0, D, [&](int z) {
        box_min_1d_x(input.data().data() + static_cast<size_t>(z) * slice_size,
                     buf1.data().data() + static_cast<size_t>(z) * slice_size, W, H, rx);
    });

    // Y-pass
    parallel_for(0, D, [&](int z) {
        box_min_1d_y(buf1.data().data() + static_cast<size_t>(z) * slice_size,
                     buf2.data().data() + static_cast<size_t>(z) * slice_size, W, H, ry);
    });

    // Z-pass
    box_min_1d_z(buf2.data().data(), out.data().data(), W, H, D, rz);
    return out;
}

Volume3D dilate_3d_box(const Volume3D& input, int rx, int ry, int rz) {
    int W = input.width();
    int H = input.height();
    int D = input.depth();
    size_t slice_size = static_cast<size_t>(W) * H;

    Volume3D buf1(W, H, D);
    Volume3D buf2(W, H, D);
    Volume3D out(W, H, D);
    out.metadata() = input.metadata();

    parallel_for(0, D, [&](int z) {
        box_max_1d_x(input.data().data() + static_cast<size_t>(z) * slice_size,
                     buf1.data().data() + static_cast<size_t>(z) * slice_size, W, H, rx);
    });

    parallel_for(0, D, [&](int z) {
        box_max_1d_y(buf1.data().data() + static_cast<size_t>(z) * slice_size,
                     buf2.data().data() + static_cast<size_t>(z) * slice_size, W, H, ry);
    });

    box_max_1d_z(buf2.data().data(), out.data().data(), W, H, D, rz);
    return out;
}

Volume3D erode_3d_ball(const Volume3D& input, int rx, int ry, int rz) {
    int W = input.width();
    int H = input.height();
    int D = input.depth();
    size_t slice_size = static_cast<size_t>(W) * H;
    Volume3D out(W, H, D);
    out.metadata() = input.metadata();

    struct Offset3D { int dz, dy, dx; };
    std::vector<Offset3D> offsets;
    for (int dz = -rz; dz <= rz; ++dz) {
        for (int dy = -ry; dy <= ry; ++dy) {
            for (int dx = -rx; dx <= rx; ++dx) {
                double dist_sq = (static_cast<double>(dx * dx) / (rx * rx)) +
                                 (static_cast<double>(dy * dy) / (ry * ry)) +
                                 (static_cast<double>(dz * dz) / (rz * rz));
                if (dist_sq <= 1.0) {
                    offsets.push_back({dz, dy, dx});
                }
            }
        }
    }

    const float* in_ptr = input.data().data();
    float* out_ptr = out.data().data();

    parallel_for(0, D, [&](int z) {
        size_t z_start = static_cast<size_t>(z) * slice_size;
        for (int y = 0; y < H; ++y) {
            size_t row_start = z_start + static_cast<size_t>(y) * W;
            for (int x = 0; x < W; ++x) {
                float min_v = std::numeric_limits<float>::infinity();
                for (const auto& off : offsets) {
                    int sz = reflect_index(z + off.dz, D);
                    int sy = reflect_index(y + off.dy, H);
                    int sx = reflect_index(x + off.dx, W);
                    float v = in_ptr[(static_cast<size_t>(sz) * H + sy) * W + sx];
                    if (v < min_v) min_v = v;
                }
                out_ptr[row_start + x] = min_v;
            }
        }
    });
    return out;
}

Volume3D dilate_3d_ball(const Volume3D& input, int rx, int ry, int rz) {
    int W = input.width();
    int H = input.height();
    int D = input.depth();
    size_t slice_size = static_cast<size_t>(W) * H;
    Volume3D out(W, H, D);
    out.metadata() = input.metadata();

    struct Offset3D { int dz, dy, dx; };
    std::vector<Offset3D> offsets;
    for (int dz = -rz; dz <= rz; ++dz) {
        for (int dy = -ry; dy <= ry; ++dy) {
            for (int dx = -rx; dx <= rx; ++dx) {
                double dist_sq = (static_cast<double>(dx * dx) / (rx * rx)) +
                                 (static_cast<double>(dy * dy) / (ry * ry)) +
                                 (static_cast<double>(dz * dz) / (rz * rz));
                if (dist_sq <= 1.0) {
                    offsets.push_back({dz, dy, dx});
                }
            }
        }
    }

    const float* in_ptr = input.data().data();
    float* out_ptr = out.data().data();

    parallel_for(0, D, [&](int z) {
        size_t z_start = static_cast<size_t>(z) * slice_size;
        for (int y = 0; y < H; ++y) {
            size_t row_start = z_start + static_cast<size_t>(y) * W;
            for (int x = 0; x < W; ++x) {
                float max_v = -std::numeric_limits<float>::infinity();
                for (const auto& off : offsets) {
                    int sz = reflect_index(z + off.dz, D);
                    int sy = reflect_index(y + off.dy, H);
                    int sx = reflect_index(x + off.dx, W);
                    float v = in_ptr[(static_cast<size_t>(sz) * H + sy) * W + sx];
                    if (v > max_v) max_v = v;
                }
                out_ptr[row_start + x] = max_v;
            }
        }
    });
    return out;
}

}  // namespace

Image2D morphology_filter(const Image2D& input, const MorphologyParams2D& params) {
    if (input.empty()) {
        throw std::invalid_argument("morphology_filter: input Image2D is empty");
    }
    if (params.radius_x < 1 || params.radius_y < 1) {
        throw std::invalid_argument("morphology_filter: radius must be >= 1");
    }

    auto do_erode = [&](const Image2D& im) {
        return (params.element == MorphElement::Box) ?
               erode_2d_box(im, params.radius_x, params.radius_y) :
               erode_2d_ball(im, params.radius_x, params.radius_y);
    };

    auto do_dilate = [&](const Image2D& im) {
        return (params.element == MorphElement::Box) ?
               dilate_2d_box(im, params.radius_x, params.radius_y) :
               dilate_2d_ball(im, params.radius_x, params.radius_y);
    };

    switch (params.op) {
        case MorphOp::Erosion:
            return do_erode(input);
        case MorphOp::Dilation:
            return do_dilate(input);
        case MorphOp::Opening:
            return do_dilate(do_erode(input));
        case MorphOp::Closing:
            return do_erode(do_dilate(input));
        case MorphOp::WhiteTopHat: {
            Image2D opened = do_dilate(do_erode(input));
            Image2D out = input;
            for (size_t i = 0; i < input.size(); ++i) {
                out.data()[i] = input.data()[i] - opened.data()[i];
            }
            return out;
        }
        case MorphOp::BlackTopHat: {
            Image2D closed = do_erode(do_dilate(input));
            Image2D out = input;
            for (size_t i = 0; i < input.size(); ++i) {
                out.data()[i] = closed.data()[i] - input.data()[i];
            }
            return out;
        }
    }
    throw std::invalid_argument("morphology_filter: unknown MorphOp");
}

Volume3D morphology_filter(const Volume3D& input, const MorphologyParams3D& params) {
    if (input.empty()) {
        throw std::invalid_argument("morphology_filter: input Volume3D is empty");
    }
    if (params.radius_x < 1 || params.radius_y < 1 || params.radius_z < 1) {
        throw std::invalid_argument("morphology_filter: radius must be >= 1");
    }

    auto do_erode = [&](const Volume3D& vol) {
        return (params.element == MorphElement::Box) ?
               erode_3d_box(vol, params.radius_x, params.radius_y, params.radius_z) :
               erode_3d_ball(vol, params.radius_x, params.radius_y, params.radius_z);
    };

    auto do_dilate = [&](const Volume3D& vol) {
        return (params.element == MorphElement::Box) ?
               dilate_3d_box(vol, params.radius_x, params.radius_y, params.radius_z) :
               dilate_3d_ball(vol, params.radius_x, params.radius_y, params.radius_z);
    };

    switch (params.op) {
        case MorphOp::Erosion:
            return do_erode(input);
        case MorphOp::Dilation:
            return do_dilate(input);
        case MorphOp::Opening:
            return do_dilate(do_erode(input));
        case MorphOp::Closing:
            return do_erode(do_dilate(input));
        case MorphOp::WhiteTopHat: {
            Volume3D opened = do_dilate(do_erode(input));
            Volume3D out = input;
            for (size_t i = 0; i < input.size(); ++i) {
                out.data()[i] = input.data()[i] - opened.data()[i];
            }
            return out;
        }
        case MorphOp::BlackTopHat: {
            Volume3D closed = do_erode(do_dilate(input));
            Volume3D out = input;
            for (size_t i = 0; i < input.size(); ++i) {
                out.data()[i] = closed.data()[i] - input.data()[i];
            }
            return out;
        }
    }
    throw std::invalid_argument("morphology_filter: unknown MorphOp");
}

namespace {
MorphElement parse_element(const std::string& str) {
    return (str == "ball" || str == "disk") ? MorphElement::Ball : MorphElement::Box;
}
}  // namespace

Image2D grayscale_erosion(const Image2D& input, int radius, const std::string& element) {
    return morphology_filter(input, MorphologyParams2D(MorphOp::Erosion, parse_element(element), radius, radius));
}

Image2D grayscale_dilation(const Image2D& input, int radius, const std::string& element) {
    return morphology_filter(input, MorphologyParams2D(MorphOp::Dilation, parse_element(element), radius, radius));
}

Image2D grayscale_opening(const Image2D& input, int radius, const std::string& element) {
    return morphology_filter(input, MorphologyParams2D(MorphOp::Opening, parse_element(element), radius, radius));
}

Image2D grayscale_closing(const Image2D& input, int radius, const std::string& element) {
    return morphology_filter(input, MorphologyParams2D(MorphOp::Closing, parse_element(element), radius, radius));
}

Image2D white_tophat(const Image2D& input, int radius, const std::string& element) {
    return morphology_filter(input, MorphologyParams2D(MorphOp::WhiteTopHat, parse_element(element), radius, radius));
}

Image2D black_tophat(const Image2D& input, int radius, const std::string& element) {
    return morphology_filter(input, MorphologyParams2D(MorphOp::BlackTopHat, parse_element(element), radius, radius));
}

Volume3D grayscale_erosion(const Volume3D& input, int radius, const std::string& element) {
    return morphology_filter(input, MorphologyParams3D(MorphOp::Erosion, parse_element(element), radius, radius, radius));
}

Volume3D grayscale_dilation(const Volume3D& input, int radius, const std::string& element) {
    return morphology_filter(input, MorphologyParams3D(MorphOp::Dilation, parse_element(element), radius, radius, radius));
}

Volume3D grayscale_opening(const Volume3D& input, int radius, const std::string& element) {
    return morphology_filter(input, MorphologyParams3D(MorphOp::Opening, parse_element(element), radius, radius, radius));
}

Volume3D grayscale_closing(const Volume3D& input, int radius, const std::string& element) {
    return morphology_filter(input, MorphologyParams3D(MorphOp::Closing, parse_element(element), radius, radius, radius));
}

Volume3D white_tophat(const Volume3D& input, int radius, const std::string& element) {
    return morphology_filter(input, MorphologyParams3D(MorphOp::WhiteTopHat, parse_element(element), radius, radius, radius));
}

Volume3D black_tophat(const Volume3D& input, int radius, const std::string& element) {
    return morphology_filter(input, MorphologyParams3D(MorphOp::BlackTopHat, parse_element(element), radius, radius, radius));
}

Volume3D grayscale_erosion_slice_by_slice(const Volume3D& input, int radius, const std::string& element) {
    return apply_slice_by_slice(input, [radius, element](const Image2D& s) {
        return grayscale_erosion(s, radius, element);
    });
}

Volume3D grayscale_dilation_slice_by_slice(const Volume3D& input, int radius, const std::string& element) {
    return apply_slice_by_slice(input, [radius, element](const Image2D& s) {
        return grayscale_dilation(s, radius, element);
    });
}

Volume3D grayscale_opening_slice_by_slice(const Volume3D& input, int radius, const std::string& element) {
    return apply_slice_by_slice(input, [radius, element](const Image2D& s) {
        return grayscale_opening(s, radius, element);
    });
}

Volume3D grayscale_closing_slice_by_slice(const Volume3D& input, int radius, const std::string& element) {
    return apply_slice_by_slice(input, [radius, element](const Image2D& s) {
        return grayscale_closing(s, radius, element);
    });
}

Volume3D white_tophat_slice_by_slice(const Volume3D& input, int radius, const std::string& element) {
    return apply_slice_by_slice(input, [radius, element](const Image2D& s) {
        return white_tophat(s, radius, element);
    });
}

Volume3D black_tophat_slice_by_slice(const Volume3D& input, int radius, const std::string& element) {
    return apply_slice_by_slice(input, [radius, element](const Image2D& s) {
        return black_tophat(s, radius, element);
    });
}

}  // namespace ct
