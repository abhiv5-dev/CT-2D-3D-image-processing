#include "ct/preview.hpp"

#include <tiffio.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

struct TIFFCloser {
    void operator()(TIFF* tif) const noexcept {
        if (tif) {
            TIFFClose(tif);
        }
    }
};

using ScopedTIFF = std::unique_ptr<TIFF, TIFFCloser>;

}  // namespace

void write_preview_tiff(const Image2D& image,
                        const std::string& path,
                        double display_min,
                        double display_max) {
    if (image.empty()) {
        throw std::runtime_error("Cannot write preview of empty Image2D: " + path);
    }

    if (std::isnan(display_min)) {
        display_min = image.min();
    }
    if (std::isnan(display_max)) {
        display_max = image.max();
    }

    bool is_constant = (display_max <= display_min);

    ScopedTIFF tif(TIFFOpen(path.c_str(), "w"));
    if (!tif) {
        throw std::runtime_error("Failed to open TIFF for writing: " + path);
    }

    TIFFSetField(tif.get(), TIFFTAG_IMAGEWIDTH, static_cast<uint32_t>(image.width()));
    TIFFSetField(tif.get(), TIFFTAG_IMAGELENGTH, static_cast<uint32_t>(image.height()));
    TIFFSetField(tif.get(), TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif.get(), TIFFTAG_BITSPERSAMPLE, 8);
    TIFFSetField(tif.get(), TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
    TIFFSetField(tif.get(), TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif.get(), TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif.get(), TIFFTAG_ROWSPERSTRIP, static_cast<uint32_t>(image.height()));

    const double range = is_constant ? 1.0 : (display_max - display_min);
    std::vector<unsigned char> row_buf(static_cast<size_t>(image.width()));

    for (int row = 0; row < image.height(); ++row) {
        for (int col = 0; col < image.width(); ++col) {
            if (is_constant) {
                row_buf[static_cast<size_t>(col)] = 128;
            } else {
                double v = static_cast<double>(image.at(row, col));
                if (std::isnan(v)) {
                    v = 0.0;
                }
                double normalized = (v - display_min) / range * 255.0;
                normalized = std::clamp(normalized, 0.0, 255.0);
                row_buf[static_cast<size_t>(col)] = static_cast<unsigned char>(normalized + 0.5);
            }
        }
        if (TIFFWriteScanline(tif.get(), row_buf.data(), static_cast<uint32_t>(row)) < 0) {
            throw std::runtime_error("Failed writing preview scanline " + std::to_string(row));
        }
    }
}
