#include "ct/tiff_io.hpp"

#include <tiffio.h>
#include <cstdint>
#include <cstring>
#include <regex>
#include <iostream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <filesystem>
#include <algorithm>

namespace {

struct TIFFCloser {
    void operator()(TIFF* tif) const noexcept {
        if (tif) {
            TIFFClose(tif);
        }
    }
};

using ScopedTIFF = std::unique_ptr<TIFF, TIFFCloser>;

std::pair<double, double> parse_slope_offset(const char* description) {
    if (!description) {
        return {1.0, 0.0};
    }
    std::string desc(description);
    std::smatch match;
    double slope = 1.0;
    double offset = 0.0;

    static const std::regex slope_re(R"(slope\s*=\s*([-+0-9.eE]+))");
    static const std::regex offset_re(R"(offset\s*=\s*([-+0-9.eE]+))");

    if (std::regex_search(desc, match, slope_re)) {
        try {
            slope = std::stod(match[1].str());
        } catch (const std::exception& e) {
            std::cerr << "Warning: malformed slope value '" << match[1].str()
                      << "', falling back to 1.0 (" << e.what() << ")\n";
            slope = 1.0;
        }
    }
    if (std::regex_search(desc, match, offset_re)) {
        try {
            offset = std::stod(match[1].str());
        } catch (const std::exception& e) {
            std::cerr << "Warning: malformed offset value '" << match[1].str()
                      << "', falling back to 0.0 (" << e.what() << ")\n";
            offset = 0.0;
        }
    }
    return {slope, offset};
}

void read_current_directory_into(TIFF* tif, float* out_pixels, uint32_t expected_width, uint32_t expected_height,
                                ImageMetadata* meta_out, const std::string& debug_name) {
    if (TIFFIsTiled(tif)) {
        throw std::runtime_error("Tiled TIFFs are unsupported: " + debug_name);
    }

    uint32_t width = 0, height = 0;
    if (!TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width) || width == 0) {
        throw std::runtime_error("Invalid or missing TIFF image width: " + debug_name);
    }
    if (!TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height) || height == 0) {
        throw std::runtime_error("Invalid or missing TIFF image height: " + debug_name);
    }

    if (expected_width > 0 && (width != expected_width || height != expected_height)) {
        throw std::runtime_error("Dimension mismatch in " + debug_name + " (expected " +
                                 std::to_string(expected_width) + "x" + std::to_string(expected_height) +
                                 ", got " + std::to_string(width) + "x" + std::to_string(height) + ")");
    }

    uint16_t compression = COMPRESSION_NONE;
    if (TIFFGetField(tif, TIFFTAG_COMPRESSION, &compression)) {
        if (!TIFFIsCODECConfigured(compression)) {
            std::string comp_name = "Unknown (" + std::to_string(compression) + ")";
            switch (compression) {
                case COMPRESSION_NONE: comp_name = "None (uncompressed)"; break;
                case COMPRESSION_CCITTRLE: comp_name = "CCITT RLE"; break;
                case COMPRESSION_CCITTFAX3: comp_name = "CCITT Group 3 fax"; break;
                case COMPRESSION_CCITTFAX4: comp_name = "CCITT Group 4 fax"; break;
                case COMPRESSION_LZW: comp_name = "LZW"; break;
                case COMPRESSION_OJPEG: comp_name = "Old JPEG"; break;
                case COMPRESSION_JPEG: comp_name = "JPEG"; break;
                case 8: /* COMPRESSION_ADOBE_DEFLATE */
                case COMPRESSION_DEFLATE: comp_name = "Deflate/Zlib (ZIP)"; break;
                case COMPRESSION_PACKBITS: comp_name = "PackBits"; break;
                case 50000: comp_name = "ZSTD"; break;
                case 50001: comp_name = "WEBP"; break;
                case 34887: comp_name = "LERC"; break;
                case 34676: comp_name = "LZMA"; break;
                case 34677: comp_name = "JBIG"; break;
                default: break;
            }
            throw std::runtime_error(
                "TIFF file '" + debug_name + "' uses compression '" + comp_name +
                "' which is not configured in this libtiff build. "
                "To decode this file, rebuild the project with the required codec enabled "
                "(e.g., install system libtiff with codecs, or configure CMake with -Dzlib=ON -Djpeg=ON etc.)."
            );
        }
    }

    uint16_t samples_per_pixel = 1;
    if (TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samples_per_pixel)) {
        if (samples_per_pixel != 1) {
            throw std::runtime_error("Unsupported multi-channel TIFF (samples per pixel = " +
                                     std::to_string(samples_per_pixel) + "): " + debug_name);
        }
    }

    uint16_t bits_per_sample = 0;
    if (!TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bits_per_sample)) {
        throw std::runtime_error("Missing TIFF bits-per-sample: " + debug_name);
    }
    if (bits_per_sample == 64) {
        throw std::runtime_error("64-bit TIFFs are unsupported: " + debug_name);
    }

    uint16_t sample_format = SAMPLEFORMAT_UINT;
    if (!TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &sample_format)) {
        sample_format = SAMPLEFORMAT_UINT;
    }

    uint16_t photometric = PHOTOMETRIC_MINISBLACK;
    TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photometric);

    char* description = nullptr;
    TIFFGetField(tif, TIFFTAG_IMAGEDESCRIPTION, &description);
    auto [slope, offset] = parse_slope_offset(description);

    if (meta_out) {
        meta_out->rescale_slope = slope;
        meta_out->rescale_offset = offset;
        meta_out->spacing = {1.0, 1.0};
        meta_out->spacing_is_known = false;
        meta_out->origin = {0.0, 0.0};
        switch (bits_per_sample) {
            case 8:  meta_out->raw_dtype = (sample_format == SAMPLEFORMAT_INT) ? "int8" : "uint8"; break;
            case 16: meta_out->raw_dtype = (sample_format == SAMPLEFORMAT_INT) ? "int16" : "uint16"; break;
            case 32:
                if (sample_format == SAMPLEFORMAT_IEEEFP) meta_out->raw_dtype = "float32";
                else if (sample_format == SAMPLEFORMAT_INT) meta_out->raw_dtype = "int32";
                else meta_out->raw_dtype = "uint32";
                break;
            default:
                throw std::runtime_error("Unsupported TIFF bits-per-sample: " + std::to_string(bits_per_sample));
        }
    }

    tmsize_t scanline_size = TIFFScanlineSize(tif);
    std::vector<unsigned char> buf(static_cast<size_t>(scanline_size));

    for (uint32_t row = 0; row < height; ++row) {
        if (TIFFReadScanline(tif, buf.data(), row) < 0) {
            throw std::runtime_error("Failed reading scanline " + std::to_string(row) + " of " + debug_name);
        }
        float* row_out = out_pixels + static_cast<size_t>(row) * width;
        for (uint32_t col = 0; col < width; ++col) {
            double raw_value = 0.0;
            if (bits_per_sample == 8) {
                if (sample_format == SAMPLEFORMAT_INT) {
                    raw_value = static_cast<double>(static_cast<int8_t>(buf[col]));
                } else {
                    raw_value = static_cast<double>(buf[col]);
                    if (photometric == PHOTOMETRIC_MINISWHITE) {
                        raw_value = 255.0 - raw_value;
                    }
                }
            } else if (bits_per_sample == 16) {
                if (sample_format == SAMPLEFORMAT_INT) {
                    int16_t v = 0;
                    std::memcpy(&v, buf.data() + static_cast<size_t>(col) * 2, sizeof(v));
                    raw_value = static_cast<double>(v);
                } else {
                    uint16_t v = 0;
                    std::memcpy(&v, buf.data() + static_cast<size_t>(col) * 2, sizeof(v));
                    raw_value = static_cast<double>(v);
                    if (photometric == PHOTOMETRIC_MINISWHITE) {
                        raw_value = 65535.0 - raw_value;
                    }
                }
            } else if (bits_per_sample == 32 && sample_format == SAMPLEFORMAT_IEEEFP) {
                float v = 0.0f;
                std::memcpy(&v, buf.data() + static_cast<size_t>(col) * 4, sizeof(v));
                raw_value = static_cast<double>(v);
            } else if (bits_per_sample == 32 && sample_format == SAMPLEFORMAT_INT) {
                int32_t v = 0;
                std::memcpy(&v, buf.data() + static_cast<size_t>(col) * 4, sizeof(v));
                raw_value = static_cast<double>(v);
            } else if (bits_per_sample == 32) {
                uint32_t v = 0;
                std::memcpy(&v, buf.data() + static_cast<size_t>(col) * 4, sizeof(v));
                raw_value = static_cast<double>(v);
                if (photometric == PHOTOMETRIC_MINISWHITE) {
                    raw_value = 4294967295.0 - raw_value;
                }
            }
            row_out[col] = static_cast<float>(raw_value * slope + offset);
        }
    }
}

bool natural_sort_comparator(const std::string& a, const std::string& b) {
    size_t ia = 0, ib = 0;
    while (ia < a.size() && ib < b.size()) {
        if (std::isdigit(static_cast<unsigned char>(a[ia])) && std::isdigit(static_cast<unsigned char>(b[ib]))) {
            size_t end_a = ia;
            while (end_a < a.size() && std::isdigit(static_cast<unsigned char>(a[end_a]))) end_a++;
            size_t end_b = ib;
            while (end_b < b.size() && std::isdigit(static_cast<unsigned char>(b[end_b]))) end_b++;

            std::string sa = a.substr(ia, end_a - ia);
            std::string sb = b.substr(ib, end_b - ib);
            try {
                unsigned long long va = std::stoull(sa);
                unsigned long long vb = std::stoull(sb);
                if (va != vb) return va < vb;
            } catch (...) {
                if (sa != sb) return sa < sb;
            }
            if (sa.size() != sb.size()) return sa.size() < sb.size();
            ia = end_a;
            ib = end_b;
        } else {
            if (a[ia] != b[ib]) {
                return a[ia] < b[ib];
            }
            ia++;
            ib++;
        }
    }
    return a.size() < b.size();
}

}  // namespace

Image2D load_tiff(const std::string& path) {
    ScopedTIFF tif(TIFFOpen(path.c_str(), "r"));
    if (!tif) {
        throw std::runtime_error("Failed to open TIFF: " + path);
    }

    uint32_t width = 0, height = 0;
    if (!TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &width) || width == 0) {
        throw std::runtime_error("Invalid or missing TIFF image width: " + path);
    }
    if (!TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &height) || height == 0) {
        throw std::runtime_error("Invalid or missing TIFF image height: " + path);
    }

    Image2D image(static_cast<int>(width), static_cast<int>(height));
    read_current_directory_into(tif.get(), image.data().data(), width, height, &image.metadata(), path);
    return image;
}

void save_tiff(const Image2D& image, const std::string& path) {
    if (image.empty()) {
        throw std::runtime_error("Cannot save empty Image2D to TIFF: " + path);
    }

    ScopedTIFF tif(TIFFOpen(path.c_str(), "w"));
    if (!tif) {
        throw std::runtime_error("Failed to open TIFF for writing: " + path);
    }

    TIFFSetField(tif.get(), TIFFTAG_IMAGEWIDTH, static_cast<uint32_t>(image.width()));
    TIFFSetField(tif.get(), TIFFTAG_IMAGELENGTH, static_cast<uint32_t>(image.height()));
    TIFFSetField(tif.get(), TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif.get(), TIFFTAG_BITSPERSAMPLE, 32);
    TIFFSetField(tif.get(), TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
    TIFFSetField(tif.get(), TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif.get(), TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif.get(), TIFFTAG_ROWSPERSTRIP, static_cast<uint32_t>(image.height()));

    for (int row = 0; row < image.height(); ++row) {
        const float* row_ptr = &image.data()[static_cast<size_t>(row) * image.width()];
        if (TIFFWriteScanline(tif.get(), const_cast<float*>(row_ptr), static_cast<uint32_t>(row)) < 0) {
            throw std::runtime_error("Failed writing scanline " + std::to_string(row) + " of " + path);
        }
    }
}

Volume3D load_volume_from_directory(const std::string& dir_path, const std::string& pattern) {
    namespace fs = std::filesystem;
    if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) {
        throw std::runtime_error("Directory does not exist: " + dir_path);
    }

    std::vector<std::string> file_paths;
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (!entry.is_regular_file()) continue;
        std::string filename = entry.path().filename().string();
        std::string ext = entry.path().extension().string();
        for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (ext == ".tif" || ext == ".tiff") {
            if (pattern.empty() || filename.find(pattern) != std::string::npos) {
                file_paths.push_back(entry.path().string());
            }
        }
    }

    if (file_paths.empty()) {
        throw std::runtime_error("No matching TIFF files found in directory: " + dir_path);
    }

    std::sort(file_paths.begin(), file_paths.end(), natural_sort_comparator);

    Image2D first_slice = load_tiff(file_paths[0]);
    int width = first_slice.width();
    int height = first_slice.height();
    int depth = static_cast<int>(file_paths.size());

    Volume3D volume(width, height, depth);
    volume.metadata().spacing = {first_slice.metadata().spacing[0], first_slice.metadata().spacing[1], 1.0};
    volume.metadata().spacing_is_known = false;
    volume.metadata().raw_dtype = first_slice.metadata().raw_dtype;
    volume.metadata().rescale_slope = first_slice.metadata().rescale_slope;
    volume.metadata().rescale_offset = first_slice.metadata().rescale_offset;

    volume.set_slice(0, first_slice);

    for (int z = 1; z < depth; ++z) {
        Image2D s = load_tiff(file_paths[static_cast<size_t>(z)]);
        if (s.width() != width || s.height() != height) {
            throw std::runtime_error("Mismatched slice dimensions in " + file_paths[static_cast<size_t>(z)] +
                                     " (expected " + std::to_string(width) + "x" + std::to_string(height) +
                                     ", got " + std::to_string(s.width()) + "x" + std::to_string(s.height()) + ")");
        }
        volume.set_slice(z, s);
    }

    return volume;
}

Volume3D load_volume_multipage(const std::string& path) {
    ScopedTIFF tif(TIFFOpen(path.c_str(), "r"));
    if (!tif) {
        throw std::runtime_error("Failed to open multi-page TIFF: " + path);
    }

    int num_pages = 0;
    do {
        num_pages++;
    } while (TIFFReadDirectory(tif.get()));

    if (num_pages <= 0) {
        throw std::runtime_error("No directories found in TIFF: " + path);
    }

    TIFFSetDirectory(tif.get(), 0);

    uint32_t width = 0, height = 0;
    if (!TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &width) || width == 0) {
        throw std::runtime_error("Invalid or missing TIFF image width: " + path);
    }
    if (!TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &height) || height == 0) {
        throw std::runtime_error("Invalid or missing TIFF image height: " + path);
    }

    Volume3D volume(static_cast<int>(width), static_cast<int>(height), num_pages);

    for (int z = 0; z < num_pages; ++z) {
        size_t slice_offset = static_cast<size_t>(z) * width * height;
        ImageMetadata meta;
        read_current_directory_into(tif.get(), volume.data().data() + slice_offset, width, height,
                                    (z == 0 ? &meta : nullptr), path + " (page " + std::to_string(z) + ")");
        if (z == 0) {
            volume.metadata().spacing = {meta.spacing[0], meta.spacing[1], 1.0};
            volume.metadata().spacing_is_known = false;
            volume.metadata().raw_dtype = meta.raw_dtype;
            volume.metadata().rescale_slope = meta.rescale_slope;
            volume.metadata().rescale_offset = meta.rescale_offset;
        }
        if (z + 1 < num_pages) {
            if (!TIFFReadDirectory(tif.get())) {
                throw std::runtime_error("Failed advancing to page " + std::to_string(z + 1) + " of " + path);
            }
        }
    }

    return volume;
}

void save_volume_directory(const Volume3D& volume, const std::string& dir_path, const std::string& prefix) {
    if (volume.empty()) {
        throw std::runtime_error("Cannot save empty Volume3D to directory: " + dir_path);
    }

    std::filesystem::create_directories(dir_path);

    for (int z = 0; z < volume.depth(); ++z) {
        std::ostringstream ss;
        ss << dir_path << "/" << prefix << std::setw(4) << std::setfill('0') << z << ".tif";
        save_tiff(volume.slice(z), ss.str());
    }
}

void save_volume_multipage(const Volume3D& volume, const std::string& path) {
    if (volume.empty()) {
        throw std::runtime_error("Cannot save empty Volume3D to multi-page TIFF: " + path);
    }

    ScopedTIFF tif(TIFFOpen(path.c_str(), "w"));
    if (!tif) {
        throw std::runtime_error("Failed to open TIFF for writing: " + path);
    }

    for (int z = 0; z < volume.depth(); ++z) {
        TIFFSetField(tif.get(), TIFFTAG_IMAGEWIDTH, static_cast<uint32_t>(volume.width()));
        TIFFSetField(tif.get(), TIFFTAG_IMAGELENGTH, static_cast<uint32_t>(volume.height()));
        TIFFSetField(tif.get(), TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(tif.get(), TIFFTAG_BITSPERSAMPLE, 32);
        TIFFSetField(tif.get(), TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
        TIFFSetField(tif.get(), TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(tif.get(), TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField(tif.get(), TIFFTAG_ROWSPERSTRIP, static_cast<uint32_t>(volume.height()));

        for (int row = 0; row < volume.height(); ++row) {
            size_t offset = (static_cast<size_t>(z) * volume.height() + row) * volume.width();
            const float* row_ptr = &volume.data()[offset];
            if (TIFFWriteScanline(tif.get(), const_cast<float*>(row_ptr), static_cast<uint32_t>(row)) < 0) {
                throw std::runtime_error("Failed writing scanline " + std::to_string(row) + " of slice " + std::to_string(z));
            }
        }

        if (!TIFFWriteDirectory(tif.get())) {
            throw std::runtime_error("Failed writing TIFF directory for slice " + std::to_string(z) + " of " + path);
        }
    }
}
