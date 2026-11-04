#include "ct/image2d.hpp"
#include "ct/tiff_io.hpp"
#include "ct/preview.hpp"

#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <string>

void print_usage(const char* prog) {
    std::cerr << "Usage:\n"
              << "  " << prog << " <input.tif> <output_preview.tif>\n"
              << "      Auto-range: normalizes using the input's own min/max.\n\n"
              << "  " << prog << " <input.tif> <output_preview.tif> <range_min> <range_max>\n"
              << "      Shared-range: normalizes using an explicit range instead --\n"
              << "      use this to make two previews (e.g. before/after a filter)\n"
              << "      visually comparable. Pass the ORIGINAL image's min/max for both.\n";
}

int main(int argc, char** argv) {
    if (argc != 3 && argc != 5) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        std::string input_path = argv[1];
        std::string output_path = argv[2];
        double range_min = std::numeric_limits<double>::quiet_NaN();
        double range_max = std::numeric_limits<double>::quiet_NaN();
        if (argc == 5) {
            range_min = std::stod(argv[3]);
            range_max = std::stod(argv[4]);
        }

        Image2D image = load_tiff(input_path);
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Loaded " << input_path << ": min=" << image.min()
                  << " max=" << image.max() << "\n";

        double used_min = std::isnan(range_min) ? image.min() : range_min;
        double used_max = std::isnan(range_max) ? image.max() : range_max;
        std::cout << "Normalizing display range [" << used_min << ", " << used_max << "]\n";

        write_preview_tiff(image, output_path, range_min, range_max);
        std::cout << "Saved viewable preview -> " << output_path << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
