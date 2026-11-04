#include "ct/image2d.hpp"
#include "ct/tiff_io.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path-to-tiff>\n";
        return 1;
    }
    std::string path = argv[1];

    try {
        Image2D image = load_tiff(path);
        const ImageMetadata& meta = image.metadata();

        std::cout << "=== Loaded: " << path << " ===\n";
        std::cout << "  size: (" << image.width() << ", " << image.height() << ")\n";
        std::cout << "  raw_dtype: " << meta.raw_dtype << "\n";
        std::cout << "  spacing: (" << meta.spacing[0] << ", " << meta.spacing[1] << ")"
                  << "  spacing_is_known: " << std::boolalpha << meta.spacing_is_known << "\n";
        std::cout << "  rescale_slope: " << meta.rescale_slope
                  << "  rescale_offset: " << meta.rescale_offset << "\n";

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "\n=== Array stats ===\n";
        std::cout << "  min=" << image.min() << " max=" << image.max()
                  << " mean=" << image.mean() << "\n";

        // Round-trip check, same as the Python prototype. Write next to the
        // current directory rather than next to the source file, since
        // source data directories are often read-only (as this project's
        // upload folder is).
        std::string filename = path.substr(path.find_last_of("/\\") + 1);
        std::string out_path = "./" + filename + ".roundtrip.tif";
        save_tiff(image, out_path);
        Image2D reloaded = load_tiff(out_path);  // note: slope/offset won't re-apply (already-rescaled float file)

        bool shapes_match = (reloaded.width() == image.width() && reloaded.height() == image.height());
        double max_abs_diff = 0.0;
        if (shapes_match) {
            for (int r = 0; r < image.height(); ++r) {
                for (int c = 0; c < image.width(); ++c) {
                    max_abs_diff = std::max(max_abs_diff,
                        std::abs(static_cast<double>(image.at(r, c)) - static_cast<double>(reloaded.at(r, c))));
                }
            }
        }
        std::cout << "\nRound-trip -> " << out_path << "\n";
        std::cout << "  shapes_match=" << shapes_match << " max_abs_diff=" << max_abs_diff << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
