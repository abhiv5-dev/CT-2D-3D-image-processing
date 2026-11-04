#include "ct/image2d.hpp"
#include "ct/tiff_io.hpp"
#include "ct/filters/gaussian.hpp"

#include <iostream>
#include <iomanip>
#include <string>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input.tif> <sigma> <output.tif>\n";
        return 1;
    }

    try {
        std::string input_path = argv[1];
        double sigma = std::stod(argv[2]);
        std::string output_path = argv[3];

        Image2D input = load_tiff(input_path);
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Input:  min=" << input.min() << " max=" << input.max()
                  << " mean=" << input.mean() << "\n";

        Image2D blurred = ct::gaussian_blur(input, sigma);
        std::cout << "Output: min=" << blurred.min() << " max=" << blurred.max()
                  << " mean=" << blurred.mean() << " (sigma=" << sigma << ")\n";

        save_tiff(blurred, output_path);
        std::cout << "Saved -> " << output_path << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
