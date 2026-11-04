#pragma once

#include "ct/image2d.hpp"
#include "ct/volume3d.hpp"
#include <string>

// 2D TIFF I/O
Image2D load_tiff(const std::string& path);
void save_tiff(const Image2D& image, const std::string& path);

// 3D Volume I/O
// Load a 3D volume from a directory of numbered slice TIFFs, sorted numerically.
// Checks that all slices have identical dimensions.
Volume3D load_volume_from_directory(const std::string& dir_path, const std::string& pattern = "");

// Load a 3D volume from a multi-page TIFF file using TIFFReadDirectory.
Volume3D load_volume_multipage(const std::string& path);

// Save a 3D volume as individual slice TIFFs into a directory.
void save_volume_directory(const Volume3D& volume, const std::string& dir_path, const std::string& prefix = "slice_");

// Save a 3D volume as a single multi-page 32-bit float TIFF file.
void save_volume_multipage(const Volume3D& volume, const std::string& path);

namespace ct {
    using ::load_tiff;
    using ::save_tiff;
    using ::load_volume_from_directory;
    using ::load_volume_multipage;
    using ::save_volume_directory;
    using ::save_volume_multipage;
}
