#pragma once

#include <string>

namespace ct {

enum class DeviceBackend {
    CPU,
    Metal,
    CUDA
};

struct DeviceInfo {
    DeviceBackend backend = DeviceBackend::CPU;
    std::string name = "Host CPU (Apple Silicon)";
    bool unified_memory = true;
    size_t memory_bytes = 24ULL * 1024ULL * 1024ULL * 1024ULL; // 24 GB
};

inline DeviceInfo get_active_device_info() {
    DeviceInfo info;
    info.backend = DeviceBackend::CPU;
    info.name = "Apple Silicon CPU (Multi-threaded NEON)";
    info.unified_memory = true;
    return info;
}

}  // namespace ct
