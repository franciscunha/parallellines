#pragma once

#include <string>

namespace paths {
    std::string output(const char *filename) {
        return std::string("build/out/") + std::string(filename);
    }
}