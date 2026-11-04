#pragma once

#include <cctype>
#include <cstdlib>
#include <string>

namespace ct {

/**
 * Normalises an input path string entered interactively:
 * - Trims leading and trailing whitespace
 * - Strips one pair of surrounding single or double quotes
 * - Turns backslash-escaped characters into plain characters
 *   (on Windows paths where backslash is a directory separator, only unescapes '\ ', '\(', '\)')
 * - Expands a leading '~' to the user's home directory (using HOME or USERPROFILE, or home_override)
 */
inline std::string normalize_input_path(std::string s, const std::string& home_override = "") {
    // 1. Trim leading and trailing whitespace
    size_t first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    size_t last = s.find_last_not_of(" \t\r\n");
    s = s.substr(first, last - first + 1);

    // 2. Strip one pair of surrounding single or double quotes
    if (s.size() >= 2) {
        if ((s.front() == '"' && s.back() == '"') ||
            (s.front() == '\'' && s.back() == '\'')) {
            s = s.substr(1, s.size() - 2);
            // Trim any inner whitespace padding
            first = s.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) return "";
            last = s.find_last_not_of(" \t\r\n");
            s = s.substr(first, last - first + 1);
        }
    }

    // Detect Windows drive letter (e.g. C:\... or c:/...)
    bool is_windows_drive = (s.size() >= 2 && std::isalpha(static_cast<unsigned char>(s[0])) && s[1] == ':');

    // 3. Turn backslash-escaped characters into plain character
    std::string unescaped;
    unescaped.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char next = s[i + 1];
            if (is_windows_drive) {
                // On Windows paths, backslash is separator: only unescape '\ ', '\(', '\)'
                if (next == ' ' || next == '(' || next == ')') {
                    unescaped.push_back(next);
                    ++i;
                } else {
                    unescaped.push_back('\\');
                }
            } else {
                // POSIX paths: unescape backslash-escaped character
                unescaped.push_back(next);
                ++i;
            }
        } else {
            unescaped.push_back(s[i]);
        }
    }
    s = std::move(unescaped);

    // 4. Expand a leading '~'
    if (!s.empty() && s[0] == '~') {
        if (s.size() == 1 || s[1] == '/' || s[1] == '\\') {
            std::string home = home_override;
            if (home.empty()) {
                const char* h = std::getenv("HOME");
                if (!h || *h == '\0') {
                    h = std::getenv("USERPROFILE");
                }
                if (h && *h != '\0') {
                    home = h;
                }
            }
            if (!home.empty()) {
                if ((home.back() == '/' || home.back() == '\\') && s.size() > 1) {
                    s = home + s.substr(2);
                } else {
                    s = home + s.substr(1);
                }
            }
        }
    }

    return s;
}

} // namespace ct
