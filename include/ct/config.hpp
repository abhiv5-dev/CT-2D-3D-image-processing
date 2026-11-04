#pragma once

#include "ct/noise.hpp"

#if __has_include("ct/default_params_embedded.hpp")
#include "ct/default_params_embedded.hpp"
#define CT_HAS_EMBEDDED_JSON 1
#endif

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace ct {

// Built-in fallback identical to config/default_params.json
inline constexpr const char* FALLBACK_DEFAULT_PARAMS_JSON = R"JSON_RAW({
  "_metadata": {
    "description": "Calibrated noise-relative default filter parameters for ctfilter",
    "calibration_date": "2026-10-26T11:39:36.343685+00:00",
    "stack_slices_total": 847,
    "tuning_block_indices": [
      350,
      358,
      366,
      374,
      382,
      390,
      398,
      406,
      414,
      422,
      430,
      438
    ],
    "selection_rule": "Maximize air-ROI noise reduction subject to edge broadening <= 10% on aligned rim and normalized leakage ratio <= 1.30x; if no candidate admissible, select candidate minimizing leakage subject to edge broadening <= 10% (marked 'no admissible setting; mildest/lowest-leakage default')",
    "noise_estimator": "Immerkaer / Laplacian MAD: sigma_hat = median(|I * M|) / (0.67448975 * 6.0)"
  },
  "filters": {
    "Gaussian": {
      "selected_setting": "sigma=1.0",
      "mean_air_noise_reduction_pct": 55.63,
      "mean_edge_broadening_pct": 4.2,
      "mean_structure_leakage_ratio": 1.951,
      "mean_inner_leakage_ratio": 2.016,
      "selection_rule": "uncalibrated practical default; no candidate met leakage <= 1.30x (calibrated minimum-leakage setting sigma=0.3 was near-identity with 1.19% noise reduction)",
      "type": "uncalibrated",
      "sigma": 1.0,
      "radius": 3,
      "calibrated_finding": "Grid search found sigma=0.3 minimizes leakage (1.715x) among broadening <= 10%, but achieves only 1.19% noise reduction. Sigma=1.0 provides 55.63% noise reduction with +4.20% broadening."
    },
    "Bilateral": {
      "selected_setting": "m=3.0x",
      "mean_air_noise_reduction_pct": 69.2,
      "mean_edge_broadening_pct": -0.9,
      "mean_structure_leakage_ratio": 1.204,
      "mean_inner_leakage_ratio": 1.248,
      "selection_rule": "Maximizes air noise reduction among candidates satisfying edge broadening <= 10% and leakage <= 1.30x",
      "type": "noise_relative",
      "multiple_sigma_hat": 3.0,
      "sigma_spatial": 2.0,
      "radius": 5
    },
    "Diffusion": {
      "selected_setting": "m=1.5x,it=15",
      "mean_air_noise_reduction_pct": 74.68,
      "mean_edge_broadening_pct": -0.49,
      "mean_structure_leakage_ratio": 1.248,
      "mean_inner_leakage_ratio": 1.291,
      "selection_rule": "Maximizes air noise reduction among candidates satisfying edge broadening <= 10% and leakage <= 1.30x",
      "type": "noise_relative",
      "multiple_sigma_hat": 1.5,
      "iterations": 15,
      "dt": 0.2
    },
    "Wiener": {
      "selected_setting": "r=2",
      "mean_air_noise_reduction_pct": 66.53,
      "mean_edge_broadening_pct": 2.06,
      "mean_structure_leakage_ratio": 1.175,
      "mean_inner_leakage_ratio": 1.2,
      "selection_rule": "Maximizes air noise reduction among candidates satisfying edge broadening <= 10% and leakage <= 1.30x",
      "type": "window_radius",
      "radius": 2,
      "noise_variance": "automatic"
    },
    "Median": {
      "selected_setting": "r=1",
      "mean_air_noise_reduction_pct": 45.16,
      "mean_edge_broadening_pct": 1.7,
      "mean_structure_leakage_ratio": 2.019,
      "mean_inner_leakage_ratio": 2.062,
      "selection_rule": "uncalibrated default (single candidate r=1 px; no candidate met leakage <= 1.30x)",
      "type": "uncalibrated",
      "radius": 1
    },
    "Morphology": {
      "selected_setting": "opening,r=1",
      "mean_air_noise_reduction_pct": 41.9,
      "mean_edge_broadening_pct": 1.15,
      "mean_structure_leakage_ratio": 1.701,
      "mean_inner_leakage_ratio": 1.732,
      "selection_rule": "uncalibrated default (single candidate opening r=1 px; no candidate met leakage <= 1.30x)",
      "type": "uncalibrated",
      "operation": "opening",
      "radius": 1,
      "element": "box"
    },
    "TV_Chambolle": {
      "selected_setting": "m=1.0x",
      "mean_air_noise_reduction_pct": 73.99,
      "mean_edge_broadening_pct": 1.05,
      "mean_structure_leakage_ratio": 1.198,
      "mean_inner_leakage_ratio": 1.228,
      "selection_rule": "Maximizes air noise reduction among candidates satisfying edge broadening <= 10% and leakage <= 1.30x",
      "type": "noise_relative",
      "multiple_sigma_hat": 1.0,
      "eps": 0.0002,
      "max_iterations": 30
    },
    "NLM": {
      "selected_setting": "m=0.75x",
      "mean_air_noise_reduction_pct": 59.41,
      "mean_edge_broadening_pct": 0.06,
      "mean_structure_leakage_ratio": 1.406,
      "mean_inner_leakage_ratio": 1.46,
      "selection_rule": "no admissible setting; mildest/lowest-leakage default",
      "type": "noise_relative",
      "multiple_sigma_hat": 0.75,
      "patch_radius": 2,
      "search_radius": 5
    }
  }
})JSON_RAW";

inline const char* get_fallback_default_params_json() {
    return FALLBACK_DEFAULT_PARAMS_JSON;
}

inline const char* get_embedded_default_params_json() {
#ifdef CT_HAS_EMBEDDED_JSON
    return embedded::DEFAULT_PARAMS_JSON;
#else
    return FALLBACK_DEFAULT_PARAMS_JSON;
#endif
}

namespace json {

enum class Type { Null, Boolean, Number, String, Array, Object };

struct Value {
    Type type = Type::Null;
    bool bool_val = false;
    double num_val = 0.0;
    std::string str_val;
    std::vector<Value> arr_val;
    std::map<std::string, Value> obj_val;

    bool is_null() const { return type == Type::Null; }
    bool is_bool() const { return type == Type::Boolean; }
    bool is_number() const { return type == Type::Number; }
    bool is_string() const { return type == Type::String; }
    bool is_array() const { return type == Type::Array; }
    bool is_object() const { return type == Type::Object; }

    double as_double(double def = 0.0) const {
        if (type == Type::Number) return num_val;
        if (type == Type::String) {
            try { return std::stod(str_val); } catch (...) {}
        }
        return def;
    }

    int as_int(int def = 0) const {
        if (type == Type::Number) return static_cast<int>(std::round(num_val));
        if (type == Type::String) {
            try { return std::stoi(str_val); } catch (...) {}
        }
        return def;
    }

    std::string as_string(const std::string& def = "") const {
        if (type == Type::String) return str_val;
        if (type == Type::Number) {
            std::ostringstream ss;
            ss << num_val;
            return ss.str();
        }
        if (type == Type::Boolean) return bool_val ? "true" : "false";
        return def;
    }

    const Value& operator[](const std::string& key) const {
        static const Value null_val;
        if (type != Type::Object) return null_val;
        auto it = obj_val.find(key);
        if (it != obj_val.end()) return it->second;
        return null_val;
    }

    bool has_key(const std::string& key) const {
        return (type == Type::Object) && (obj_val.find(key) != obj_val.end());
    }
};

class Parser {
    const std::string& src;
    size_t pos = 0;

    void skip_whitespace() {
        while (pos < src.size() && (src[pos] == ' ' || src[pos] == '\t' || src[pos] == '\n' || src[pos] == '\r')) {
            ++pos;
        }
    }

    char peek() {
        skip_whitespace();
        return (pos < src.size()) ? src[pos] : '\0';
    }

    char get() {
        skip_whitespace();
        return (pos < src.size()) ? src[pos++] : '\0';
    }

public:
    explicit Parser(const std::string& s) : src(s) {}

    Value parse() {
        skip_whitespace();
        Value v = parse_value();
        skip_whitespace();
        return v;
    }

private:
    Value parse_value() {
        char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return parse_string();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        return Value{};
    }

    Value parse_object() {
        Value val;
        val.type = Type::Object;
        if (get() != '{') return val;
        while (true) {
            char c = peek();
            if (c == '}' || c == '\0') {
                if (c == '}') get();
                break;
            }
            Value key = parse_string();
            char colon = peek();
            if (colon == ':') get();
            Value child = parse_value();
            val.obj_val[key.str_val] = std::move(child);
            char sep = peek();
            if (sep == ',') {
                get();
            } else if (sep == '}') {
                get();
                break;
            } else {
                break;
            }
        }
        return val;
    }

    Value parse_array() {
        Value val;
        val.type = Type::Array;
        if (get() != '[') return val;
        while (true) {
            char c = peek();
            if (c == ']' || c == '\0') {
                if (c == ']') get();
                break;
            }
            val.arr_val.push_back(parse_value());
            char sep = peek();
            if (sep == ',') {
                get();
            } else if (sep == ']') {
                get();
                break;
            } else {
                break;
            }
        }
        return val;
    }

    Value parse_string() {
        Value val;
        val.type = Type::String;
        if (get() != '"') return val;
        std::string s;
        while (pos < src.size()) {
            char c = src[pos++];
            if (c == '"') break;
            if (c == '\\' && pos < src.size()) {
                char esc = src[pos++];
                if (esc == '"') s += '"';
                else if (esc == '\\') s += '\\';
                else if (esc == '/') s += '/';
                else if (esc == 'n') s += '\n';
                else if (esc == 't') s += '\t';
                else if (esc == 'r') s += '\r';
                else s += esc;
            } else {
                s += c;
            }
        }
        val.str_val = std::move(s);
        return val;
    }

    Value parse_bool() {
        Value val;
        val.type = Type::Boolean;
        if (src.compare(pos, 4, "true") == 0) {
            pos += 4;
            val.bool_val = true;
        } else if (src.compare(pos, 5, "false") == 0) {
            pos += 5;
            val.bool_val = false;
        }
        return val;
    }

    Value parse_null() {
        Value val;
        val.type = Type::Null;
        if (src.compare(pos, 4, "null") == 0) {
            pos += 4;
        }
        return val;
    }

    Value parse_number() {
        Value val;
        val.type = Type::Number;
        size_t start = pos;
        if (pos < src.size() && src[pos] == '-') ++pos;
        while (pos < src.size() && ((src[pos] >= '0' && src[pos] <= '9') || src[pos] == '.' || src[pos] == 'e' || src[pos] == 'E' || src[pos] == '+' || src[pos] == '-')) {
            ++pos;
        }
        std::string num_str = src.substr(start, pos - start);
        try {
            val.num_val = std::stod(num_str);
        } catch (...) {
            val.num_val = 0.0;
        }
        return val;
    }
};

inline Value parse(const std::string& s) {
    Parser p(s);
    return p.parse();
}

} // namespace json

// Calibration config structures
struct FilterConfig {
    double gaussian_sigma = 1.0;
    int gaussian_radius = 3;

    double bilateral_m = 3.0;
    double bilateral_spatial = 2.0;
    int bilateral_radius = 5;

    double diffusion_m = 1.5;
    int diffusion_iters = 15;
    double diffusion_dt = 0.2;

    int wiener_radius = 2;
    double wiener_noise_var = -1.0; // automatic

    int median_radius = 1;

    std::string morphology_operation = "opening";
    int morphology_radius = 1;
    std::string morphology_element = "box";

    double tv_m = 1.0;
    double tv_eps = 0.0002;
    int tv_iters = 30;

    double nlm_m = 0.75;
    int nlm_patch_radius = 2;
    int nlm_search_radius = 5;

    static FilterConfig load(const std::string& json_str = "") {
        FilterConfig cfg;
        const std::string src = json_str.empty() ? std::string(get_embedded_default_params_json()) : json_str;
        json::Value root = json::parse(src);
        const json::Value& filters = root["filters"];

        if (filters.has_key("Gaussian")) {
            const auto& g = filters["Gaussian"];
            cfg.gaussian_sigma = g["sigma"].as_double(cfg.gaussian_sigma);
            cfg.gaussian_radius = g["radius"].as_int(cfg.gaussian_radius);
        }
        if (filters.has_key("Bilateral")) {
            const auto& b = filters["Bilateral"];
            cfg.bilateral_m = b["multiple_sigma_hat"].as_double(cfg.bilateral_m);
            cfg.bilateral_spatial = b["sigma_spatial"].as_double(cfg.bilateral_spatial);
            cfg.bilateral_radius = b["radius"].as_int(cfg.bilateral_radius);
        }
        if (filters.has_key("Diffusion")) {
            const auto& d = filters["Diffusion"];
            cfg.diffusion_m = d["multiple_sigma_hat"].as_double(cfg.diffusion_m);
            cfg.diffusion_iters = d["iterations"].as_int(cfg.diffusion_iters);
            cfg.diffusion_dt = d["dt"].as_double(cfg.diffusion_dt);
        }
        if (filters.has_key("Wiener")) {
            const auto& w = filters["Wiener"];
            cfg.wiener_radius = w["radius"].as_int(cfg.wiener_radius);
        }
        if (filters.has_key("Median")) {
            const auto& m = filters["Median"];
            cfg.median_radius = m["radius"].as_int(cfg.median_radius);
        }
        if (filters.has_key("Morphology")) {
            const auto& mo = filters["Morphology"];
            cfg.morphology_operation = mo["operation"].as_string(cfg.morphology_operation);
            cfg.morphology_radius = mo["radius"].as_int(cfg.morphology_radius);
            cfg.morphology_element = mo["element"].as_string(cfg.morphology_element);
        }
        if (filters.has_key("TV_Chambolle")) {
            const auto& tv = filters["TV_Chambolle"];
            cfg.tv_m = tv["multiple_sigma_hat"].as_double(cfg.tv_m);
            cfg.tv_eps = tv["eps"].as_double(cfg.tv_eps);
            cfg.tv_iters = tv["max_iterations"].as_int(cfg.tv_iters);
        }
        if (filters.has_key("NLM")) {
            const auto& nlm = filters["NLM"];
            cfg.nlm_m = nlm["multiple_sigma_hat"].as_double(cfg.nlm_m);
            cfg.nlm_patch_radius = nlm["patch_radius"].as_int(cfg.nlm_patch_radius);
            cfg.nlm_search_radius = nlm["search_radius"].as_int(cfg.nlm_search_radius);
        }
        return cfg;
    }
};

// Flags indicating which parameters were explicitly supplied by the caller
struct UserFilterFlags {
    bool has_sigma = false;
    bool has_radius = false;
    bool has_sigma_spatial = false;
    bool has_sigma_range = false;
    bool has_kappa = false;
    bool has_iters = false;
    bool has_dt = false;
    bool has_noise_var = false;
    bool has_weight = false;
    bool has_eps = false;
    bool has_element = false;
    bool has_patch_radius = false;
    bool has_search_radius = false;
    bool has_h = false;
};

// Concrete resolved filter parameters with automated notes
struct ResolvedParams {
    std::string filter_name;
    double sigma = 1.0;
    int radius = 0;
    double sigma_spatial = 2.0;
    double sigma_range = 0.1;
    double kappa = 0.05;
    int iters = 10;
    double dt = -1.0;
    double noise_var = -1.0;
    double weight = 0.1;
    double eps = 2.0e-4;
    std::string element = "box";
    int patch_radius = 2;
    int search_radius = 5;
    double h = 0.05;

    std::vector<std::string> auto_notes;
    std::string uncalibrated_note;

    void print_notes(std::ostream& os = std::cout) const {
        for (const auto& note : auto_notes) {
            os << note << "\n";
        }
        if (!uncalibrated_note.empty()) {
            os << uncalibrated_note << "\n";
        }
    }
};

inline ResolvedParams resolve_filter_parameters(
    const std::string& filter_name,
    double sigma_hat,
    const UserFilterFlags& flags,
    const FilterConfig& cfg = FilterConfig::load(),
    bool is_3d = false
) {
    ResolvedParams p;
    p.filter_name = filter_name;
    std::string f = filter_name;
    std::transform(f.begin(), f.end(), f.begin(), ::tolower);
    // If estimated noise is near zero (e.g. noiseless synthetic image), use a positive floor
    const double eff_sigma = (sigma_hat > 1e-6) ? sigma_hat : 0.01;

    if (f == "gaussian") {
        if (!flags.has_sigma) {
            p.sigma = cfg.gaussian_sigma;
            p.auto_notes.push_back("  [auto] sigma = " + std::to_string(p.sigma));
            p.uncalibrated_note = "  Note: Gaussian default sigma=" + std::to_string(p.sigma) +
                                  " is uncalibrated (no setting met leakage <= 1.30x; calibrated sigma=0.3 was near-identity).";
        }
        if (!flags.has_radius) {
            p.radius = cfg.gaussian_radius;
            p.auto_notes.push_back("  [auto] radius = " + std::to_string(p.radius));
        }
    } else if (f == "bilateral") {
        if (!flags.has_sigma_range) {
            p.sigma_range = cfg.bilateral_m * eff_sigma;
            p.auto_notes.push_back("  [auto] sigma_range = " + std::to_string(p.sigma_range) +
                                   " (" + std::to_string(cfg.bilateral_m) + "x sigma_hat " + std::to_string(eff_sigma) + ")");
        }
        if (!flags.has_sigma_spatial) {
            p.sigma_spatial = cfg.bilateral_spatial;
            p.auto_notes.push_back("  [auto] sigma_spatial = " + std::to_string(p.sigma_spatial));
        }
        if (!flags.has_radius) {
            p.radius = cfg.bilateral_radius;
            p.auto_notes.push_back("  [auto] radius = " + std::to_string(p.radius));
        }
    } else if (f == "diffusion") {
        if (!flags.has_kappa) {
            p.kappa = cfg.diffusion_m * eff_sigma;
            p.auto_notes.push_back("  [auto] kappa = " + std::to_string(p.kappa) +
                                   " (" + std::to_string(cfg.diffusion_m) + "x sigma_hat " + std::to_string(eff_sigma) + ")");
        }
        if (!flags.has_iters) {
            p.iters = cfg.diffusion_iters;
            p.auto_notes.push_back("  [auto] iters = " + std::to_string(p.iters));
        }
        if (!flags.has_dt) {
            p.dt = cfg.diffusion_dt;
            p.auto_notes.push_back("  [auto] dt = " + std::to_string(p.dt));
        }
    } else if (f == "wiener") {
        if (!flags.has_radius) {
            p.radius = cfg.wiener_radius;
            p.auto_notes.push_back("  [auto] radius = " + std::to_string(p.radius));
        }
        if (!flags.has_noise_var) {
            p.noise_var = cfg.wiener_noise_var;
            p.auto_notes.push_back("  [auto] noise_var = estimated");
        }
    } else if (f == "median") {
        if (!flags.has_radius) {
            p.radius = cfg.median_radius;
            p.auto_notes.push_back("  [auto] radius = " + std::to_string(p.radius));
            p.uncalibrated_note = "  Note: Median default radius=" + std::to_string(p.radius) +
                                  " is uncalibrated (single candidate; no setting met leakage <= 1.30x).";
        }
    } else if (f == "opening" || f == "closing" || f == "erosion" || f == "dilation" ||
               f == "white_tophat" || f == "black_tophat" || f == "morphology") {
        if (!flags.has_radius) {
            p.radius = cfg.morphology_radius;
            p.auto_notes.push_back("  [auto] radius = " + std::to_string(p.radius));
            p.uncalibrated_note = "  Note: Morphology default radius=" + std::to_string(p.radius) +
                                  " is uncalibrated (single candidate; no setting met leakage <= 1.30x).";
        }
        if (!flags.has_element) {
            p.element = cfg.morphology_element;
            p.auto_notes.push_back("  [auto] element = " + p.element);
        }
    } else if (f == "tv" || f == "tv_chambolle") {
        if (!flags.has_weight) {
            p.weight = cfg.tv_m * eff_sigma;
            p.auto_notes.push_back("  [auto] weight = " + std::to_string(p.weight) +
                                   " (" + std::to_string(cfg.tv_m) + "x sigma_hat " + std::to_string(eff_sigma) + ")");
        }
        if (!flags.has_iters) {
            p.iters = cfg.tv_iters;
            p.auto_notes.push_back("  [auto] iters = " + std::to_string(p.iters));
        }
        if (!flags.has_eps) {
            p.eps = cfg.tv_eps;
            p.auto_notes.push_back("  [auto] eps = " + std::to_string(p.eps));
        }
    } else if (f == "nlm") {
        if (!flags.has_h) {
            p.h = cfg.nlm_m * eff_sigma;
            p.auto_notes.push_back("  [auto] h = " + std::to_string(p.h) +
                                   " (" + std::to_string(cfg.nlm_m) + "x sigma_hat " + std::to_string(eff_sigma) + ")");
        }
        if (!flags.has_patch_radius) {
            p.patch_radius = is_3d ? 1 : cfg.nlm_patch_radius;
            p.auto_notes.push_back("  [auto] patch_radius = " + std::to_string(p.patch_radius));
        }
        if (!flags.has_search_radius) {
            p.search_radius = is_3d ? 3 : cfg.nlm_search_radius;
            p.auto_notes.push_back("  [auto] search_radius = " + std::to_string(p.search_radius));
        }
    }
    return p;
}

} // namespace ct
