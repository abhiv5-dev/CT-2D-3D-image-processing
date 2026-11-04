cmake_minimum_required(VERSION 3.16)
project(ct_filters VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

if (MSVC)
    option(CT_SANITIZE "Enable AddressSanitizer and UndefinedBehaviorSanitizer on tests" OFF)
    add_compile_options(/W3 /permissive- /Zc:__cplusplus)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
else()
    option(CT_SANITIZE "Enable AddressSanitizer and UndefinedBehaviorSanitizer on tests" ON)
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CT_BUILD_PYTHON "AUTO" CACHE STRING "Build Python bindings extension (ON, OFF, or AUTO)")

# Search paths for macOS Homebrew
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew" "/opt/homebrew/opt/libtiff")

# TIFF dependency
option(CT_FETCH_LIBTIFF "Download and build libtiff with FetchContent if not found" ON)

find_package(TIFF QUIET)

if (TIFF_FOUND)
    message(STATUS "Found system TIFF: ${TIFF_LIBRARIES} (include: ${TIFF_INCLUDE_DIRS})")
elseif (CT_FETCH_LIBTIFF)
    message(STATUS "System TIFF not found. Fetching pinned libtiff 4.6.0 via FetchContent...")
    include(FetchContent)

    # Disable all optional codecs and extras for a zero-dependency static build
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libtiff" FORCE)
    set(tiff-tools OFF CACHE BOOL "Disable libtiff tools" FORCE)
    set(tiff-tests OFF CACHE BOOL "Disable libtiff tests" FORCE)
    set(tiff-contrib OFF CACHE BOOL "Disable libtiff contrib" FORCE)
    set(tiff-docs OFF CACHE BOOL "Disable libtiff docs" FORCE)
    set(zlib OFF CACHE BOOL "Disable zlib codec" FORCE)
    set(libdeflate OFF CACHE BOOL "Disable libdeflate codec" FORCE)
    set(jpeg OFF CACHE BOOL "Disable jpeg codec" FORCE)
    set(old-jpeg OFF CACHE BOOL "Disable old jpeg codec" FORCE)
    set(jpeg12 OFF CACHE BOOL "Disable jpeg12 codec" FORCE)
    set(lzma OFF CACHE BOOL "Disable lzma codec" FORCE)
    set(jbig OFF CACHE BOOL "Disable jbig codec" FORCE)
    set(lerc OFF CACHE BOOL "Disable lerc codec" FORCE)
    set(zstd OFF CACHE BOOL "Disable zstd codec" FORCE)
    set(webp OFF CACHE BOOL "Disable webp codec" FORCE)
    set(pixarlog OFF CACHE BOOL "Disable pixarlog codec" FORCE)

    FetchContent_Declare(
        tiff
        URL https://download.osgeo.org/libtiff/tiff-4.6.0.tar.gz
        URL_HASH SHA256=88b3979e6d5c7e32b50d7ec72fb15af724f6ab2cbf7e10880c360a77e4b5d99a
    )
    FetchContent_MakeAvailable(tiff)

    if (TARGET tiff AND NOT TARGET TIFF::TIFF)
        add_library(TIFF::TIFF ALIAS tiff)
    endif()
else()
    message(FATAL_ERROR "TIFF not found and CT_FETCH_LIBTIFF is OFF. Please install libtiff or enable CT_FETCH_LIBTIFF.")
endif()

# Optional OpenMP
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, enabling multi-threading via OpenMP")
else()
    message(STATUS "OpenMP not found, will use std::thread fallback")
endif()

# Embed config/default_params.json at build time into a standalone C++ header
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/config/default_params.json" CONFIG_JSON_CONTENT)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/ct/default_params_embedded.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include/ct/default_params_embedded.hpp"
    @ONLY
)

# Core static library sources
set(CTFILTERS_SOURCES
    src/image2d.cpp
    src/volume3d.cpp
    src/tiff_io.cpp
    src/preview.cpp
    src/filters/gaussian.cpp
    src/filters/gaussian_reference.cpp
    src/filters/median.cpp
    src/filters/bilateral.cpp
    src/filters/diffusion.cpp
    src/filters/nlm.cpp
    src/filters/wiener.cpp
    src/filters/tv_chambolle.cpp
    src/filters/morphology.cpp
    src/intensity.cpp
)

# Helper function to configure ctfilters variants without code duplication
function(setup_ctfilters_target target_name is_sanitized)
    add_library(${target_name} STATIC ${CTFILTERS_SOURCES})
    target_include_directories(${target_name} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated/include>
        $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(${target_name} PUBLIC TIFF::TIFF)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(${target_name} PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(${target_name} PUBLIC CT_HAS_OPENMP=1)
    endif()
    if (is_sanitized AND NOT MSVC)
        target_compile_options(${target_name} PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer -g)
        target_link_options(${target_name} PUBLIC -fsanitize=address,undefined)
    endif()
endfunction()

# Default optimized library
setup_ctfilters_target(ctfilters FALSE)

# CLI tools
add_executable(demo_io apps/demo_io.cpp)
target_link_libraries(demo_io PRIVATE ctfilters)

add_executable(run_gaussian apps/run_gaussian.cpp)
target_link_libraries(run_gaussian PRIVATE ctfilters)

add_executable(make_preview apps/make_preview.cpp)
target_link_libraries(make_preview PRIVATE ctfilters)

add_executable(ctfilter apps/ctfilter.cpp)
target_link_libraries(ctfilter PRIVATE ctfilters)

add_executable(benchmark apps/benchmark.cpp)
target_link_libraries(benchmark PRIVATE ctfilters)
if (WIN32)
    target_link_libraries(benchmark PRIVATE psapi)
endif()


# Python bindings
set(BUILD_PYBIND11_BINDINGS OFF)

if (NOT CT_BUILD_PYTHON STREQUAL "OFF")
    if (NOT Python_EXECUTABLE AND DEFINED ENV{VIRTUAL_ENV})
        if (WIN32)
            set(Python_EXECUTABLE "$ENV{VIRTUAL_ENV}/Scripts/python.exe")
        else()
            set(Python_EXECUTABLE "$ENV{VIRTUAL_ENV}/bin/python")
        endif()
    endif()

    find_package(Python 3.9 COMPONENTS Interpreter Development.Module QUIET)

    if (Python_FOUND)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
            OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE PYBIND11_DIR_RESULT
            ERROR_QUIET
        )
        if (PYBIND11_DIR_RESULT EQUAL 0 AND EXISTS "${PYBIND11_CMAKE_DIR}")
            set(pybind11_DIR "${PYBIND11_CMAKE_DIR}")
            list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
        endif()

        find_package(pybind11 QUIET)
        if (pybind11_FOUND)
            set(BUILD_PYBIND11_BINDINGS ON)
            message(STATUS "Python ${Python_VERSION} and pybind11 found: building 'ct' Python module")
        else()
            if (CT_BUILD_PYTHON STREQUAL "ON")
                message(FATAL_ERROR "CT_BUILD_PYTHON is ON, but pybind11 was not found.")
            else()
                message(STATUS "pybind11 not found; skipping Python bindings (core library and CLI available)")
            endif()
        endif()
    else()
        if (CT_BUILD_PYTHON STREQUAL "ON")
            message(FATAL_ERROR "CT_BUILD_PYTHON is ON, but Python 3.9+ (Interpreter and Development.Module) was not found.")
        else()
            message(STATUS "Python 3.9+ development headers not found; skipping Python bindings (core library and CLI available)")
        endif()
    endif()
else()
    message(STATUS "Python bindings disabled by CT_BUILD_PYTHON=OFF")
endif()

if (BUILD_PYBIND11_BINDINGS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/python/bindings.cpp")
    pybind11_add_module(ct python/bindings.cpp)
    target_link_libraries(ct PRIVATE ctfilters)
    target_include_directories(ct PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    set_target_properties(ct PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()
