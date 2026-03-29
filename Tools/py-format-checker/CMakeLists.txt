cmake_minimum_required(VERSION 3.15)
project(PyFormatChecker CXX)

# ---------------------------------------------------------------------------
# Find LLVM / Clang
# Adjust the HINTS path if your LLVM install is elsewhere.
# ---------------------------------------------------------------------------
find_package(LLVM REQUIRED CONFIG
    HINTS /usr/lib/llvm-21/lib/cmake/llvm
    NO_DEFAULT_PATH)
find_package(Clang REQUIRED CONFIG
    HINTS /usr/lib/llvm-21/lib/cmake/clang
    NO_DEFAULT_PATH)

message(STATUS "LLVM version : ${LLVM_PACKAGE_VERSION}")
message(STATUS "LLVM headers : ${LLVM_INCLUDE_DIRS}")
message(STATUS "Clang headers: ${CLANG_INCLUDE_DIRS}")

# ---------------------------------------------------------------------------
# Plugin target
# ---------------------------------------------------------------------------
add_library(PyFormatChecker MODULE py_format_checker.cpp)

target_include_directories(PyFormatChecker PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${CLANG_INCLUDE_DIRS}
)

target_compile_definitions(PyFormatChecker PRIVATE
    ${LLVM_DEFINITIONS}
)

# Clang plugins must not link against libclang-cpp or libLLVM directly on
# Linux – they are loaded into a clang process that already has those symbols.
# We only need the headers; leave LLVM_LINK_COMPONENTS empty on purpose.

target_compile_features(PyFormatChecker PRIVATE cxx_std_23)

# Clang on Linux won't accept a plugin SO that defines main, and it must be
# a MODULE (not SHARED).  No "lib" prefix, keep .so suffix.
set_target_properties(PyFormatChecker PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

target_compile_options(PyFormatChecker PRIVATE -fno-rtti)


