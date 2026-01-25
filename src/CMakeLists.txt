cmake_minimum_required(VERSION 3.24)

project(mifrost LANGUAGES CXX)

# --- C++ Standard ---
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Dependencies ---
# Ensure Conan's CMakeDeps generator folder is on the prefix path (helps CLion).
#if(NOT DEFINED CMAKE_PREFIX_PATH)
#    set(CMAKE_PREFIX_PATH "")
#endif()
#set(_conan_generators "${CMAKE_BINARY_DIR}/conan/build/${CMAKE_BUILD_TYPE}/generators")
#if(EXISTS "${_conan_generators}")
#    list(APPEND CMAKE_PREFIX_PATH "${_conan_generators}")
#else()
#    foreach(_cfg Release Debug RelWithDebInfo MinSizeRel)
#        set(_conan_generators "${CMAKE_BINARY_DIR}/conan/build/${_cfg}/generators")
#        if(EXISTS "${_conan_generators}")
#            list(APPEND CMAKE_PREFIX_PATH "${_conan_generators}")
#            break()
#        endif()
#    endforeach()
#endif()
#unset(_conan_generators)

# Nanobind requires finding Python first
find_package(Python 3.8 COMPONENTS Interpreter Development.Module Development.Embed REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# Dependencies via Conan
find_package(Boost REQUIRED COMPONENTS iostreams)
find_package(fmt REQUIRED)
find_package(ZLIB REQUIRED)
find_package(loki REQUIRED)
find_package(nauty REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(TBB REQUIRED)
find_package(phmap CONFIG REQUIRED)
find_package(unordered_dense CONFIG REQUIRED)
find_package(GTest REQUIRED)

# Conan include fallbacks (some recipes omit include dirs).
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/ConanFallbacks.cmake")
# Ensure nanobind's core library target exists for non-module targets.
nanobind_build_library(nanobind-static)

set(_python_embed_target "")
if(TARGET Python::Python)
    set(_python_embed_target Python::Python)
elseif(TARGET Python::Embed)
    set(_python_embed_target Python::Embed)
endif()

# Valla is provided by Conan; create an alias expected by mimir's CMake config.
find_package(valla CONFIG REQUIRED)
if(TARGET valla::valla AND NOT TARGET valla::core)
    add_library(valla::core ALIAS valla::valla)
endif()

# Loki is provided by Conan; create an alias expected by mimir's CMake config.
if(TARGET loki::loki AND NOT TARGET loki::parsers)
    add_library(loki::parsers ALIAS loki::loki)
endif()

# Mimir Extended Linkage (Found via CMAKE_PREFIX_PATH pointing to wheel contents)
find_package(mimir REQUIRED)

# --- Targets ---
add_library(mifrost_core STATIC
    "mifrost/core/batch_builder.cpp"
    "mifrost/core/hgraph_stream_encoder.cpp"
)

target_include_directories(mifrost_core PUBLIC
    "${CMAKE_CURRENT_LIST_DIR}"
    "${CMAKE_CURRENT_LIST_DIR}/mifrost"
)

target_link_libraries(mifrost_core PUBLIC
    # Public Mimir dependencies (needed if not transitively linked correctly)
    Boost::iostreams
    fmt::fmt
    ZLIB::ZLIB
    loki::loki
    nauty::nauty
    # The Core Mimir Library
    mimir::core
    # Ensure absl headers are available to Loki/Mimir
    absl::node_hash_set
    # Ensure TBB headers are available to Valla/Mimir
    TBB::tbb
    # Ensure parallel-hashmap headers are available for valla's gtl wrapper
    phmap
    # Use unordered_dense for our internal maps
    unordered_dense::unordered_dense
    # Nanobind runtime for core symbols used by tests and module
    nanobind-static
)

conan_add_package_include_fallback(mifrost_core abseil)
conan_add_package_include_fallback(mifrost_core onetbb)

nanobind_add_module(mifrost "mifrost.cpp")
target_link_libraries(mifrost PRIVATE mifrost_core)

# Loki's AST uses >20 variant types; bump Boost limits to avoid MPL list errors.
target_compile_definitions(mifrost_core PUBLIC
    BOOST_MPL_LIMIT_LIST_SIZE=30
    BOOST_MPL_LIMIT_VECTOR_SIZE=30
    BOOST_VARIANT_LIMIT_TYPES=30
    BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
)

# --- Tests ---
include(CTest)
if(BUILD_TESTING)
    add_executable(mifrost_tests
        "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/batch_builder_test.cpp"
    )
    target_include_directories(mifrost_tests PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
    target_link_libraries(mifrost_tests PRIVATE GTest::gtest_main mifrost_core ${_python_embed_target})
    add_test(NAME mifrost_tests COMMAND mifrost_tests)
endif()
