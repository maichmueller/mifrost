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
if(NOT COMMAND nanobind_build_library)
    if(DEFINED nanobind_BUILD_DIRS_RELEASE AND EXISTS "${nanobind_BUILD_DIRS_RELEASE}/nanobind-config.cmake")
        include("${nanobind_BUILD_DIRS_RELEASE}/nanobind-config.cmake")
    elseif(DEFINED nanobind_BUILD_DIRS_DEBUG AND EXISTS "${nanobind_BUILD_DIRS_DEBUG}/nanobind-config.cmake")
        include("${nanobind_BUILD_DIRS_DEBUG}/nanobind-config.cmake")
    endif()
endif()

# Dependencies via Conan
find_package(Boost CONFIG REQUIRED COMPONENTS iostreams)
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
# Ensure nanobind's shared library target exists for cross-module interop.
nanobind_build_library(nanobind)

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

# Help pymimir's FindNauty by supplying Conan paths.
if(NOT NAUTY_INCLUDE_DIR)
    if(DEFINED nauty_INCLUDE_DIRS_RELEASE)
        set(NAUTY_INCLUDE_DIR "${nauty_INCLUDE_DIRS_RELEASE}")
    elseif(DEFINED nauty_INCLUDE_DIRS_DEBUG)
        set(NAUTY_INCLUDE_DIR "${nauty_INCLUDE_DIRS_DEBUG}")
    endif()
endif()
if(NOT NAUTY_LIBRARY)
    if(DEFINED nauty_LIB_DIRS_RELEASE)
        find_library(NAUTY_LIBRARY NAMES nauty PATHS "${nauty_LIB_DIRS_RELEASE}" NO_DEFAULT_PATH)
    elseif(DEFINED nauty_LIB_DIRS_DEBUG)
        find_library(NAUTY_LIBRARY NAMES nauty PATHS "${nauty_LIB_DIRS_DEBUG}" NO_DEFAULT_PATH)
    endif()
endif()

# Mimir Extended Linkage (Found via CMAKE_PREFIX_PATH pointing to wheel contents)
find_package(mimir REQUIRED)

# --- RPATH mode (dev vs wheel) ---
set(MIFROST_RPATH_MODE "dev" CACHE STRING "RPATH mode: dev or wheel")
set_property(CACHE MIFROST_RPATH_MODE PROPERTY STRINGS dev wheel)

# --- RPATH helpers for editable installs ---
function(_mifrost_append_rpath_from_target out_var target_name)
    if(NOT TARGET "${target_name}")
        return()
    endif()
    foreach(prop IMPORTED_LOCATION IMPORTED_LOCATION_RELEASE IMPORTED_LOCATION_DEBUG)
        get_target_property(_loc "${target_name}" "${prop}")
        if(_loc AND EXISTS "${_loc}")
            get_filename_component(_dir "${_loc}" DIRECTORY)
            list(APPEND "${out_var}" "${_dir}")
        endif()
    endforeach()
    set("${out_var}" "${${out_var}}" PARENT_SCOPE)
endfunction()

# --- Targets ---
add_library(mifrost_core STATIC
    "mifrost/core/batch_builder.cpp"
    "mifrost/core/hgraph_stream_encoder.cpp"
    "mifrost/core/transition_dag.cpp"
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
    nanobind
)

conan_add_package_include_fallback(mifrost_core abseil)
conan_add_package_include_fallback(mifrost_core onetbb)

nanobind_add_module(mifrost_core_module
    NB_DOMAIN pymimir_abi_domain
    "mifrost.cpp"
)
set_target_properties(mifrost_core_module PROPERTIES OUTPUT_NAME "_core")
target_link_libraries(mifrost_core_module PRIVATE mifrost_core)

set(_mifrost_rpaths "")
if(APPLE)
    list(APPEND _mifrost_rpaths "@loader_path")
elseif(UNIX)
    list(APPEND _mifrost_rpaths "$ORIGIN")
endif()
if(MIFROST_RPATH_MODE STREQUAL "dev")
    _mifrost_append_rpath_from_target(_mifrost_rpaths mimir::core)
    _mifrost_append_rpath_from_target(_mifrost_rpaths TBB::tbb)
    _mifrost_append_rpath_from_target(_mifrost_rpaths nanobind)
endif()
list(REMOVE_DUPLICATES _mifrost_rpaths)

set_target_properties(mifrost_core_module PROPERTIES
    BUILD_RPATH "${_mifrost_rpaths}"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${_mifrost_rpaths}"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
if(MIFROST_RPATH_MODE STREQUAL "dev")
    set_target_properties(mifrost_core_module PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/mifrost"
    )
    if(TARGET nanobind)
        set_target_properties(nanobind PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/mifrost"
        )
    endif()
endif()

install(TARGETS mifrost_core_module LIBRARY DESTINATION "mifrost")
if(TARGET nanobind)
    install(TARGETS nanobind LIBRARY DESTINATION "mifrost")
endif()

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
