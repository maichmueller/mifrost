cmake_minimum_required(VERSION 3.24)

project(mifrost LANGUAGES CXX)

include(CMakePackageConfigHelpers)

# --- C++ Standard ---
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(MIFROST_PYTHON_PACKAGE_DIR "mifrost")
set(MIFROST_SDK_INCLUDE_DIR "${MIFROST_PYTHON_PACKAGE_DIR}/include")
set(MIFROST_SDK_LIBRARY_DIR "${MIFROST_PYTHON_PACKAGE_DIR}/lib")
set(MIFROST_SDK_CMAKE_DIR "${MIFROST_SDK_LIBRARY_DIR}/cmake/mifrost")

set(_mifrost_package_version "${SKBUILD_PROJECT_VERSION}")
if (NOT _mifrost_package_version)
    set(_mifrost_package_version "0.0.0.dev0")
endif ()

option(MIFROST_BUILD_BENCHMARKS "Build C++ benchmarks (Google Benchmark)." OFF)
option(MIFROST_GENERATE_STUBS "Generate nanobind .pyi stubs in dev mode." ON)
option(MIFROST_EXPORT_LOCAL_CONAN_RECIPES "Export local Conan recipes before Conan provider install." ON)
option(MIFROST_FREE_THREADED "Enable nanobind FREE_THREADED mode (if compatible python version)." ON)


# In provider mode, export local recipes (loki/nauty/valla, etc.) before the
# first find_package() triggers Conan resolution on clean machines/CI runners.
if (MIFROST_EXPORT_LOCAL_CONAN_RECIPES AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(_mifrost_repo_root "${CMAKE_CURRENT_LIST_DIR}/..")
    if (DEFINED Python_EXECUTABLE AND EXISTS "${Python_EXECUTABLE}")
        set(_mifrost_python "${Python_EXECUTABLE}")
    else ()
        find_program(_mifrost_python NAMES python3 python)
    endif ()

    if (NOT _mifrost_python)
        message(FATAL_ERROR "Could not locate Python to run conan_export.py. Set Python_EXECUTABLE.")
    endif ()

    set(_mifrost_conan_cmd "")
    if (DEFINED CONAN_COMMAND AND NOT CONAN_COMMAND STREQUAL "")
        set(_mifrost_conan_cmd "${CONAN_COMMAND}")
    endif ()
    if (_mifrost_conan_cmd STREQUAL "")
        set(_mifrost_conan_cmd "$ENV{CONAN_COMMAND}")
    endif ()
    if (_mifrost_conan_cmd STREQUAL "")
        set(_mifrost_conan_cmd "$ENV{CONAN_CMD}")
    endif ()
    if (_mifrost_conan_cmd STREQUAL "")
        find_program(_mifrost_conan_cmd NAMES conan)
    endif ()
    if (NOT _mifrost_conan_cmd)
        set(_mifrost_conan_cmd "conan")
    endif ()

    execute_process(
            COMMAND "${_mifrost_python}" "${_mifrost_repo_root}/conan_export.py" "--conan_cmd" "${_mifrost_conan_cmd}"
            WORKING_DIRECTORY "${_mifrost_repo_root}"
            RESULT_VARIABLE _mifrost_export_result
    )
    if (NOT _mifrost_export_result EQUAL 0)
        message(FATAL_ERROR "Failed to export local Conan recipes with ${_mifrost_conan_cmd}.")
    endif ()

    unset(_mifrost_repo_root)
    unset(_mifrost_python)
    unset(_mifrost_conan_cmd)
    unset(_mifrost_export_result)
endif ()

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

# Nanobind requires finding Python first.
# Prefer the active virtualenv when present, otherwise fall back to the active
# conda env if no explicit Python is configured.
if (NOT DEFINED Python_EXECUTABLE AND DEFINED ENV{VIRTUAL_ENV})
    set(Python_EXECUTABLE "$ENV{VIRTUAL_ENV}/bin/python")
    set(Python_ROOT_DIR "$ENV{VIRTUAL_ENV}")
elseif (NOT DEFINED Python_EXECUTABLE AND DEFINED ENV{CONDA_PREFIX})
    set(Python_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
    set(Python_ROOT_DIR "$ENV{CONDA_PREFIX}")
endif ()
find_package(
        Python 3.12
        REQUIRED
        COMPONENTS Interpreter Development.Module
        OPTIONAL_COMPONENTS Development.Embed
)

find_package(nanobind CONFIG REQUIRED)
if (NOT COMMAND nanobind_build_library)
    if (DEFINED nanobind_BUILD_DIRS_RELEASE AND EXISTS "${nanobind_BUILD_DIRS_RELEASE}/nanobind-config.cmake")
        include("${nanobind_BUILD_DIRS_RELEASE}/nanobind-config.cmake")
    elseif (DEFINED nanobind_BUILD_DIRS_DEBUG AND EXISTS "${nanobind_BUILD_DIRS_DEBUG}/nanobind-config.cmake")
        include("${nanobind_BUILD_DIRS_DEBUG}/nanobind-config.cmake")
    endif ()
endif ()

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
find_package(strong_type REQUIRED)
find_package(range-v3 REQUIRED)
find_package(dlpack REQUIRED)
if (MIFROST_BUILD_BENCHMARKS)
    find_package(benchmark REQUIRED)
    find_package(argparse REQUIRED)
endif ()
if (NOT TARGET dlpack::dlpack)
    message(FATAL_ERROR "Expected Conan package dlpack to define target dlpack::dlpack")
endif ()

# Conan include fallbacks (some recipes omit include dirs).
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/ConanFallbacks.cmake")
set(_python_embed_target "")
if (TARGET Python::Python)
    set(_python_embed_target Python::Python)
elseif (TARGET Python::Embed)
    set(_python_embed_target Python::Embed)
endif ()

# Valla is provided by Conan; create an alias expected by mimir's CMake config.
find_package(valla CONFIG REQUIRED)
if (TARGET valla::valla AND NOT TARGET valla::core)
    add_library(valla::core ALIAS valla::valla)
endif ()

# Loki is provided by Conan; create an alias expected by mimir's CMake config.
if (TARGET loki::loki AND NOT TARGET loki::parsers)
    add_library(loki::parsers ALIAS loki::loki)
endif ()

# Help pymimir's FindNauty by supplying Conan paths.
if (NOT NAUTY_INCLUDE_DIR)
    if (DEFINED nauty_INCLUDE_DIRS_RELEASE)
        set(NAUTY_INCLUDE_DIR "${nauty_INCLUDE_DIRS_RELEASE}")
    elseif (DEFINED nauty_INCLUDE_DIRS_DEBUG)
        set(NAUTY_INCLUDE_DIR "${nauty_INCLUDE_DIRS_DEBUG}")
    endif ()
endif ()
if (NOT NAUTY_LIBRARY)
    if (DEFINED nauty_LIB_DIRS_RELEASE)
        find_library(NAUTY_LIBRARY NAMES nauty PATHS "${nauty_LIB_DIRS_RELEASE}" NO_DEFAULT_PATH)
    elseif (DEFINED nauty_LIB_DIRS_DEBUG)
        find_library(NAUTY_LIBRARY NAMES nauty PATHS "${nauty_LIB_DIRS_DEBUG}" NO_DEFAULT_PATH)
    endif ()
endif ()

# Resolve mimir CMake package from explicit env override or Python runtime.
if (NOT DEFINED mimir_DIR)
    if (DEFINED ENV{MIMIR_CMAKE_DIR} AND NOT "$ENV{MIMIR_CMAKE_DIR}" STREQUAL "")
        set(mimir_DIR "$ENV{MIMIR_CMAKE_DIR}" CACHE PATH "Path to mimir CMake package")
    else ()
        execute_process(
                COMMAND "${Python_EXECUTABLE}" "-c" "import pymimir; print(pymimir.get_cmake_dir())"
                RESULT_VARIABLE _mifrost_mimir_result
                OUTPUT_VARIABLE _mifrost_mimir_dir
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if (_mifrost_mimir_result EQUAL 0 AND
        (EXISTS "${_mifrost_mimir_dir}/mimir-config.cmake" OR EXISTS "${_mifrost_mimir_dir}/mimirConfig.cmake"))
            set(mimir_DIR "${_mifrost_mimir_dir}" CACHE PATH "Path to mimir CMake package")
        else ()
            message(STATUS "mimir auto-detect failed with Python_EXECUTABLE=${Python_EXECUTABLE}; set MIMIR_CMAKE_DIR or mimir_DIR.")
        endif ()
        unset(_mifrost_mimir_result)
        unset(_mifrost_mimir_dir)
    endif ()
endif ()

if (DEFINED mimir_DIR AND NOT mimir_DIR STREQUAL "")
    message(STATUS "Using mimir_DIR=${mimir_DIR}")
endif ()

# Mimir Extended Linkage (Found via CMAKE_PREFIX_PATH or pymimir.get_cmake_dir()).
find_package(mimir REQUIRED)

if (NOT DEFINED mimir_VERSION)
    message(FATAL_ERROR "mimir was found, but did not report mimir_VERSION")
endif()

if (mimir_VERSION VERSION_LESS "0.13.60")
    message(FATAL_ERROR "mimir >= 0.13.60 required, found ${mimir_VERSION}")
endif()

# --- RPATH mode (dev vs wheel) ---
set(MIFROST_RPATH_MODE "dev" CACHE STRING "RPATH mode: dev or wheel")
set_property(CACHE MIFROST_RPATH_MODE PROPERTY STRINGS dev wheel)

# --- RPATH helpers for editable installs ---
function(_mifrost_append_rpath_from_target out_var target_name)
    if (NOT TARGET "${target_name}")
        return()
    endif ()
    foreach (prop
            IMPORTED_LOCATION
            IMPORTED_LOCATION_RELEASE
            IMPORTED_LOCATION_DEBUG
            INTERFACE_LINK_DIRECTORIES)
        get_target_property(_loc "${target_name}" "${prop}")
        if (_loc)
            foreach (_entry IN LISTS _loc)
                if (_entry MATCHES "\\$<")
                    # Keep generator expressions (e.g. config-specific Conan dirs).
                    list(APPEND "${out_var}" "${_entry}")
                elseif (IS_DIRECTORY "${_entry}")
                    list(APPEND "${out_var}" "${_entry}")
                elseif (EXISTS "${_entry}")
                    get_filename_component(_dir "${_entry}" DIRECTORY)
                    list(APPEND "${out_var}" "${_dir}")
                endif ()
            endforeach ()
        endif ()
    endforeach ()
    set("${out_var}" "${${out_var}}" PARENT_SCOPE)
endfunction()

# --- Targets ---
add_library(
    mifrost_core
    SHARED
        "_core/mifrost/core/batch_builder.cpp"
        "_core/mifrost/core/schema.cpp"
        "_core/mifrost/input_handling/batch_input_goal_inputs.cpp"
        "_core/mifrost/core/encoders/homo/color_encoder.cpp"
        "_core/mifrost/core/encoders/flat/flat_encoder_common.cpp"
        "_core/mifrost/core/encoders/flat/flat_entity_context.cpp"
        "_core/mifrost/core/encoders/flat/flat_goal_helpers.cpp"
        "_core/mifrost/core/encoders/flat/flat_horizon_context.cpp"
        "_core/mifrost/core/encoders/flat/flat_horizon_encoder.cpp"
        "_core/mifrost/core/encoders/flat/flat_lgan.cpp"
        "_core/mifrost/core/encoders/flat/flat_relation_context.cpp"
        "_core/mifrost/core/encoders/flat/flat_relation_encoder.cpp"
        "_core/mifrost/core/encoders/flat/flat_relation_schema.cpp"
        "_core/mifrost/core/encoders/hetero/hgraph_stream_encoder.cpp"
        "_core/mifrost/core/encoders/common/target_metadata.cpp"
        "_core/mifrost/core/encoders/common/transition_dag.cpp"
        "_core/mifrost/core/encoders/hetero/horizon_hgraph_encoder.cpp"
        "_core/mifrost/core/encoders/hetero/successor_hgraph_encoder.cpp"
)
set_target_properties(
    mifrost_core
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME core
)
target_compile_definitions(
    mifrost_core
    PUBLIC
    MIFROST_BUILD_SHARED
    PRIVATE
    MIFROST_EXPORTS
)
if (NOT TARGET mifrost::core)
    add_library(mifrost::core ALIAS mifrost_core)
endif ()


target_include_directories(mifrost_core PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/_core>"
    "$<INSTALL_INTERFACE:${MIFROST_SDK_INCLUDE_DIR}>"
)

target_link_libraries(
        mifrost_core
        PUBLIC
    # Public SDK dependencies surfaced by installed headers.
        fmt::fmt
    loki::parsers
        mimir::core
        absl::btree
        phmap
        unordered_dense::unordered_dense
        rollbear::strong_type
        range-v3::range-v3
    PRIVATE
    Boost::iostreams
    ZLIB::ZLIB
    nauty::nauty
    absl::node_hash_set
    dlpack::dlpack
)

conan_add_package_include_fallback(mifrost_core abseil SCOPE PUBLIC)
conan_add_package_include_fallback(mifrost_core onetbb SCOPE PRIVATE)

add_library(
    mifrost_python_support
    STATIC
        "_core/mifrost/common.cpp"
        "_core/mifrost/core/batch_builder_python.cpp"
        "_core/mifrost/input_handling/batch_input_parser.cpp"
        "_core/mifrost/input_handling/batch_input_hgraph_parser.cpp"
        "_core/mifrost/input_handling/batch_input_color_parser.cpp"
        "_core/mifrost/input_handling/batch_input_flat_parser.cpp"
        "_core/mifrost/input_handling/batch_input_successor_parser.cpp"
        "_core/mifrost/input_handling/batch_input_horizon_parser.cpp"
)
set_target_properties(mifrost_python_support PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(
    mifrost_python_support
    PRIVATE
    MIFROST_BUILD_SHARED
    MIFROST_ENABLE_PYTHON_API
    NB_DOMAIN=pymimir_abi_domain
    MIFROST_EXPORTS
)
target_link_libraries(
    mifrost_python_support
    PRIVATE
    mifrost_core
    Python::Module
    nanobind::nanobind
    dlpack::dlpack
)
target_include_directories(
    mifrost_python_support
    PRIVATE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/_core>"
)
conan_add_package_include_fallback(mifrost_python_support abseil SCOPE PRIVATE)

set(_mifrost_core_rpaths "")
if (APPLE)
    list(APPEND _mifrost_core_rpaths "@loader_path")
    # Keep a stable relative lookup for pymimir in both wheel and editable installs.
    list(APPEND _mifrost_core_rpaths "@loader_path/../../pymimir/lib")
elseif (UNIX)
    list(APPEND _mifrost_core_rpaths "$ORIGIN")
    # Keep a stable relative lookup for pymimir in both wheel and editable installs.
    list(APPEND _mifrost_core_rpaths "$ORIGIN/../../pymimir/lib")
endif ()
if (MIFROST_RPATH_MODE STREQUAL "wheel")
    # Wheel mode keeps this path as well; duplicates are removed below.
endif ()
if (MIFROST_RPATH_MODE STREQUAL "dev")
    _mifrost_append_rpath_from_target(_mifrost_core_rpaths mimir::core)

    set(_mifrost_conan_home "")
    if (DEFINED ENV{CONAN_HOME} AND NOT "$ENV{CONAN_HOME}" STREQUAL "")
        set(_mifrost_conan_home "$ENV{CONAN_HOME}")
    elseif (DEFINED ENV{HOME} AND EXISTS "$ENV{HOME}/.conan2")
        set(_mifrost_conan_home "$ENV{HOME}/.conan2")
    endif ()
    if (NOT _mifrost_conan_home STREQUAL "" AND IS_DIRECTORY "${_mifrost_conan_home}")
        file(GLOB _mifrost_conan_lib_dirs LIST_DIRECTORIES true
                "${_mifrost_conan_home}/p/*/p/lib"
                "${_mifrost_conan_home}/p/b/*/p/lib"
        )
        foreach (_dir IN LISTS _mifrost_conan_lib_dirs)
            if (IS_DIRECTORY "${_dir}")
                list(APPEND _mifrost_core_rpaths "${_dir}")
            endif ()
        endforeach ()
        unset(_mifrost_conan_lib_dirs)
    endif ()
    unset(_mifrost_conan_home)
endif ()
list(REMOVE_DUPLICATES _mifrost_core_rpaths)
set_target_properties(
        mifrost_core
        PROPERTIES
        BUILD_RPATH "${_mifrost_core_rpaths}"
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "${_mifrost_core_rpaths}"
        INSTALL_RPATH_USE_LINK_PATH FALSE
)

if(MIFROST_FREE_THREADED)
    set(_mifrost_freethreading "FREE_THREADED")  # automatically ignored if python is not built for free-threading
endif ()


nanobind_add_module(mifrost_core_module
        ${_mifrost_freethreading}
        STABLE_ABI  # free-threading does not have stable api, but nanobind handles this.
        NB_DOMAIN pymimir_abi_domain
        "_core/mifrost/mifrost.cpp"
        "_core/mifrost/pyg_views.cpp"
        "_core/mifrost/batch_encoding_graph_field_access.cpp"
        "_core/mifrost/batch_encoding_graph_field_mutation.cpp"
        "_core/mifrost/batch_encoding_graph_field_serialization.cpp"
        "_core/mifrost/batch_encoding_python_collation.cpp"
        "_core/mifrost/batch_encoding_repr.cpp"
        "_core/mifrost/batch_encoding_schema.cpp"
        "_core/mifrost/batch_encoding_state.cpp"
        "_core/mifrost/batch_encoding_tensor_cache.cpp"
        "_core/mifrost/init_common.cpp"
        "_core/mifrost/init_map_view.cpp"
        "_core/mifrost/init_relation_formatter.cpp"
        "_core/mifrost/init_color_encoder.cpp"
        "_core/mifrost/init_schema.cpp"
        "_core/mifrost/schema_bindings.cpp"
        "_core/mifrost/init_batch_encoding.cpp"
        "_core/mifrost/init_hgraph_encoder.cpp"
        "_core/mifrost/init_flat_encoder.cpp"
        "_core/mifrost/init_horizon_encoder.cpp"
        "_core/mifrost/init_successor_encoders.cpp"
        "_core/mifrost/init_transition_dag.cpp"
)
set_target_properties(
        mifrost_core_module
        PROPERTIES
        OUTPUT_NAME "_core"
)
target_link_libraries(
        mifrost_core_module
        PRIVATE
    dlpack::dlpack
        mifrost_core
    mifrost_python_support
)
target_compile_definitions(mifrost_core_module PRIVATE MIFROST_ENABLE_PYTHON_API)

set(_mifrost_rpaths "")
if (APPLE)
    list(APPEND _mifrost_rpaths "@loader_path" "@loader_path/lib")
    # Keep a stable relative lookup for pymimir in both wheel and editable installs.
    list(APPEND _mifrost_rpaths "@loader_path/../pymimir/lib")
elseif (UNIX)
    list(APPEND _mifrost_rpaths "$ORIGIN" "$ORIGIN/lib")
    # Keep a stable relative lookup for pymimir in both wheel and editable installs.
    list(APPEND _mifrost_rpaths "$ORIGIN/../pymimir/lib")
endif ()
if (MIFROST_RPATH_MODE STREQUAL "wheel")
    # Wheel mode keeps this path as well; duplicates are removed below.
endif ()
if (MIFROST_RPATH_MODE STREQUAL "dev")
    _mifrost_append_rpath_from_target(_mifrost_rpaths mimir::core)

    # Fallback: if Conan is used to provide runtime libraries, bake the Conan cache
    # lib dirs into the RPATH. This avoids relying on imported-target properties
    # (which vary across CMake package configs) and makes CI source installs
    # deterministic.
    set(_mifrost_conan_home "")
    if (DEFINED ENV{CONAN_HOME} AND NOT "$ENV{CONAN_HOME}" STREQUAL "")
        set(_mifrost_conan_home "$ENV{CONAN_HOME}")
    elseif (DEFINED ENV{HOME} AND EXISTS "$ENV{HOME}/.conan2")
        set(_mifrost_conan_home "$ENV{HOME}/.conan2")
    endif ()
    if (NOT _mifrost_conan_home STREQUAL "" AND IS_DIRECTORY "${_mifrost_conan_home}")
        file(GLOB _mifrost_conan_lib_dirs LIST_DIRECTORIES true
                "${_mifrost_conan_home}/p/*/p/lib"
                "${_mifrost_conan_home}/p/b/*/p/lib"
        )
        foreach (_dir IN LISTS _mifrost_conan_lib_dirs)
            if (IS_DIRECTORY "${_dir}")
                list(APPEND _mifrost_rpaths "${_dir}")
            endif ()
        endforeach ()
        unset(_mifrost_conan_lib_dirs)
    endif ()
    unset(_mifrost_conan_home)
endif ()
list(REMOVE_DUPLICATES _mifrost_rpaths)

set_target_properties(
        mifrost_core_module
        PROPERTIES
        BUILD_RPATH "${_mifrost_rpaths}"
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "${_mifrost_rpaths}"
        # Avoid embedding transient link directories (e.g. pip-build-env overlays).
        INSTALL_RPATH_USE_LINK_PATH FALSE
)
if (MIFROST_RPATH_MODE STREQUAL "dev")
    set_target_properties(mifrost_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/mifrost/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/mifrost/lib"
    )
    set_target_properties(mifrost_core_module PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/mifrost"
    )
endif ()

configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/../cmake/mifrostConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/mifrostConfig.cmake"
    INSTALL_DESTINATION "${MIFROST_SDK_CMAKE_DIR}"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/mifrostConfigVersion.cmake"
    VERSION "${_mifrost_package_version}"
    COMPATIBILITY SameMajorVersion
)

install(TARGETS mifrost_core
    EXPORT mifrostTargets
    LIBRARY DESTINATION "${MIFROST_SDK_LIBRARY_DIR}"
    ARCHIVE DESTINATION "${MIFROST_SDK_LIBRARY_DIR}"
    RUNTIME DESTINATION "${MIFROST_SDK_LIBRARY_DIR}"
    INCLUDES DESTINATION "${MIFROST_SDK_INCLUDE_DIR}"
)
install(TARGETS mifrost_core_module LIBRARY DESTINATION "${MIFROST_PYTHON_PACKAGE_DIR}")

foreach (_mifrost_private_sdk_header IN ITEMS
    dlpack_utils.hpp
    map_view.hpp
    nanobind_unordered_dense.hpp
    nb_instance.hpp
)
    install(CODE
        "file(REMOVE \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${MIFROST_SDK_INCLUDE_DIR}/mifrost/core/${_mifrost_private_sdk_header}\")"
    )
endforeach ()
install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/_core/mifrost/core/"
    DESTINATION "${MIFROST_SDK_INCLUDE_DIR}/mifrost/core"
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "dlpack_utils.hpp" EXCLUDE
    PATTERN "map_view.hpp" EXCLUDE
    PATTERN "nanobind_unordered_dense.hpp" EXCLUDE
    PATTERN "nb_instance.hpp" EXCLUDE
)
install(EXPORT mifrostTargets
    FILE mifrostTargets.cmake
    NAMESPACE mifrost::
    DESTINATION "${MIFROST_SDK_CMAKE_DIR}"
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/mifrostConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/mifrostConfigVersion.cmake"
    DESTINATION "${MIFROST_SDK_CMAKE_DIR}"
)

if (MIFROST_GENERATE_STUBS)
    set(_mifrost_stub_output "mifrost/_core.pyi")
    if (MIFROST_RPATH_MODE STREQUAL "dev")
        # Dev mode: keep stubs in source tree so IDEs can pick them up directly.
        STRING(PREPEND _mifrost_stub_output "${CMAKE_SOURCE_DIR}/src/")
        nanobind_add_stub(
                mifrost_core_module_stubs
                MODULE mifrost._core
                OUTPUT "${_mifrost_stub_output}"
                PYTHON_PATH "${CMAKE_SOURCE_DIR}/src"
                DEPENDS mifrost_core_module
        )
    else ()
        # Wheel/install mode: generate stubs at install time in the install prefix.
        nanobind_add_stub(
                mifrost_core_module_stubs
                MODULE mifrost._core
                OUTPUT "${_mifrost_stub_output}"
                DEPENDS mifrost_core_module
                INSTALL_TIME
        )
    endif ()
endif ()

# Loki's AST uses >20 variant types; bump Boost limits to avoid MPL list errors.
target_compile_definitions(mifrost_core PUBLIC
        BOOST_MPL_LIMIT_LIST_SIZE=30
        BOOST_MPL_LIMIT_VECTOR_SIZE=30
        BOOST_VARIANT_LIMIT_TYPES=30
        BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
)

if (MIFROST_BUILD_BENCHMARKS)
    add_executable(
            mifrost_bench_hgraph
            "${CMAKE_SOURCE_DIR}/bench/bench_hgraph_encoder.cpp"
    )
    target_link_libraries(
            mifrost_bench_hgraph
            PRIVATE
            mifrost_core
            benchmark::benchmark
            argparse::argparse
            ${_python_embed_target}
    )
    target_compile_definitions(
            mifrost_bench_hgraph
            PRIVATE
            MIFROST_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
    )
    target_compile_definitions(mifrost_bench_hgraph PRIVATE MIFROST_BENCH_DEFAULT_DOMAIN="blocks")
    target_compile_definitions(mifrost_bench_hgraph PRIVATE MIFROST_BENCH_DEFAULT_PROBLEM="smedium")

    add_executable(
            mifrost_bench_flat_relation
            "${CMAKE_SOURCE_DIR}/bench/bench_flat_relation_encoder.cpp"
    )
    target_link_libraries(
            mifrost_bench_flat_relation
            PRIVATE
            mifrost_core
            benchmark::benchmark
            argparse::argparse
            ${_python_embed_target}
    )
    target_compile_definitions(
            mifrost_bench_flat_relation
            PRIVATE
            MIFROST_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
    )
    target_compile_definitions(mifrost_bench_flat_relation PRIVATE MIFROST_BENCH_DEFAULT_DOMAIN="blocks")
    target_compile_definitions(mifrost_bench_flat_relation PRIVATE MIFROST_BENCH_DEFAULT_PROBLEM="smedium")

    add_executable(
            mifrost_bench_relation_encoders
            "${CMAKE_SOURCE_DIR}/bench/bench_relation_encoder_compare.cpp"
    )
    target_link_libraries(
            mifrost_bench_relation_encoders
            PRIVATE
            mifrost_core
            benchmark::benchmark
            argparse::argparse
            ${_python_embed_target}
    )
    target_compile_definitions(
            mifrost_bench_relation_encoders
            PRIVATE
            MIFROST_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
    )
    target_compile_definitions(mifrost_bench_relation_encoders PRIVATE MIFROST_BENCH_DEFAULT_DOMAIN="blocks")
    target_compile_definitions(mifrost_bench_relation_encoders PRIVATE MIFROST_BENCH_DEFAULT_PROBLEM="smedium")
endif ()

# --- Tests ---
if (BUILD_TESTING)
    find_package(GTest REQUIRED)
    add_executable(
            mifrost_tests
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/batch_builder_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/color_encoder_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/flat_horizon_encoder_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/flat_relation_encoder_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/flat_stream_encoder_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/hgraph_encoder_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/hgraph_history_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/hgraph_stream_cache_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/hgraph_stream_contract_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/horizon_hgraph_encoder_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/horizon_stream_cache_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/successor_hgraph_encoder_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/transition_dag_test.cpp"
    )
    target_include_directories(mifrost_tests PRIVATE "${CMAKE_CURRENT_LIST_DIR}/_core")
    target_link_libraries(mifrost_tests PRIVATE GTest::gtest_main mifrost_core ${_python_embed_target})
    target_compile_definitions(mifrost_tests PRIVATE MIFROST_DATA_DIR="${CMAKE_SOURCE_DIR}/data")
    add_test(NAME mifrost_tests COMMAND mifrost_tests)
endif ()
