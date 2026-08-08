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
option(MIFROST_BUILD_PYTHON "Build Python extensions for the neutral core and selected adapters." ON)
option(MIFROST_BUILD_PYMIMIR_ADAPTER "Build the Pymimir compatibility adapter." ON)
option(MIFROST_BUILD_PYTYR_ADAPTER "Build the optional PyTyr adapter." OFF)
option(MIFROST_GENERATE_STUBS "Generate nanobind .pyi stubs in dev mode." ON)
option(MIFROST_EXPORT_LOCAL_CONAN_RECIPES "Export local Conan recipes before Conan provider install." ON)
option(MIFROST_FREE_THREADED "Enable nanobind FREE_THREADED mode (requires backend-compatible Python extensions)." OFF)


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

if (MIFROST_BUILD_PYTHON)
    # Nanobind requires finding Python first. Prefer the active virtualenv when
    # present, otherwise fall back to the active conda environment.
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
    # Conan's generated package contributes this header-only convenience target,
    # while nanobind's upstream CMake package intentionally does not. Keep the
    # rest of the build independent of which package provider supplied nanobind.
    if (NOT TARGET nanobind::nanobind)
        add_library(nanobind::nanobind INTERFACE IMPORTED)
        set_property(
            TARGET nanobind::nanobind
            PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${NB_DIR}/include"
        )
    endif ()
endif ()

# Dependencies via Conan
find_package(Boost CONFIG REQUIRED)
find_package(fmt REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(phmap CONFIG REQUIRED)
find_package(unordered_dense CONFIG REQUIRED)
find_package(strong_type REQUIRED)
find_package(range-v3 REQUIRED)
if (MIFROST_BUILD_PYTHON OR MIFROST_BUILD_PYMIMIR_ADAPTER)
    find_package(dlpack REQUIRED)
endif ()
if (MIFROST_BUILD_PYTYR_ADAPTER)
    # PyYggdrasil's exported CMake target links Boost.JSON. Request the
    # component before PyTyr/PyPDDL transitively import that target.
    find_package(Boost CONFIG REQUIRED COMPONENTS json)
endif ()
if (MIFROST_BUILD_PYMIMIR_ADAPTER)
    find_package(Boost CONFIG REQUIRED COMPONENTS iostreams)
    find_package(ZLIB REQUIRED)
    # Do not let PyTyr's PyPDDL prefix satisfy this lookup in a both-backend
    # build. Mimir requires its Conan-built Loki variant; using PyPDDL's Loki
    # injects PyYggdrasil's forward-only GTL declarations into Pymimir sources.
    # Boost is resolved from the same CMakeDeps generator directory, so use it
    # as the authoritative package root for the entire Pymimir dependency set.
    if (NOT EXISTS "${Boost_DIR}/loki-config.cmake")
        message(FATAL_ERROR
            "The selected Conan CMakeDeps directory does not contain loki-config.cmake: ${Boost_DIR}"
        )
    endif ()
    if (DEFINED loki_DIR AND NOT loki_DIR STREQUAL "${Boost_DIR}")
        unset(loki_DIR CACHE)
    endif ()
    find_package(loki CONFIG REQUIRED NO_DEFAULT_PATH PATHS "${Boost_DIR}")
    find_package(nauty REQUIRED)
    find_package(TBB REQUIRED)
    find_package(valla CONFIG REQUIRED)
endif ()

if (MIFROST_BUILD_PYTYR_ADAPTER)
    # Resolve the native runtime directly from PyTyr. Importing Tyr's complete
    # CMake package also imports its Loki/Yggdrasil dependency graph, whose Loki
    # target name collides with Pymimir's independently versioned Loki package
    # in a both-backend build. This narrow imported target is sufficient for an
    # adapter that uses Tyr's public views and links libtyr_core.
    if (DEFINED ENV{PYTYR_NATIVE_PREFIX} AND NOT "$ENV{PYTYR_NATIVE_PREFIX}" STREQUAL "" AND
        NOT DEFINED MIFROST_PYTYR_NATIVE_PREFIX)
        set(MIFROST_PYTYR_NATIVE_PREFIX "$ENV{PYTYR_NATIVE_PREFIX}"
            CACHE PATH "Path to the PyTyr native prefix")
    endif ()
    if (DEFINED ENV{PYYGGDRASIL_NATIVE_PREFIX} AND NOT "$ENV{PYYGGDRASIL_NATIVE_PREFIX}" STREQUAL "" AND
        NOT DEFINED MIFROST_PYYGGDRASIL_NATIVE_PREFIX)
        set(MIFROST_PYYGGDRASIL_NATIVE_PREFIX "$ENV{PYYGGDRASIL_NATIVE_PREFIX}"
            CACHE PATH "Path to the PyYggdrasil native prefix")
    endif ()
    if (DEFINED ENV{PYPDDL_NATIVE_PREFIX} AND NOT "$ENV{PYPDDL_NATIVE_PREFIX}" STREQUAL "" AND
        NOT DEFINED MIFROST_PYPDDL_NATIVE_PREFIX)
        set(MIFROST_PYPDDL_NATIVE_PREFIX "$ENV{PYPDDL_NATIVE_PREFIX}"
            CACHE PATH "Path to the PyPDDL native prefix")
    endif ()
    if (NOT DEFINED MIFROST_PYTYR_NATIVE_PREFIX OR
        "${MIFROST_PYTYR_NATIVE_PREFIX}" STREQUAL "" OR
        NOT DEFINED MIFROST_PYYGGDRASIL_NATIVE_PREFIX OR
        "${MIFROST_PYYGGDRASIL_NATIVE_PREFIX}" STREQUAL "" OR
        NOT DEFINED MIFROST_PYPDDL_NATIVE_PREFIX OR
        "${MIFROST_PYPDDL_NATIVE_PREFIX}" STREQUAL "")
        execute_process(
            COMMAND "${Python_EXECUTABLE}"
                "${CMAKE_CURRENT_LIST_DIR}/../scripts/resolve_native_prefixes.py"
            RESULT_VARIABLE _mifrost_tyr_result
            OUTPUT_VARIABLE _mifrost_tyr_paths
            ERROR_VARIABLE _mifrost_tyr_error
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if (_mifrost_tyr_result EQUAL 0)
            string(REPLACE "\n" ";" _mifrost_tyr_paths "${_mifrost_tyr_paths}")
            list(LENGTH _mifrost_tyr_paths _mifrost_tyr_path_count)
            if (_mifrost_tyr_path_count LESS 3)
                message(FATAL_ERROR
                    "Native planner prefix resolver returned too few paths: ${_mifrost_tyr_paths}"
                )
            endif ()
            list(GET _mifrost_tyr_paths 0 _mifrost_tyr_prefix)
            list(GET _mifrost_tyr_paths 1 _mifrost_yggdrasil_prefix)
            list(GET _mifrost_tyr_paths 2 _mifrost_pypddl_prefix)
            if (NOT DEFINED MIFROST_PYTYR_NATIVE_PREFIX OR
                "${MIFROST_PYTYR_NATIVE_PREFIX}" STREQUAL "")
                set(MIFROST_PYTYR_NATIVE_PREFIX "${_mifrost_tyr_prefix}"
                    CACHE PATH "Path to the PyTyr native prefix")
            endif ()
            if (NOT DEFINED MIFROST_PYYGGDRASIL_NATIVE_PREFIX OR
                "${MIFROST_PYYGGDRASIL_NATIVE_PREFIX}" STREQUAL "")
                set(MIFROST_PYYGGDRASIL_NATIVE_PREFIX "${_mifrost_yggdrasil_prefix}"
                    CACHE PATH "Path to the PyYggdrasil native prefix")
            endif ()
            if (NOT DEFINED MIFROST_PYPDDL_NATIVE_PREFIX OR
                "${MIFROST_PYPDDL_NATIVE_PREFIX}" STREQUAL "")
                set(MIFROST_PYPDDL_NATIVE_PREFIX "${_mifrost_pypddl_prefix}"
                    CACHE PATH "Path to the PyPDDL native prefix")
            endif ()
        else ()
            message(FATAL_ERROR
                "Could not resolve native planner package prefixes. ${_mifrost_tyr_error}"
            )
        endif ()
        unset(_mifrost_tyr_result)
        unset(_mifrost_tyr_paths)
        unset(_mifrost_tyr_error)
        unset(_mifrost_tyr_path_count)
        unset(_mifrost_tyr_prefix)
        unset(_mifrost_yggdrasil_prefix)
        unset(_mifrost_pypddl_prefix)
    endif ()

    set(
        _mifrost_tyr_library
        "${MIFROST_PYTYR_NATIVE_PREFIX}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}tyr_core${CMAKE_SHARED_LIBRARY_SUFFIX}"
    )
    if (NOT EXISTS "${_mifrost_tyr_library}")
        message(FATAL_ERROR
            "Could not find the PyTyr native library at ${_mifrost_tyr_library}. "
            "Install pytyr or set PYTYR_NATIVE_PREFIX/MIFROST_PYTYR_NATIVE_PREFIX."
        )
    endif ()
    foreach (_mifrost_tyr_include_prefix IN ITEMS
        "${MIFROST_PYTYR_NATIVE_PREFIX}"
        "${MIFROST_PYYGGDRASIL_NATIVE_PREFIX}"
        "${MIFROST_PYPDDL_NATIVE_PREFIX}"
    )
        if (NOT EXISTS "${_mifrost_tyr_include_prefix}/include")
            message(FATAL_ERROR
                "Could not find a required PyTyr adapter include prefix at "
                "${_mifrost_tyr_include_prefix}/include. Install pytyr, "
                "pyyggdrasil, and pypddl or set their MIFROST_*_NATIVE_PREFIX values."
            )
        endif ()
    endforeach ()
    if (NOT TARGET tyr::core)
        add_library(tyr::core SHARED IMPORTED GLOBAL)
        set_target_properties(tyr::core PROPERTIES
            IMPORTED_LOCATION "${_mifrost_tyr_library}"
            INTERFACE_COMPILE_DEFINITIONS "FMT_HEADER_ONLY;BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS;BOOST_MPL_LIMIT_LIST_SIZE=50;TYR_STATE_STORAGE_TREE"
            INTERFACE_INCLUDE_DIRECTORIES "${MIFROST_PYTYR_NATIVE_PREFIX}/include;${MIFROST_PYYGGDRASIL_NATIVE_PREFIX}/include;${MIFROST_PYPDDL_NATIVE_PREFIX}/include"
        )
    endif ()
    message(STATUS "Using PyTyr native prefix=${MIFROST_PYTYR_NATIVE_PREFIX}")
    unset(_mifrost_tyr_library)
endif ()
if (MIFROST_BUILD_BENCHMARKS AND MIFROST_BUILD_PYMIMIR_ADAPTER)
    find_package(benchmark REQUIRED)
    find_package(argparse REQUIRED)
endif ()
if (MIFROST_BUILD_BENCHMARKS AND NOT MIFROST_BUILD_PYMIMIR_ADAPTER)
    message(FATAL_ERROR "MIFROST_BUILD_BENCHMARKS currently requires MIFROST_BUILD_PYMIMIR_ADAPTER=ON")
endif ()
if ((MIFROST_BUILD_PYTHON OR MIFROST_BUILD_PYMIMIR_ADAPTER) AND NOT TARGET dlpack::dlpack)
    message(FATAL_ERROR "Expected Conan package dlpack to define target dlpack::dlpack")
endif ()

# Conan include fallbacks (some recipes omit include dirs).
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/ConanFallbacks.cmake")
set(_python_embed_target "")
if (MIFROST_BUILD_PYTHON AND TARGET Python::Python)
    set(_python_embed_target Python::Python)
elseif (MIFROST_BUILD_PYTHON AND TARGET Python::Embed)
    set(_python_embed_target Python::Embed)
endif ()

if (MIFROST_BUILD_PYMIMIR_ADAPTER)
    # Valla and Loki aliases are expected by Mimir's CMake config.
    if (TARGET valla::valla AND NOT TARGET valla::core)
        add_library(valla::core ALIAS valla::valla)
    endif ()
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

    # Resolve Mimir from an explicit override or the Pymimir runtime.
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

    # Mimir linkage found via CMAKE_PREFIX_PATH or pymimir.get_cmake_dir().
    find_package(mimir REQUIRED)

    if (NOT DEFINED mimir_VERSION)
        message(FATAL_ERROR "mimir was found, but did not report mimir_VERSION")
    endif()

    if (mimir_VERSION VERSION_LESS "0.13.60")
        message(FATAL_ERROR "mimir >= 0.13.60 required, found ${mimir_VERSION}")
    endif()
endif ()

# --- RPATH mode (dev vs wheel) ---
set(MIFROST_RPATH_MODE "dev" CACHE STRING "RPATH mode: dev or wheel")
set_property(CACHE MIFROST_RPATH_MODE PROPERTY STRINGS dev wheel)

# --- Dev staging root ---
# A dev-mode build stages an importable `mifrost` package: the native modules,
# the shared libraries they load at runtime, and the generated stubs. Two build
# directories that stage into the same root overwrite each other's artifacts,
# and the symptom is not a build error but a corrupt runtime: a sanitizer or
# Debug build replaces the shared library that another build directory's already
# linked tests and benchmarks load, so those binaries then report spurious
# failures. Each configuration therefore stages inside its own binary directory
# unless it is explicitly one of the two builds that own the source tree -- the
# scikit-build-core editable install, and the stub build that has to produce
# `src/mifrost/*.pyi` for packaging.
if (SKBUILD)
    set(_mifrost_default_dev_root "${CMAKE_SOURCE_DIR}/src")
else ()
    set(_mifrost_default_dev_root "${CMAKE_BINARY_DIR}/python")
endif ()
set(MIFROST_DEV_PACKAGE_ROOT "${_mifrost_default_dev_root}" CACHE PATH
    "Directory holding the staged 'mifrost' Python package in dev RPATH mode.")
unset(_mifrost_default_dev_root)
set(MIFROST_DEV_PACKAGE_DIR "${MIFROST_DEV_PACKAGE_ROOT}/mifrost")
set(MIFROST_DEV_PACKAGE_LIB_DIR "${MIFROST_DEV_PACKAGE_DIR}/lib")
if (MIFROST_DEV_PACKAGE_ROOT STREQUAL "${CMAKE_SOURCE_DIR}/src")
    set(MIFROST_DEV_PACKAGE_IN_SOURCE TRUE)
else ()
    set(MIFROST_DEV_PACKAGE_IN_SOURCE FALSE)
endif ()

# Claim the staging root. Two build directories may legitimately share one root
# when they produce interchangeable artifacts (the editable install and the stub
# build are both dev-mode Release builds); they may not when their compile
# settings differ, because then the shared libraries are not interchangeable.
if (MIFROST_RPATH_MODE STREQUAL "dev")
    set(_mifrost_dev_owner_stamp "${MIFROST_DEV_PACKAGE_DIR}/.mifrost-dev-owner")
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _mifrost_dev_config)
    string(JOIN "|" _mifrost_dev_signature
        "${CMAKE_BUILD_TYPE}"
        "${CMAKE_CXX_FLAGS}"
        "${CMAKE_CXX_FLAGS_${_mifrost_dev_config}}"
        "${CMAKE_EXE_LINKER_FLAGS}"
    )
    unset(_mifrost_dev_config)
    if (EXISTS "${_mifrost_dev_owner_stamp}")
        file(READ "${_mifrost_dev_owner_stamp}" _mifrost_dev_owner_contents)
        string(REGEX MATCH "^([^\n]*)\n([^\n]*)" _mifrost_dev_owner_match "${_mifrost_dev_owner_contents}")
        set(_mifrost_dev_owner_dir "${CMAKE_MATCH_1}")
        set(_mifrost_dev_owner_signature "${CMAKE_MATCH_2}")
        if (NOT _mifrost_dev_owner_dir STREQUAL "${CMAKE_BINARY_DIR}"
                AND NOT _mifrost_dev_owner_signature STREQUAL "${_mifrost_dev_signature}")
            message(FATAL_ERROR
                "Build directory '${CMAKE_BINARY_DIR}' would stage the mifrost dev package "
                "into '${MIFROST_DEV_PACKAGE_DIR}', which is already staged by "
                "'${_mifrost_dev_owner_dir}' with different compile settings.\n"
                "  this build:  ${_mifrost_dev_signature}\n"
                "  other build: ${_mifrost_dev_owner_signature}\n"
                "Overwriting it would leave the other build's tests and benchmarks loading "
                "incompatible shared libraries. Point this build elsewhere with "
                "-DMIFROST_DEV_PACKAGE_ROOT=<dir>, or remove the other build directory."
            )
        endif ()
        unset(_mifrost_dev_owner_contents)
        unset(_mifrost_dev_owner_match)
        unset(_mifrost_dev_owner_dir)
        unset(_mifrost_dev_owner_signature)
    endif ()
    file(WRITE "${_mifrost_dev_owner_stamp}" "${CMAKE_BINARY_DIR}\n${_mifrost_dev_signature}\n")
    unset(_mifrost_dev_owner_stamp)
    unset(_mifrost_dev_signature)
endif ()

# A staging root outside the source tree holds only the compiled artifacts, so
# the pure-Python package has to be mirrored into it before anything can import
# `mifrost` from there (stub generation does). Only the tracked `.py` sources are
# copied: mirroring the whole directory would drag another build's compiled
# modules along and recreate the collision this staging root exists to avoid.
add_custom_target(mifrost_stage_python)
if (MIFROST_RPATH_MODE STREQUAL "dev" AND NOT MIFROST_DEV_PACKAGE_IN_SOURCE)
    file(GLOB_RECURSE _mifrost_pure_python
        RELATIVE "${CMAKE_SOURCE_DIR}/src"
        CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/src/mifrost/*.py"
    )
    set(_mifrost_staged_python "")
    foreach (_mifrost_rel IN LISTS _mifrost_pure_python)
        add_custom_command(
            OUTPUT "${MIFROST_DEV_PACKAGE_ROOT}/${_mifrost_rel}"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${CMAKE_SOURCE_DIR}/src/${_mifrost_rel}"
                "${MIFROST_DEV_PACKAGE_ROOT}/${_mifrost_rel}"
            DEPENDS "${CMAKE_SOURCE_DIR}/src/${_mifrost_rel}"
            COMMENT "Staging ${_mifrost_rel}"
            VERBATIM
        )
        list(APPEND _mifrost_staged_python "${MIFROST_DEV_PACKAGE_ROOT}/${_mifrost_rel}")
    endforeach ()
    add_custom_target(mifrost_stage_python_files DEPENDS ${_mifrost_staged_python})
    add_dependencies(mifrost_stage_python mifrost_stage_python_files)
    unset(_mifrost_pure_python)
    unset(_mifrost_staged_python)
    unset(_mifrost_rel)
endif ()

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
# Backend-neutral encoding and batching code lives in a separately loadable
# library. Backend adapters may depend on it, but it must not gain a dependency
# on a planning library.
add_library(
    mifrost_neutral_core
    SHARED
        "_core/mifrost/core/batch_builder.cpp"
        "_core/mifrost/core/schema.cpp"
        "_core/mifrost/core/encoders/common/relation_catalog.cpp"
        "_core/mifrost/core/encoders/common/relation_key.cpp"
        "_core/mifrost/core/encoders/hetero/hetero_relation_schema.cpp"
        "_core/mifrost/core/encoders/flat/flat_encoder_common.cpp"
        "_core/mifrost/core/encoders/flat/flat_lgan.cpp"
        "_core/mifrost/core/encoders/flat/flat_composition.cpp"
        "_core/mifrost/core/encoders/flat/flat_relation_schema.cpp"
        "_core/mifrost/core/encoders/flat/semantic_flat_relation_encoder.cpp"
        "_core/mifrost/core/encoders/flat/semantic_flat_relation_view_bridge.cpp"
        "_core/mifrost/core/encoders/flat/semantic_flat_horizon_encoder.cpp"
        "_core/mifrost/core/encoders/homo/semantic_color_encoder.cpp"
        "_core/mifrost/core/encoders/hetero/semantic_hgraph_encoder.cpp"
        "_core/mifrost/core/encoders/hetero/semantic_horizon_hgraph_encoder.cpp"
        "_core/mifrost/core/encoders/hetero/semantic_successor_hgraph_encoder.cpp"
        "_core/mifrost/core/encoders/common/target_metadata.cpp"
        "_core/mifrost/core/semantic/semantic_transition_dag.cpp"
)
set_target_properties(
    mifrost_neutral_core
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME neutral_core
)
target_compile_definitions(
    mifrost_neutral_core
    PUBLIC
    MIFROST_BUILD_SHARED
    PRIVATE
    MIFROST_EXPORTS
)
if (NOT TARGET mifrost::neutral_core)
    add_library(mifrost::neutral_core ALIAS mifrost_neutral_core)
endif ()
target_include_directories(mifrost_neutral_core PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/_core>"
    "$<INSTALL_INTERFACE:${MIFROST_SDK_INCLUDE_DIR}>"
)
target_link_libraries(
    mifrost_neutral_core
    PUBLIC
        fmt::fmt
        Boost::headers
        absl::btree
        phmap
        unordered_dense::unordered_dense
        rollbear::strong_type
        range-v3::range-v3
    PRIVATE
        absl::node_hash_set
)
conan_add_package_include_fallback(mifrost_neutral_core abseil SCOPE PUBLIC)

set(_mifrost_neutral_rpaths "")
if (APPLE)
    list(APPEND _mifrost_neutral_rpaths "@loader_path")
elseif (UNIX)
    list(APPEND _mifrost_neutral_rpaths "$ORIGIN")
endif ()
set_target_properties(
    mifrost_neutral_core
    PROPERTIES
    BUILD_RPATH "${_mifrost_neutral_rpaths}"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${_mifrost_neutral_rpaths}"
    INSTALL_RPATH_USE_LINK_PATH FALSE
)
if (MIFROST_RPATH_MODE STREQUAL "dev")
    set_target_properties(mifrost_neutral_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${MIFROST_DEV_PACKAGE_LIB_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${MIFROST_DEV_PACKAGE_LIB_DIR}"
    )
endif ()

if (MIFROST_FREE_THREADED)
    set(_mifrost_freethreading "FREE_THREADED")
endif ()

if (MIFROST_BUILD_PYTHON)
    nanobind_add_module(mifrost_neutral_core_module
        ${_mifrost_freethreading}
        NB_DOMAIN pymimir_abi_domain
        "_core/mifrost/neutral_module.cpp"
        "_core/mifrost/common.cpp"
        "_core/mifrost/pyg_views.cpp"
        "_core/mifrost/core/batch_builder_python.cpp"
        "_core/mifrost/batch_encoding_attributes.cpp"
        "_core/mifrost/batch_encoding_collection.cpp"
        "_core/mifrost/batch_encoding_conversion.cpp"
        "_core/mifrost/batch_encoding_graph_field_access.cpp"
        "_core/mifrost/batch_encoding_graph_field_mutation.cpp"
        "_core/mifrost/batch_encoding_graph_field_serialization.cpp"
        "_core/mifrost/batch_encoding_python_collation.cpp"
        "_core/mifrost/batch_encoding_repr.cpp"
        "_core/mifrost/batch_encoding_schema.cpp"
        "_core/mifrost/batch_encoding_state.cpp"
        "_core/mifrost/batch_encoding_tensor_cache.cpp"
        "_core/mifrost/init_map_view.cpp"
        "_core/mifrost/init_schema.cpp"
        "_core/mifrost/schema_bindings.cpp"
        "_core/mifrost/init_batch_encoding.cpp"
        "_core/mifrost/init_semantic_flat_encoder.cpp"
        "_core/mifrost/init_semantic_color_encoder.cpp"
        "_core/mifrost/init_semantic_hgraph_encoder.cpp"
        "_core/mifrost/init_semantic_transition_dag.cpp"
    )
    set_target_properties(
        mifrost_neutral_core_module
        PROPERTIES
        OUTPUT_NAME "_neutral_core"
    )
    target_link_libraries(
        mifrost_neutral_core_module
        PRIVATE
        dlpack::dlpack
        mifrost_neutral_core
        Python::Module
        nanobind::nanobind
    )
    target_compile_definitions(
        mifrost_neutral_core_module
        PRIVATE
        MIFROST_BUILD_SHARED
        MIFROST_ENABLE_PYTHON_API
    )
    conan_add_package_include_fallback(mifrost_neutral_core_module abseil SCOPE PRIVATE)

    set(_mifrost_neutral_module_rpaths "")
    if (APPLE)
        list(APPEND _mifrost_neutral_module_rpaths "@loader_path" "@loader_path/lib")
    elseif (UNIX)
        list(APPEND _mifrost_neutral_module_rpaths "$ORIGIN" "$ORIGIN/lib")
    endif ()
    set_target_properties(
        mifrost_neutral_core_module
        PROPERTIES
        BUILD_RPATH "${_mifrost_neutral_module_rpaths}"
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "${_mifrost_neutral_module_rpaths}"
        INSTALL_RPATH_USE_LINK_PATH FALSE
    )
    if (MIFROST_RPATH_MODE STREQUAL "dev")
        set_target_properties(
            mifrost_neutral_core_module
            PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${MIFROST_DEV_PACKAGE_DIR}"
        )
    endif ()
endif ()

if (MIFROST_BUILD_PYTYR_ADAPTER)
    add_library(
        mifrost_pytyr_adapter
        SHARED
            "_core/mifrost/backends/pytyr/semantic_flat_encoder.cpp"
    )
    set_target_properties(
        mifrost_pytyr_adapter
        PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        EXPORT_NAME pytyr_adapter
    )
    target_compile_definitions(
        mifrost_pytyr_adapter
        PUBLIC
            MIFROST_BUILD_SHARED
            MIFROST_USE_EXTERNAL_GTL
        PRIVATE MIFROST_EXPORTS
    )
    add_library(mifrost::pytyr_adapter ALIAS mifrost_pytyr_adapter)
    target_include_directories(mifrost_pytyr_adapter PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/_core>"
        "$<INSTALL_INTERFACE:${MIFROST_SDK_INCLUDE_DIR}>"
    )
    target_link_libraries(
        mifrost_pytyr_adapter
        PUBLIC
            mifrost_neutral_core
            tyr::core
    )

    set(_mifrost_pytyr_rpaths "")
    if (APPLE)
        list(APPEND _mifrost_pytyr_rpaths "@loader_path" "@loader_path/../../pytyr/native/lib")
    elseif (UNIX)
        list(APPEND _mifrost_pytyr_rpaths "$ORIGIN" "$ORIGIN/../../pytyr/native/lib")
    endif ()
    if (MIFROST_RPATH_MODE STREQUAL "dev")
        _mifrost_append_rpath_from_target(_mifrost_pytyr_rpaths tyr::core)
    endif ()
    list(REMOVE_DUPLICATES _mifrost_pytyr_rpaths)
    set_target_properties(
        mifrost_pytyr_adapter
        PROPERTIES
        BUILD_RPATH "${_mifrost_pytyr_rpaths}"
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "${_mifrost_pytyr_rpaths}"
        INSTALL_RPATH_USE_LINK_PATH FALSE
    )

    if (MIFROST_BUILD_PYTHON)
        set(
            _mifrost_pytyr_nanobind_library
            "${MIFROST_PYYGGDRASIL_NATIVE_PREFIX}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}nanobind${CMAKE_SHARED_LIBRARY_SUFFIX}"
        )
        set(
            _mifrost_pytyr_nanobind_header
            "${MIFROST_PYYGGDRASIL_NATIVE_PREFIX}/include/nanobind/nanobind.h"
        )
        if (NOT EXISTS "${_mifrost_pytyr_nanobind_library}" OR
            NOT EXISTS "${_mifrost_pytyr_nanobind_header}")
            message(FATAL_ERROR
                "PyTyr adapter requires PyYggdrasil's bundled nanobind runtime and headers; "
                "expected ${_mifrost_pytyr_nanobind_library} and ${_mifrost_pytyr_nanobind_header}."
            )
        endif ()
        add_library(mifrost_pytyr_nanobind SHARED IMPORTED)
        set_target_properties(mifrost_pytyr_nanobind PROPERTIES
            IMPORTED_LOCATION "${_mifrost_pytyr_nanobind_library}"
            INTERFACE_INCLUDE_DIRECTORIES "${MIFROST_PYYGGDRASIL_NATIVE_PREFIX}/include"
            INTERFACE_COMPILE_DEFINITIONS NB_SHARED
        )
        Python_add_library(
            mifrost_pytyr_adapter_module MODULE WITH_SOABI
            "_core/mifrost/pytyr_module.cpp"
        )
        set_target_properties(
            mifrost_pytyr_adapter_module
            PROPERTIES OUTPUT_NAME "_pytyr_adapter"
        )
        target_link_libraries(
            mifrost_pytyr_adapter_module
            PRIVATE
                mifrost_pytyr_adapter
                Python::Module
                mifrost_pytyr_nanobind
        )
        target_compile_definitions(
            mifrost_pytyr_adapter_module
            PRIVATE MIFROST_ENABLE_PYTHON_API
        )

        set(_mifrost_pytyr_module_rpaths "")
        if (APPLE)
            list(APPEND _mifrost_pytyr_module_rpaths
                "@loader_path/../pyyggdrasil/lib"
                "@loader_path/lib"
                "@loader_path/../pytyr/native/lib"
            )
        elseif (UNIX)
            list(APPEND _mifrost_pytyr_module_rpaths
                "$ORIGIN/../pyyggdrasil/lib"
                "$ORIGIN/lib"
                "$ORIGIN/../pytyr/native/lib"
            )
        endif ()
        if (MIFROST_RPATH_MODE STREQUAL "dev")
            list(PREPEND
                _mifrost_pytyr_module_rpaths
                "${MIFROST_PYYGGDRASIL_NATIVE_PREFIX}/lib"
            )
            _mifrost_append_rpath_from_target(_mifrost_pytyr_module_rpaths tyr::core)
        endif ()
        list(REMOVE_DUPLICATES _mifrost_pytyr_module_rpaths)
        set_target_properties(
            mifrost_pytyr_adapter_module
            PROPERTIES
            BUILD_RPATH "${_mifrost_pytyr_module_rpaths}"
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "${_mifrost_pytyr_module_rpaths}"
            INSTALL_RPATH_USE_LINK_PATH FALSE
        )

        if (MIFROST_RPATH_MODE STREQUAL "dev")
            set_target_properties(mifrost_pytyr_adapter_module PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY "${MIFROST_DEV_PACKAGE_DIR}"
            )
        endif ()
        unset(_mifrost_pytyr_nanobind_library)
        unset(_mifrost_pytyr_nanobind_header)
    endif ()
    if (MIFROST_RPATH_MODE STREQUAL "dev")
        set_target_properties(mifrost_pytyr_adapter PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${MIFROST_DEV_PACKAGE_LIB_DIR}"
            RUNTIME_OUTPUT_DIRECTORY "${MIFROST_DEV_PACKAGE_LIB_DIR}"
        )
    endif ()
endif ()

if (MIFROST_BUILD_PYMIMIR_ADAPTER)
# Compatibility target for the current Pymimir-backed extension. This is the
# first adapter target and will narrow further as bindings migrate.
add_library(
    mifrost_pymimir_adapter
    SHARED
        "_core/mifrost/input_handling/batch_input_goal_inputs.cpp"
        "_core/mifrost/backends/pymimir/encoders/homo/color_encoder.cpp"
        "_core/mifrost/backends/pymimir/encoders/flat/flat_entity_context.cpp"
        "_core/mifrost/backends/pymimir/encoders/flat/flat_goal_helpers.cpp"
        "_core/mifrost/backends/pymimir/encoders/flat/flat_horizon_context.cpp"
        "_core/mifrost/backends/pymimir/encoders/flat/flat_horizon_encoder.cpp"
        "_core/mifrost/backends/pymimir/encoders/flat/flat_relation_context.cpp"
        "_core/mifrost/backends/pymimir/encoders/flat/flat_relation_encoder.cpp"
        "_core/mifrost/backends/pymimir/encoders/hetero/hgraph_stream_encoder.cpp"
        "_core/mifrost/backends/pymimir/transition_target_metadata.cpp"
        "_core/mifrost/backends/pymimir/encoders/common/transition_dag.cpp"
        "_core/mifrost/backends/pymimir/encoders/hetero/horizon_hgraph_encoder.cpp"
        "_core/mifrost/backends/pymimir/encoders/hetero/successor_hgraph_encoder.cpp"
)
set_target_properties(
    mifrost_pymimir_adapter
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME pymimir_adapter
)
target_compile_definitions(
    mifrost_pymimir_adapter
    PUBLIC
    MIFROST_BUILD_SHARED
    PRIVATE
    MIFROST_EXPORTS
)
if (NOT TARGET mifrost::pymimir_adapter)
    add_library(mifrost::pymimir_adapter ALIAS mifrost_pymimir_adapter)
endif ()
if (NOT TARGET mifrost::core)
    add_library(mifrost::core ALIAS mifrost_pymimir_adapter)
endif ()


target_include_directories(mifrost_pymimir_adapter PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/_core>"
    "$<INSTALL_INTERFACE:${MIFROST_SDK_INCLUDE_DIR}>"
)

target_link_libraries(
        mifrost_pymimir_adapter
        PUBLIC
    # Public SDK dependencies surfaced by installed headers.
        mifrost_neutral_core
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

conan_add_package_include_fallback(mifrost_pymimir_adapter abseil SCOPE PUBLIC)
conan_add_package_include_fallback(mifrost_pymimir_adapter onetbb SCOPE PRIVATE)

if (MIFROST_BUILD_PYTHON)
    add_library(
        mifrost_python_support
        OBJECT
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
        mifrost_pymimir_adapter
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
endif ()

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
        mifrost_pymimir_adapter
        PROPERTIES
        BUILD_RPATH "${_mifrost_core_rpaths}"
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "${_mifrost_core_rpaths}"
        INSTALL_RPATH_USE_LINK_PATH FALSE
)

if (MIFROST_BUILD_PYTHON)
    nanobind_add_module(mifrost_pymimir_adapter_module
            ${_mifrost_freethreading}
            NB_DOMAIN pymimir_abi_domain
            "_core/mifrost/mifrost.cpp"
            "_core/mifrost/pyg_views.cpp"
            "_core/mifrost/batch_encoding_attributes.cpp"
            "_core/mifrost/batch_encoding_collection.cpp"
            "_core/mifrost/batch_encoding_conversion.cpp"
            "_core/mifrost/batch_encoding_graph_field_access.cpp"
            "_core/mifrost/batch_encoding_graph_field_mutation.cpp"
            "_core/mifrost/batch_encoding_graph_field_serialization.cpp"
            "_core/mifrost/batch_encoding_python_collation.cpp"
            "_core/mifrost/batch_encoding_repr.cpp"
            "_core/mifrost/batch_encoding_schema.cpp"
            "_core/mifrost/batch_encoding_state.cpp"
            "_core/mifrost/batch_encoding_tensor_cache.cpp"
            "_core/mifrost/init_common.cpp"
            "_core/mifrost/init_relation_formatter.cpp"
            "_core/mifrost/init_color_encoder.cpp"
            "_core/mifrost/schema_bindings.cpp"
            "_core/mifrost/init_hgraph_encoder.cpp"
            "_core/mifrost/init_flat_encoder.cpp"
            "_core/mifrost/init_horizon_encoder.cpp"
            "_core/mifrost/init_successor_encoders.cpp"
            "_core/mifrost/init_transition_dag.cpp"
            $<TARGET_OBJECTS:mifrost_python_support>
    )
    set_target_properties(
            mifrost_pymimir_adapter_module
            PROPERTIES
            OUTPUT_NAME "_pymimir_adapter"
    )
    target_link_libraries(
            mifrost_pymimir_adapter_module
            PRIVATE
            dlpack::dlpack
            mifrost_pymimir_adapter
            Python::Module
            nanobind::nanobind
    )
    target_compile_definitions(mifrost_pymimir_adapter_module PRIVATE MIFROST_ENABLE_PYTHON_API)

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
            mifrost_pymimir_adapter_module
            PROPERTIES
            BUILD_RPATH "${_mifrost_rpaths}"
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "${_mifrost_rpaths}"
            # Avoid embedding transient link directories (e.g. pip-build-env overlays).
            INSTALL_RPATH_USE_LINK_PATH FALSE
    )
    if (MIFROST_RPATH_MODE STREQUAL "dev")
        set_target_properties(mifrost_pymimir_adapter_module PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY "${MIFROST_DEV_PACKAGE_DIR}"
        )
    endif ()
endif ()
if (MIFROST_RPATH_MODE STREQUAL "dev")
    set_target_properties(mifrost_pymimir_adapter PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${MIFROST_DEV_PACKAGE_LIB_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${MIFROST_DEV_PACKAGE_LIB_DIR}"
    )
endif ()
add_custom_command(
    TARGET mifrost_pymimir_adapter
    POST_BUILD
    COMMAND
        "${CMAKE_COMMAND}" -E copy_if_different
        "$<TARGET_FILE:mifrost_pymimir_adapter>"
        "$<TARGET_FILE_DIR:mifrost_pymimir_adapter>/${CMAKE_SHARED_LIBRARY_PREFIX}mifrost_core${CMAKE_SHARED_LIBRARY_SUFFIX}"
    COMMENT "Creating the legacy mifrost core library compatibility artifact"
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

install(TARGETS mifrost_neutral_core
    EXPORT mifrostTargets
    LIBRARY DESTINATION "${MIFROST_SDK_LIBRARY_DIR}"
    ARCHIVE DESTINATION "${MIFROST_SDK_LIBRARY_DIR}"
    RUNTIME DESTINATION "${MIFROST_SDK_LIBRARY_DIR}"
    INCLUDES DESTINATION "${MIFROST_SDK_INCLUDE_DIR}"
)
if (MIFROST_BUILD_PYTHON)
    install(
        TARGETS mifrost_neutral_core_module
        LIBRARY DESTINATION "${MIFROST_PYTHON_PACKAGE_DIR}"
    )
endif ()
if (MIFROST_BUILD_PYMIMIR_ADAPTER)
    install(TARGETS mifrost_pymimir_adapter
        EXPORT mifrostTargets
        LIBRARY DESTINATION "${MIFROST_SDK_LIBRARY_DIR}"
        ARCHIVE DESTINATION "${MIFROST_SDK_LIBRARY_DIR}"
        RUNTIME DESTINATION "${MIFROST_SDK_LIBRARY_DIR}"
        INCLUDES DESTINATION "${MIFROST_SDK_INCLUDE_DIR}"
    )
    if (MIFROST_BUILD_PYTHON)
        install(TARGETS mifrost_pymimir_adapter_module LIBRARY DESTINATION "${MIFROST_PYTHON_PACKAGE_DIR}")
    endif ()
    install(
        FILES "$<TARGET_FILE:mifrost_pymimir_adapter>"
        DESTINATION "${MIFROST_SDK_LIBRARY_DIR}"
        RENAME "${CMAKE_SHARED_LIBRARY_PREFIX}mifrost_core${CMAKE_SHARED_LIBRARY_SUFFIX}"
    )
endif ()
if (MIFROST_BUILD_PYTYR_ADAPTER)
    install(TARGETS mifrost_pytyr_adapter
        EXPORT mifrostTargets
        LIBRARY DESTINATION "${MIFROST_SDK_LIBRARY_DIR}"
        ARCHIVE DESTINATION "${MIFROST_SDK_LIBRARY_DIR}"
        RUNTIME DESTINATION "${MIFROST_SDK_LIBRARY_DIR}"
        INCLUDES DESTINATION "${MIFROST_SDK_INCLUDE_DIR}"
    )
    if (MIFROST_BUILD_PYTHON)
        install(TARGETS mifrost_pytyr_adapter_module LIBRARY DESTINATION "${MIFROST_PYTHON_PACKAGE_DIR}")
    endif ()
    install(
        FILES
            "${CMAKE_CURRENT_LIST_DIR}/_core/mifrost/backends/pytyr/semantic_flat_encoder.hpp"
            "${CMAKE_CURRENT_LIST_DIR}/_core/mifrost/backends/pytyr/views.hpp"
        DESTINATION "${MIFROST_SDK_INCLUDE_DIR}/mifrost/backends/pytyr"
    )
endif ()

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
if (MIFROST_BUILD_PYMIMIR_ADAPTER)
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
    install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/_core/mifrost/backends/pymimir/"
        DESTINATION "${MIFROST_SDK_INCLUDE_DIR}/mifrost/backends/pymimir"
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
    )
    install(
        FILES "${CMAKE_CURRENT_LIST_DIR}/_core/gtl/phmap.hpp"
        DESTINATION "${MIFROST_SDK_INCLUDE_DIR}/gtl"
    )
else ()
    set(_mifrost_neutral_public_headers
        "core/api.hpp"
        "core/batch_builder.hpp"
        "core/common_types.hpp"
        "core/graph_fields.hpp"
        "core/schema.hpp"
        "core/schema_key_separators.hpp"
        "core/utils/macro.hpp"
        "core/utils/type_traits.hpp"
        "core/views/concepts.hpp"
        "core/views/ids.hpp"
        "core/views/ranges.hpp"
        "core/views/canonical.hpp"
        "core/encoders/common/default_relations.hpp"
        "core/encoders/common/goal_derivation.hpp"
        "core/encoders/common/goal_level.hpp"
        "core/encoders/common/relation_catalog.hpp"
        "core/encoders/common/relation_dict_types.hpp"
        "core/encoders/common/relation_key.hpp"
        "core/encoders/common/root_policy.hpp"
        "core/encoders/common/target_metadata.hpp"
        "core/encoders/common/target_source.hpp"
        "core/encoders/flat/flat_encoder_common.hpp"
        "core/encoders/flat/flat_lgan.hpp"
        "core/encoders/flat/flat_composition.hpp"
        "core/encoders/flat/flat_relation_config.hpp"
        "core/encoders/flat/flat_relation_schema.hpp"
        "core/encoders/flat/flat_tuple_layout.hpp"
        "core/encoders/flat/view_flat_relation_encoder.hpp"
        "core/encoders/flat/semantic_flat_horizon_encoder.hpp"
        "core/encoders/flat/semantic_flat_relation_encoder.hpp"
        "core/encoders/flat/semantic_flat_relation_view_bridge.hpp"
        "core/encoders/homo/semantic_color_encoder.hpp"
        "core/encoders/hetero/hetero_relation_schema.hpp"
        "core/encoders/hetero/semantic_hgraph_encoder.hpp"
        "core/encoders/hetero/semantic_horizon_hgraph_encoder.hpp"
        "core/encoders/hetero/semantic_successor_hgraph_encoder.hpp"
        "core/semantic/records.hpp"
        "core/semantic/semantic_transition_dag.hpp"
        "core/semantic/views.hpp"
        "core/views/semantic_preparation.hpp"
    )
    foreach (_mifrost_header IN LISTS _mifrost_neutral_public_headers)
        get_filename_component(_mifrost_header_dir "${_mifrost_header}" DIRECTORY)
        install(
            FILES "${CMAKE_CURRENT_LIST_DIR}/_core/mifrost/${_mifrost_header}"
            DESTINATION "${MIFROST_SDK_INCLUDE_DIR}/mifrost/${_mifrost_header_dir}"
        )
    endforeach ()
endif ()
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

set(_mifrost_dev_stub_targets "")

if (MIFROST_BUILD_PYTHON AND MIFROST_GENERATE_STUBS)
    set(_mifrost_neutral_stub_output "mifrost/_neutral_core.pyi")
    if (MIFROST_RPATH_MODE STREQUAL "dev")
        string(PREPEND _mifrost_neutral_stub_output "${MIFROST_DEV_PACKAGE_ROOT}/")
        nanobind_add_stub(
            mifrost_neutral_core_module_stubs
            INCLUDE_PRIVATE
            MODULE mifrost._neutral_core
            OUTPUT "${_mifrost_neutral_stub_output}"
            PYTHON_PATH "${MIFROST_DEV_PACKAGE_ROOT}"
            DEPENDS mifrost_neutral_core_module
        )
        list(APPEND _mifrost_dev_stub_targets mifrost_neutral_core_module_stubs)
    else ()
        nanobind_add_stub(
            mifrost_neutral_core_module_stubs
            INCLUDE_PRIVATE
            MODULE mifrost._neutral_core
            OUTPUT "${_mifrost_neutral_stub_output}"
            DEPENDS mifrost_neutral_core_module
            INSTALL_TIME
        )
    endif ()
endif ()

if (MIFROST_BUILD_PYTHON AND MIFROST_BUILD_PYMIMIR_ADAPTER AND MIFROST_GENERATE_STUBS)
    set(_mifrost_stub_output "mifrost/_pymimir_adapter.pyi")
    if (MIFROST_RPATH_MODE STREQUAL "dev")
        # Dev mode: keep stubs in source tree so IDEs can pick them up directly.
        STRING(PREPEND _mifrost_stub_output "${MIFROST_DEV_PACKAGE_ROOT}/")
        nanobind_add_stub(
                mifrost_pymimir_adapter_module_stubs
                INCLUDE_PRIVATE
                MODULE mifrost._pymimir_adapter
                OUTPUT "${_mifrost_stub_output}"
                PYTHON_PATH "${MIFROST_DEV_PACKAGE_ROOT}"
                DEPENDS mifrost_pymimir_adapter_module
        )
        list(APPEND _mifrost_dev_stub_targets mifrost_pymimir_adapter_module_stubs)
    else ()
        # Wheel/install mode: generate stubs at install time in the install prefix.
        nanobind_add_stub(
                mifrost_pymimir_adapter_module_stubs
                INCLUDE_PRIVATE
                MODULE mifrost._pymimir_adapter
                OUTPUT "${_mifrost_stub_output}"
                DEPENDS mifrost_pymimir_adapter_module
                INSTALL_TIME
        )
    endif ()
endif ()

if (MIFROST_BUILD_PYTHON AND MIFROST_BUILD_PYTYR_ADAPTER AND MIFROST_GENERATE_STUBS)
    set(_mifrost_pytyr_stub_output "mifrost/_pytyr_adapter.pyi")
    if (MIFROST_RPATH_MODE STREQUAL "dev")
        string(PREPEND _mifrost_pytyr_stub_output "${MIFROST_DEV_PACKAGE_ROOT}/")
        nanobind_add_stub(
            mifrost_pytyr_adapter_module_stubs
            INCLUDE_PRIVATE
            MODULE mifrost._pytyr_adapter
            OUTPUT "${_mifrost_pytyr_stub_output}"
            PYTHON_PATH "${MIFROST_DEV_PACKAGE_ROOT}"
            DEPENDS mifrost_pytyr_adapter_module
        )
        list(APPEND _mifrost_dev_stub_targets mifrost_pytyr_adapter_module_stubs)
    else ()
        nanobind_add_stub(
            mifrost_pytyr_adapter_module_stubs
            INCLUDE_PRIVATE
            MODULE mifrost._pytyr_adapter
            OUTPUT "${_mifrost_pytyr_stub_output}"
            DEPENDS mifrost_pytyr_adapter_module
            INSTALL_TIME
        )
    endif ()
endif ()

# Generate the historical compatibility surface by introspecting the Python
# facade after its split native modules are available. This keeps nanobind as
# the source of truth instead of maintaining a merged compatibility stub.
if (MIFROST_BUILD_PYTHON AND MIFROST_BUILD_PYMIMIR_ADAPTER AND MIFROST_GENERATE_STUBS)
    set(_mifrost_compat_stub_output "mifrost/_core.pyi")
    if (MIFROST_RPATH_MODE STREQUAL "dev")
        string(PREPEND _mifrost_compat_stub_output "${MIFROST_DEV_PACKAGE_ROOT}/")
        nanobind_add_stub(
            mifrost_core_module_stubs
            INCLUDE_PRIVATE
            MODULE mifrost._core
            OUTPUT "${_mifrost_compat_stub_output}"
            PYTHON_PATH "${MIFROST_DEV_PACKAGE_ROOT}"
            DEPENDS
                mifrost_neutral_core_module
                mifrost_pymimir_adapter_module
                "${CMAKE_SOURCE_DIR}/src/mifrost/_core.py"
        )
        list(APPEND _mifrost_dev_stub_targets mifrost_core_module_stubs)
    else ()
        nanobind_add_stub(
            mifrost_core_module_stubs
            INCLUDE_PRIVATE
            MODULE mifrost._core
            OUTPUT "${_mifrost_compat_stub_output}"
            DEPENDS mifrost_neutral_core_module mifrost_pymimir_adapter_module
            INSTALL_TIME
        )
    endif ()
endif ()

if (_mifrost_dev_stub_targets)
    # nanobind imports the staged package to introspect it, so the pure-Python
    # modules have to be in place before any stub target runs.
    foreach (_mifrost_stub_target IN LISTS _mifrost_dev_stub_targets)
        add_dependencies("${_mifrost_stub_target}" mifrost_stage_python)
    endforeach ()
    unset(_mifrost_stub_target)
    add_custom_target(mifrost_module_stubs DEPENDS ${_mifrost_dev_stub_targets})
endif ()

if (MIFROST_BUILD_PYMIMIR_ADAPTER)
    # Loki's AST uses >20 variant types; bump Boost limits to avoid MPL list errors.
    target_compile_definitions(mifrost_pymimir_adapter PUBLIC
            BOOST_MPL_LIMIT_LIST_SIZE=30
            BOOST_MPL_LIMIT_VECTOR_SIZE=30
            BOOST_VARIANT_LIMIT_TYPES=30
            BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
    )
endif ()

if (MIFROST_BUILD_BENCHMARKS)
    add_executable(
            mifrost_bench_hgraph
            "${CMAKE_SOURCE_DIR}/bench/bench_hgraph_encoder.cpp"
    )
    target_link_libraries(
            mifrost_bench_hgraph
            PRIVATE
            mifrost_pymimir_adapter
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
            mifrost_pymimir_adapter
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
            mifrost_bench_encoder_phases
            "${CMAKE_SOURCE_DIR}/bench/bench_encoder_phases.cpp"
    )
    target_link_libraries(
            mifrost_bench_encoder_phases
            PRIVATE
            mifrost_pymimir_adapter
            benchmark::benchmark
            argparse::argparse
            ${_python_embed_target}
    )
    target_compile_definitions(
            mifrost_bench_encoder_phases
            PRIVATE
            MIFROST_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
    )
    target_compile_definitions(mifrost_bench_encoder_phases PRIVATE MIFROST_BENCH_DEFAULT_DOMAIN="blocks")
    target_compile_definitions(mifrost_bench_encoder_phases PRIVATE MIFROST_BENCH_DEFAULT_PROBLEM="smedium")

    add_executable(
            mifrost_bench_relation_encoders
            "${CMAKE_SOURCE_DIR}/bench/bench_relation_encoder_compare.cpp"
    )
    target_link_libraries(
            mifrost_bench_relation_encoders
            PRIVATE
            mifrost_pymimir_adapter
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

    add_executable(
            mifrost_bench_semantic_flat_composition
            "${CMAKE_SOURCE_DIR}/bench/bench_semantic_flat_composition.cpp"
    )
    target_link_libraries(
            mifrost_bench_semantic_flat_composition
            PRIVATE
            mifrost_neutral_core
            benchmark::benchmark
    )
endif ()

# --- Tests ---
if (BUILD_TESTING)
    find_package(GTest REQUIRED)
    add_executable(
            mifrost_neutral_tests
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/batch_builder_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/flat_composition_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/hetero_relation_schema_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/relation_catalog_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/relation_key_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/relation_schema_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/semantic_flat_horizon_encoder_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/semantic_transition_dag_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/view_preparation_scaling_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/views_test.cpp"
    )
    target_include_directories(mifrost_neutral_tests PRIVATE "${CMAKE_CURRENT_LIST_DIR}/_core")
    target_link_libraries(mifrost_neutral_tests PRIVATE GTest::gtest_main mifrost_neutral_core)
    add_test(NAME mifrost_neutral_tests COMMAND mifrost_neutral_tests)

    if (MIFROST_BUILD_PYMIMIR_ADAPTER)
        add_executable(
            mifrost_tests
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/batch_builder_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/color_encoder_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/cross_problem_batch_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/direct_view_color_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/direct_view_encoder_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/direct_view_flat_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/direct_view_hgraph_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/flat_horizon_encoder_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/flat_relation_encoder_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/flat_stream_encoder_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/hgraph_encoder_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/hgraph_history_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/hgraph_stream_cache_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/hgraph_stream_contract_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/horizon_hgraph_encoder_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/horizon_stream_cache_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/stream_view_lifetime_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/successor_hgraph_encoder_test.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/transition_dag_test.cpp"
        )
        target_include_directories(mifrost_tests PRIVATE "${CMAKE_CURRENT_LIST_DIR}/_core")
        target_link_libraries(mifrost_tests PRIVATE GTest::gtest_main mifrost_pymimir_adapter ${_python_embed_target})
        target_compile_definitions(mifrost_tests PRIVATE MIFROST_DATA_DIR="${CMAKE_SOURCE_DIR}/data")
        add_test(NAME mifrost_tests COMMAND mifrost_tests)
    endif ()
endif ()
