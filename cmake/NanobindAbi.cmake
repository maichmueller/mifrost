# Configure-time guard against the one nanobind failure mode that reports nothing.
#
# Every nanobind extension looks its shared C++-type registry up in the builtins
# dict under `__nb_internals_<abi_tag>_<domain>__`, and that tag leads with
# NB_INTERNALS_VERSION -- a counter over the layout of nanobind's internal
# structs which moves independently of the release number:
#
#     nanobind 2.7.0 -> 16    2.12.0 -> 19    2.15.0 -> 21
#     nanobind 2.9.2 -> 16    2.13.0 -> 20
#
# Two modules built against different generations get *separate* registries and
# can no longer cast each other's C++ types. Nothing fails at link time and
# nothing fails at import time: the first pymimir object handed to this
# extension raises a TypeError instead of converting.
#
# This project deliberately joins pymimir's registry (NB_DOMAIN below), so the
# two generations must agree. Both numbers are readable before a single
# translation unit is compiled -- nanobind's from its own `nb_abi.h`, pymimir's
# from the tag string baked into the installed binary -- so the mismatch is
# turned into a configure error here rather than a TypeError at the first
# cross-module call.

# Reads NB_INTERNALS_VERSION out of the nanobind that `nanobind_dir` belongs to.
function (nanobind_abi_internals_version nanobind_dir out_var)
    set(${out_var} "" PARENT_SCOPE)
    get_filename_component(_root "${nanobind_dir}" DIRECTORY)
    foreach (_candidate "${_root}/src/nb_abi.h" "${_root}/include/nanobind/nb_abi.h")
        if (EXISTS "${_candidate}")
            file(STRINGS "${_candidate}" _defines
                 REGEX "define[ \t]+NB_INTERNALS_VERSION[ \t]+[0-9]+")
            if (_defines)
                list(GET _defines 0 _define)
                string(REGEX MATCH "[0-9]+$" _version "${_define}")
                set(${out_var} "${_version}" PARENT_SCOPE)
            endif ()
            return ()
        endif ()
    endforeach ()
endfunction ()

# Reads the ABI tag (e.g. `v21_system_libcpp_abi1`) out of the installed
# pymimir extension. Empty when pymimir is not importable.
function (nanobind_abi_pymimir_tag python_executable out_var)
    set(${out_var} "" PARENT_SCOPE)
    execute_process(
        COMMAND "${python_executable}" "-c"
                "import itertools, pathlib, pymimir; d = pathlib.Path(pymimir.__file__).resolve().parent; c = sorted(itertools.chain(d.glob('*.so'), d.glob('*.pyd'))); print(c[0] if c else '')"
        RESULT_VARIABLE _result
        OUTPUT_VARIABLE _module
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (NOT _result EQUAL 0 OR NOT _module OR NOT EXISTS "${_module}")
        return ()
    endif ()
    # `file(STRINGS)` pulls ASCII runs out of the binary the way `strings` does;
    # the tag is the only `v<N>_<compiler>...` run nanobind emits.
    file(STRINGS "${_module}" _matches
         REGEX "v[0-9]+_(system|msvc|mingw|gcc_cygwin)[A-Za-z0-9_]*")
    foreach (_match IN LISTS _matches)
        string(REGEX MATCH "v[0-9]+_(system|msvc|mingw|gcc_cygwin)[A-Za-z0-9_]*" _tag "${_match}")
        if (_tag)
            set(${out_var} "${_tag}" PARENT_SCOPE)
            return ()
        endif ()
    endforeach ()
endfunction ()

# Fails the configure when this build's nanobind and the installed pymimir sit
# in different internals generations. Skips quietly when either number cannot be
# determined -- pymimir need not be installed to compile the C++ core, and an
# unrecognised nanobind layout should not block a build that would otherwise
# work.
function (nanobind_abi_require_match nanobind_dir python_executable)
    nanobind_abi_internals_version("${nanobind_dir}" _nanobind_generation)
    nanobind_abi_pymimir_tag("${python_executable}" _pymimir_tag)

    if (NOT _nanobind_generation)
        message(STATUS "nanobind ABI check skipped: no nb_abi.h next to ${nanobind_dir}.")
        return ()
    endif ()
    if (NOT _pymimir_tag)
        message(STATUS "nanobind ABI check skipped: pymimir not importable from ${python_executable}.")
        return ()
    endif ()

    string(REGEX MATCH "^v([0-9]+)_" _matched "${_pymimir_tag}")
    set(_pymimir_generation "${CMAKE_MATCH_1}")

    if (NOT _pymimir_generation STREQUAL _nanobind_generation)
        message(FATAL_ERROR
            "nanobind ABI mismatch: this build would use nanobind internals "
            "generation ${_nanobind_generation}, but the installed pymimir is "
            "'${_pymimir_tag}' (generation ${_pymimir_generation}).\n"
            "  nanobind: ${nanobind_dir}\n"
            "  python:   ${python_executable}\n"
            "Both modules share NB_DOMAIN=pymimir_abi_domain, so mismatched "
            "generations would build separate type registries and every "
            "pymimir object crossing into this extension would raise a "
            "TypeError -- with no build or import error to point at it.\n"
            "Install a pymimir and a nanobind from the same generation "
            "(pymimir >= 0.14.3 pins nanobind 2.15.x, generation 21), then "
            "reconfigure from a clean build directory."
        )
    endif ()

    message(STATUS
            "nanobind ABI generation ${_nanobind_generation} matches installed "
            "pymimir (${_pymimir_tag}).")
endfunction ()
