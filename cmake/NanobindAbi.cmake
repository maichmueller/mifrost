# Configure-time guard against the one nanobind failure mode that reports nothing.
#
# Every nanobind extension looks its shared C++-type registry up in the builtins
# dict under `__nb_internals_<abi_tag>_<domain>__`, and that tag leads with
# NB_INTERNALS_VERSION -- a counter over the layout of nanobind's internal
# structs which moves independently of the release number:
#
#     nanobind 2.7.0 -> 16    2.12.0 -> 19    2.15.0 -> 21
#     nanobind 2.9.2 -> 16    2.13.0 -> 20    3.0.0 -> 22
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
    # nanobind 3 deleted nb_abi.h and moved the counter into nb_internals.h.
    # Checking only the 2.x filenames made this guard SKIP silently on 3.x --
    # reporting "no nb_abi.h next to <dir>" and then allowing exactly the
    # mismatch it exists to catch. Both spellings are checked, newest first.
    foreach (_candidate
             "${_root}/src/nb_internals.h"
             "${_root}/include/nanobind/nb_internals.h"
             "${_root}/src/nb_abi.h"
             "${_root}/include/nanobind/nb_abi.h")
        if (EXISTS "${_candidate}")
            file(STRINGS "${_candidate}" _defines
                 REGEX "define[ \t]+NB_INTERNALS_VERSION[ \t]+[0-9]+")
            # Return only once the counter is actually FOUND, not merely because
            # a candidate file exists: nanobind 2.x ships an nb_internals.h that
            # does NOT carry the define (it lives in nb_abi.h there), so
            # returning on mere existence would skip the guard on exactly the
            # versions it used to protect.
            if (_defines)
                list(GET _defines 0 _define)
                string(REGEX MATCH "[0-9]+$" _version "${_define}")
                set(${out_var} "${_version}" PARENT_SCOPE)
                return ()
            endif ()
        endif ()
    endforeach ()
endfunction ()

# Reads the ABI tag out of the installed pymimir extension. Empty when pymimir
# is not importable.
#
# The two nanobind generations spell the tag in OPPOSITE orders:
#   nanobind 2.x -> `v21_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1`
#   nanobind 3.x -> `system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1_a1_v22`
# Matching only the 2.x shape found nothing in a 3.x pymimir, so the guard
# reported "pymimir not importable" and skipped -- the same silent-skip failure
# the internals-version lookup above had. Both orders are matched here, and
# `nanobind_abi_require_match` parses the generation out of either.
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
         REGEX "(v[0-9]+_)?(system|msvc|mingw|gcc_cygwin)[A-Za-z0-9_]*")
    foreach (_match IN LISTS _matches)
        # 3.x first: its tag ends in `_v<N>` and would otherwise be truncated by
        # the 2.x pattern, which stops at the first non-tag character.
        string(REGEX MATCH "(system|msvc|mingw|gcc_cygwin)[A-Za-z0-9_]*_v[0-9]+" _tag "${_match}")
        if (NOT _tag)
            string(REGEX MATCH "v[0-9]+_(system|msvc|mingw|gcc_cygwin)[A-Za-z0-9_]*" _tag "${_match}")
        endif ()
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

    # 2.x puts the generation first, 3.x last -- accept either.
    set(_pymimir_generation "")
    if ("${_pymimir_tag}" MATCHES "^v([0-9]+)_")
        set(_pymimir_generation "${CMAKE_MATCH_1}")
    elseif ("${_pymimir_tag}" MATCHES "_v([0-9]+)$")
        set(_pymimir_generation "${CMAKE_MATCH_1}")
    endif ()
    if (NOT _pymimir_generation)
        message(STATUS
                "nanobind ABI check skipped: could not parse a generation out "
                "of the installed pymimir tag '${_pymimir_tag}'.")
        return ()
    endif ()

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
            "(pymimir >= 0.15.0 pins nanobind 3.0.x, generation 22), then "
            "reconfigure from a clean build directory."
        )
    endif ()

    message(STATUS
            "nanobind ABI generation ${_nanobind_generation} matches installed "
            "pymimir (${_pymimir_tag}).")
endfunction ()
