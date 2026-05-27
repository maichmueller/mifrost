function(conan_add_package_include_fallback target package_prefix)
    set(options "")
    set(oneValueArgs SCOPE)
    set(multiValueArgs "")
    cmake_parse_arguments(CONAN_FALLBACK "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if(NOT CONAN_FALLBACK_SCOPE)
        set(CONAN_FALLBACK_SCOPE PUBLIC)
    endif()

    set(_fallback_includes "")
    if(DEFINED ${package_prefix}_PACKAGE_FOLDER_DEBUG)
        list(APPEND _fallback_includes
            "$<BUILD_INTERFACE:$<$<CONFIG:Debug>:${${package_prefix}_PACKAGE_FOLDER_DEBUG}/include>>")
    endif()
    if(DEFINED ${package_prefix}_PACKAGE_FOLDER_RELEASE)
        list(APPEND _fallback_includes
            "$<BUILD_INTERFACE:$<$<CONFIG:Release>:${${package_prefix}_PACKAGE_FOLDER_RELEASE}/include>>")
    endif()
    if(DEFINED ${package_prefix}_PACKAGE_FOLDER_RELWITHDEBINFO)
        list(APPEND _fallback_includes
            "$<BUILD_INTERFACE:$<$<CONFIG:RelWithDebInfo>:${${package_prefix}_PACKAGE_FOLDER_RELWITHDEBINFO}/include>>")
    endif()
    if(DEFINED ${package_prefix}_PACKAGE_FOLDER_MINSIZEREL)
        list(APPEND _fallback_includes
            "$<BUILD_INTERFACE:$<$<CONFIG:MinSizeRel>:${${package_prefix}_PACKAGE_FOLDER_MINSIZEREL}/include>>")
    endif()
    if(DEFINED ${package_prefix}_PACKAGE_FOLDER)
        list(APPEND _fallback_includes "$<BUILD_INTERFACE:${${package_prefix}_PACKAGE_FOLDER}/include>")
    endif()

    if(_fallback_includes)
        target_include_directories(${target} SYSTEM ${CONAN_FALLBACK_SCOPE} ${_fallback_includes})
    endif()
endfunction()
