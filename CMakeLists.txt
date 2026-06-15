cmake_minimum_required(VERSION 3.24)

# --- Conan Integration ---
# Use the Conan provider when a toolchain is not already in play.
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    # Must be included before project() if we want it to manage dependencies for the whole tree
    set(CMAKE_PROJECT_TOP_LEVEL_INCLUDES "${CMAKE_CURRENT_LIST_DIR}/conan_provider.cmake")
endif()

project(mifrost_root LANGUAGES CXX)

include(CTest)

add_subdirectory(src)
