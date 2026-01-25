cmake_minimum_required(VERSION 3.24)

# --- Conan Integration ---
# Must be included before project() if we want it to manage dependencies for the whole tree
set(CMAKE_PROJECT_TOP_LEVEL_INCLUDES "${CMAKE_CURRENT_LIST_DIR}/conan_provider.cmake")

project(bifrost_root LANGUAGES CXX)

add_subdirectory(src)
