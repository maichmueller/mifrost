cmake_minimum_required(VERSION 3.24)

project(mifrost_pytyr_consumer LANGUAGES CXX)

find_package(mifrost CONFIG REQUIRED)

if (NOT mifrost_WITH_PYTYR_ADAPTER)
    message(FATAL_ERROR "Expected a Mifrost package with the PyTyr adapter")
endif ()
if (NOT TARGET mifrost::pytyr_adapter)
    message(FATAL_ERROR "Package does not export mifrost::pytyr_adapter")
endif ()

add_executable(mifrost_pytyr_consumer main.cpp)
target_compile_features(mifrost_pytyr_consumer PRIVATE cxx_std_23)
target_link_libraries(mifrost_pytyr_consumer PRIVATE mifrost::pytyr_adapter)
