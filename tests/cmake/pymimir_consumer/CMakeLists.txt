cmake_minimum_required(VERSION 3.24)

project(mifrost_pymimir_consumer LANGUAGES CXX)

find_package(mifrost CONFIG REQUIRED)

if (NOT mifrost_WITH_PYMIMIR_ADAPTER)
    message(FATAL_ERROR "Expected a Mifrost package with the Pymimir adapter")
endif ()
if (NOT TARGET mifrost::pymimir_adapter)
    message(FATAL_ERROR "Package does not export mifrost::pymimir_adapter")
endif ()
if (NOT TARGET mifrost::core)
    message(FATAL_ERROR "Package does not preserve the mifrost::core target")
endif ()

add_executable(mifrost_pymimir_consumer main.cpp)
target_compile_features(mifrost_pymimir_consumer PRIVATE cxx_std_23)
target_link_libraries(mifrost_pymimir_consumer PRIVATE mifrost::core)
