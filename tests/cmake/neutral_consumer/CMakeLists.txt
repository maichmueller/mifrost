cmake_minimum_required(VERSION 3.24)

project(mifrost_neutral_consumer LANGUAGES CXX)

find_package(mifrost CONFIG REQUIRED)

if (mifrost_WITH_PYMIMIR_ADAPTER)
    message(FATAL_ERROR "Expected a neutral-only Mifrost package")
endif ()
if (TARGET mifrost::core)
    message(FATAL_ERROR "Neutral-only package unexpectedly exports mifrost::core")
endif ()
if (NOT TARGET mifrost::neutral_core)
    message(FATAL_ERROR "Neutral-only package does not export mifrost::neutral_core")
endif ()

add_executable(mifrost_neutral_consumer main.cpp)
target_compile_features(mifrost_neutral_consumer PRIVATE cxx_std_23)
target_link_libraries(mifrost_neutral_consumer PRIVATE mifrost::neutral_core)
