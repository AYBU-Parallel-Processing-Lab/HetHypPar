cmake_minimum_required(VERSION 3.10)
project(patoh)

# Create an interface library target
add_library(patoh STATIC IMPORTED GLOBAL)

# Set the path to the pre-compiled library
set_target_properties(patoh PROPERTIES
    IMPORTED_LOCATION "${PROJECT_SOURCE_DIR}/libpatoh.a"
    INTERFACE_INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/include"
)