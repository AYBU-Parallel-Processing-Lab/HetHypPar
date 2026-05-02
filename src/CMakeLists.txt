cmake_minimum_required(VERSION 3.24)

project(inner LANGUAGES C CUDA)

enable_testing()

find_package(MPI REQUIRED)
add_definitions(-DOMPI_SKIP_MPICXX)

find_package(MKL REQUIRED)
include_directories(${MKL_INCLUDE_DIRS})
link_directories(${MKL_LIBRARIES})

# Use modern CUDA support (built into CMake for CUDA language)
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

find_package(OpenMP REQUIRED)

# Add Unity testing framework
include(FetchContent)
FetchContent_Declare(
    unity
    GIT_REPOSITORY https://github.com/ThrowTheSwitch/Unity.git
    GIT_TAG v2.6.1
)
FetchContent_MakeAvailable(unity)

add_subdirectory(dependencies)

set(C_SOURCE_FILES
  util/hhp_matrix.c
  util/hhp_wrap_bora.c
  util/hhp_cpu.c
  util/hhp_cuda.c
)

include_directories(
    include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${MPI_C_INCLUDE_DIRS}
    /usr/include/openmpi-x86_64
)

link_libraries(
  mkl_intel_lp64
  mkl_intel_thread
  bora-spmxv
  mmio
  patoh
  iomp5
  mkl_core
  MPI::MPI_C
  OpenMP::OpenMP_C
  CUDA::cudart
  CUDA::cusparse
  CUDA::cublas
  m
)

# Create library target for source files
add_library(inner_lib STATIC ${C_SOURCE_FILES})

add_executable(bicgstab-cpu entry/bicgstab_cpu.c)
add_executable(bicgstab-mpi entry/bicgstab_mpi.c)
add_executable(bicgstab-gpu entry/bicgstab_gpu.c)
add_executable(bicgstab-mpi-gpu entry/bicgstab_mpi_gpu.c)
add_executable(bicgstab-mpi-gpu-pipeline entry/bicgstab_mpi_gpu_pipeline.c)
add_executable(bicgstab-mpi-gpu-fusion entry/bicgstab_mpi_gpu_fusion.cu)
add_executable(bicgstab-mpi-gpu-pipeline-fusion entry/bicgstab_mpi_gpu_pipeline_fusion.cu)

target_compile_options(bicgstab-mpi-gpu-pipeline PRIVATE -g -O0)
target_compile_options(bicgstab-mpi-gpu-fusion PRIVATE -g -O3)
target_include_directories(bicgstab-mpi-gpu-fusion PRIVATE ${MPI_C_INCLUDE_DIRS} /usr/include/openmpi-x86_64)
target_include_directories(bicgstab-mpi-gpu-pipeline-fusion PRIVATE ${MPI_C_INCLUDE_DIRS} /usr/include/openmpi-x86_64)

target_link_libraries(bicgstab-cpu inner_lib)
target_link_libraries(bicgstab-mpi inner_lib)
target_link_libraries(bicgstab-mpi-gpu inner_lib)
target_link_libraries(bicgstab-mpi-gpu-pipeline inner_lib)
target_link_libraries(bicgstab-mpi-gpu-fusion inner_lib)
target_link_libraries(bicgstab-mpi-gpu-pipeline-fusion inner_lib)
target_link_libraries(bicgstab-gpu inner_lib)

add_subdirectory(tests)
