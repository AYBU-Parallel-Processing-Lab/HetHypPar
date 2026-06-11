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
  util/hhp_dp_kernels.cu
)

include_directories(
    include
    ${CUDAToolkit_INCLUDE_DIRS}
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
add_executable(bicgstab-gpu-dp entry/bicgstab_gpu_dp.c)
add_executable(bicgstab-mpi-gpu entry/bicgstab_mpi_gpu.c)
add_executable(bicgstab-mpi-gpu-mt entry/bicgstab_mpi_gpu_mt.c)
add_executable(bicgstab-hybrid-async entry/bicgstab_hybrid_async.c)
add_executable(bicgstab-hybrid-async-dp entry/bicgstab_hybrid_async_dp.c)
add_executable(spmv-cpu entry/spmv_cpu.c)
add_executable(spmv-cpu-mt entry/spmv_cpu_mt.c)
add_executable(spmv-cpu-mkl entry/spmv_cpu_mkl.c)
add_executable(spmv-mpi entry/spmv_mpi.c)
add_executable(spmv-hybrid entry/spmv_hybrid.c)
add_executable(spmv-hybrid-async entry/spmv_hybrid_async.c)
add_executable(spmv-gpu entry/spmv_gpu.c)

target_link_libraries(bicgstab-cpu inner_lib)
target_link_libraries(bicgstab-mpi inner_lib)
target_link_libraries(bicgstab-mpi-gpu inner_lib)
target_link_libraries(bicgstab-mpi-gpu-mt inner_lib)
target_link_libraries(bicgstab-hybrid-async inner_lib)
target_link_libraries(bicgstab-hybrid-async-dp inner_lib)
target_link_libraries(bicgstab-gpu inner_lib)
target_link_libraries(bicgstab-gpu-dp inner_lib)
target_link_libraries(spmv-cpu inner_lib)
target_link_libraries(spmv-cpu-mt inner_lib)
target_link_libraries(spmv-cpu-mkl inner_lib)
target_link_libraries(spmv-mpi inner_lib)
target_link_libraries(spmv-hybrid inner_lib)
target_link_libraries(spmv-hybrid-async inner_lib)
target_link_libraries(spmv-gpu inner_lib)

add_subdirectory(tests)
