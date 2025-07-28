project(inner LANGUAGES C CUDA)

enable_testing()

find_package(MPI REQUIRED)
add_definitions(-DOMPI_SKIP_MPICXX)

find_package(MKL REQUIRED)
include_directories(${MKL_INCLUDE_DIRS})
link_directories(${MKL_LIBRARIES})

# Force CUDA to use specific installation
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.9")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.9/bin/nvcc")

# Find CUDA package
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

# Find cuBLAS (part of CUDA toolkit)
find_library(CUBLAS_LIBRARY 
    NAMES cublas
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    NO_DEFAULT_PATH
    REQUIRED
)

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
    ${CUDA_INCLUDE_DIRS}
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
		${CUDA_LIBRARIES}
		${CUDA_cusparse_LIBRARY}
		${CUBLAS_LIBRARY}
		m
)

# Create library target for source files
add_library(inner_lib STATIC ${C_SOURCE_FILES})

# add_executable(mxv-gpu-bicgstab entry/hybrid-bicgstab.c)  # BICG-Stab using GPU-CPU hybrid SpMxV
add_executable(bicgstab-cpu entry/bicgstab_cpu.c)  # BICG-Stab using CPU SpMxV
add_executable(bicgstab-mpi entry/bicgstab_mpi.c)  # BICG-Stab using MPI SpMxV
add_executable(bicgstab-gpu entry/bicgstab_gpu.c)  # BICG-Stab using GPU hybrid SpMxV
add_executable(bicgstab-mpi-gpu entry/bicgstab_mpi_gpu.c)  # BICG-Stab using MPI SpMxV
# target_link_libraries(mxv-gpu-bicgstab inner_lib)
target_link_libraries(bicgstab-cpu inner_lib)
target_link_libraries(bicgstab-mpi inner_lib)
# target_link_libraries(mxv-gpu-bicgstab-gpu inner_lib)
target_link_libraries(bicgstab-mpi-gpu inner_lib)
target_link_libraries(bicgstab-gpu inner_lib)

add_subdirectory(tests)