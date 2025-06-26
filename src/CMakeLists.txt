project(inner LANGUAGES C CUDA)

enable_testing()

find_package(MPI REQUIRED)
add_definitions(-DOMPI_SKIP_MPICXX)

find_package(MKL REQUIRED)
include_directories(${MKL_INCLUDE_DIRS})
link_directories(${MKL_LIBRARIES})



# Find CUDA package
find_package(CUDA REQUIRED)
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
		m
)


# Create library target for source files
add_library(inner_lib STATIC ${C_SOURCE_FILES})


# add_executable(mxv-gpu-bicgstab entry/hybrid-bicgstab.c)  # BICG-Stab using GPU-CPU hybrid SpMxV
add_executable(mxv-gpu-bicgstab-cpu entry/hybrid-bicgstab_cpu.c)  # BICG-Stab using GPU-CPU hybrid SpMxV
add_executable(mxv-gpu-bicgstab-mpi entry/hybrid-bicgstab_mpi.c)  # BICG-Stab using GPU-CPU hybrid SpMxV
# target_link_libraries(mxv-gpu-bicgstab inner_lib)
target_link_libraries(mxv-gpu-bicgstab-cpu inner_lib)
target_link_libraries(mxv-gpu-bicgstab-mpi inner_lib)


add_subdirectory(tests)