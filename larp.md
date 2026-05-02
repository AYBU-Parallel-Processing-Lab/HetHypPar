# How to compile for noobs

## Obtaining intel MKL through intel's openAPI Toolkit
Download [Intel oneAPI Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneapi-toolkit-download.html) and follow the instalation instructions

## Obtaining cuBLAS through nvidia
Download [nvidia cuBLAS](https://developer.nvidia.com/cublas-downloads) and follow the instalation instructions

## Compilation
1. Load the intel environment
```
source /opt/intel/oneapi/setvars.sh;
```
Why not just set this in the user's .bashrc? because it breaks user login for me...

2. Export OpenMPI's binary
```
export PATH=/usr/lib64/openmpi/bin:$PATH;
```
Intel ships its own MPI runtime, why do I need this? Because intel's MPI runtime doesn't support `--report-bindings` and `-bind-to-core` flags, which we use to pin certain nodes to certain ranks.

3. Export CC and CXX for CMake
```
export CC=/usr/lib64/openmpi/bin/mpicc;
export CXX=/usr/lib64/openmpi/bin/mpicxx;
```
Why? idk. Having them in my .bashrc breaks my PATH variable and that isn't good.

4. Clean and Build
```
rm -rf build/;
cmake -S src -B build -G Ninja;
cmake --build build;
```

## Running

### Exports openmpi to LD_LIBRARY_PATH
```
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:$LD_LIBRARY_PATH 
```

### Run the solver
```
mpirun --report-bindings -bind-to core -n 2 \
./build/bicgstab-mpi-gpu \
-g data/matrices/circuit5M_dc/in/is_gpu.txt \
-m data/matrices/circuit5M_dc/in/circuit5M_dc.mtx \
-n 100 \
-o data/matrices/circuit5M_dc/out/X_mpi_gpu_pipe.txt \
-p data/matrices/circuit5M_dc/in/row_part_2 \
-x data/matrices/circuit5M_dc/in/X_init.txt \
-y data/matrices/circuit5M_dc/in/B.txt
```
