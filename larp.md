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
