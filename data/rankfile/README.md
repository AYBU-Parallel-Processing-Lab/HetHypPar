# Advanced OpenMPI rank binding strategies for custom core assignment

OpenMPI's process binding system offers sophisticated control over how MPI ranks are assigned to CPU cores, enabling optimal performance for diverse workloads including mixed CPU/GPU applications. This comprehensive guide provides practical solutions for implementing custom core assignments, allowing multiple ranks to share cores, and optimizing heterogeneous computing environments.

## Understanding OpenMPI's three-phase binding process

OpenMPI employs a systematic three-phase procedure for process placement that provides fine-grained control over rank distribution. **First, the mapping phase assigns processes to nodes and resources based on available hardware**. Second, the ranking phase assigns MPI_COMM_WORLD rank values to establish process identity. Finally, the binding phase constrains processes to specific processors, preventing unwanted migration and ensuring consistent performance.

The default binding behavior varies based on process count: jobs with 2 or fewer processes bind to individual cores, while larger jobs bind to sockets. Oversubscribed nodes receive no binding by default. This automatic behavior often requires override for specialized workloads, particularly when implementing custom core assignments or allowing core sharing between ranks.

Modern OpenMPI versions (4.x and 5.x) have introduced more flexible syntax while maintaining backward compatibility. The transition from simple sequential binding to topology-aware placement enables significant performance improvements, with NUMA-aware binding showing 15-30% performance gains over random placement in memory-intensive applications.

## Arbitrary core binding with rankfiles

Rankfiles provide the most precise control over process-to-core binding, enabling arbitrary assignments that override default sequential placement. **The basic rankfile syntax follows the pattern `rank N=hostname slot=core_list`, where core_list can specify individual cores, ranges, or socket-specific assignments**.

A basic rankfile for custom core assignment looks like this:
```
rank 0=node1 slot=0
rank 1=node1 slot=3
rank 2=node1 slot=7
rank 3=node2 slot=1:0-2
```

This assigns rank 0 to core 0, rank 1 to core 3, rank 2 to core 7 on node1, while rank 3 gets cores 0-2 on socket 1 of node2. The slot specification supports complex patterns including socket:core notation for NUMA-aware placement.

For hardware thread specification, append the `:hwtcpus` qualifier:
```bash
mpirun --map-by rankfile:file=myrankfile:hwtcpus ./application
```

Relative hostname specification using `+n0`, `+n1` notation enables rankfiles to work with dynamically allocated nodes from job schedulers:
```
rank 0=+n0 slot=1:0-2
rank 1=+n1 slot=0:0,1
rank 2=+n2 slot=1-2
```

When combined with hostfiles, this provides maximum flexibility for heterogeneous cluster configurations.

## Enabling multiple ranks per core

Core sharing, or oversubscription, requires explicit override of OpenMPI's default protection mechanisms. **The key flag `--bind-to :overload-allowed` permits multiple processes to bind to the same core**, essential for I/O-bound workloads or when running more processes than available cores.

To enable basic oversubscription:
```bash
mpirun --oversubscribe -np 16 --bind-to none ./application  # On 8-core system
```

For controlled core sharing with specific assignments:
```bash
# Rankfile for shared cores
cat > shared_cores.rf << EOF
rank 0=localhost slot=0
rank 1=localhost slot=0
rank 2=localhost slot=1
rank 3=localhost slot=1
EOF

mpirun --bind-to :overload-allowed --map-by rankfile:file=shared_cores.rf -np 4 ./app
```

Environment variables provide persistent oversubscription settings:
```bash
export OMPI_MCA_rmaps_base_oversubscribe=1
export OMPI_MCA_hwloc_base_binding_policy=none
```

Performance considerations for oversubscription include using `--mca mpi_yield_when_idle 1` to enable cooperative multitasking and reduce CPU contention between ranks sharing cores.

## Command line syntax for custom binding scenarios

OpenMPI provides extensive command-line options for binding control. **The modern syntax uses `--map-by`, `--bind-to`, and `--rank-by` flags to control each phase of the placement process independently**.

### Binding one CPU rank to every core on localhost:
```bash
# Basic one-to-one core binding
mpirun -np 8 --map-by core --bind-to core --report-bindings ./application

# Explicit core assignment using cpu-set
mpirun -np 4 --bind-to cpu-list:ordered --cpu-set "0,2,4,6" ./application

# NUMA-aware distribution
mpirun -np 8 --map-by ppr:4:numa --bind-to core ./application
```

### Advanced binding patterns:
```bash
# Multiple cores per process (hybrid MPI/OpenMP)
export OMP_NUM_THREADS=4
mpirun -np 4 --map-by core:PE=4 --bind-to core ./hybrid_app

# Socket-level distribution with core binding
mpirun -np 16 --map-by ppr:8:socket --bind-to core ./application

# L3 cache-aware binding for AMD systems
mpirun -np 32 --map-by ppr:2:l3cache --bind-to core ./cache_optimized_app
```

Verification flags `--report-bindings` and `--display-map` provide immediate feedback on binding decisions, essential for confirming correct placement.

## Optimizing mixed CPU/GPU workloads

Mixed CPU/GPU workloads require careful coordination between compute resources. **GPU ranks should bind to cores within the same NUMA domain as their assigned GPU to minimize memory latency**, while CPU-only ranks can be distributed across remaining cores.

A comprehensive wrapper script for mixed workloads:
```bash
#!/bin/bash
# gpu_cpu_wrapper.sh

LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
GPU_COUNT=$(nvidia-smi -L | wc -l)

if [ $LOCAL_RANK -lt $GPU_COUNT ]; then
    # GPU ranks
    export CUDA_VISIBLE_DEVICES=$LOCAL_RANK
    # Bind to cores near GPU (example for 4 GPUs on 64-core system)
    case $LOCAL_RANK in
        0) cpus="0-7" ;;    # GPU 0 - NUMA 0
        1) cpus="16-23" ;;  # GPU 1 - NUMA 1
        2) cpus="32-39" ;;  # GPU 2 - NUMA 2
        3) cpus="48-55" ;;  # GPU 3 - NUMA 3
    esac
else
    # CPU-only ranks
    export CUDA_VISIBLE_DEVICES=""
    # Distribute across remaining cores
    OFFSET=$((($LOCAL_RANK - $GPU_COUNT) * 2 + 8))
    cpus="$OFFSET-$(($OFFSET + 1))"
fi

numactl --physcpubind=$cpus "$@"
```

For CUDA-aware MPI configurations:
```bash
export UCX_TLS=rc,sm,cuda_copy,gdr_copy,cuda_ipc
mpirun -np 8 --map-by ppr:2:numa --bind-to numa \
       -x UCX_TLS -x CUDA_VISIBLE_DEVICES \
       ./wrapper.sh ./gpu_application
```

## Configuration file examples

### Complex rankfile for heterogeneous workload:
```
# Mixed CPU/GPU rankfile for 2-node cluster
# Node 1: 4 GPU ranks + 4 CPU ranks
rank 0=node01 slot=0:0-3     # GPU rank, NUMA 0
rank 1=node01 slot=0:4-7     # GPU rank, NUMA 0
rank 2=node01 slot=1:16-19   # GPU rank, NUMA 1
rank 3=node01 slot=1:20-23   # GPU rank, NUMA 1
rank 4=node01 slot=0:8-11    # CPU rank, NUMA 0
rank 5=node01 slot=0:12-15   # CPU rank, NUMA 0
rank 6=node01 slot=1:24-27   # CPU rank, NUMA 1
rank 7=node01 slot=1:28-31   # CPU rank, NUMA 1

# Node 2: Similar distribution
rank 8=node02 slot=0:0-3     # GPU rank, NUMA 0
rank 9=node02 slot=0:4-7     # GPU rank, NUMA 0
```

### Hostfile with oversubscription support:
```
node1 slots=32 max_slots=64
node2 slots=32 max_slots=64
gpu-node1 slots=16 max_slots=32
gpu-node2 slots=16 max_slots=32
```

## Essential environment variables and MCA parameters

Key environment variables for binding control:
```bash
# OpenMPI 4.x
export OMPI_MCA_rmaps_base_mapping_policy=core
export OMPI_MCA_hwloc_base_binding_policy=core
export OMPI_MCA_rmaps_base_oversubscribe=1
export OMPI_MCA_rmaps_base_report_bindings=1

# OpenMPI 5.x (new prefix)
export PRTE_MCA_rmaps_default_mapping_policy=core
export PRTE_MCA_hwloc_default_binding_policy=core
```

Critical MCA parameters for fine-tuning:
```bash
# Hardware thread handling
mpirun --mca hwloc_base_use_hwthreads_as_cpus 1 ./app

# Physical CPU interpretation for rankfiles
mpirun --mca rmaps_rank_file_physical 1 --rankfile physical.rf ./app

# Binding verification
mpirun --mca rmaps_base_display_map 1 --mca rmaps_base_report_bindings 1 ./app
```

## Troubleshooting custom rank binding

Common binding errors and solutions include addressing oversubscription warnings by adding `--bind-to :overload-allowed`, resolving hostname mismatches in rankfiles by using FQDN or relative notation, and fixing cgroup limitations from resource managers by checking `/sys/fs/cgroup/cpuset/cpuset.cpus`.

**Diagnostic workflow for binding issues starts with hardware verification using `lstopo`, followed by binding validation with `--report-bindings`, system-level confirmation via `hwloc-ps` or `taskset -pc`, and performance comparison between different binding strategies**.

Performance problems manifest as 30-50% degradation from poor NUMA locality, excessive context switches from unbound processes, or cache thrashing from incorrect L3 domain assignment. Monitor with `numastat`, `pidstat -w`, and application-specific benchmarks to identify and resolve binding-related performance issues.

Error messages like "binding more processes than cpus" indicate resource conflicts requiring explicit overload permission or adjusted process counts. The "rankfile claimed host not allocated" error typically stems from hostname resolution issues or mismatched resource allocations between the scheduler and rankfile.

## Conclusion

OpenMPI's rank binding capabilities provide the fine-grained control necessary for optimizing modern HPC applications. By combining rankfiles for precise placement, oversubscription flags for flexible core sharing, and NUMA-aware strategies for mixed workloads, administrators can achieve significant performance improvements over default configurations. The key to success lies in understanding your hardware topology, workload characteristics, and systematically applying the appropriate binding strategies while monitoring their impact on application performance.