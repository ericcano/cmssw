Remote CUDA over MPI (openMPI + UCX)
===

OpenMPI provides support to directly access the memory of the CUDA. For example, an MPI process can receive a message directly in the GPU memory.

In addition, with proper hardware and software support, the data can be copied directly from an RDMA-supporting NIC to the GPU. Without RDMA support, the peer to peer management layer (PML) of MPI will still handle the transport automatically, yet with an intermediate copy in CPU memory.

The [UCX](https://openucx.readthedocs.io/en/master/index.html) PML provides the support for all those direct memory copies.

The code reading directly to GPU memory looks like:
```C++
//allocate memory space for vector in the global memory of the evice.
cudaMalloc((void**)&d_vect1, mpiInput.vectorWorkers1.size()*sizeoffloat));

MPI_Irecv(d_vect1,
          mpiInput.numberToSend[mpiInput.rank],
          MPI_FLOAT,
          root,
          0,
          MPI_COMM_WORLD,
          &requestWorkerRecv[0]);
```

In order to use UCX with openMPI, the foolowing options shall be used:
```bash
mpirun --mca pml ucx -x UCX_TLS=all -np 4 myProgram
```

The program (which depends on the CMSSW environment) was run on multiple hosts in the CMS DAQ cluster with the following command:
```bash
mpirun  -H `echo fu-c2a02-39-0{1..4}|tr " " ,` --launch-agent ~/ortedWithEnv.sh --mca pml ucx -x UCX_TLS=all -np 4 myProgram
```
with the help of the `~/ortedWithEnv.sh` wrapper:
```bash
#!/bin/bash

# Wrapper to open MPI tool orted from a CMSSW environment in the DAQ cluster
source /data/cmssw/cmsset_default.sh
cd /data/cmssw/slc7_amd64_gcc900/cms/cmssw/CMSSW_12_1_0_pre3/
eval `scramv1 runtime -sh`
cd
orted $@
```
The detailed routing of messages can be found by requesting log from UCX through extra environment variables (set with the `-x` option of `mpirun`): `-x UCX_LOG_FILE=~/MPI/mpi_ucx-%h-%p.log -x UCX_LOG_LEVEL=INFO`

The interface to use can be set using the variable: `-x UCX_NET_DEVICES=enp59s0`

Without the option, the log file indicates routing through the all available network (control on 10.174.0.0/16, data on 10.180.0.0/16):
```
[1635950815.628538] [fu-c2a02-39-01:186353:0]         tcp_cm.c:96   UCX  DEBUG tcp_ep 0x7fa4e00008c0: ACCEPTING -> CONNECTED for the [10.180.47.40:46878]<->[10.180.47.41:53221]:0 connection [-:Rx]
[1635950815.628554] [fu-c2a02-39-01:186353:0]         tcp_cm.c:96   UCX  DEBUG tcp_ep 0x7fa4e0000970: ACCEPTING -> CONNECTED for the [10.176.14.185:56378]<->[10.176.14.195:49971]:0 connection [-:Rx]
```
While specifying the interface constraints the communication to the data network:
```
[1635950846.394465] [fu-c2a02-39-01:188231:0]         tcp_cm.c:96   UCX  DEBUG tcp_ep 0x7f8d6c0008c0: ACCEPTING -> CONNECTED for the [10.180.47.40:43938]<->[10.180.47.41:40168]:0 connection [-:Rx]
```
In the absence of hardwar support for RDMA, the set of transports used can be reduced to the minimal set of `-x UCX_TLS=tcp,cuda` instead of `-x UCX_TLS=all`


