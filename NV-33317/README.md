Context: I have been porting my research group’s MFC project to CMake. It is an OpenACC accelerated high fidelity CFD code written in Fortran. My current developement fork is available here: GitHub - henryleberre/MFC: High-fidelity multiphase flow simulation 1.

We encounter what seems to be an internal compiler error from NVHPC 22.5 when attempting to compile the “simulation” executable on some systems. On ORNL’s Summit no error is generated, and all tests pass. However, any other system I have tried it on produces (with NVHPC 22.5):

```console
NVFORTRAN-W-0435-Array declared with zero size (/home/henryleberre/MFC/src/common/autogen/m_global_parameters.f90: 822)
NVFORTRAN-W-0435-Array declared with zero size (/home/henryleberre/MFC/src/common/autogen/m_global_parameters.f90: 823)
NVFORTRAN-W-0435-Array declared with zero size (/home/henryleberre/MFC/src/common/autogen/m_global_parameters.f90: 824)
NVFORTRAN-W-0435-Array declared with zero size (/home/henryleberre/MFC/src/common/autogen/m_global_parameters.f90: 825)
NVFORTRAN-W-0435-Array declared with zero size (/home/henryleberre/MFC/src/common/autogen/m_global_parameters.f90: 826)
NVFORTRAN-W-0435-Array declared with zero size (/home/henryleberre/MFC/src/common/autogen/m_global_parameters.f90: 1123)
NVFORTRAN-W-0155-Constant or Parameter used in data clause - weno_polyn (/home/henryleberre/MFC/src/common/autogen/m_global_parameters.f90: 483)
NVFORTRAN-W-0155-Constant or Parameter used in data clause - nb (/home/henryleberre/MFC/src/common/autogen/m_global_parameters.f90: 484)
  0 inform,   2 warnings,   0 severes, 0 fatal for s_initialize_global_parameters_module
s_initialize_global_parameters_module:
    750, Generating update device(re_idx(:,:),re_size(:))
    786, Generating update device(startz,starty,startx)
s_comp_n_from_cons:
   1029, Generating acc routine seq
         Generating NVIDIA GPU code
s_comp_n_from_prim:
   1069, Generating acc routine seq
         Generating NVIDIA GPU code
s_quad:
   1099, Generating acc routine seq
         Generating NVIDIA GPU code
nvfortran-Fatal-/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/compilers/bin/tools/fort2 TERMINATED by signal 11
Arguments to /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/compilers/bin/tools/fort2
/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/compilers/bin/tools/fort2 /tmp/nvfortranmhIKgCQT5tQ.ilm -fn /home/henryleberre/MFC/src/common/autogen/m_global_parameters.f90 -debug -x 120 0x200 -x 123 0x400 -opt 0 -terse 1 -inform warn -x 51 0x20 -x 119 0xa10000 -x 122 0x40 -x 123 0x1000 -x 127 4 -x 127 17 -x 19 0x400000 -x 28 0x40000 -x 120 0x10000000 -x 70 0x8000 -x 122 1 -x 125 0x20000 -quad -x 59 4 -tp haswell -x 124 0x1400 -y 15 2 -x 57 0x3b0000 -x 58 0x48000000 -x 49 0x100 -astype 0 -x 121 1 -x 183 4 -x 121 0x800 -x 68 0x1 -x 8 0x40000000 -x 70 0x40000000 -x 56 0x10 -x 54 0x10 -x 120 0x2000000 -x 120 0x2000000 -x 249 140 -x 68 0x20 -x 70 0x40000000 -x 8 0x40000000 -x 164 0x800000 -x 71 0x2000 -x 71 0x4000 -x 34 0x40000000 -x 83 0x1 -x 85 0x1 -x 206 0x02 -x 68 0x1 -x 39 4 -x 56 0x10 -x 26 0x10 -x 26 1 -x 56 0x4000 -x 124 1 -accel tesla -accel host -x 197 0 -x 175 0 -x 203 0 -x 204 0 -x 180 0x4000400 -x 121 0xc00 -x 186 0x80 -x 180 0x4000400 -x 121 0xc00 -x 194 0x40000 -x 163 0x1 -x 186 0x80000 -cudaver 11070 -x 176 0x100 -cudacap 35 -cudacap 50 -cudacap 60 -cudacap 61 -cudacap 70 -cudacap 75 -cudacap 80 -cudacap 86 -cudaroot /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7 -x 189 0x8000 -y 163 0xc0000000 -x 163 0x800000 -x 189 0x10 -y 189 0x4000000 -cudaroot /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7 -x 187 0x40000 -x 187 0x8000000 -x 60 512 -x 0 0x1000000 -x 2 0x100000 -x 0 0x2000000 -x 161 16384 -x 162 16384 -x 124 0x20 -x 62 8 -cci /tmp/nvfortranShIe78ZaBXo.cci -cmdline '+nvfortran /home/henryleberre/MFC/src/common/autogen/m_global_parameters.f90 -I/home/henryleberre/MFC/build/install/include -isystem /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/comm_libs/openmpi/openmpi-3.1.5/include -isystem /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/comm_libs/openmpi/openmpi-3.1.5/lib -isystem /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/include -isystem /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/math_libs/include -g -gopt -r8 -cpp -Mpreprocess -Mfreeform -lcutensor -Minfo=accel -Mr8intrinsics -fPIC -acc -Mpreprocess -c -o CMakeFiles/simulation.dir/__/common/autogen/m_global_parameters.f90.o' -stbfile /tmp/nvfortranmhIKtK5KqRN.stb -asm /tmp/nvfortranmhIKD56NGRl.ll
make[3]: *** [src/simulation_code/CMakeFiles/simulation.dir/build.make:145: src/simulation_code/CMakeFiles/simulation.dir/__/common/autogen/m_global_parameters.f90.o] Error 127
make[2]: *** [CMakeFiles/Makefile2:113: src/simulation_code/CMakeFiles/simulation.dir/all] Error 2
make[1]: *** [CMakeFiles/Makefile2:120: src/simulation_code/CMakeFiles/simulation.dir/rule] Error 2
make: *** [Makefile:169: simulation] Error 2
```

Other online posts I have seen on this issue referenced earlier versions of NVHPC and all seemed to be associated with a compiler bug.

The simplest way to replicate this error would be, provided Python 3.8 or newer is installed, to:

+ `git clone https://github.com/henryleberre/MFC`
+ `cd MFC`
+ `pip3 install pyyaml rich fypp`
+ `pip3 install -e toolchain/`
+ `mkdir build`
+ `python3 toolchain/mfc/main.py test -j $(nproc) -m release-gpu -o 5EB1467A`

This is (thankfully) not how regular users would interact with MFC but this alternative will produce the error with fewer steps. The last command instructs the code to run the first test case with OpenACC enabled. We use a preprocessor for Fortran (Fypp) that converts .fpp files into autogen/.f90 files. The case that a user wishes to run is passed to the FYPP prior to compilation.

The last command will compile (and run) the pre_process code first, and then attempt to build the “simulation” code. Once an error occurs, the entire output will be printed to the console. You can now henceforth run only the simulation component of MFC on this test case with

```console
./mfc.sh run tests/5EB1467A/case.json -j 8 -t simulation
```

I would greatly appreciate any help you could provide us. If this is indeed a compiler bug, are there ways to circumvent it?

