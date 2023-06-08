# OLCFDEV-infinity

One can reproduce many of CCE's Cray Fortran compiler bugs while running a
basic MFC example case on OLCF Crusher with a `--gpu` build of MFC.

```console
$ git clone -b amdgpu git@github.com:henryleberre/MFC.git
$ cd MFC && . ./mfc.sh load -c c -m g
$ ./mfc.sh run examples/1D_bubblescreen/case.py -t pre_process simulation -j 8 --gpu --debug
```

Notes:

- Running `pre_process` is only necessary when running for the first time (this setting is saved).
- Including `--gpu --debug` is only necessary when running for the first time (this setting is saved).
- `. ./mfc.sh load -c c -m g` loads the appropriate modules for the Crusher (c) computer in GPU (g) mode.
- There is a separate build command available: `./mfc.sh build -t pre_process simulation --gpu --debug`.
- The use of `--debug` is recommended (see CMakeLists.txt).
