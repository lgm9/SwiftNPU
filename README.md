# SwiftNPU
Source Code for "SwiftNPU: Scalable Shape-Flexible Allocation for Inter-Core Connected NPUs" (EuroMLsys '26)
[Paper link](https://dl.acm.org/doi/10.1145/3805621.3807614)

This code is based on Tenstorrent Blackhole p150a device, but is easily adaptable to other Tenstorrent devices.

## Setup
First, init the submodule & build tt-metal 
```bash
git submodule update --init --recursive
cd tt-metal
cmake -S . -B build
cd ..
```

Now, build the main project. You might need to run make build several times if error comes out.
```bash
make configure
make build
```

## Algorithm overhead test

Move to the algorithm_test directory.
```
Run ./build.sh
```
Scripts are based on 2.9GHz, you could change the clock frequency by --ghz option on the build.sh file.

## Main test

Move to the root directory.

First, make shapes.txt from running the gen_shapes.py under the scripts/swiftnpu_euromlsys.

```
python3 ./scripts/swiftnpu_euromlsys/gen_shapes.py
```
You should run on the root directory with the command above.
This will produce 500 randomly generated shapes.
Initially, this will produce the 'MIXED' workload.
You could use other workload by changing the list on this python script.

After generating the input shapes, running the command below will print out the results on /results directory.
```
export TT_METAL_HOME="$HOME/SwiftNPU/tt-metal" # Directory of SwiftNPU/tt-metal
env TT_METAL_HOME="$TT_METAL_HOME" make run_SwiftNPU
```
Default algorithm for this run is First-Fit.
You could change the allocation algorithm by changing the alloc_algorithm option from the makefile.

Options for the alloc_algorithm are:
0 - FF
1 - BF
2 - LSSA
3 - ASFF
4 - ASBF
5 - NAS
6 - GED
