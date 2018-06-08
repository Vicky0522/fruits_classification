#########################################################################
# File Name: install_cpu.sh
# Author: ouyangwenqi
# Mail: vinkeyoy@gmail.com
# Created Time: 2018年01月10日 星期三 16时31分03秒
#########################################################################
#!/bin/bash

mkdir -p build

# g++ src/interp_cpu_kernel.cpp -fPIC -shared -o build/interp_cpu_kernel.so
nvcc -c -o build/roi_putin_kernel.o src/cuda/roi_putin_kernel.cu --gpu-architecture=compute_52 --gpu-code=compute_52 --compiler-options -fPIC
g++ -shared -o build/roi_putin_kernel.so build/roi_putin_kernel.o

python build.py



