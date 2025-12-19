## sudo - run with administrative privileges 
## apt - Ubuntu package manager
## update - download latest information about avaliable packages
## upgrade - upgrade installed packages to newest versions
## -y - assume yes to everything

## sudo add-apt-repository ppa:graphics-drivers/ppa - added new software repositroy to my system specifically the Ubuntu Graphics Drivers PPA for NVIDIA

## nvidia-smi - gives GPU information
## nvcc - nvidia compiler for cuda code
    - compiles c++ code with c++ compiler
    - compiles GPU kerneles
## -o device_query - name the output executable device_query

## major.minor = 8.6 for example and this is computable capability (what features the GPU has and how code should be compiled for it) 
# major is what arhitecture and what big features exist
# minor how thoese features are implemented