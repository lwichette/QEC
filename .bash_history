exit
ls
cd periodic_boundary
ls
exit
exit
which nvcc
ls -ll /usr/local/cuda/lib64
exit
nvidia-toolkit-container
which nvidia-toolkit-container
exit
ls -ll /etc/nvidia-container-runtime/host-files-for-container.d/
cd /etc/
ls
exit
cd /usr/local/cuda
ls
cd lib64
ls
exit
ls
cd periodic_boundary
make
ls
./cluster
vim Makefile 
exit
cd periodic_boundary
make
ldd cluster
exit
ldd cluster
cd periodic_boundary
ldd cluster
exit
