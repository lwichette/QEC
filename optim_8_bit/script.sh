#!/bin/bash
./cuIsing --X 24 --Y 24 --prob 0.1 --step 0.1 --nie 100 --up 0 --nit 1024 --nw 250000 --temp 1.5 --ndev 1 --leave_out 1 --v 1 --folder test --binning_order 10
./cuIsing --X 24 --Y 24 --prob 0.1 --step 0.1 --nie 100 --up 0 --nit 2048 --nw 250000 --temp 1.5 --ndev 1 --leave_out 1 --v 1 --folder test --binning_order 11
./cuIsing --X 24 --Y 24 --prob 0.1 --step 0.1 --nie 100 --up 0 --nit 4096 --nw 250000 --temp 1.5 --ndev 1 --leave_out 1 --v 1 --folder test --binning_order 12
./cuIsing --X 24 --Y 24 --prob 0.1 --step 0.1 --nie 100 --up 0 --nit 8192 --nw 250000 --temp 1.5 --ndev 1 --leave_out 1 --v 1 --folder test --binning_order 12
./cuIsing --X 24 --Y 24 --prob 0.1 --step 0.1 --nie 100 --up 0 --nit 16384 --nw 250000 --temp 1.5 --ndev 1 --leave_out 1 --v 1 --folder test --binning_order 12
./cuIsing --X 24 --Y 24 --prob 0.1 --step 0.1 --nie 100 --up 0 --nit 32768 --nw 250000 --temp 1.5 --ndev 1 --leave_out 1 --v 1 --folder test --binning_order 12
