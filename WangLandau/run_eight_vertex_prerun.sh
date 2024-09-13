#!/bin/bash

LOGFILE="script.log"
exec > >(tee -a "$LOGFILE") 2>&1

SECONDS=0

alpha=0.8

beta=0.000001

walker_wl=5

overlap_wl=0.25

seed_hist=1

seed_run=1000

num_interactions=3

replica_exchange_steps=50

intervals_wl=10

iterations=10000

time_limit=600

xval=4
yval=8

timeout $time_limit ./prerun_eight_vertex_-10 -x $xval -y $yval --prob_x 0.1 --prob_y 0.1 --prob_z 0.1 --nit 1000 --nl 1000 -w 128 --seed $seed_hist --num_intervals 20  --hist_scale 1 --replicas $num_interactions
if [ $? -eq 124 ]; then
    echo "prerun timed out after $time_limit seconds."
fi

seed_run=$(($seed_run + 1))

duration=$SECONDS
echo "Total runtime: $((duration / 60)) minutes and $((duration % 60)) seconds."