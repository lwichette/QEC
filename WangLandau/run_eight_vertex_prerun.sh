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

num_interactions=2

replica_exchange_steps=50

intervals_wl=10

iterations=10000

time_limit=600

for size in 4
do
    xval=$size
    yval=$size
    timeout $time_limit ./prerun_eight_vertex_-10 -x $xval -y $yval --prob_x 0.1 --prob_y 0.1 --prob_z 0.1 --nit $iterations --nl 100 --num_walker_total 10 --seed $seed_hist --num_intervals 3  --replicas $num_interactions
    if [ $? -eq 124 ]; then
        echo "prerun timed out after $time_limit seconds."
    fi

    seed_run=$(($seed_run + 1))

    echo "Done with size $size, probability $probability, error type $error_type"
done

duration=$SECONDS
echo "Total runtime: $((duration / 60)) minutes and $((duration % 60)) seconds."