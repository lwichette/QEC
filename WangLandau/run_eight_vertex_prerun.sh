#!/bin/bash

prob_x_default=0.0002
prob_z_default=0.0002
prob_y_default=0.0002

# Check if arguments for pX pZ are provided for decoupled scenario
if [ "$#" -eq 2 ]; then
    echo "Usage: $0 pX pZ"

    # Read input parameters
    pX=$1
    pZ=$2

    prob_i=$(echo "(1 - $pX) * (1 - $pZ)" | bc -l)
    prob_x=$(echo "$pX * (1 - $pZ)" | bc -l)
    prob_z=$(echo "(1 - $pX) * $pZ" | bc -l)
    prob_y=$(echo "$pX * $pZ" | bc -l)

    echo "Decoupled scenario with provided arguments: pX = $pX and pZ = $pZ."
else
    prob_x=$prob_x_default
    prob_z=$prob_z_default
    prob_y=$prob_y_default

    echo "Non decoupled scenario."
fi

LOGFILE="script.log"
exec > >(tee -a "$LOGFILE") 2>&1

SECONDS=0

alpha=0.8

beta=0.000001

walker_wl=5

overlap_wl=0.25

seed_hist=1

seed_run=1000

num_interactions=1

replica_exchange_steps=50

intervals_wl=10

iterations=10000

time_limit=600

x_horizontal_error=0

x_vertical_error=0

z_horizontal_error=0

z_vertical_error=0

xval=4

yval=4

timeout $time_limit ./prerun_eight_vertex_-10 -x $xval -y $yval --prob_x $prob_x --prob_y $prob_y --prob_z $prob_z --nit 1000 --nl 100 -w 128 --seed $seed_hist --num_intervals 20  --hist_scale 1 --replicas $num_interactions --x_horizontal_error $x_horizontal_error  --x_vertical_error $x_vertical_error  --z_horizontal_error $z_horizontal_error --z_vertical_error $z_vertical_error
if [ $? -eq 124 ]; then
    echo "prerun timed out after $time_limit seconds."
fi

seed_run=$(($seed_run + 1))

duration=$SECONDS
echo "Total runtime: $((duration / 60)) minutes and $((duration % 60)) seconds."