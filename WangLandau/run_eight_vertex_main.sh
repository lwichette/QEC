#!/bin/bash

# # This block triggers decoupled scenario from Flammia Chubb paper when handing two params to shell execution from cli

# prob_x_default=0.1
# prob_z_default=0.1
# prob_y_default=0.1

# # Check if arguments for pX pZ are provided for decoupled scenario
# if [ "$#" -eq 2 ]; then
#     echo "Usage: $0 pX pZ"

#     # Read input parameters
#     pX=$1
#     pZ=$2

#     prob_i=$(echo "(1 - $pX) * (1 - $pZ)" | bc -l)
#     prob_x=$(echo "$pX * (1 - $pZ)" | bc -l)
#     prob_z=$(echo "(1 - $pX) * $pZ" | bc -l)
#     prob_y=$(echo "$pX * $pZ" | bc -l)

#     echo "Decoupled scenario with provided arguments: pX = $pX and pZ = $pZ."
# else
#     prob_x=$prob_x_default
#     prob_z=$prob_z_default
#     prob_y=$prob_y_default

#     echo "Non decoupled scenario."
# fi

LOGFILE="script.log"
exec > >(tee -a "$LOGFILE") 2>&1

SECONDS=0

alpha=0.8

beta=0.000001

walker_wl=1

overlap_wl=0.25

seed_hist=1

seed_run=1000

num_interactions=1

replica_exchange_steps=10

intervals_main=10

iterations=1000

time_limit=600

x_horizontal_error=0

x_vertical_error=0

z_horizontal_error=0

z_vertical_error=0

error_mean=0.3

error_variance=0.001

histogram_scale=1

prob_x=0.1

prob_z=0.1

prob_y=0.1

xval=4

yval=4

# ./prerun_eight_vertex_-10 -x $xval -y $yval --prob_x $prob_x --prob_y $prob_y --prob_z $prob_z --nit 10000 --nl 1000 -w 512 --seed $seed_hist --num_intervals 20  --hist_scale $histogram_scale --replicas $num_interactions --x_horizontal_error $x_horizontal_error  --x_vertical_error $x_vertical_error  --z_horizontal_error $z_horizontal_error --z_vertical_error $z_vertical_error  --error_mean $error_mean --error_variance $error_variance

timeout $time_limit ./main_eight_vertex_-10 -a $alpha -b $beta -c $x_horizontal_error -d $x_vertical_error -e $z_horizontal_error -f $z_vertical_error -g $prob_x -h $prob_y -i $prob_z --replica_exchange_offsets $replica_exchange_steps --num_intervals $intervals_main --num_iterations $iterations --overlap_decimal $overlap_wl --seed_histogram $seed_hist --seed_run 42 --num_interactions $num_interactions --error_mean $error_mean --error_variance $error_variance --walker_per_interval $walker_wl -x $xval -y $yval
if [ $? -eq 124 ]; then
    echo "prerun timed out after $time_limit seconds."
fi

# seed_run=$(($seed_run + 1))

duration=$SECONDS
echo "Total runtime: $((duration / 60)) minutes and $((duration % 60)) seconds."