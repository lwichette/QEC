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

beta=0.00000001

walker_wl=5

overlap_wl=0.25

seed_hist=1

seed_run=1

num_interactions=500

replica_exchange_steps=20

intervals_main=32

iterations=1000

time_limit=3600

histogram_scale=1

qubit_specific_noise=0

end_seed=$((seed_hist + num_interactions - 1))

for prob in 0.08 0.1 0.12 0.13
do
    for size in 4 6
    do
        xval=$size
        yval=$size
        for xh_err in 0 1
        do
            for xv_err in 0 1
            do
                for zh_err in 0
                do
                    for zv_err in 0
                    do
                        ./prerun_eight_vertex_-10 -x $xval -y $yval --prob_x $prob --prob_y 0.000001 --prob_z 0.000001 --nit 1000 --nl 100 -w 512 --seed $seed_hist --num_intervals 64  --hist_scale $histogram_scale --replicas $num_interactions --x_horizontal_error $xh_err  --x_vertical_error $xv_err  --z_horizontal_error $zh_err --z_vertical_error $zv_err

                        timeout $time_limit ./main_eight_vertex_-10 -a $alpha -b $beta -c $xh_err -d $xv_err -e $zh_err -f $zv_err --prob_x $prob --prob_y 000001 --prob_z 000001 --replica_exchange_offsets $replica_exchange_steps --num_intervals $intervals_main --num_iterations $iterations --overlap_decimal $overlap_wl --seed_histogram $seed_hist --seed_run $seed_run --num_interactions $num_interactions --walker_per_interval $walker_wl -x $xval -y $yval
                        if [ $? -eq 124 ]; then
                            echo "prerun timed out after $time_limit seconds."
                        fi

                        seed_run=$(($seed_run + 1))

                        init_dir="init"
                        rm -rf "$init_dir"/*

                        echo "Done with size $size, probability $prob, error type $xh_err $xv_err $zh_err $zv_err, $(($SECONDS / 60)) minutes"
                    done
                done
            done
        done
    done
done

duration=$SECONDS
echo "Total runtime: $((duration / 60)) minutes and $((duration % 60)) seconds."
