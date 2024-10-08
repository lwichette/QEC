#!/bin/bash

LOGFILE="script.log"
exec > >(tee -a "$LOGFILE") 2>&1

SECONDS=0

alpha=0.8

task_id=1

beta=0.000001

walker_wl=4

overlap_wl=0.25

seed_hist=1

seed_run=1

num_interactions=500

replica_exchange_steps=20

intervals_main=16

iterations=1000

time_limit=3600

histogram_scale=1

qubit_specific_noise=0

end_seed=$((seed_hist + num_interactions - 1))

for prob in 0.05 0.06 0.08
do
    for size in 4 6
    do
        xval=$size
        yval=$size
        for xh_err in 0 1
        do
            for xv_err in 0 1
            do
                for zh_err in 0 1
                do
                    for zv_err in 0 1
                    do
                        ./prerun_eight_vertex_-10 -x $xval -y $yval --prob_x $prob --prob_y $prob --prob_z $prob --nit 1000 --nl 100 -w 512 --seed $seed_hist --num_intervals 64  --hist_scale $histogram_scale --replicas $num_interactions --x_horizontal_error $xh_err  --x_vertical_error $xv_err  --z_horizontal_error $zh_err --z_vertical_error $zv_err -v $task_id

                        timeout $time_limit ./main_eight_vertex_-10 -a $alpha -b $beta -c $xh_err -d $xv_err -e $zh_err -f $zv_err --prob_x $prob --prob_y $prob --prob_z $prob --replica_exchange_offsets $replica_exchange_steps --num_intervals $intervals_main --num_iterations $iterations --overlap_decimal $overlap_wl --seed_histogram $seed_hist --seed_run $seed_run --num_interactions $num_interactions --walker_per_interval $walker_wl -x $xval -y $yval -v $task_id
                        if [ $? -eq 124 ]; then
                            echo "prerun timed out after $time_limit seconds."
                        fi

                        seed_run=$(($seed_run + 1))

                        rm -rf ./init/*

                        echo "Done with size $size, probability $prob, error type $xh_err $xv_err $zh_err $zv_err, $(($SECONDS / 60)) minutes"
                    done
                done
            done
        done
    done
done

duration=$SECONDS
echo "Total runtime: $((duration / 60)) minutes and $((duration % 60)) seconds."
