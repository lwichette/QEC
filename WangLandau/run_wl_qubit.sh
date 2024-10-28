#!/bin/bash

LOGFILE="script.log"
exec > >(tee -a "$LOGFILE") 2>&1

SECONDS=0

slurm_job_array_id=2

alpha=0.8

walker_wl=4

overlap_wl=0.25

seed_hist=1

seed_run=1

num_interactions=10

replica_exchange_steps=20

boundary_type=0

intervals_wl=20

iterations=1000

time_limit=3600

num_loops=200

num_walker_prerun=128

num_intervals_prerun=30

hist_scale=1

error_variance=0.0001

beta=0.00000001

for error_mean in 0.1
    do
    for size in 6
        do
        for error_type in I
            do
                ./prerun_qubit_-10 -x $size -y $size -n $iterations -l $num_loops -w $num_walker_prerun -s $seed_hist -i $num_intervals_prerun -e "$error_type" -b $boundary_type -r $num_interactions -d $slurm_job_array_id -m $error_mean -v $error_variance -h $hist_scale

                timeout $time_limit ./wl_qubit_-10 -x $size -y $size -n $iterations -a $alpha -b $beta -i $intervals_wl -w $walker_wl -o $overlap_wl -h $seed_hist -s $seed_run -e "$error_type" -t $boundary_type -r $num_interactions -c $replica_exchange_steps -d $slurm_job_array_id -m $error_mean -v $error_variance

                if [ $? -eq 124 ]; then
                    echo "timed out after $time_limit seconds."
                fi

                seed_run=$(($seed_run + 1))

                echo "Done with size $size, error_mean $error_mean, error type $error_type"

                # base_dir="init/task_id_${slurm_job_array_id}"

                # rm -r $base_dir
            done
        done
    done

duration=$SECONDS
echo "Total runtime: $((duration / 60)) minutes and $((duration % 60)) seconds."

