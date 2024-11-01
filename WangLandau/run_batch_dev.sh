#!/bin/bash

LOGFILE="script.log"
exec > >(tee -a "$LOGFILE") 2>&1

SECONDS=0

slurm_job_array_id=1

alpha=0.8

walker_wl=8

overlap_wl=0.25

seed_hist=1

seed_run=42

num_interactions=1

replica_exchange_steps=20

boundary_type=0

intervals_wl=10

iterations=1000

time_limit=3600

num_loops=100

num_walker_prerun=250

num_intervals_prerun=20

end_seed=$((seed_hist + num_interactions - 1))

case $boundary_type in
    1)
        result="open"
        ;;
    0)
        result="periodic"
        ;;
    2)
        result="cylinder"
        ;;
esac

for beta in 0.00000001
  do
  for probability in 0.05
    do
      for xval in 5
      do
        for yval in 5
        do
          for error_type in X
          do

            ./prerun_-10 -x $xval -y $yval -p $probability -n $iterations -l $num_loops -w $num_walker_prerun -s $seed_hist -i $num_intervals_prerun -e "$error_type" -b $boundary_type -r $num_interactions -d $slurm_job_array_id

            timeout $time_limit ./wl_-10 -x $xval -y $yval -n $iterations -p $probability -a $alpha -b $beta -i $intervals_wl -w $walker_wl -o $overlap_wl -s $seed_run -e "$error_type" -t $boundary_type -h $seed_hist -r $num_interactions -c $replica_exchange_steps -d $slurm_job_array_id
            if [ $? -eq 124 ]; then
                echo "wl_-10 timed out after $time_limit seconds."
            fi

            # seed_run=$(($seed_run + 1))

            # echo "Done with size $size, probability $probability, error type $error_type"

            # formatted_prob=$(LC_NUMERIC=C printf "%.6f" "$probability")

            # base_dir="init/task_id_${slurm_job_array_id}/${result}/prob_${formatted_prob}/X_${xval}_Y_${yval}/error_class_${error_type}"

            # python3 delete_folders.py $base_dir $seed_hist $end_seed

          done
        done
      done
  done
done

duration=$SECONDS
echo "Total runtime: $((duration / 60)) minutes and $((duration % 60)) seconds."
