#!/bin/bash

LOGFILE="script.log"
exec > >(tee -a "$LOGFILE") 2>&1

SECONDS=0

alpha=0.8

beta=0.000001

walker_wl=8

overlap_wl=0.25

seed_hist=1000

seed_run=42

num_interactions=1000

replica_exchange_steps=50

boundary_type=0

intervals_wl=10

iterations=1000

time_limit=600

num_loops=200

num_walker_prerun=150

num_intervals_prerun=30

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

for probability in 0.1
  do
    for size in 4
    do
      xval=$size
      yval=$size
      for error_type in I
      do
        # ./prerun_-10 -x $xval -y $yval -p $probability -n $iterations -l $num_loops -w $num_walker_prerun -s $seed_hist -i $num_intervals_prerun -e "$error_type" -b $boundary_type -r $num_interactions

        timeout $time_limit ./wl_-10 -x $xval -y $yval -n $iterations -p $probability -a $alpha -b $beta -i $intervals_wl -w $walker_wl -o $overlap_wl -s $seed_run -e "$error_type" -t $boundary_type -h $seed_hist -r $num_interactions -c $replica_exchange_steps
        if [ $? -eq 124 ]; then
            echo "wl_-10 timed out after $time_limit seconds."
        fi

        seed_run=$(($seed_run + 1))

        echo "Done with size $size, probability $probability, error type $error_type"

        formatted_prob=$(LC_NUMERIC=C printf "%.6f" "$probability")

        base_dir="init/${result}/prob_${formatted_prob}/X_${size}_Y_${size}/error_class_${error_type}"

        # python3 delete_folders.py $base_dir $seed_hist $end_seed

      done
    done
done

duration=$SECONDS
echo "Total runtime: $((duration / 60)) minutes and $((duration % 60)) seconds."
