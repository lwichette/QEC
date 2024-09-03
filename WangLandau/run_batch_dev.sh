#!/bin/bash

SECONDS=0

alpha=0.8

beta=0.0001

walker_wl=8

overlap_wl=0.25

seed_hist=1

seed_run=1000

num_interactions=500

replica_exchange_steps=50

boundary_type=0

intervals_wl=10

iterations=1000

time_limit=360  # Time limit in seconds

for probability in 0.02 0.06 0.08 0.12
  do
    for size in 4 6 10
    do
      xval=$size
      yval=$size
      for error_type in I X Y Z
      do
        timeout $time_limit ./prerun_-10 -x $xval -y $yval -p $probability -n $iterations -l 100 -w 100 -s $seed_hist -i 20 -e "$error_type" -b $boundary_type -r $num_interactions
        if [ $? -eq 124 ]; then
            echo "prerun timed out after $time_limit seconds."
        fi

        timeout $time_limit ./wl_-10 -x $xval -y $yval -n $iterations -p $probability -a $alpha -b $beta -i $intervals_wl -w $walker_wl -o $overlap_wl -s $seed_run -e "$error_type" -t $boundary_type -h $seed_hist -r $num_interactions -c $replica_exchange_steps
        if [ $? -eq 124 ]; then
            echo "wl_-10 timed out after $time_limit seconds."
        fi

        seed_run=$(($seed_run + 1))

        echo "Done with size $size, probability $probability, error type $error_type"
      done
    done
done

# Calculate and display the total runtime
duration=$SECONDS
echo "Total runtime: $((duration / 60)) minutes and $((duration % 60)) seconds."