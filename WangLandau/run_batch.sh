#!/bin/bash

SECONDS=0

#Note: should make sure we have "fresh" histograms from the prerun
beta=0.000001

for probability in {0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12}
do


seed=400
repetitions=100


for size in {4,6}
do

xval=$size
yval=$size


intervals=10

#10000 used for seeds 100-400, 1000 now
iterations=1000


echo $seed

./prerun_-10 -x $xval -y $yval -p $probability -n $iterations -l 10 -w 16 -s $seed -i $intervals -e "I" -b 0 -r $repetitions
wait
./wl_-10 -x $xval -y $yval -n $iterations -p $probability -a 0.8 -b $beta -i $intervals -w 8 -o 0.25 -s $seed -e "I" -t 0 -h $seed -r $repetitions -c 1
wait
wait

./prerun_-10 -x $xval -y $yval -p $probability -n $iterations -l 10 -w 16 -s $seed -i $intervals -e "X" -b 0 -r $repetitions
wait
./wl_-10 -x $xval -y $yval -n $iterations -p $probability -a 0.8 -b $beta -i $intervals -w 8 -o 0.25 -s $seed -e "X" -t 0 -h $seed -r $repetitions -c 1
wait

./prerun_-10 -x $xval -y $yval -p $probability -n $iterations -l 10 -w 16 -s $seed -i $intervals -e "Z" -b 0 -r $repetitions
wait
./wl_-10 -x $xval -y $yval -n $iterations -p $probability -a 0.8 -b $beta -i $intervals -w 8 -o 0.25 -s $seed -e "Z" -t 0 -h $seed -r $repetitions -c 1
wait

./prerun_-10 -x $xval -y $yval -p $probability -n $iterations -l 10 -w 16 -s $seed -i $intervals -e "Y" -b 0 -r $repetitions
wait
./wl_-10 -x $xval -y $yval -n $iterations -p $probability -a 0.8 -b $beta -i $intervals -w 8 -o 0.25 -s $seed -e "Y" -t 0 -h $seed -r $repetitions -c 1
wait
echo "Done"

done
done

duration=$SECONDS
echo "Total runtime: $((duration / 60)) minutes and $((duration % 60)) seconds."