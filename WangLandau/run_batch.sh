#!/bin/bash

SECONDS=0


iterations=1000
preruniterations=10000
beta=0.000001
repetitions=100

intervals=40

for seed in {100,200} 
do

for probability in {0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12}
do

# for size in {8}
# do
size=8

xval=$size

yval=$size

echo $seed

./prerun_-10 -x $xval -y $yval -p $probability -n $preruniterations -l 10 -w 16 -s $seed -i $intervals -e "I" -b 0 -r $repetitions

wait

./wl_-10 -x $xval -y $yval -n $iterations -p $probability -a 0.8 -b $beta -i $intervals -w 8 -o 0.25 -s 0 -e "I" -t 0 -h $seed -r $repetitions -c 4

wait


./prerun_-10 -x $xval -y $yval -p $probability -n $preruniterations -l 10 -w 16 -s $seed -i $intervals -e "X" -b 0 -r $repetitions

wait

./wl_-10 -x $xval -y $yval -n $iterations -p $probability -a 0.8 -b $beta -i $intervals -w 8 -o 0.25 -s 0 -e "X" -t 0 -h $seed -r $repetitions -c 4

wait

./prerun_-10 -x $xval -y $yval -p $probability -n $preruniterations -l 10 -w 16 -s $seed -i $intervals -e "Z" -b 0 -r $repetitions

wait

./wl_-10 -x $xval -y $yval -n $iterations -p $probability -a 0.8 -b $beta -i $intervals -w 8 -o 0.25 -s 0 -e "Z" -t 0 -h $seed -r $repetitions -c 4

wait

./prerun_-10 -x $xval -y $yval -p $probability -n $preruniterations -l 10 -w 16 -s $seed -i $intervals -e "Y" -b 0 -r $repetitions

wait

./wl_-10 -x $xval -y $yval -n $iterations -p $probability -a 0.8 -b $beta -i $intervals -w 8 -o 0.25 -s 0 -e "Y" -t 0 -h $seed -r $repetitions -c 4

wait

echo "Done"

# done

duration=$SECONDS

echo "Total runtime: $((duration / 60)) minutes and $((duration % 60)) seconds."

done

done