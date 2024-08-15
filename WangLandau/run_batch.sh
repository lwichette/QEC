#!/bin/bash

SECONDS=0

xval=8
yval=8

for seed in {100..130}
do
echo $seed

./prerun -x $xval -y $yval -p 0.107 -n 100000 -l 10 -w 16 -s $seed -i 20 -e "I" -b 0
wait
./wl -x $xval -y $yval -n 100000 -p 0.107 -a 0.8 -b 0.000001 -i 20 -w 8 -o 0.25 -s $seed -e "I" -t 0 -h $seed
wait

./prerun -x $xval -y $yval -p 0.107 -n 100000 -l 10 -w 16 -s $seed -i 20 -e "X" -b 0
wait
./wl -x $xval -y $yval -n 100000 -p 0.107 -a 0.8 -b 0.000001 -i 20 -w 8 -o 0.25 -s $seed -e "X" -t 0 -h $seed
wait

./prerun -x $xval -y $yval -p 0.107 -n 100000 -l 10 -w 16 -s $seed -i 20 -e "Z" -b 0
wait
./wl -x $xval -y $yval -n 100000 -p 0.107 -a 0.8 -b 0.000001 -i 20 -w 8 -o 0.25 -s $seed -e "Z" -t 0 -h $seed
wait

./prerun -x $xval -y $yval -p 0.107 -n 100000 -l 10 -w 16 -s $seed -i 20 -e "Y" -b 0
wait
./wl -x $xval -y $yval -n 100000 -p 0.107 -a 0.8 -b 0.000001 -i 20 -w 8 -o 0.25 -s $seed -e "Y" -t 0 -h $seed
wait
echo "Done"

done

duration=$SECONDS
echo "Total runtime: $((duration / 60)) minutes and $((duration % 60)) seconds."