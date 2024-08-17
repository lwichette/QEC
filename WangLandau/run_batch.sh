#!/bin/bash

SECONDS=0

#Note: should make sure we have "fresh" histograms from the prerun, probably?
beta=0.0001

for seed in {100..500}
do

for size in {12,14}
do

xval=$size
yval=$size

#Might adjust later
if [[ $size > 10 ]]; then
intervals=20
else
intervals=5
fi

echo $seed

./prerun_-10 -x $xval -y $yval -p 0.107 -n 100000 -l 10 -w 16 -s $seed -i $intervals -e "I" -b 0
wait
./wl_-10 -x $xval -y $yval -n 100000 -p 0.107 -a 0.8 -b $beta -i $intervals -w 8 -o 0.25 -s $seed -e "I" -t 0 -h $seed
wait

./prerun_-10 -x $xval -y $yval -p 0.107 -n 100000 -l 10 -w 16 -s $seed -i $intervals -e "X" -b 0
wait
./wl_-10 -x $xval -y $yval -n 100000 -p 0.107 -a 0.8 -b $beta -i $intervals -w 8 -o 0.25 -s $seed -e "X" -t 0 -h $seed
wait

./prerun_-10 -x $xval -y $yval -p 0.107 -n 100000 -l 10 -w 16 -s $seed -i $intervals -e "Z" -b 0
wait
./wl_-10 -x $xval -y $yval -n 100000 -p 0.107 -a 0.8 -b $beta -i $intervals -w 8 -o 0.25 -s $seed -e "Z" -t 0 -h $seed
wait

./prerun_-10 -x $xval -y $yval -p 0.107 -n 100000 -l 10 -w 16 -s $seed -i $intervals -e "Y" -b 0
wait
./wl_-10 -x $xval -y $yval -n 100000 -p 0.107 -a 0.8 -b $beta -i $intervals -w 8 -o 0.25 -s $seed -e "Y" -t 0 -h $seed
wait
echo "Done"

done
done
duration=$SECONDS
echo "Total runtime: $((duration / 60)) minutes and $((duration % 60)) seconds."