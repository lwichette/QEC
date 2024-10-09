#!/bin/bash

filename="$1"

line_count=$(awk 'NR > 1 {print $1}' ../configs/${filename}.txt | sort -n | tail -1)

echo "The file '$filename' has $line_count lines."

sbatch --array=1-1 $filename.slr
