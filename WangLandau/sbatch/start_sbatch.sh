#!/bin/bash

filename="$1"

line_count=$(wc -l < "../configs/${filename}.txt")

line_count=$((line_count - 1))

echo "The file '$filename' has $line_count lines."

sbatch --array=1-4 $filename.slr
