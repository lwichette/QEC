#!/bin/bash

filename="$1"

line_count=$(wc -l < "../configs/${filename}.txt")

echo "The file '$filename' has $line_count lines."

sbatch --array=1-$line_count $filename.slr