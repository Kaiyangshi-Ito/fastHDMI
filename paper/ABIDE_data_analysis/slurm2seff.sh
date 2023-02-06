#!/bin/bash

# Loop through all files that match the pattern "slurm-*.out"
for file in slurm-*.out; do
  # Extract the number from the filename
  number=$(echo "$file" | grep -o '[0-9]\+')

  # Run the seff command and redirect the output to a new file
  seff "$number" >> "seff-${number}.out"
done
