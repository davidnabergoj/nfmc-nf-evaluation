#!/usr/bin/env sh



# Print total jobs
echo "Total jobs: $(squeue --me | tail -n +2 | wc -l)"

# Print number of running jobs
echo "Running jobs: $(squeue --me -t RUNNING | tail -n +2 | wc -l)"
squeue --me --format="%j %i" --state RUNNING | tail -n +2 | cut -f 1 -d'-' | sort | uniq -c

# Print number of pending jobs
echo "Pending jobs: $(squeue --me -t PENDING | tail -n +2 | wc -l)"
squeue --me --format="%j %i" --state PENDING | tail -n +2 | cut -f 1 -d'-' | sort | uniq -c

echo "-------------------------------------"
echo "Running jobs head:"
echo "$(squeue --me --sort=+j -t RUNNING --format="%.8i %.9P %.25j %.2t %.10M %.5D %R" | head -n 5)"
echo "-------------------------------------"
echo "Pending jobs head:"
echo "$(squeue --me --sort=+j -t PENDING --format="%.8i %.9P %.25j %.2t %.10M %.5D %R" | head -n 5)"
echo "-------------------------------------"

echo "File status:"
counter=1
for d in `echo */*/results */results`;
do
  echo "${counter} {T: $(cat $(dirname $d)/submit.log), L: $(ls $d/../logs | wc -l), R: $(ls $d | wc -l), E: $(ls $d/../errors | wc -l)} [$(dirname $d)]";
  counter=$((counter+1))
done
