#!/bin/bash

source ~/.bashrc

export PYTHONPATH=$HOME/Python3/bin

multiple_to_take=6
host=$HOSTNAME
num=${host:6}
num=$((num % multiple_to_take))

i=0
while read line; do
    i=$((i + 1))
    if [[ $((i % multiple_to_take)) == "$num" ]]; then
        echo "run line $i"
        python3.6 -m VGGsiamese $line  # > "run_${host}.out"
    else
        echo "skip $i as '$num' != '$((i % multiple_to_take))'"
    fi
done < cmd.txt

