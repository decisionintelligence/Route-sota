#!/bin/bash

declare -a sigma=(30 60 120 240)
declare -a eta=(333 170 83 41)


for ((n=0; n<=3; n++))
do
    echo ${sigma[$n]}
    echo ${eta[$n]}
    nohup time python T-BS.py --sig ${sigma[$n]} --eta ${eta[$n]} --tm peak  > peak_crout_d"${sigma[$n]}"_1 & 
done

