#!/bin/bash

declare -a sigma=60
declare -a eta=170
declare -a tm=(peak, offpeak)

echo ${sigma}
echo ${eta}
nohup time python V-BS.py --sig ${sigma} --eta ${eta} --tm ${tm[0]}  > ${tm[0]}_crout_d"${sigma}"_1 & 


