#!/bin/bash

declare -a tm=(peak offpeak)

nohup python V-B-P.py --tm ${tm[0]}> ${tm[0]}_V-B-P & 


