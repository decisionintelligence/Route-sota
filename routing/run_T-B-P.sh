#!/bin/bash

declare -a tm=(peak offpeak)

nohup python T-B-P.py --tm ${tm[0]}> ${tm[0]}_T-B-P & 


