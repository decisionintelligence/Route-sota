#!/bin/bash

declare -a tm=(peak offpeak)

nohup python T-None.py --tm ${tm[0]}> ${tm[0]}_T-None & 


