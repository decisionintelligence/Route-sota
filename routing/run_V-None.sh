#!/bin/bash

declare -a tm=(peak offpeak)

nohup python V-None.py --tm ${tm[0]}> ${tm[0]}_V-None & 


