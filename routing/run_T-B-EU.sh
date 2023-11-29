#!/bin/bash

declare -a tm=(peak offpeak)


nohup python T-B-EU.py --tm ${tm[0]}> ${tm[0]}_T-B-EU & 



