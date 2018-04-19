#!/bin/bash

source ~/.bashrc
source activate emerge
export PROJECTROOT="~/emergent_comms_negotiation"
export PYTHONPATH=$PROJECTROOT:$PYTHONPATH

python ecn.py --enable-cuda
