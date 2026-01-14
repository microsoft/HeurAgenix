#!/bin/bash

while true; do
    echo "==========================================" >> "gpu.log"
    date >> "gpu.log"
    nvidia-smi >> "gpu.log"
    # Wait for 10 seconds
    sleep 10
done
