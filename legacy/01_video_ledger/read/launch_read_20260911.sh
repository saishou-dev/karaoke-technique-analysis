#!/bin/bash
cd ~/karaoke-vision
for i in 0 1 2 3; do setsid nohup bash ./run_read_20260911.sh "$i" > /dev/null 2>&1 < /dev/null & done
sleep 3; pgrep -fa "run_read_20260911.sh [0-3]"
