import subprocess
import sys

day=sys.argv[1]
seq=sys.argv[2]

print("Starting data sampling")
subprocess.run(['python3','sample_cadcd.py',day,seq])

print("Starting pre processing")
subprocess.run(['python3','pre_processing.py',day,seq])

print("Starting lidar autolabeling")
subprocess.run(['python3','lidar_autolabeling.py',day,seq])


print("Starting dino autolabeling")
subprocess.run(['python3','dino_autolabeling.py',day,seq])

print("Starting post processing")
subprocess.run(['python3','post_processing.py',day,seq])
