import subprocess
import sys

seq=sys.argv[1]

#print("Starting data sampling")
#subprocess.run(['python3','sample_kitti360.py',seq])

print("Starting pre processing")
subprocess.run(['python3','pre_processing.py',seq])

print("Starting lidar autolabeling")
subprocess.run(['python3','lidar_autolabeling.py',seq])

print("Starting camera autolabeling")
subprocess.run(['python3','camera_autolabeling.py',seq])

print("Starting post processing")
subprocess.run(['python3','post_processing.py',seq])
