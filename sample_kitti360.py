import numpy as np
import os
import matplotlib.pyplot as plt
import open3d
import cv2
import yaml
import utils
import shutil
import time
import sys
from datetime import datetime

def combine_pose_files(
    input_dir: str  #path to directory containing pose csv files
) ->np.ndarray:     # shape (number of poses,3). lat,lon,azimuth for each pose
    
    """
    Combines poses from diffefent files to single array
    """    

    combined_poses=[]
    num_files=len(os.listdir(input_dir))
    for i in range(num_files):
        file=str(i).zfill(10)+'.txt'
        while not os.path.exists(os.path.join(input_dir,file)):
            i-=1
            file=str(i).zfill(10)+'.txt'
        poses=np.loadtxt(os.path.join(input_dir,file),delimiter=' ',dtype='float')
        poses=poses[[0,1,5]]
        poses[-1]= 90-np.rad2deg(poses[-1]) #change to azimuth
        combined_poses.append(poses)
    return np.array(combined_poses)


def get_quadrant(
    point: np.ndarray #shape: (3). lidar point. x,y.z
) -> int: #points quadrant
    
    '''
    Define points quadrant. Original source: https://github.com/VincentCheungM/Run_based_segmentation/issues/3#issuecomment-452686184 
    '''
    res = 0
    x = point[0]
    y = point[1]
    if x > 0 and y >= 0:
        res = 1
    elif x <= 0 and y > 0:
        res = 2
    elif x < 0 and y <= 0:
        res = 3
    elif x >= 0 and y < 0:
        res = 4
    return res

def separate_scan_rings(
    cartesian_coords: np.ndarray    #shape: (number of points, 3). x,y,z for each point. 
) ->np.ndarray:                     #shape: (number of rings). Pointcloud organized to scan rings. Each element contains list of points in that scan ring.
    
    """
    separates scan into scan rings based on the point order. Original source: https://github.com/VincentCheungM/Run_based_segmentation/issues/3#issuecomment-452686184 
    """
    grouped_coords = [[] for _ in range(64)]
    num_of_points = cartesian_coords.shape[0]
    velodyne_rings_count = 64
    previous_quadrant = 0
    ring = 0

    for num in range(num_of_points-1,-1,-1):
        quadrant = get_quadrant(cartesian_coords[num])
        if ring < velodyne_rings_count-1 and quadrant == 4 and previous_quadrant == 1:
            ring += 1
        grouped_coords[ring].append(cartesian_coords[num])
        previous_quadrant = quadrant
    

    for i in range(64):
        if len(grouped_coords[i])==0:
            grouped_coords[i].append(np.empty(3))
        grouped_coords[i] = np.array(grouped_coords[i])
    
    grouped_coords=np.array(grouped_coords,dtype='object')

    return grouped_coords[7:]

def main():

    seq=sys.argv[1]
    param_file=sys.argv[2]

    #read sampling params
    with open(param_file) as file:
        params=yaml.safe_load(file)
        config=params['data_sampling']
        dataset_path=params['dataset_path']
        dist_between_samples=config['dist_between_samples']
        trajectory_length=params['pre_processing']['trajectory_length']

    output_path=os.path.join(dataset_path,'processed',seq,'data')
    os.makedirs(output_path,exist_ok=True)

    #Input paths
    input_pcd_path=os.path.join(dataset_path,'raw','data_3d_raw',seq+'_sync','velodyne_points','data')
    input_gnss_path=os.path.join(dataset_path,'raw','data_poses_oxts','data_poses',seq+'_sync','oxts','data')
    input_camera_path=os.path.join(dataset_path,'raw','data_2d_raw',seq+'_sync','image_00','data_rect')


    GT_label_path=os.path.join(dataset_path,'raw','data_2d_semantics','train',seq+'_sync','image_00','semantic_rgb')

    #Output paths
    output_pcd_path=os.path.join(output_path,'scans')
    output_camera_path=os.path.join(output_path,'imgs')
    output_gnss_path=os.path.join(output_path,'poses.csv')
    output_gnss_id_path=os.path.join(output_path,'pose_ids.txt')

    # gnns data combined to single file
    gnss_data=combine_pose_files(input_gnss_path)

    np.savetxt(output_gnss_path,gnss_data,delimiter=',')

    #create dirs if doesnt exist
    os.makedirs(output_pcd_path,exist_ok=True)
    os.makedirs(output_camera_path,exist_ok=True)


    gnss_id=0
    data_id=0
    gnss_data[:,0:2] = utils.latlon2utm(gnss_data[:,0:2])


    # we want to cleanly exit if the user stops execution with keyboard interrupt
    with open(output_gnss_id_path,'w') as file:

        while True:

            #check if GT label is available. If not incrase by one until found.
            GT_label=False
            while not GT_label:
                if gnss_id>=len(gnss_data):
                    break

                if os.path.exists(os.path.join(GT_label_path,str(gnss_id).zfill(10)+'.png')):
                    GT_label=True
                else:
                    gnss_id += 1

            # If pose data for 100 following meters doesnt exist exit
            if not utils.take_following_N_m(gnss_data[gnss_id:,:2],trajectory_length):
                print("Exiting")
                break
            
            file.write(str(gnss_id)+'\n')

            # Read lidar scan corresponding to the current gnss_id and separate scan rings based on polar angle
            scan_data = np.fromfile(os.path.join(input_pcd_path,str(gnss_id).zfill(10)+'.bin'), dtype=np.float32)
            scan_data = scan_data.reshape((-1, 4))[:,:3]
            scan_rings=separate_scan_rings(scan_data)
            np.save(os.path.join(output_pcd_path,str(data_id)+'.npy'),scan_rings)

            # Copy the image corresponding to the current gnss_id
            shutil.copy(os.path.join(input_camera_path,str(gnss_id).zfill(10)+'.png'),os.path.join(output_camera_path,str(data_id)+'.png'))

            gnss_id=gnss_id+max(1,utils.take_following_N_m(gnss_data[gnss_id:],dist_between_samples)) #in every case increase at least by one

            data_id+=1

        
if __name__=="__main__": 
    main()

























