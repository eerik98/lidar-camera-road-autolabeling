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
) ->np.ndarray:     # shape (number of poses,4). lat,lon,z,azimuth for each pose
    
    """
    Combines poses from diffefent files to single array
    """    

    combined_poses=[]
    num_files=len(os.listdir(input_dir))
    for i in range(num_files):
        file=str(i).zfill(10)+'.txt'
        poses=np.loadtxt(os.path.join(input_dir,file),delimiter=' ',dtype='float')
        poses=poses[[0,1,9]]
        combined_poses.append(poses)
    return np.array(combined_poses)

def dates_to_unix(
    date_string: str    #date as string
) ->int:                #unix timestap
    
    """
    converts date string to unix timestamp
    """

    date_object = datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S.%f')
    unix_timestamp = date_object.timestamp()
    return unix_timestamp

def get_timestamps(
    date_file: str #path to file containing the timestamps as dates
) ->np.ndarray:    #shape(number of timestamps). Timestamps in unix format
    timestamps=[]
    with open(date_file, 'r') as file:
        for line in file:
            date=line.strip()
            date=date[:-3] # accuracy up to microseconds supported by datetime lib. More than enough here. 
            unix_timestamp=dates_to_unix(date)
            timestamps.append(unix_timestamp)
    return np.array(timestamps)

# Assume `cartesian_coords` is an Nx3 array with columns [x, y, z]
def cartesian_to_spherical(
    cartesian_coords: np.ndarray    #shape: (number of points,3). x,y,z for each point. 
) ->np.ndarray:                     #shape (number of points,3). r,azimuth angle,polar angle for each point. 
    
    """
    Transforms cartesian coords to spherical
    """

    x = cartesian_coords[:, 0]
    y = cartesian_coords[:, 1]
    z = cartesian_coords[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    polar_coords = np.stack((r, theta, phi), axis=1)
    
    return polar_coords

def separate_scan_rings(
    cartesian_coords: np.ndarray    #shape: (number of points, 3). x,y,z for each point. 
) ->np.ndarray:                     #shape: (number of rings). Pointcloud organized to scan rings. Each element contains list of points in that scan ring.
    
    """
    separates scan into scan rings based on polar angle.  
    """

    polar_coords = cartesian_to_spherical(cartesian_coords)
    theta = polar_coords[:, 1]
    angles=90-np.rad2deg(theta) # convert from polar angle(rads) to elevation angle(deg)
    #List of known reference values (from VLP-32C datasheet)
    reference_values = [-25.00,-15.64,-11.31,-8.84,-7.25,-6.15,-5.33,-4.67,-4.00,-3.67,-3.33,-3.00,-2.67,-2.33,
                        -2.00,-1.67,-1.33,-1.00,-0.67,-0.33,0.00,0.33,0.67,1.00,1.33,1.67,2.33,3.33,4.67,7.00,10.33,15.00]

    grouped_coords = [[] for _ in range(len(reference_values))]  # Create empty lists to store the subarrays
    for i in range(len(cartesian_coords)):  # Loop through each angle
        angle=angles[i]
        closest_ref = np.argmin([abs(angle - ref) for ref in reference_values])  # Find the reference value that is closest to the current angle
        grouped_coords[closest_ref].append(cartesian_coords[i])   # Assign the angle to the corresponding reference group



    # If scan ring has no values append empty(3) to it. This way it is easier to use the data (for example vstack)
    for i in range(32):
        if len(grouped_coords[i])==0:
            grouped_coords[i].append(np.empty(3))
        grouped_coords[i] = np.array(grouped_coords[i])

    grouped_coords=np.array(grouped_coords,dtype='object')

    return grouped_coords


def main():

    seq=sys.argv[1]
    param_file=sys.argv[2]

    #read sampling params
    with open(param_file) as file:
        params=yaml.safe_load(file)
        config=params['data_sampling']
        crop_start=config['crop_start']
        crop_end=config['crop_end']
        dataset_path=params['dataset_path']
        dist_between_samples=config['dist_between_samples']
        trajectory_length=params['pre_processing']['trajectory_length']

    input_seq=os.path.join(dataset_path,'raw',seq,'raw')
    output_path=os.path.join(dataset_path,'processed',seq,'data')
    os.makedirs(output_path,exist_ok=True)

    day=seq.split('/')[0]
    camera_matrix,extrinsics,dist_coeffs,gnss2lidar=utils.load_calib_cadcd(dataset_path,day)

    #Input paths
    input_pcd_path=os.path.join(input_seq,'lidar_points')
    input_gnss_path=os.path.join(input_seq,'novatel_rtk')
    input_camera_path=os.path.join(input_seq,'image_00')

    # Timestamps in unnix format
    pcd_timestamps=get_timestamps(os.path.join(input_pcd_path,'timestamps.txt'))
    gnss_timestamps=get_timestamps(os.path.join(input_gnss_path,'timestamps.txt'))
    camera_timestamps=get_timestamps(os.path.join(input_camera_path,'timestamps.txt'))

    #Output paths
    output_pcd_path=os.path.join(output_path,'scans')
    output_img_path=os.path.join(output_path,'imgs')
    output_gnss_path=os.path.join(output_path,'poses.csv')
    output_gnss_id_path=os.path.join(output_path,'pose_ids.txt')

    # gnns data combined to single file
    gnss_data=combine_pose_files(os.path.join(input_gnss_path,'data'))
    np.savetxt(output_gnss_path,gnss_data,delimiter=',')
    gnss_data[:,0:2] = utils.latlon2utm(gnss_data[:,0:2])

    #create dirs if doesnt exist
    os.makedirs(output_pcd_path,exist_ok=True)
    os.makedirs(output_img_path,exist_ok=True)


    gnss_id=0
    data_id=0

    # we want to cleanly exit if the user stops execution with keyboard interrupt
    with open(output_gnss_id_path,'w') as file:
        while True:
            
            # If pose data for 100 following meters doesnt exist exit
            if not utils.take_following_N_m(gnss_data[gnss_id:,:2],trajectory_length):
                break
                
            file.write(str(gnss_id)+'\n')

            # Find img id and pcd id corresponding to the pose id. 
            img_id=utils.find_closest_index(camera_timestamps,gnss_timestamps[gnss_id])
            pcd_id=utils.find_closest_index(pcd_timestamps,gnss_timestamps[gnss_id])

            # Read image and undistort
            img=cv2.imread(os.path.join(input_camera_path,'data/'+str(img_id).zfill(10)+'.png'))
            img=cv2.undistort(img,distCoeffs=dist_coeffs,cameraMatrix=camera_matrix)
            img=img[crop_start:crop_end,:,:]     
            cv2.imwrite(os.path.join(output_img_path,str(data_id)+'.png'),img)
            
            # Read lidar scan corresponding to the current gnss_id and separate scan rings based on polar angle
            scan_data = np.fromfile(os.path.join(input_pcd_path,'data/'+str(pcd_id).zfill(10)+'.bin'), dtype=np.float32)
            scan_data = scan_data.reshape((-1, 4))[:,:3]
            scan_rings=separate_scan_rings(scan_data)
            np.save(os.path.join(output_pcd_path,str(data_id)+'.npy'),scan_rings)

            gnss_id=gnss_id+utils.take_following_N_m(gnss_data[gnss_id:],dist_between_samples)
            data_id+=1

if __name__=="__main__": 
    main()

























