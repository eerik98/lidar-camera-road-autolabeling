import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from scipy.interpolate import griddata
import utils
import yaml
from tqdm import tqdm
import sys
from typing import Tuple


def get_height_score(    
    scan: np.ndarray,             # shape: (number of rings,). Pointcloud organized to scan rings. Each element contains list of points in that scan ring.
    trajectory_data: np.ndarray,  # shape: (number of trajectory points, 4). Ring, center point, left wheel point and right wheel point id for each trajectory point
    height_std: float,            # standard deviation for the height score 
    max_radial_dist: float        # max radial distance allowed for points in the same scan ring
) -> np.ndarray:                  # shape: (number of rings,). Height based score. Same order as input scan.
     
    """
    Computes height based score
    """

    # init with nan
    score=[]
    for ring_id in range(len(scan)):
        sim=np.empty(len(scan[ring_id]))
        sim[:]=np.nan
        score.append(sim)
    score=np.array(score,dtype='object')
    
    # compute similarity scores
    for (ring,center,left,right) in trajectory_data:
        scan_ring=scan[ring]
        xyz=scan_ring[center]
        d_ref=np.sqrt(xyz[0]**2+xyz[1]**2+xyz[2]**2)
        d_scan=np.sqrt(scan_ring[:,0]**2+scan_ring[:,1]**2+scan_ring[:,2]**2)
        
        diff=scan_ring[:,2]-xyz[2]
        score[ring][diff>0]=np.exp(-diff[diff>0]**2/height_std**2)#diff>0
        score[ring][diff<0]=1 # if lower than refrence set to 1

        #if the radial distance is much different compared to refrence, likely obstacle point even when the height is similar
        score[ring][(np.abs(d_ref-d_scan)>max_radial_dist)]=0 

    return score

def get_gradient_score(
    scan: np.ndarray,               #shape: (number of rings,). Pointcloud organized to scan rings. Each element contains list of points in that scan ring.
    trajectory_data: np.ndarray,    # shape: (number of trajectory points, 4). Ring, center point, left wheel point and right wheel point id for each trajectory point
    gradient_std: float             # standar deviation for gradient based score
) -> np.ndarray:                    #shape: (number of rings,). Gradient based score for each point. Same order as input scan. 
     
    """
    Computes gradient based score
    """
    
    #Init with nan
    score=[]
    for ring_id in range(len(scan)):
        sim=np.empty(len(scan[ring_id]))
        sim[:]=np.nan
        score.append(sim)

    score=np.array(score,dtype='object')

    for (ring,center,left,right) in trajectory_data:

        scan_ring=scan[ring]
        x = scan_ring[:, 0]
        y = scan_ring[:, 1]

        theta=np.arctan2(y,x)
        theta_center= theta[center] 
        theta=theta-theta_center
        theta = np.mod(theta, 2 * np.pi) #change range to [0,2pi]
        theta_left=theta[left]
        theta_right=theta[right]
        sorted_idx=np.argsort(theta) #sorted list starts from trajectory center and goes counter clockwise from 0 to 2pi
        sorted_theta=theta[sorted_idx]
        sorted_z=scan_ring[sorted_idx,2]

        diff=-np.diff(sorted_z)
        diff=np.append(diff,diff[-1])
        diff[sorted_theta<np.pi]=-diff[sorted_theta<np.pi]

        between_wheels=(sorted_theta>theta_right)|(sorted_theta<theta_left)

        if np.sum(between_wheels)==0:
            between_wheels_max_diff=0
        else:
            between_wheels_max_diff=(diff[between_wheels]).max()

        diff[(diff<between_wheels_max_diff)]=0

        diff[sorted_theta>np.pi]=np.cumsum(diff[sorted_theta>np.pi][::-1])[::-1] #compute in reversed order
        diff[sorted_theta<np.pi]=np.cumsum(diff[sorted_theta<np.pi])

        unordered_score=np.exp(-(diff**2)/gradient_std**2)

        ordered=np.empty_like(unordered_score)
        ordered[sorted_idx]=unordered_score
        score[ring]=ordered

    return score
    
def interpolate(
    coords: np.ndarray,     #shape: (number of points,2). Pixel coordinates in format (x,y).
    values: np.ndarray,     #shape: (number of points,). Score value for each point. 
    image_shape: Tuple      #in format (height,width)
) ->np.ndarray:             #shape: (height,width). Interpolated score in image.  
    
    """
    Creates interpolated image given sample coordinates and their values. 
    """

    grid_x, grid_y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
    interpolated=griddata(coords[~np.isnan(values)],values[~np.isnan(values)],(grid_x,grid_y),method='linear')
    
    #set values above last certain road pixel to nan
    indices = np.where(np.any((interpolated == 1), axis=1))[0]
    if len(indices) == 0:
        row_id = -1  
    else:
        row_id = indices[0]  
    interpolated[:row_id,:]=np.nan

    return interpolated


def main():

    day=sys.argv[1]
    seq=sys.argv[2]

    with open('params_cadcd.yaml', 'r') as file:
        params = yaml.safe_load(file)
        crop_start=params['data_sampling']['crop_start']
        l_wheel_0=params['origin']['l_wheel_scan_loc']
        r_wheel_0=params['origin']['r_wheel_scan_loc']
        dataset_path=params['dataset_path']
        
        config=params['lidar_auto_labeling']
        slope_std=config['slope_label_std']
        height_std=config['height_label_std']
        use_height=config['use_height']
        use_slope=config['use_slope']
        max_radial_dist=config['max_radial_dist']

    camera_matrix,extrinsics,dist_coeffs,gnss2lidar=utils.load_calib_cadcd(dataset_path,day)
    camera_matrix[1,2]=camera_matrix[1,2]-crop_start #add the effect off cropping

    l_wheel_img_location=utils.lidar_points_to_image(np.array(l_wheel_0).reshape(1,-1),extrinsics,camera_matrix).flatten() 
    r_wheel_img_location=utils.lidar_points_to_image(np.array(r_wheel_0).reshape(1,-1),extrinsics,camera_matrix).flatten()

    seq_path=os.path.join(dataset_path,'processed',day,seq)

    #input paths
    scan_path=os.path.join(seq_path,'autolabels/pre_processing/filtered_scans')
    trajectory_path=os.path.join(seq_path,'autolabels/pre_processing/trajectory_data')
    img_path=os.path.join(seq_path,'data/imgs')

    #output paths
    auto_label_path=os.path.join(seq_path,'autolabels/lidar/labels')
    auto_label_overlaid_path=os.path.join(seq_path,'autolabels/lidar/labels_overlaid')

    #create dirs if doesnt exists
    os.makedirs(auto_label_overlaid_path,exist_ok=True)
    os.makedirs(auto_label_path,exist_ok=True)


    num_frames = len(os.listdir(scan_path))
    for data_id in tqdm(range(num_frames)):

        #Read data
        scan=np.load(os.path.join(scan_path,str(data_id)+'.npy'),allow_pickle=True)
        frame=cv2.imread(os.path.join(img_path,str(data_id)+'.png'))
        trajectory_data = np.loadtxt(os.path.join(trajectory_path,str(data_id)+'.csv'), delimiter=',',dtype='int')

        #Check if enough trajectory points
        if len(trajectory_data)<2:
            print("frame skipped. Not enough trajectory data")
            label=np.zeros(frame.shape)
            label[:,:,2]=255
            cv2.imwrite(os.path.join(auto_label_path,str(data_id)+'.png'),label)
            continue
        
        #Compute height score
        height_score=get_height_score(scan,trajectory_data,height_std,max_radial_dist)

        #Compute gradient score
        slope_score=get_gradient_score(scan,trajectory_data,slope_std)

        #Combine scores
        if use_height and use_slope:
            score=utils.nanmean(np.concatenate(height_score),np.concatenate(slope_score),require_both=True)
        if use_height and not use_slope:
            score=np.concatenate(height_score)
        if use_slope and not use_height:
            score=np.concatenate(slope_score)
        if not use_height and not use_slope:
            print("Set at least one of the options: use_height or use_slope to True")
            break
        
        #Project scan points to image
        image_points=utils.lidar_points_to_image(np.vstack(scan)[:,:3],extrinsics,camera_matrix)
        image_points=np.vstack([image_points,[l_wheel_img_location,r_wheel_img_location]])
        score=np.hstack([score,[1,1]])

        #Interpolate between points
        interpolated_label=interpolate(image_points,score,frame.shape)

        #Save outputs
        rgb_label=utils.make_rgb_label(interpolated_label*255)
        overlaid_img=utils.draw_points_to_img(frame,image_points,score)
        overlaid_img=utils.overlay_heatmap(overlaid_img,interpolated_label)
        cv2.imwrite(os.path.join(auto_label_path,str(data_id)+'.png'),rgb_label)
        cv2.imwrite(os.path.join(auto_label_overlaid_path,str(data_id)+'.png'),overlaid_img)

main()