import numpy as np
import open3d
import os
import cv2
import utils
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
import sys

def ENU2Lidar(
    ENU_poses: np.ndarray,  #shape: (number of poses, 4). x,y,z and azimuth in ENU frame for each pose. 
    gnss2lidar: np.ndarray  #shape: (4,4). Trasformation between GPS and Lidar frame. 
) ->np.ndarray:             #shape: (number of poses,4). x,y,z and orientation in Lidar frame for each pose.
    
    """
    Transform ENU poses to the Lidar frame.
    """

    #change x and y coords to the GPSIMU frame at the time scan is taken
    azimuth=ENU_poses[0,3]
    #azimuth is clockwise angle from North (y). Lets rotate all x,y coordinates counter-clockwise by first azimuth the get coordinates w.r.t the GPS frame at the time scan was taken
    angle_rad=np.deg2rad(azimuth) 
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])
    ENU_poses[:,0:2] = np.dot(ENU_poses[:,0:2], rotation_matrix.T)
    homogeneus_xyz=np.hstack((ENU_poses[:,:3],np.ones((ENU_poses.shape[0], 1))))
    lidar_frame_xyz=(homogeneus_xyz@gnss2lidar.T)[:,:3]
    lidar_frame_orientation=azimuth-ENU_poses[:,3]
    lidar_frame_coords=np.hstack((lidar_frame_xyz,lidar_frame_orientation.reshape(-1,1)))
    return lidar_frame_coords

def closest_pcd_point(
    poses: np.ndarray,              #shape: (number of poses,4). x,y,z and heading for for each pose. 
    scan: np.ndarray,               #shape: (number of rings,). Pointcloud organized to scan rings. Each element contains list of points in that scan ring.
    max_scan2pose_distance: float,  #maximum allowed distance between scan point and pose to create a match
    height_threshold: float,        #maximum allowed heigth difference between trajectory points in consecutive scan rings
    distance_threshold: float       #minimum allowed distance between trajectory points in consecutive scan rings
) ->np.ndarray:                     #shape: (number of point matches,3). Ring, point id and heading for each match.
    
    """
    Finds the closest match between poses and scan points at each scan ring. 
    """
    trajectory=[]
    prev_point=np.array([0,0,-2.1])
    for ring in range(len(scan)):
      if len(scan[ring])>0:
        ring_points=scan[ring]

        height_mask=(ring_points[:,2]-prev_point[2])<height_threshold
        dist_mask=(np.linalg.norm(ring_points[:,:2]-[0,0],axis=1)-np.linalg.norm(prev_point[:2]-[0,0]))>distance_threshold

        filtered_idx=np.where(height_mask & dist_mask)[0]
        if len(filtered_idx)==0:
            continue

        filtered_ring_points=ring_points[filtered_idx]
        closest_idx,min_distance=utils.closest_pair(filtered_ring_points[:,0:2],poses[:,:2])
        scan_id=filtered_idx[closest_idx[0]]
        gnss_id=closest_idx[1]

        if min_distance<max_scan2pose_distance:
           trajectory.append([ring,scan_id,poses[gnss_id,3]])
           prev_point=ring_points[scan_id]

    return trajectory

def wheel_contact_points(
    scan: np.ndarray,                   #shape: (number of rings,). Pointcloud organized to scan rings. Each element contains list of points in that scan ring.
    trajectory: np.ndarray,             #shape: (number of trajectory points,3). Ring,center and heading for each trajectory point. 
    max_center2wheel_distance: float    #maximum allowed distance from center point to the wheel
)-> np.ndarray:                         #shape: (number of trajectory points,4). Ring,center,left wheel and right wheel id for each trajectory point. 
    
    """
    Computes wheel points based on center points and heading
    """

    wheel_point_idx=[]

    for (ring,center_id,heading) in trajectory:
        scan_ring=scan[ring]
        center=scan_ring[center_id]
        x, y = center[0:2]

        theta = np.deg2rad(heading)

        x_left = x + np.cos(theta+np.pi/2)
        y_left = y + np.sin(theta+np.pi/2)

        x_right = x - np.cos(theta+np.pi/2)
        y_right = y - np.sin(theta+np.pi/2)
        scan_ring_thetas=np.arctan2(scan_ring[:,1],scan_ring[:,0])
        theta_left=np.arctan2(y_left,x_left)
        l_idx=utils.find_closest_index(theta_left,scan_ring_thetas)
        l_wheel_point=scan_ring[l_idx,:3]
        theta_right=np.arctan2(y_right,x_right)
        r_idx=utils.find_closest_index(theta_right,scan_ring_thetas)
        r_wheel_point=scan_ring[r_idx,:3]

        if (np.linalg.norm(l_wheel_point[:2]-center[:2])<max_center2wheel_distance) and (np.linalg.norm(r_wheel_point[:2]-center[:2])<max_center2wheel_distance):
            wheel_point_idx.append([ring,center_id,l_idx,r_idx])

    return np.array(wheel_point_idx)

def get_future_poses(
    ENU_poses: np.ndarray,      #shape: (number of poses, 4). x,y,z and azimuth in ENU frame for each pose. 
    pose_id: int,               #index of the pose we are at
    trajectory_length: float,   #length of the future trajectory to extract
    gnss2lidar: np.ndarray,     #shape: (4,4). Trasformation between GPS and Lidar frame.  ,
) -> np.ndarray:                #shape: (number of poses, 4). x,y,z and heading in lidar frame for each pose in the future trajectory
    """
    Finds the poses in lidar frame for the future trajectory
    """
    poses=ENU_poses[pose_id:]
    poses[:,0:2]=poses[:,0:2]-poses[0,0:2] #shift origin
    limit=utils.take_following_N_m(poses[:,:2],trajectory_length)
    poses=poses[:limit]
    poses_at_lidar_frame=ENU2Lidar(poses,gnss2lidar)

    return poses_at_lidar_frame

def occlusion_filtering(
    wheel_idx: np.ndarray,                      # shape: (number of trajectory points, 4). Trajectory data for each wheel (ring, center, left wheel, right wheel)
    scan: np.ndarray,                           # #shape: (number of rings,). Pointcloud organized to scan rings. Each element contains list of points in that scan ring.
    extrinsics: np.ndarray,                     # shape: (4, 4). Extrinsic calibration matrix from lidar to camera
    camera_matrix: np.ndarray,                  # shape: (3, 3). Intrinsic calibration matrix of the camera
    occlusion_filter_pixel_threshold: float     # pixel proximity threshold for occlusion filtering
) -> np.ndarray:                                # shape: (number of non occluded trajectory points, 4). Filtered trajectory data points where occluded points are removed
    
    """
    Detect if any wheel points are occluded and remove them. 
    """
    
    for i,(ring,center,left,right) in enumerate(wheel_idx[1:],start=1):
        prev_ring=np.vstack(scan[:ring])[:,:3]

        l_wheel=scan[ring][left][:3]
        r_wheel=scan[ring][right][:3]

        # Project points from lidar to image plane
        prev_pixels = utils.lidar_points_to_image(prev_ring, extrinsics, camera_matrix)
        l_wheel_pixel = utils.lidar_points_to_image(l_wheel.reshape(1, 3), extrinsics, camera_matrix)
        r_wheel_pixel = utils.lidar_points_to_image(r_wheel.reshape(1, 3), extrinsics, camera_matrix)

        # Compute pixel proximity masks
        r_proximity_mask = np.abs(prev_pixels[:, 0] - r_wheel_pixel[0, 0]) < occlusion_filter_pixel_threshold
        l_proximity_mask = np.abs(prev_pixels[:, 0] - l_wheel_pixel[0, 0]) < occlusion_filter_pixel_threshold

        # If there are previous points close to the right wheel, check for occlusion
        if np.any(r_proximity_mask):
            r_lim = np.min(prev_pixels[r_proximity_mask, 1])
            if r_lim < r_wheel_pixel[0, 1]:  # Check if the right wheel is occluded
                return wheel_idx[:i]

        # If there are previous points close to the left wheel, check for occlusion
        if np.any(l_proximity_mask):
            l_lim = np.min(prev_pixels[l_proximity_mask, 1])
            if l_lim < l_wheel_pixel[0, 1]:  # Check if the left wheel is occluded
                return wheel_idx[:i]

    # If no occlusion detected, return the full wheel_idx
    return wheel_idx

def noise_filtering(
    scan: np.ndarray,   #shape: (number of rings,). Pointcloud organized to scan rings. Each element contains list of points in that scan ring.
    nb_neighbors: int,  #number of neighbors for statistical outlier filtering
    std_ratio: float,   #standard deviation ratio for statistical outlier filtering  
) -> np.ndarray:        #shape: (number of rings,). Pointcloud organized to scan rings. Each element contains list of filtered points in that scan ring.
    
    """
    Filters out noise points from the pointcloud
    """

    xyzi=np.vstack(scan)
    xyz=xyzi[:,:3]
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)
    cl,ind=pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,std_ratio=std_ratio)

    filtered_scans=[]
    start_index=0

    for ring in range(len(scan)):
        scan_ring=scan[ring]
        num_points=len(scan_ring)
        if num_points==0:
            filtered_scans.append(np.empty((0,4)))
        else:
            mask=ind[start_index:start_index + num_points]
            filtered_points = np.asarray(xyzi)[mask]
            filtered_scans.append(filtered_points)
            start_index+=num_points

    return np.array(filtered_scans,dtype=object)


def fov_filtering(
    scan: np.ndarray,   #shape: (number of rings,). Pointcloud organized to scan rings. Each element contains list of points in that scan ring.
    fov: float          #Field of view in degrees. 
) ->np.ndarray:         #shape: (number of rings,). Pointcloud organized to scan rings. Each element contains list of filtered points in that scan ring.
    
    """
    Filters out points outside the field of view
    """

    fov=np.deg2rad(fov)
    filtered_scan=np.empty(len(scan),dtype='object')

    for i in range(len(scan)):
        ring=scan[i]
        theta = np.arctan2(ring[:,1], ring[:,0])    
        filtered_ring=ring[((theta<0.5*fov) & (theta>=0))|((theta>-0.5*fov) & (theta<0))]
        filtered_scan[i]=filtered_ring

    return filtered_scan


def main():
    day=sys.argv[1]
    seq=sys.argv[2]

    #Parameters
    with open('params_own_data.yaml', 'r') as file:
        params = yaml.safe_load(file)
        config=params['pre_processing']
        crop_start=params['data_sampling']['crop_start']
        l_wheel_0=params['wheels']['l_wheel_scan_loc']
        r_wheel_0=params['wheels']['r_wheel_scan_loc']
        dataset_path=params['dataset_path']
        distance_threshold = config['distance_filter_threshold']
        height_threshold=config['height_filter_threshold']
        max_pcd2gnss_distance=config['max_pcd2gnss_distance']
        max_center2wheel_distance=config['max_center2wheel_distance']
        occlusion_filter_pixel_threshold=config['occlusion_filter_pixel_threshold']
        noise_filtering_nb_neighbors=config['noise_filtering_nb_neighbors']
        noise_filtering_std_ratio=config['noise_filtering_std_ratio']
        trajectory_length = config['trajectory_length']
        lidar_fov=config['lidar_fov']

    #camera_matrix,extrinsics,dist_coeffs,gnss2lidar=utils.load_calib_own_data(dataset_path,day)

    camera_matrix,extrinsics,dist_coeffs,gnss2lidar=utils.load_calib_cadcd(dataset_path,day)
    camera_matrix[1,2]=camera_matrix[1,2]-crop_start #add the effect off cropping
        

    seq_path=os.path.join(dataset_path,'processed',day,seq)

    #input paths
    pcd_path=os.path.join(seq_path,'data/scans')
    pose_path=os.path.join(seq_path,'data/poses.csv')
    pose_id_path=os.path.join(seq_path,'data/pose_ids.txt')
    img_path=os.path.join(seq_path,'data/imgs')

    #output paths
    mask_path=os.path.join(seq_path,'autolabels/pre_processing/trajectory_masks')
    mask_overlaid_path=os.path.join(seq_path,'autolabels/pre_processing/trajectory_masks_overlaid')
    trajectory_data_path=os.path.join(seq_path,'autolabels/pre_processing/trajectory_data')
    filtered_scan_path=os.path.join(seq_path,'autolabels/pre_processing/filtered_scans')

    #create dirs if doesnt exist
    os.makedirs(mask_path,exist_ok=True)
    os.makedirs(mask_overlaid_path,exist_ok=True)
    os.makedirs(trajectory_data_path,exist_ok=True)
    os.makedirs(filtered_scan_path,exist_ok=True)

    poses=np.loadtxt(pose_path, delimiter=',',skiprows=1) 
    poses[:,0:2] = utils.latlon2utm(poses[:,0:2])
    pose_idx=np.loadtxt(pose_id_path,delimiter=',',dtype='int')

    for i in tqdm(range(len(pose_idx))):
        
        current_pose_id=pose_idx[i]
        frame=cv2.imread(os.path.join(img_path,str(i)+'.png'))
        scan=np.load(os.path.join(pcd_path,str(i)+'.npy'),allow_pickle=True)

        scan=fov_filtering(scan,lidar_fov)
        filtered_scan=noise_filtering(scan,noise_filtering_nb_neighbors,noise_filtering_std_ratio)
        future_poses=get_future_poses(poses.copy(),current_pose_id,trajectory_length,gnss2lidar)
        trajectory=closest_pcd_point(future_poses,filtered_scan,max_pcd2gnss_distance,height_threshold,distance_threshold)

        wheel_point_idx=wheel_contact_points(filtered_scan,trajectory,max_center2wheel_distance) 
        wheel_point_idx=occlusion_filtering(wheel_point_idx,filtered_scan,extrinsics,camera_matrix,occlusion_filter_pixel_threshold)

        l_wheel_points=utils.ring_and_id2xyz(wheel_point_idx[:,[0,2]],filtered_scan)
        r_wheel_points=utils.ring_and_id2xyz(wheel_point_idx[:,[0,3]],filtered_scan)

        wheel_points=np.concatenate((l_wheel_points,np.flip(r_wheel_points,axis=0)))
        wheel_points=np.vstack([l_wheel_0,wheel_points,r_wheel_0])

        image_points=utils.lidar_points_to_image(wheel_points,extrinsics,camera_matrix) # x,y
        
        mask=utils.get_polygon_mask(image_points, image_shape=(frame.shape[0],frame.shape[1]))

        # utils.visualize_pcd(filtered_scan,wheel_point_idx) #uncomment if you want to visualize the trajecotry in the pointcloud

        overlaid_mask=utils.overlay_mask(frame,mask)

        np.savetxt(os.path.join(trajectory_data_path,str(i)+'.csv'),wheel_point_idx,delimiter=',',fmt='%d, %d,%d,%d')
        cv2.imwrite(os.path.join(mask_path,str(i)+'.png'),(mask.astype('uint8'))*255)
        cv2.imwrite(os.path.join(mask_overlaid_path,str(i)+'.png'),overlaid_mask)
        np.save(os.path.join(filtered_scan_path,str(i)+'.npy'),filtered_scan)

if __name__=="__main__": 
    main()

