import numpy as np
from matplotlib.path import Path
from scipy.spatial import KDTree
import utm
import open3d
import cv2
from matplotlib.colors import LinearSegmentedColormap
import os
import yaml
from typing import Tuple
import matplotlib.pyplot as plt 


def load_calib_kitti360(
    dataset_path:str        #path to the dataset root 
) ->Tuple[np.ndarray,...]:  #calibration params as numpy arrays
    
    """
    Fetches calibration data for kitti360
    """

    lastrow = np.array([0,0,0,1]).reshape(1,4)

    cam_to_velo=np.loadtxt(os.path.join(dataset_path,'raw','calibration','calib_cam_to_velo.txt')).reshape((3,4))
    cam_to_velo=np.concatenate((cam_to_velo, lastrow))
    extrinsics=np.linalg.inv(cam_to_velo)

    with open(os.path.join(dataset_path,'raw','calibration','calib_cam_to_pose.txt')) as f:
        for line in f:
            line = line.split(' ')
            if line[0] == 'image_00:':
                cam_to_pose = [float(x) for x in line[1:13]]
                cam_to_pose = np.reshape(cam_to_pose, [3,4])
                cam_to_pose=np.concatenate((cam_to_pose,lastrow))
                gnss2lidar=cam_to_velo@(np.linalg.inv(cam_to_pose))
                gnss2lidar=gnss2lidar@np.array([[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]]) #we assume gnss coordinates have y-forward, x-right similarly to cadcd. 
                gnss2lidar=gnss2lidar[[0, 1, 3], :][:, [0, 1, 3]] #we only consider x,y in this transform

    with open(os.path.join(dataset_path,'raw','calibration','perspective.txt')) as f:
        for line in f:
            line = line.split(' ')
            if line[0] == 'P_rect_00:':
                K = [float(x) for x in line[1:13]]             
                K = np.reshape(K, [3,4])
                camera_matrix = K[:,[0,1,2]] # lets use 3x3 camera matrix

    return camera_matrix,extrinsics,gnss2lidar


def load_calib_own_data(
    dataset_path:str,       #path to the dataset root
    day:str                 #the collection date
) ->Tuple[np.ndarray,...]:  #calibration params as numpy arrays
    
    """
    Fetches calibration data for own dataset
    """

    with open(os.path.join(dataset_path,'raw',day,'calib.yaml')) as file:
        calib=yaml.safe_load(file)
    camera_matrix=read_matrix_from_dict(calib['camera_matrix'])
    dist_coeffs=read_matrix_from_dict(calib['dist_coeffs'])
    extrinsics=read_matrix_from_dict(calib['extrinsics'])
    gnss2lidar=read_matrix_from_dict(calib['gnss2lidar'])
    gnss2lidar=gnss2lidar[[0, 1, 3], :][:, [0, 1, 3]] # we only consider x,y in this transform

    return camera_matrix,extrinsics,dist_coeffs,gnss2lidar


def load_calib_cadcd(
    dataset_path:str,       #path to the dataset root
    day:str                 #the collection date
) ->Tuple[np.ndarray,...]:  #calibration params as numpy arrays
    
    """
    Fetches calibration data for the CADCD dataset
    """

    with open(os.path.join(dataset_path,'raw',day,'calib','00.yaml')) as file:
        cam_calib=yaml.safe_load(file)
        camera_matrix=read_matrix_from_dict(cam_calib['camera_matrix'])
        dist_coeffs=read_matrix_from_dict(cam_calib['distortion_coefficients'])

    with open(os.path.join(dataset_path,'raw',day,'calib','extrinsics.yaml')) as file:
        ext_calib=yaml.safe_load(file)
        extrinsics=np.linalg.inv(np.array(ext_calib['T_LIDAR_CAM00']))
        gnss2lidar=np.array(ext_calib['T_LIDAR_GPSIMU'])
        gnss2lidar=gnss2lidar[[0, 1, 3], :][:, [0, 1, 3]] # we only consider x,y in this transform

    return camera_matrix,extrinsics,dist_coeffs,gnss2lidar


def read_matrix_from_dict(
    data_dict: dict # matrix in dictionary form
) ->np.ndarray:     # shape: (rows,columns). Matrix as numpy array
    
    """
    Convert data from a dictionary to a NumPy array
    """

    rows = data_dict.get('rows')
    cols = data_dict.get('cols')
    data = data_dict.get('data')
    matrix = np.array(data).reshape((rows, cols))
    return matrix

def take_following_N_m(
    gnss: np.ndarray,   # shape: (number of poses,2). x,y for each pose. 
    N: float            # limit in meters
) ->int:                # index of the first pose that exceeds the limit
    
    """
    Computes the first pose that exceeds the given distance limit
    """
    distDriven=0
    for idx in range(len(gnss)-1):
        distDriven=distDriven+np.linalg.norm(gnss[idx+1]-gnss[idx])
        if distDriven>N:
            return idx
    return False


def find_closest_index(
    array: np.ndarray,      #shape: (any length). 
    target: float           #target value 
) ->int:                    #index of the closest value
    
    """
    Finds the closes value in the array given a target value
    """

    differences = np.abs(array - target)
    index = np.argmin(differences)
    return index

def closest_pair(
    array1: np.ndarray,             #shape: (any length,any dim)
    array2: np.ndarray              #shape: (any length,same dim as array1)
) ->Tuple[Tuple[int,int],float]:    #indeces of the closest match and the corresponding distance.
    
    """
    Given two arrays find the closest pair between them based on euclidian distance
    """

    tree = KDTree(array2)
    distances, indices = tree.query(array1)
    min_index = np.argmin(distances)
    index_array1 = min_index
    index_array2 = indices[min_index]
    closest_pair=(index_array1,index_array2)
    min_distance=np.linalg.norm(array1[index_array1]-array2[index_array2])
    
    return closest_pair,min_distance

def ring_and_id2xyz(
    ring_and_id: np.ndarray,    #shape(number of points,2). Ring and id for each point
    scan: np.ndarray,           #shape: (number of rings,). Pointcloud organized to scan rings. Each element contains list of points in that scan ring.
) ->np.ndarray:                 #shape(number of points,3). x,y,z for each point.
    
    """
    Returns the corresponding 3d coordinates for given ring and point index
    """

    xyz_coords=[]
    for (ring,id) in ring_and_id: 
        xyz=scan[ring][id][:3]
        xyz_coords.append(xyz)
    return np.array(xyz_coords)

def get_polygon_mask(
    coordinates: np.ndarray,        #shape: (number of corner points,2). Pixel coordinates for the corner points 
    image_shape: Tuple[int,int]     #height and width of the image
) ->np.ndarray:                     #shape (height,width). Mask that is true if pixel is inside the polygon
    
    """
    Defines polygon mask from given corner points
    """

    mask = np.zeros(image_shape, dtype=bool)
    path = Path(coordinates)
    X, Y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
    points = np.vstack((X.flatten(), Y.flatten())).T
    mask = path.contains_points(points).reshape(image_shape)
    return mask


def lidar_points_to_image(
    points_3d: np.ndarray,          #shape: (number of points,3). x,y,z for each lidar point
    extrinsic_matrix: np.ndarray,   #shape: (4,4). Transform form lidar frame to image frame.
    projection_matrix               #shape: (3,3). Projectrion matrix from camera coords to image coords. 
) ->np.ndarray:                     #shape: (number of points,2). Pixel coords for each point. 
    
    """
    Projects 3d coordinates to pixel coordinates
    """

    homogeneous_points_3d = np.hstack((points_3d, np.ones((points_3d.shape[0], 1), dtype=np.float32)))
    transformed_points = np.dot(extrinsic_matrix, homogeneous_points_3d.T).T[:, :3]
    homogeneous_points_2d = np.dot(projection_matrix, transformed_points.T).T
    projected_points = homogeneous_points_2d[:, :2] / homogeneous_points_2d[:, 2:]
    image_points = np.round(projected_points)
    return image_points.astype('int')

def latlon2utm(
    latlon: np.ndarray      #shape: (number of poses,2). lat and lon for each pose
) ->np.ndarray:             #shape (number of poses,2). UTM coords for each pose
    
    """
    Convert latitude and longitude to utm coordinates
    """

    utm_data=[]
    for message in latlon:
      utm_coords = utm.from_latlon(float(message[0]), float(message[1]))[:2]
      utm_data.append(utm_coords)
    return np.array(utm_data)

def make_rgb_label(
    label: np.ndarray       #shape: (height,width). Continuos label.
) ->np.ndarray:             #shape: (height,width,3). Bgr label. 
    """
    Saves the autolabel so that pixels that we can't estimate based on lidar are red
    and the pixels we can estimate are green and go from 0-255 based on the similarity
    (255->maximum similarity).
    """
    image= np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    nan_mask=np.isnan(label)
    label[nan_mask]=0
    image[:,:,1]=label
    image[nan_mask,2]=255
    return image

def nanmean(
    arr1:np.ndarray,    #shape: (any size)
    arr2:np.ndarray,    #shape: (same as arr1)
    require_both: bool  #if true mean set to nan if either input has element that is nan.   
) ->np.ndarray:         # shape(same as arr1 and arr2). The mean of arr1 and arr2. 
    
    """
    Computes the mean of arr1 and arr2 and handles nans correctly. 
    """
    only_arr1_nan = np.isnan(arr1) & ~np.isnan(arr2)
    only_arr2_nan = ~np.isnan(arr1) & np.isnan(arr2)
    neither_nan = ~np.isnan(arr1) & ~np.isnan(arr2)
    result = np.full(arr1.shape, np.nan)
    if not require_both:
        result[only_arr1_nan] = arr2[only_arr1_nan]  # Use arr2 where arr1 is NaN
        result[only_arr2_nan] = arr1[only_arr2_nan]  # Use arr1 where arr2 is NaN
    result[neither_nan] = (arr1[neither_nan] + arr2[neither_nan]) / 2  # Take mean where both are not NaN
    return result

def visualize_pcd(
    scan: np.ndarray,            #shape: (number of rings,). Pointcloud organized to scan rings. Each element contains list of points in that scan ring.
    wheel_point_idx: np.ndarray, #shape: (number of trajectory points,4). Ring,center,left wheel and right wheel if for each trajectory point
    future_poses: np.ndarray     #shape: (number of poses,2). x,y for each pose given in the lidar frame. 
) ->None:
    
    """
    visualize the pointcloud with trajectory and wheel points.     
    """

    # Create an empty list to store the point clouds for each scan ring
    geometries = []


    # Loop through each scan ring
    for i, ring in enumerate(scan):

        # Convert each scan ring to an Open3D point cloud
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(ring[:, :3])  # Set points for the current scan ring

        if i % 3 == 0:
            color=[1,0,0]
        if i % 3 == 1: 
            color=[0,1,0]
        if i % 3 == 2:
            color=[0,0,1]

        pcd.colors = open3d.utility.Vector3dVector(np.tile(color, (ring.shape[0], 1)))  # Set the color for all points in this ring

        # Add the colored point cloud to the geometries list
        geometries.append(pcd)

 
    # Loop through trajectory points of each ring
    for (ring,center,left,right) in wheel_point_idx:
        
        #add left wheel
        sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.3)
        sphere.paint_uniform_color([0, 1, 0])  # Set color to green
        sphere.translate(scan[ring][left][:3])
        geometries.append(sphere)

        #add center
        sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.3)
        sphere.paint_uniform_color([0, 1, 0])  # Set color to green
        sphere.translate(scan[ring][center][:3])
        geometries.append(sphere)

        #add right wheel
        sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.3)
        sphere.paint_uniform_color([0, 1, 0])  # Set color to green
        sphere.translate(scan[ring][right][:3])
        geometries.append(sphere)
    
    poses = open3d.geometry.PointCloud()
    poses.points = open3d.utility.Vector3dVector(np.hstack((future_poses,np.zeros((len(future_poses),1)))))
    poses.colors = open3d.utility.Vector3dVector(np.ones((len(future_poses), 3)) * np.array([0, 1, 0]))
    geometries.append(poses)
        
    # Set visualization parameters
    visualizer = open3d.visualization.Visualizer()
    visualizer.create_window()

    # Add sphere geometries to the visualization
    for geometry in geometries:
        visualizer.add_geometry(geometry)

    visualizer.run()
    visualizer.destroy_window()

def overlay_mask(
    image: np.ndarray,  #shape: (height,width,3). Bgr image
    mask: np.ndarray    #shape: (height,widht). True if pixel belongs to mask. 
) ->np.ndarray:         #shape (height,widht,3). Image overlaid with the mask.
    
    """
    overlay binary mask with image
    """

    color_mask=np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color_mask[mask == 0] = [0, 0, 0]
    color_mask[mask == 1] = [0, 255, 0]
    blended = cv2.addWeighted(image, 1.0, color_mask, 0.3, 0)
    return blended


def overlay_heatmap(
    image: np.ndarray,  #shape (height,width,3). Bgr image
    label: np.ndarray   #shape (height,width). Continuos label in the range 0-1. 
) ->np.ndarray:         # shape(height,width,3). Image overlaid with heatmap generated from the label.
    
    """
    Make heatmap from continuos label and overlay with the image
    """

    green_to_red = LinearSegmentedColormap.from_list('red_orange_green', ['red','orange', 'green'])
    heatmap_colored = green_to_red(label)
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    heatmap_colored_bgr = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(image, 0.5, heatmap_colored_bgr, 0.5, 0)
    return overlay


def draw_points_to_img(
    image: np.ndarray,      #shape: (height,width,3). Brg image
    points: np.ndarray,     #shape: (number of points, 2). Pixel coords of each point in x,y format.
    scores: np.ndarray      #shape: (number of points). Score of each point. 
) ->np.ndarray:             #shape: (heigth,width,3). Image with the points drawn.

    """
    Draw points to an image
    """

    green_to_red = LinearSegmentedColormap.from_list('red_orange_green', ['red','orange', 'green'])
    for point,score in zip(points[~np.isnan(scores)],scores[~np.isnan(scores)]):
        if score>0:
            color = green_to_red(score)  # returns a tuple (R, G, B, A)
            color = tuple([int(255*color[2]),int(255*color[1]),int(255*color[0])])
            cv2.circle(image, point, 5, color, -1)
    return image

