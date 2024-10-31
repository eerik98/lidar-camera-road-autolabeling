import cv2
import numpy as np
import os
from crf import dense_crf
import utils
import yaml
from tqdm import tqdm
import sys

def combine_labels(
    lidar_label: np.ndarray, #shape: (height,width). lidar autolabel
    dino_label: np.ndarray #shape: (height,width). visual autolabel
) -> np.ndarray: #shape (height,width). Combined autolabel
    
    """
    Combines the continous lidar label and dino label to single continuos label by taking the mean. 
    If label value is defined for only one label in some image area that value is retained without modifications. 
    """

    lidar_label_cat=np.concatenate(lidar_label[:,:,1]/255,dtype=float)
    dino_label_cat=np.concatenate(dino_label[:,:,1]/255,dtype=float)

    lidar_nan_mask=np.concatenate(lidar_label[:, :, 2] == 255)
    dino_nan_mask=np.concatenate(dino_label[:, :, 2] == 255) 

    lidar_label_cat[lidar_nan_mask]=np.nan
    dino_label_cat[dino_nan_mask]=np.nan

    combined_label_cat=utils.nanmean(lidar_label_cat,dino_label_cat,require_both=False)
    combined_label=combined_label_cat.reshape(lidar_label.shape[0],lidar_label.shape[1])

    return combined_label


def crf_post_processing(
    image:np.ndarray, #shape (heigh,width,3). Bgr image.
    label: np.ndarray #shape (height,width). Continuos label in the range 0-1
) ->np.ndarray: #shape (height,width). Binary label,each value 0 or 1. 
    
    """
    Creates discrete label from the continuos label using crf
    """

    rgb_img=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    label=np.stack((1-label,label),axis=0)
    out=dense_crf(rgb_img,label).argmax(0)
    return out

def main():

    day=sys.argv[1]
    seq=sys.argv[2]

    with open('params_cadcd.yaml', 'r') as file:
        params = yaml.safe_load(file)
        config=params['post_processing']
        use_lidar_label=config['use_lidar_label']
        use_dino_label=config['use_camera_label']
        use_crf=config['use_crf']
        dataset_path=params['dataset_path']

    seq_path=os.path.join(dataset_path,'processed',day,seq)

    #inputs
    dino_label_path=os.path.join(seq_path,'autolabels/camera/labels')
    lidar_label_path=os.path.join(seq_path,'autolabels/lidar/labels')
    img_path=os.path.join(seq_path,'data/imgs')

    #outputs
    label_path=os.path.join(seq_path,'autolabels/post_processing/labels')
    label_overlaid_path=os.path.join(seq_path,'autolabels/post_processing/labels_overlaid')

    #create dirs if doesnt exists
    os.makedirs(label_path,exist_ok=True)
    os.makedirs(label_overlaid_path,exist_ok=True)

    if not use_lidar_label and not use_dino_label:
        print("Set at least one of the options: use_lidar_label or use_camera_label to True")
        exit()

    num_files = len(os.listdir(img_path))
    for data_id in tqdm(range(num_files)):
        
        #Read data
        frame=cv2.imread(os.path.join(img_path,str(data_id)+'.png'))
        if use_dino_label:
            dino_label=cv2.imread(os.path.join(dino_label_path,str(data_id)+'.png'))
  
        if use_lidar_label:
            lidar_label=cv2.imread(os.path.join(lidar_label_path,str(data_id)+'.png'))

        #Combine labels
        if use_dino_label and use_lidar_label:
            label=combine_labels(lidar_label,dino_label)
        
        if use_dino_label and not use_lidar_label:
            label=dino_label[:,:,1]/255
        
        if use_lidar_label and not use_dino_label:
            label=lidar_label[:,:,1]/255

        #Post process with crf if wanted
        if use_crf:
            bin_label=crf_post_processing(frame,label)
        else:
            bin_label=(label>0.5)

        #Save outputs
        overlaid=utils.overlay_mask(frame,bin_label)
        cv2.imwrite(os.path.join(label_overlaid_path,str(data_id)+'.png'),overlaid)
        cv2.imwrite(os.path.join(label_path,str(data_id)+'.png'),(bin_label.astype('uint8'))*255)
main()