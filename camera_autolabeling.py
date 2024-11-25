import cv2
import numpy as np
import torch
import torchvision.transforms as T
import os
import utils
import torch.nn.functional as F
import yaml
from tqdm import tqdm
import sys

def dino_features(
    model: torch.nn.Module, #dinov2 pre-trained feature extractor
    norm_image: np.ndarray  #shape: (height,width,3). Input image in BGR format
) ->torch.tensor:           #shape: (height//patch_size,width//patch_size,size of patch feature). Features for each image patch
    
    """
    Computes features for an image
    """
    with torch.no_grad():
        features = model.get_intermediate_layers(norm_image, n=1)[0]
    features=features.reshape(norm_image.shape[2]//14,norm_image.shape[3]//14,1536)
    return features

def mean_feature(
    features: torch.tensor, #shape: (height//patch_size,width//patch_size,size of patch feature)
    mask: torch.tensor      #shape: (height//patch_size,width//patch_size). Mask that is true if image patch belongs to trajectory
) ->torch.tensor:           #shape: (size of patch feature). Mean feature of the trajectory
    
    """
    Computes the mean feature of the trajectory mask
    """

    mask_features=features[mask,:]
    mean_feat=mask_features.mean(dim=0).flatten()
    return mean_feat


def cosine_similarity_score(
    features: torch.tensor,     #shape: (height//patch_size,width//patch_size,size of patch feature),
    mean_feat: torch.tensor,    #shape: (size of patch feature). Mean feature of the trajectory,
    std: float                  #standard deviation for visual road score
)-> torch.tensor:               #shape: (height//patch_size,width//patch_size). Score for each image patch. 
    
    """
    Computes cosine similarity with the mean feature
    """

    cos_sim=F.cosine_similarity(features,mean_feat.reshape(1,1,-1),dim=2)
    cos_sim=cos_sim/cos_sim.max()
    cos_sim=torch.exp(-torch.pow((1-cos_sim)/std,2))
    return cos_sim

def main():

    #day=sys.argv[1]
    seq=sys.argv[1]

    with open('params_kitti360.yaml', 'r') as file:
        params = yaml.safe_load(file)
        config=params['camera_auto_labeling']
        scale=config['img_scale']
        std=config['camera_label_std']
        min_refrence_feats=config['min_refrence_feats']
        dataset_path=params['dataset_path']

    seq_path=os.path.join(dataset_path,'processed',seq)

    #input paths
    trajectory_mask_path=os.path.join(seq_path,'autolabels/pre_processing/trajectory_masks')
    img_path=os.path.join(seq_path,'data/imgs')

    #output paths
    auto_label_path=os.path.join(seq_path,'autolabels/camera/labels')
    auto_label_overlaid_path=os.path.join(seq_path,'autolabels/camera/labels_overlaid')

    #create dirs if doesnt exists
    os.makedirs(auto_label_overlaid_path,exist_ok=True)
    os.makedirs(auto_label_path,exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg',verbose=False)
    model=dinov2_vitg14.to(device)

    frame=cv2.imread(os.path.join(img_path,'0.png'))
    H_dino_in=int(((frame.shape[0]*scale)//14)*14)  
    W_dino_in=int(((frame.shape[1]*scale)//14)*14)

    img_transform=T.Compose([T.ToTensor(),T.Resize((H_dino_in,W_dino_in)),T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]) #IMAGENET MEAN AND STD (As recommended by dinov2 authors)
    mask_transfrom=T.Compose([T.ToTensor(),T.Resize((H_dino_in//14,W_dino_in//14))])

    num_files = len(os.listdir(img_path))
    prev_mean_feature=None
    
    for data_id in tqdm(range(num_files)):
        
        #Read data
        frame=cv2.imread(os.path.join(img_path,str(data_id)+'.png'))
        rgb_img=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #Dinov2 expects images in RGB format
        mask=cv2.imread(os.path.join(trajectory_mask_path,str(data_id)+'.png'),cv2.IMREAD_GRAYSCALE).astype('bool')
        img_tensor=img_transform(rgb_img).unsqueeze(0).to(device) # 1,3,H,W. This format required from dinov2
        mask_tensor=mask_transfrom(mask).squeeze(0).to(device) # H,W

        #Compute dino_features
        features=dino_features(model,img_tensor)

        #Check if enough trajectory samples
        if mask.sum()<min_refrence_feats and prev_mean_feature is not None:
            mean_feat=prev_mean_feature
            print("Not enough refrence features, using previous valid sample")
        else:
            mean_feat=mean_feature(features,mask_tensor)
            prev_mean_feature=mean_feat

        #Compute cosine similarity based score
        score=cosine_similarity_score(features,mean_feat,std).cpu().numpy()

        #Save outputs
        score=cv2.resize(score*255,(frame.shape[1],frame.shape[0]),interpolation = cv2.INTER_LINEAR)
        label=utils.make_rgb_label(score)
        overlaid=utils.overlay_heatmap(frame,score/255)
        cv2.imwrite(os.path.join(auto_label_path,str(data_id)+'.png'),label)
        cv2.imwrite(os.path.join(auto_label_overlaid_path,str(data_id)+'.png'),overlaid)
main()