import cv2
import torch
import torchvision.transforms as T
import os
import utils
import torch.nn.functional as F
import yaml
from tqdm import tqdm
import sys
import numpy as np

def dino_features(
    model: torch.nn.Module, #dinov2 pre-trained feature extractor
    transform: T.Compose,   #desired transforms for the input image
    image: np.ndarray       #shape: (height,width,3). Input image in BGR format
) ->torch.tensor:           #shape: (height//patch_size,width//patch_size,size of patch feature). Features for each image patch
    
    """
    Computes features for an image
    """

    rgb_img=cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Dinov2 expects images in RGB format
    norm_image=transform(rgb_img)
    with torch.no_grad():
        features = model.get_intermediate_layers(norm_image.unsqueeze(0).cuda(), n=1)[0]
    features=features.reshape(norm_image.shape[1]//14,norm_image.shape[2]//14,1536)
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
    day=sys.argv[1]
    seq=sys.argv[2]

    with open('params_cadcd.yaml', 'r') as file:
        params = yaml.safe_load(file)
        config=params['dino_auto_labeling']
        scale=config['img_scale']
        std=config['dino_label_std']
        min_refrence_feats=config['min_refrence_feats']
        dataset_path=params['dataset_path']

    seq_path=os.path.join(dataset_path,'processed',day,seq)

    #input paths
    trajectory_mask_path=os.path.join(seq_path,'autolabels/pre_processing/trajectory_masks')
    img_path=os.path.join(seq_path,'data/imgs')

    #output paths
    auto_label_path=os.path.join(seq_path,'autolabels/dino/labels')
    auto_label_overlaid_path=os.path.join(seq_path,'autolabels/dino/labels_overlaid')

    #create dirs if doesnt exists
    os.makedirs(auto_label_overlaid_path,exist_ok=True)
    os.makedirs(auto_label_path,exist_ok=True)

    dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg',verbose=False)
    model=dinov2_vitg14.cuda()
    transform=T.Compose([T.ToTensor(),T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]) #IMAGENET MEAN AND STD (As recommended by dinov2 authors)

    num_files = len(os.listdir(img_path))
    prev_mean_feature=None
    
    for data_id in tqdm(range(num_files)):
        frame=cv2.imread(os.path.join(img_path,str(data_id)+'.png'))
        mask=cv2.imread(os.path.join(trajectory_mask_path,str(data_id)+'.png'),cv2.IMREAD_GRAYSCALE)

        H_dino_in=int(((frame.shape[0]*scale)//14)*14)  
        W_dino_in=int(((frame.shape[1]*scale)//14)*14)  

        mask=cv2.resize(mask,(W_dino_in//14,H_dino_in//14),cv2.INTER_AREA)
        mask=torch.tensor(mask).bool()
        input_img=cv2.resize(frame,(W_dino_in,H_dino_in),cv2.INTER_AREA)
        features=dino_features(model,transform,input_img)

        if mask.sum()<min_refrence_feats and prev_mean_feature is not None:
            mean_feat=prev_mean_feature
            print("Not enough refrence features, using previous valid sample")
        else:
            mean_feat=mean_feature(features,mask)
            prev_mean_feature=mean_feat

        score=cosine_similarity_score(features,mean_feat,std).cpu().numpy()
        score=cv2.resize(score*255,(frame.shape[1],frame.shape[0]),interpolation = cv2.INTER_LINEAR)
        label=utils.make_rgb_label(score)
        overlaid=utils.overlay_heatmap(frame,score/255)

        cv2.imwrite(os.path.join(auto_label_path,str(data_id)+'.png'),label)
        cv2.imwrite(os.path.join(auto_label_overlaid_path,str(data_id)+'.png'),overlaid)
main()