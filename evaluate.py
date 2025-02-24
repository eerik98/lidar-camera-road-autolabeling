import numpy as np
import cv2
import os
import yaml
import sys

def compute_metrics(autolabel_path,GT_path):
    TP=0
    FP=0
    FN=0
    for i in range(len(os.listdir(autolabel_path))):
        
        autolabel=cv2.imread(os.path.join(autolabel_path,str(i)+'.png'),cv2.IMREAD_GRAYSCALE).astype('bool')
        GT=cv2.imread(os.path.join(GT_path,str(i)+'.png'),cv2.IMREAD_GRAYSCALE).astype('bool')
       
        TP+=(autolabel&GT).sum()
        FP+=(autolabel&(~GT)).sum()
        FN+=((~autolabel)&GT).sum()
    
    return TP,FP,FN

def compute_IoU(TP,FP,FN):
    return TP/(TP+FP+FN)

def compute_PRE(TP,FP):
    return TP/(TP+FP)

def compute_REC(TP,FN):
    return TP/(TP+FN)

def compute_F1(TP,FP,FN):
    return 2*TP/(2*TP+FP+FN)

def print_results(TP,FP,FN):
    print("IoU: {:.3f}".format(compute_IoU(TP, FP, FN)))
    print("PRE: {:.3f}".format(compute_PRE(TP, FP)))
    print("REC: {:.3f}".format(compute_REC(TP, FN)))
    print("F1:  {:.3f}".format(compute_F1(TP, FP, FN)))


def main():

    seq=sys.argv[1]
    param_file=sys.argv[2]

    with open(param_file, 'r') as file:
       params = yaml.safe_load(file)
       dataset_path=params['dataset_path']

    autolabel_path=os.path.join(dataset_path,'processed',seq,'autolabels','post_processing','labels')
    GT_path=os.path.join(dataset_path,'processed',seq,'labels')


    TP,FP,FN=compute_metrics(autolabel_path,GT_path)

    print("Evaluation for "+seq+':')
    print_results(TP,FP,FN)

if __name__=="__main__": 
    main()

