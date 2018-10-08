from skimage.measure import regionprops,label
from sklearn.metrics import jaccard_similarity_score
import numpy as np
import os
from PIL import Image

def CRAF(pred,truth):
    alpha=0.2
    for i in range(1,128+1,1):

        res=pred[i]
        res_prev=pred[i-1]
        CRS=[]
        for obj_id in range(1, np.max(res)+1, 1):
            img_reg = label(res == obj_id)
            j_objs=[]
            areas = []
            for j in range(1, np.max(img_reg) + 1, 1):
                areas.append(np.sum(img_reg == j))
                res_prev_obj=res_prev==obj_id
                tmp=img_reg==j
                j_objs.append(jaccard_similarity_score(tmp.flatten(),res_prev_obj.flatten()))
            if areas[np.argmax(j_objs)[0]]/max(areas)>alpha:
                CRS.append(img_reg == np.argmax(j_objs)[0])

def load_groundtruth(path):
    groundpath=os.path.join(path, 'groundtruth')
    det = []
    vol=[]
    for root, dirs, files in os.walk(groundpath):
        for file in files:
            if 'detections' in file:
                det.append(os.path.join(os.path.join(path, 'groundtruth'), file))
            if 'volumes' in file:
                vol.append(os.path.join(os.path.join(path, 'groundtruth'), file))
    return [det,vol]

ground_truth=load_groundtruth( 'E:/ai_challenger/validationset/Edema_validationset/')
dirval='E:/asd/val'
rea=[]
srf=[]
ped=[]


for ground_idx in range(15):
    print(ground_truth[1][ground_idx])
    vol_truth = np.load(ground_truth[1][ground_idx]).astype(np.uint8)
    for i in range(1,127+1,1):
        pre=vol_truth[i-1]
        now=vol_truth[i]
        if np.sum(now==1)>0:
            rea.append(jaccard_similarity_score((pre==1).flatten(),(now==1).flatten()))
        if np.sum(now==2)>0:
            srf.append(jaccard_similarity_score((pre == 2).flatten(), (now == 2).flatten()))
        if np.sum(now==3)>0:
            ped.append(jaccard_similarity_score((pre == 3).flatten(), (now == 3).flatten()))
for name in [rea,srf,ped]:
    print(len(name))
    print(np.median(name))
    print(np.mean(name))


rea=[]
srf=[]
ped=[]
for fp,dirs,fs in os.walk(dirval):
    for f in fs:
        if 'vol' in f:
            vol_pred=np.load(os.path.join(fp,f)).astype(np.uint8)
            for i in range(1, 127 + 1, 1):
                pre = vol_pred[i - 1]
                now = vol_pred[i]
                if np.sum(now == 1) > 0:
                    rea.append(jaccard_similarity_score((pre == 1).flatten(), (now == 1).flatten()))
                if np.sum(now == 2) > 0:
                    srf.append(jaccard_similarity_score((pre == 2).flatten(), (now == 2).flatten()))
                if np.sum(now == 3) > 0:
                    ped.append(jaccard_similarity_score((pre == 3).flatten(), (now == 3).flatten()))
for name in [rea,srf,ped]:
    print(len(name))
    print(np.median(name))
    print(np.mean(name))