import os,scipy
import numpy as np


import sys
import requests
import os
requests.packages.urllib3.disable_warnings()

def download(url, file_path):
    r1 = requests.get(url, stream=True, verify=False)
    total_size = int(r1.headers['Content-Length'])
    if os.path.exists(file_path):
        temp_size = os.path.getsize(file_path)
    else:
        temp_size = 0
    print(temp_size)
    print(total_size)
    headers = {'Range': 'bytes=%d-' % temp_size}
    r = requests.get(url, stream=True, verify=False, headers=headers)
    with open(file_path, "ab") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                temp_size += len(chunk)
                f.write(chunk)
                f.flush()
                done = int(50 * temp_size / total_size)
                sys.stdout.write("\r[%s%s] %d%%" % ('█' * done, ' ' * (50 - done), 100 * temp_size / total_size))
                sys.stdout.flush()
    print()
trainURL='http://ai-challenger.ufile.ucloud.cn/ai_challenger_fl2018_trainingset.zip?UCloudPublicKey=1zlyEEtcBCAM/exE02Mcc0eWT1fXG7t2B8CqM+Ywr8jJvyXuKn7Dog==&Expires=1538702520&Signature=dZ9qztXws7lklmNWqzbynhdWn+M='

indir='D:/ai_challenger/trainingset/Edema_trainingset/label_images'
outdir='D:/ai_challenger/trainingset/Edema_trainingset/groundtruth'

from scipy.misc import imread
for fpath,dirs,fs in os.walk(indir):
    for dir in dirs:
        saved_path=os.path.join(outdir,str(dir)+'_detections.npy')
        #print(saved_path)
        det=np.zeros((128,3),dtype=np.int64)
        for i in range(128):
            img_path=os.path.join(os.path.join(fpath,dir),str(i+1)+'.bmp')
            img_data=scipy.misc.imread(img_path, mode='L')
            #rea
            if (img_data==255).sum()>0:
                det[i][0]=1
            #srf
            if (img_data==191).sum()>0:
                det[i][1]=1
            #ped
            if (img_data==255).sum()>0:
                det[i][2]=1
        np.save(saved_path,det)
print('done1')
indir='D:/ai_challenger/testset/Edema_testset/original_images'
outdir='D:/ai_challenger/testset/Edema_testset/groundtruth'
for fpath,dirs,fsd in os.walk(indir):
    for dir in dirs:
        nname=os.path.join(outdir,dir[:-4]+'_labelMark_detections.npy')
        a=np.zeros((128,3),dtype=np.int64)
        np.save(nname,a)
print('done2')