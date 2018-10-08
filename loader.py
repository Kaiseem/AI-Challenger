from scipy.misc import imread
import scipy,random
import numpy as np
import os,gc
from scipy import ndimage
import tensorflow as tf

'''
Background：0

PED: 128

SRF: 191

REA: 255

'''


def save_images(images,label,pred, image_path):
    images=(images+1.)/2.*255
    ped = label == 1
    srf = label == 2
    rea = label == 3
    label[ped] = 128
    label[srf] = 191
    label[rea] = 255
    ped1 = pred == 1
    srf1 = pred == 2
    rea1 = pred == 3
    pred[ped1] = 128
    pred[srf1] = 191
    pred[rea1] = 255
    img=np.concatenate([images,label,pred],axis=1).astype(np.uint8)
    scipy.misc.imsave(image_path,img[:,:,-1])
    del ped1,srf1,rea1,img,images,label,pred

def load_path(path):
    if not (os.path.isdir(path) ):
        raise Exception("Wrong path! %s"%(path))
    imgpath=os.path.join(path,'original_images')
    a=[]
    for root, dirs, files in os.walk(imgpath):
        label_root = root.replace('original_images', 'label_images')[:-4] + '_labelMark'
        if files!=[]:
            det_root=root.replace('original_images', 'groundtruth')[:-4]+'_labelMark_detections.npy'
            det=np.load(det_root)
            files.sort(key=lambda num:int(num[:-4]))
            for file in files:
                index=int(os.path.splitext(file)[0])-1
                a.append([os.path.join(root,file),os.path.join(label_root,file),det[index].tolist()])
            del det
    return a

def pre_train_path(path_list):
    pre_list=[]
    count=0
    for path in path_list:
        y = scipy.misc.imread(path[1], mode='L')
        ped_sum=(y==191).sum()
        if ped_sum>0:
            pre_list.append(path)
        else:
            count+=1
            if count%2==0:
                pre_list.append(path)
        del y
    print('%4d/%4d used to pre-trained warm up'%(len(pre_list),len(path_list)))
    return pre_list



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

def load_train_data(path):#(4664,4296)
    out_image=[]
    out_label=[]
    image_path=[path[i][0] for i in range(len(path))]
    out_det=[path[i][2] for i in range(len(path))]
    for image in image_path:
        x = scipy.misc.imread(image, mode='L').astype(np.float32)
        y = scipy.misc.imread(os.path.join(os.path.dirname(image).replace('original_images', 'label_images')[:-4] + '_labelMark',os.path.basename(image)), mode='L').astype(np.int32)
        x=x/127.5-1
        ped=y==128
        srf=y==191
        rea=y==255
        y[ped]=1
        y[srf]=2
        y[rea]=3
        x,y=data_augmentation(x,y)
        x=np.expand_dims(x,axis=-1)
        out_image.append(x)
        y=np.expand_dims(y,axis=-1)
        out_label.append(y)
        del x,y,ped,srf,rea
    return out_image,out_label,out_det
def load_test_data(image_path):
    x = scipy.misc.imread(image_path[0], mode='L').astype(np.float32)
    x = x / 127.5 - 1
    x=[np.expand_dims(x,axis=-1)]
    y=np.zeros_like(x)
    return x,y,[image_path[2]]
def load_validation_data(path):#(4664,4296)
    x = scipy.misc.imread(path[0], mode='L').astype(np.float32)
    y = scipy.misc.imread(path[1], mode='L').astype(np.int32)
    z=[path[2]]
    x=x/127.5-1
    ped=y==128
    srf=y==191
    rea=y==255
    y[ped]=1
    y[srf]=2
    y[rea]=3
    out_image=[np.expand_dims(x,axis=-1)]
    out_label=[np.expand_dims(y,axis=-1)]
    del x,y,ped,srf,rea
    return out_image,out_label,z

def data_augmentation(x,y):
    mode = random.randint(0, 2)
    if mode!=0:
        lh = random.randint(128, 384)
        st = random.randint(64, 448 - lh)
        rg = random.randint(0, 30)
        offset=random.randint(0,1)
        for j in range(st,st+lh,1):
            newi = int(rg * np.sin(mode*np.pi * (j-st) / lh+np.pi*offset))
            x[max(0,-newi):min(1024,1024-newi),j]=x[max(0,newi):min(1024,1024+newi),j]
            y[max(0, -newi):min(1024, 1024 - newi), j] =y[max(0, newi):min(1024, 1024 + newi), j]
    x=np.pad(x,((256,256),(0,0)),'constant',constant_values = (-1,-1))
    y = np.pad(y, ((256, 256), (0, 0)), 'constant', constant_values=(0, 0))
    clipnum=random.randint(0,511)
    x=x[clipnum:clipnum+1024,:]
    y=y[clipnum:clipnum+1024,:]
    angle=random.randint(-15, 15)
    x=ndimage.rotate(x, angle, order=0,cval=-1, reshape=False)
    y = ndimage.rotate(y, angle, order=0, reshape=False)
    if random.random()>0.5:
        x=np.fliplr(x)
        y=np.fliplr(y)
    del clipnum,angle
    return x,y
'''
from PIL import Image
import numpy as np
fp='E:/ai_challenger/trainingset/Edema_trainingset/original_images/P0153_MacularCube512x128_7-3-2014_10-20-51_OD_sn25894_cube_z.img/1.bmp'
im=Image.open(fp)
img=np.array(im)
img2=img

for j in range(200,300):
        newi= int(60 * np.sin(np.pi * (j - 200) / 100))
        if newi>0:
            img2[0:1024-newi,j]=img[newi:1024,j]
im=Image.fromarray(img2)
im.show()


'''