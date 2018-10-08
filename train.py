import model
import loader
import tensorflow as tf
import numpy as np
import time,os,gc
from Challenger_AI.utils import SwatsOptimizer
from Challenger_AI.utils import Loss as myloss
from Challenger_AI.utils import AdamOptimizer
from Challenger_AI.utils import NAdamOptimizer
class ChallengerAI(object):
    def __init__(self,sess,flag,path):
        self.path=path
        self.sess=sess
        self.step=tf.Variable(0, trainable=False)
        self.global_step=0
        self.flags=flag
        self.image_size=[1024,512]
        self.model=model.deeplabV3plus
        #self.model=model.FCDenseNet103
        #self.model=model.Nest_Net
        self.class_num=4
        self._build_model()
    def _build_model(self):
        def get_weight(valid_label):
            valid_label=np.array(valid_label,dtype=np.float32)
            class_weight=[1.43,14.3,35,50]
            scale=0.25
            for i in  [3,2,1,0]:
                valid_label[valid_label==i]=class_weight[i]*scale
            return valid_label
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.image = tf.placeholder(tf.float32, [None, self.image_size[0], self.image_size[1], 1], name='image')
        self.label = tf.placeholder(tf.int32, [None, self.image_size[0], self.image_size[1], 1], name='label')
        self.detection= tf.placeholder(tf.int64, [None,3], name='detection')

        self.valid_labels = tf.reshape(tf.squeeze(self.label, axis=3), [-1, ])

        self.is_train=tf.placeholder(tf.bool, None, name='training_phase')
        self.logits,self.pred_detections=self.model(self.image,self.is_train, self.valid_labels,num_classes=self.class_num, reuse=False)
        self.pred_det=tf.nn.softmax(self.pred_detections,dim=-1)
        self.pred_classes = tf.expand_dims(tf.argmax(self.logits, axis=3, output_type=tf.int32), axis=3)#mask

        self.valid_pred_detection=tf.reshape(self.pred_detections,[-1,2])
        self.valid_detection=tf.reshape(self.detection,[-1,])
        self.valid_logits = tf.reshape(self.logits, [-1, self.class_num])



        #self.dice_loss=self.generalised_dice_loss(self.valid_logits, self.valid_labels, self.class_num)
        #self.cross_entropy=self.generalised_wasserstein_dice_loss(self.valid_logits, self.valid_labels, self.class_num)
        #self.cross_entropy=self.generalised_dice_loss(self.valid_logits, self.valid_labels, self.class_num)

        self.encoder_wd=tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'kernel' in v.name and 'det_model' not in v.name and 'seg_model' not in v.name and 'depthwise' not in v.name])*0.00004
        self.seg_wd=tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'kernel' in v.name and 'seg_model' in v.name and 'depthwise' not in v.name])*0.0001
        self.det_wd=tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'kernel' in v.name and 'det_model' in v.name and 'depthwise' not in v.name])*0.0001
        self.det_loss=tf.losses.sparse_softmax_cross_entropy(logits=self.valid_pred_detection,labels=self.valid_detection)+self.det_wd
        [weights]=tf.py_func(get_weight,[self.valid_labels],[tf.float32])
        self.seg_loss=tf.losses.sparse_softmax_cross_entropy(logits=self.valid_logits, labels=self.valid_labels,weights=weights)+self.encoder_wd+self.seg_wd
        #self.seg_loss=myloss.focal_loss(self.valid_logits,self.valid_labels)+self.encoder_wd+self.seg_wd
        self.loss_sum = tf.summary.scalar("loss", self.det_loss)
        self.sum = tf.summary.merge([self.loss_sum])
        b=[]
        for var in tf.trainable_variables():
            if 'kernel' in var.name:
                c=1
                a=var.get_shape().as_list()
                for i in range(len(a)):
                    c*=a[i]
                print(var.name,var.shape,c)
                b.append(c)
                del a,c
        print(sum(b))
        del b


    def train(self):
        def warm_up(step, total_step):
            lr = self.flags.lr
            return 0.05 * lr + 0.95 * lr * (step / total_step)

        def cos_decay(lr=None, gs=None, decay_steps=150000, alpha=0.05):
            from math import cos, pi
            if lr == None:
                lr = self.flags.lr
                gs = self.global_step
            global_step = min(gs, decay_steps)
            cosine_decay = 0.5 * (1 + cos(pi * global_step / decay_steps))
            decayed = (1 - alpha) * cosine_decay + alpha
            return lr * decayed
        seg_list=[]
        det_list=[]

        for var in tf.trainable_variables():
            if 'det_model' not in var.name:
                seg_list.append(var)
            else:
                det_list.append(var)
        #self.optim = tf.train.AdamOptimizer(self.lr, beta1=0.9,beta2=0.999).minimize(self.seg_loss,var_list=seg_list,global_step=self.step)
        #self.optim=SwatsOptimizer(0.001).minimize(self.seg_loss,var_list=seg_list,global_step=self.step)
        self.optim =NAdamOptimizer(self.lr, beta1=0.9,beta2=0.999).minimize(self.seg_loss,var_list=seg_list,global_step=self.step)
        self.detopt = tf.train.AdamOptimizer(0.0003).minimize(self.det_loss,var_list=det_list)
        #Nesterov动量法
        var_list=[]
        for var in tf.trainable_variables():
            var_list.append(var)

        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
        self.saver = tf.train.Saver(var_list=var_list, max_to_keep=5)


        if (self.flags.is_continue_train):
            ckpt = tf.train.get_checkpoint_state("./parameters/")
            if ckpt and ckpt.model_checkpoint_path:
                print('successfully load')
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            init = tf.global_variables_initializer()
            self.sess.run(init)
        batch_size = 1
        self.train_data=loader.load_path(self.flags.train_path)
        self.validation_data=loader.load_path(self.flags.validation_path)
        self.ground_truth=loader.load_groundtruth(self.flags.validation_path)
        self.test_data=loader.load_path(self.flags.test_path)
        self.pre_train_data = loader.pre_train_path(self.train_data)
        if False:

            self.sess.run(tf.assign(self.step,-len(self.pre_train_data)))
            np.random.shuffle(self.pre_train_data)
            for idx in range(len(self.pre_train_data)):
                pre_train_image, pre_train_label ,pre_train_det= loader.load_train_data(self.train_data[idx * batch_size:(idx + 1) * batch_size])
                lr=warm_up(idx,len(self.pre_train_data))
                if idx%200==0:
                    print('warming up %4d step with %2.5f lr'%(idx,lr))
                _= self.sess.run(self.optim, feed_dict={self.image: pre_train_image, self.label: pre_train_label, self.detection:pre_train_det,self.lr: lr, self.is_train: True})
                del pre_train_image,pre_train_label



        batch_idxs = len(self.pre_train_data)
        for epoch in range(self.flags.epoch):
            st_time=time.time()
            gc.collect()
            print('garbege collection cost time:',time.time()-st_time)
            np.random.shuffle(self.pre_train_data)
            start_time = time.time()
            print('epoch: %2d'%(epoch))
            for idx in range(int(batch_idxs / batch_size)):
            #for idx in range(128):
                train_image, train_label ,train_det= loader.load_train_data(self.pre_train_data[idx * batch_size:(idx + 1) * batch_size])
                lr = cos_decay()
                _, self.global_step, summary_str, loss, wd1,wd2,wd3 = self.sess.run([self.optim, self.step, self.sum, self.seg_loss, self.encoder_wd,self.seg_wd,self.det_wd], feed_dict={self.image: train_image, self.label: train_label, self.detection:train_det,self.lr: lr, self.is_train: True})

                if not os.path.isdir(os.path.join(self.path,'sample')):
                    os.mkdir(os.path.join(self.path,'sample'))
                if (idx * batch_size) % 8 == 0:
                    pred = self.sess.run(self.pred_classes, feed_dict={self.image: train_image, self.label: train_label, self.is_train: False})
                    loader.save_images(train_image[0], train_label[0], pred[0], os.path.join((os.path.join(self.path,'sample')), str(self.global_step) + '.png'))

                if (idx * batch_size) % 128 == 0:
                    print(("Epoch: [%2d] [%4d/%4d] [%4d] Cost time: %4.4f loss: %4.4f encoder wd: %4.4f seg wd: %4.4f det wd: %4.4f" % (
                        epoch, (idx + 1) * batch_size, batch_idxs, self.global_step, time.time() - start_time, loss,wd1/0.00004,wd2/0.0001,wd3/0.0001)))
                self.writer.add_summary(summary_str, self.global_step)
                del train_image,train_label

            print('epoch %2d end cost time %4.4f'%(epoch,time.time()-start_time))
            self.validation(epoch)
            if epoch>5:
                self.test(epoch)
            self.saver.save(self.sess, "./parameters/model" + str(epoch) + ".ckpt", global_step=self.global_step)

    def validation(self, epoch):
        def aic_fundus_lesion_segmentation(ground_truth, prediction, num_samples=128):
            assert (ground_truth.shape == (num_samples, 1024, 512))
            assert (prediction.shape == (num_samples, 1024, 512))
            ground_truth = np.reshape(ground_truth,[128*1024*512])
            prediction =np.reshape(prediction,[128*1024*512])
            ret = [0.0, 0.0, 0.0, 0.0]
            for i in range(4):
                mask1 = (ground_truth == i)
                mask2 = (prediction == i)
                if np.sum(mask1) != 0:
                    ret[i] = 2 * (np.sum(mask1 * (ground_truth == prediction))) / (np.sum(mask1) + np.sum(mask2))
                else:
                    ret[i] = float('nan')
            del ground_truth,prediction
            return ret
        def aic_fundus_lesion_classification(ground_truth, prediction, num_samples=128):
            assert (ground_truth.shape == (num_samples, 3))
            assert (prediction.shape == (num_samples, 3))
            from sklearn import metrics
            ret = [0.5, 0.5, 0.5]
            for i in range(3):
                fpr, tpr, thresholds = metrics.roc_curve(ground_truth[:, i], prediction[:, i], pos_label=1)
                ret[i] = metrics.auc(fpr, tpr)
            return ret

        if not self.validation_data:
            print('wrong validation path 嘤嘤嘤')
        else:
            gc.collect()
            no_detection = True
            self.vol_pred = []
            self.det_pred = []
            ret_segmentation = []
            for idx in range(len(self.validation_data)):
            #for idx in range(128):
                #print(self.validation_data[idx][0], idx, str(1 + (idx) // 128) + '_' + str((idx) % 128+1))
                validation_image, validation_label ,validation_det= loader.load_validation_data(self.validation_data[idx])
                det,pred, asd = self.sess.run([self.pred_det,self.pred_classes, self.logits], feed_dict={self.image: validation_image, self.label: validation_label, self.detection:validation_det,self.is_train: False})
                if not os.path.isdir(os.path.join(self.path,'validation_result')):
                    os.mkdir(os.path.join(self.path,'validation_result'))
                val_path=(os.path.join((os.path.join(self.path,'validation_result')) , str(epoch) + '_val'))

                if not os.path.isdir(val_path):
                    os.mkdir(val_path)
                if epoch >5:
                    loader.save_images(validation_image[0], validation_label[0], pred[0], os.path.join(val_path, str(1 + (idx) // 128) + '_' + str((idx) % 128+1) + '.png'))
                self.vol_pred.append(pred[0])
                self.det_pred.append([det[0][0][1],det[0][1][1],det[0][2][1]])
                if (idx + 1) % 128 == 0:
                    print('processing: '+os.path.basename(os.path.dirname(self.validation_data[idx][0])))
                    ground_idx = (idx + 1) // 128-1
                    det_truth = np.load(self.ground_truth[0][ground_idx]).astype(np.int64)
                    vol_truth = np.load(self.ground_truth[1][ground_idx]).astype(np.uint8)
                    det_pred = np.array(self.det_pred, dtype=np.float32)
                    vol_pred = np.squeeze(np.array(self.vol_pred, dtype=np.uint8),axis=-1)
                    rea = vol_pred == 255
                    srf = vol_pred == 191
                    ped = vol_pred == 128
                    vol_pred[rea] = 1
                    vol_pred[srf] = 2
                    vol_pred[ped] = 3
                    vol_result=aic_fundus_lesion_segmentation(vol_truth,vol_pred)
                    ret_segmentation.append(vol_result)
                    self.det_pred = []
                    self.vol_pred = []
                    if no_detection:
                        detection_ref_all = det_truth
                        detection_pre_all = det_pred
                        no_detection = False
                    else:
                        detection_ref_all = np.concatenate((detection_ref_all,det_truth), axis=0)
                        detection_pre_all = np.concatenate((detection_pre_all, det_pred), axis=0)
                    del vol_truth, vol_result,det_truth,validation_image,validation_det,validation_label
            det_result = aic_fundus_lesion_classification(detection_ref_all,detection_pre_all,num_samples=15*128)
            REA_detection, SRF_detection, PED_detection =det_result[0], det_result[1],det_result[2]
            REA_segementation, SRF_segementation, PED_segementation = 0.0, 0.0, 0.0
            n1, n2, n3 = 0, 0, 0
            import math
            for i in range(15):
                if not math.isnan(ret_segmentation[i][1]):
                    REA_segementation += ret_segmentation[i][1]
                    n1 += 1
                if not math.isnan(ret_segmentation[i][2]):
                    SRF_segementation += ret_segmentation[i][2]
                    n2 += 1
                if not math.isnan(ret_segmentation[i][3]):
                    PED_segementation += ret_segmentation[i][3]
                    n3 += 1
            REA_segementation /= n1
            SRF_segementation /= n2
            PED_segementation /= n3
            avg_detection = (REA_detection + SRF_detection + PED_detection) / 3
            avg_segmentation = (REA_segementation + SRF_segementation + PED_segementation) / 3
            print('epoch %2d, avg_seg: %4.8f, avg_det:%4.8f'%(epoch,  avg_segmentation,avg_detection))


    def test(self, epoch):
        if not self.test_data:
            print('wrong validation path 嘤嘤嘤')
        else:
            gc.collect()
            self.det_pred = []
            self.vol_pred = []
            for idx in range(len(self.test_data)):
                test_image,test_label ,test_det= loader.load_test_data(self.test_data[idx])

                det,pred = self.sess.run([self.pred_det,self.pred_classes], feed_dict={self.image: test_image,self.label:test_label,self.detection:test_det, self.is_train: False})

                if not os.path.isdir(os.path.join(self.path,'test_result')):
                    os.mkdir(os.path.join(self.path,'test_result'))
                test_path=os.path.join(os.path.join(self.path,'test_result'), str(epoch) + '_val')
                if not os.path.isdir(test_path):
                    os.mkdir(test_path)
                loader.save_images(test_image[0], pred[0], pred[0],  os.path.join(test_path ,str(1 + (idx) // 128) + '_' + str((idx) % 128+1) + '.png'))

                self.vol_pred.append(pred[0])
                self.det_pred.append([det[0][0][1],det[0][1][1],det[0][2][1]])

                if (idx + 1) % 128 == 0:
                    print('processing: '+os.path.basename(os.path.dirname(self.test_data[idx][0])))
                    volums_name = os.path.basename(os.path.dirname(self.test_data[idx][0])[:-4] + '_volumes.npy')
                    detect_name = os.path.basename(os.path.dirname(self.test_data[idx][0])[:-4]+ '_detections.npy')
                    path_name=os.path.join(test_path,'uploading')
                    if not os.path.isdir(path_name):
                        os.mkdir(path_name)
                    det_pred = np.array(self.det_pred, dtype=np.float32)
                    vol_pred = np.squeeze(np.array(self.vol_pred, dtype=np.uint8),axis=-1)
                    rea = vol_pred == 255
                    srf = vol_pred == 191
                    ped = vol_pred == 128
                    vol_pred[rea] = 1
                    vol_pred[srf] = 2
                    vol_pred[ped] = 3
                    np.save(os.path.join(path_name, volums_name), vol_pred)
                    np.save(os.path.join(path_name,detect_name), det_pred)
                    del rea,srf,ped,test_label,test_image,test_det
                    self.det_pred = []
                    self.vol_pred = []
