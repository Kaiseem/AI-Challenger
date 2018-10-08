import tensorflow as tf
import time,os
from train import ChallengerAI
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('epoch', 200, "total epoch")
tf.app.flags.DEFINE_float('lr',0.001,'')
tf.app.flags.DEFINE_boolean('is_continue_train',False,'')
tf.app.flags.DEFINE_string('save_path','D:','')
tf.app.flags.DEFINE_string('save_name', 'E:/GAN/Challenger_AI/saved/model.ckpt', "saved parameters")
tf.app.flags.DEFINE_string('train_path', 'E:/ai_challenger/trainingset/Edema_trainingset/', "")
tf.app.flags.DEFINE_string('test_path', 'E:/ai_challenger/testset/Edema_testset/', "")
tf.app.flags.DEFINE_string('validation_path', 'E:/ai_challenger/validationset/Edema_validationset/', "")

def main(_):
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth=True
    tf_config.gpu_options.per_process_gpu_memory_fraction = 1.0
    path=os.path.join(FLAGS.save_path,str(time.time()))
    if not os.path.isdir(path):
        os.mkdir(path)
    with tf.Session(config=tf_config) as sess:
        model=ChallengerAI(sess,FLAGS,path)
        model.train()


if __name__ == '__main__':
    tf.app.run()



'''
def cm2mIoU(cm):
    # self.valid_preds = tf.reshape(self.pred_classes, [-1, ])
    # self.valid_weight=tf.maximum(tf.minimum(tf.cast(self.valid_labels,dtype=tf.float32),tf.ones_like(self.valid_labels,dtype=tf.float32)),0.5*tf.ones_like(self.valid_labels,dtype=tf.float32))
    # self.confusion_matrix = tf.confusion_matrix(self.valid_labels, self.valid_preds, num_classes=self.class_num)
    # self.IoUloss=tf.cast(tf.diag_part(self.confusion_matrix)/(tf.reduce_sum(self.confusion_matrix+tf.transpose(self.confusion_matrix,[1,0]),axis=0)-tf.diag_part(self.confusion_matrix)+1),dtype=tf.float32)
    # self.mIoU,self.IoU=tf.py_func(cm2mIoU,[self.confusion_matrix],[tf.float32,tf.float32])
    # self.mIoU_sum=tf.summary.scalar('mIoU',self.mIoU)
    cm = np.array(cm, dtype=np.float32)
    IoU = []
    mean = []
    for i in range(4):
        if (cm[i][1] + cm[i][2] + cm[i][3] + cm[i][0] + cm[1][i] + cm[2][i] + cm[3][i] + cm[0][i] - cm[i][i]) != 0:
            IoU.append(cm[i][i] / (cm[i][1] + cm[i][2] + cm[i][3] + cm[i][0] + cm[1][i] + cm[2][i] + cm[3][i] + cm[0][i] - cm[i][i]))
            mean.append(cm[i][i] / (cm[i][1] + cm[i][2] + cm[i][3] + cm[i][0] + cm[1][i] + cm[2][i] + cm[3][i] + cm[0][i] - cm[i][i]))
        else:
            IoU.append(-1)
    mIoU = np.mean(mean, dtype=np.float32)
    IoU = np.array(IoU, dtype=np.float32)
    return mIoU, IoU
'''