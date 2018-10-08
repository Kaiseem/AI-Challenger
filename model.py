import tensorflow as tf
from ops import conv,dsconv,sn,relu,bn,selu

def deeplabV3plus(image, is_train, valid_label, num_classes=4, reuse=False, name="deeplab"):
    def resnet_block(x, out_channel, name,strides=2,repeat_num=12):
        if strides==2:
            r0 = sn(conv(x, out_channel[2], 1, name + '_res_0', s=strides), name + '_res_0_sn')
            r1 = sn(dsconv(relu(x), out_channel[0], 3, name + '_res_1'), name + '_res_1_sn')
            r2 = sn(dsconv(relu(r1), out_channel[1], 3, name + '_res_2'), name + '_res_2_sn')
            r3 = sn(dsconv(relu(r2), out_channel[2], 3, name + '_res_3', s=strides), name + '_res_3_sn')
            x=r0+r3
        elif strides==1:
            for i in range(repeat_num):
                r_name = name + '_' + str(i)
                h = sn(dsconv(relu(x), out_channel[0], 3, r_name + '_res_1'), r_name + '_res_sn_1')
                h = sn(dsconv(relu(h), out_channel[1], 3, r_name + '_res_2'), r_name + '_res_sn_2')
                h = sn(dsconv(relu(h), out_channel[2], 3, r_name + '_res_3'), r_name + '_res_sn_3')
                x = x + h
        return x

    def atrous_spatial_pyramid_pooling(inputs, is_train, depth=256,DenseMode=False):  # [6,12,18]
        inputs_size = tf.shape(inputs)[1:3]
        conv_1x1 = conv(inputs, depth, 1, 'aspp_1x1', rate=1)
        conv_3x3_1 = sn(dsconv(relu(inputs), depth, 3, 'aspp_3x3_1', rate=6))
        conv_3x3_2 = sn(dsconv(relu(inputs), depth, 3, 'aspp_3x3_2', rate=12))
        conv_3x3_3 = sn(dsconv(relu(inputs), depth, 3, 'aspp_3x3_3', rate=18))

        image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keep_dims=True)
        image_level_features = sn(conv(relu(image_level_features), depth, 1, 'aspp_ilf_1x1'), 'aspp_sn')
        image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')
        net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
        net = relu(sn(conv(net, depth, 1, 'conv_1x1_concat'), 'aspp_sn1'))
        net = tf.cond(is_train, lambda: tf.nn.dropout(net, 0.9), lambda: net)
        return net

    def DenseASPP(input, is_train, depth=512):
        inputs_size = tf.shape(input)[1:3]
        conv_1x1 = relu(sn(conv(input, 512, 1, 'aspp_1x1'), 'assp_sn1'))
        stack = conv_1x1
        drate = [1, 2, 4, 8]
        for i in range(4):
            l = relu(sn(conv(stack, 128, 3, 'aspp_3x3_' + str(i + 1), rate=drate[i]), is_train, 'aspp_3x3_bn' + str(i + 1)))
            stack = tf.concat([stack, l], axis=-1)
        net = relu(sn(conv(stack, depth, 1, 'conv_1x1_concat'), 'aspp_sn1'))
        return net

    def AMSoftMaxConv(input, valid_label, is_train, nrof_classes=4):
        m = 0.35
        s = 30
        input = tf.reshape(input, [-1, input.get_shape().as_list()[-1]])
        input_norm = tf.nn.l2_normalize(input, 1, 1e-10, name='embeddings')
        kernel = tf.get_variable(name='kernel', dtype=tf.float32, shape=[input.get_shape().as_list()[-1], nrof_classes], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        kernel_norm = tf.nn.l2_normalize(kernel, 0, 1e-10, name='kernel_norm')
        cos_theta = tf.matmul(input_norm, kernel_norm)
        cos_theta = tf.clip_by_value(cos_theta, -1, 1)
        phi = cos_theta - m
        adjust_theta = tf.cond(is_train, lambda: s * tf.where(tf.equal(tf.one_hot(valid_label, nrof_classes), 1), phi, cos_theta), lambda: s * cos_theta)
        return tf.reshape(adjust_theta, [-1, 1024, 512, 4])

    def resize_convolution(x, size, hnum, is_train, name='rszconv'):
        y = tf.image.resize_images(x, (2 * size, size), method=1)
        y = relu(sn(conv(y, hnum, 3, name + '_rszconv'), name + '_sn'))
        return y

    def Dense_Upsampling_Conv(input, nrof_classes=4, d=16):
        input_size = input.get_shape().as_list()
        return tf.reshape(conv(input, num_classes * d * d, 1, 'DUC'), [-1, input_size[1] * d, input_size[2] * d, nrof_classes])

    def det_gain(det):
        import numpy as np
        rea = det[0][0][1]
        srf = det[0][1][1]
        ped = det[0][2][1]
        if srf > 0.1:
            srf = 1
        if ped > 0.1:
            ped = 1
        output = np.array([1, ped, srf, 1], dtype=np.float32)
        return output

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        inputs_size = tf.shape(image)[1:3]

        # Block1 512*512*3>>256*256*64
        b0 = relu(sn(conv(image, 32, 3, 'block1_conv_1', s=2), 'sn_1'))
        b1 = relu(sn(conv(b0, 64, 3, 'block1_conv_2'), 'sn_2'))

        # Block2 256*256*64>>128*128*128
        b2 = resnet_block(b1, [128, 128, 128], 'block2')
        low_level_features = relu(sn(conv(b2, 48, 1, 'low_conv'), 'low_conv_sn'))
        low_level_features_size = tf.shape(low_level_features)[1:3]

        # Block3 128*128*128>>64*64*256
        b3 = resnet_block(b2, [256, 256, 256], 'block3')

        # Block4 64*64*256 (728)>>32*32*1024
        b5 = resnet_block(b3, [256,256,256], 'block4',strides=1,repeat_num= 12)

        b7 = resnet_block(b5, [384, 512, 512], 'block4_exit_conv')

        b8 = sn(dsconv(relu(b7), 512, 3, 'block4_exit_res_4'))
        b9 = sn(dsconv(relu(b8), 768, 3, 'block4_exit_res_5'))
        b10 = sn(dsconv(relu(b9), 1024, 3, 'block4_exit_res_6'))

        '''
        with tf.variable_scope('seg_model'):
            # ASPP 32*32*1024>>32*32*256
            b11 = atrous_spatial_pyramid_pooling(b10, is_train)
            b12=tf.image.resize_bilinear(b11, low_level_features_size, name='upsample_1')#256*128
            b13=tf.concat([b12,low_level_features],axis=3,name='concat')
            b14=relu(sn(dsconv(b13,256,3,'concat_3x3_1'),is_train,'concat_3x3_sn1'))
            b15=relu(sn(dsconv(b14,256,3,'concat_3x3_2'),is_train,'concat_3x3_sn2'))
            b16=conv(b15,num_classes,1,'output')
            logits = tf.image.resize_bilinear(b16, inputs_size, name='upsample_2')

            b11=atrous_spatial_pyramid_pooling(b10, is_train)
            b12= tf.layers.conv2d_transpose(b11, filters=256, kernel_size=2, strides=2, padding='same', activation=tf.nn.relu, name='deconv_1_64')
            b13= tf.layers.conv2d_transpose(b12, filters=256, kernel_size=2, strides=2, padding='same', activation=tf.nn.relu, name='deconv_2_128')
            b14=tf.concat([b13,low_level_features],axis=3,name='concat')
            b15= tf.layers.conv2d_transpose(b14, filters=128, kernel_size=2, strides=2, padding='same', activation=tf.nn.relu, name='deconv_3_256')
            b16= tf.layers.conv2d_transpose(b15, filters=64, kernel_size=2, strides=2, padding='same', activation=tf.nn.relu, name='deconv_4_512')
            logits=AMSoftMaxConv(b16,valid_label,is_train)
        '''
        with tf.variable_scope('seg_model'):
            # ASPP 32*32*102>>32*32*256
            b11 = atrous_spatial_pyramid_pooling(b10, is_train)
            b12 = tf.image.resize_bilinear(b11, low_level_features_size, name='upsample_1')  # 256*128
            b13 = tf.concat([b12, low_level_features], axis=3, name='concat')
            b14 = relu(sn(dsconv(b13, 256, 3, 'concat_3x3_1')))
            b15 = relu(sn(dsconv(b14, 128, 3, 'concat_3x3_2')))
            b16 = relu(sn(dsconv(b15, 64, 3, 'concat_3x3_3')))
            b17 = tf.image.resize_bilinear(b16, inputs_size, name='upsample_2')
            logits = AMSoftMaxConv(b17, valid_label, is_train)
        with tf.variable_scope('det_model'):
            # 64*32*512>>32*16*512
            h0 = resnet_block(b7, [512, 512, 512], 'det_block0')
            # h0 = relu(sn(dsconv(b11,512, 3, 'det_conv0', s=2), is_train, 'det_sn0'))
            # 32*16*512>>16*8*1024
            h1 = resnet_block(h0, [512, 512, 512], 'det_block1')
            # h1 = relu(sn(dsconv(h0,1024, 3, 'det_conv1', s=2), is_train, 'det_sn1'))
            # 16*8*1024>>2*1*1024
            h2 = tf.nn.avg_pool(h1, [1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
            # 2*1*1024>>2*1*3
            h3 = conv(h2, 3, 1, 'd_convend')
            det = tf.squeeze(tf.transpose(h3, [0, 3, 1, 2]), axis=3)  # batch*3*2
            # det_guide=tf.nn.softmax(det,dim=-1)
            # a = tf.concat([det_guide, tf.ones_like(det_guide)], axis=1)
            # b =tf.transpose(tf.reshape(tf.slice(a,[0,0,1],[-1,4,1]),[-1,4]),[1,0])
            # c=tf.concat([b[3], b[2], b[1], b[0]], axis=0)

            # logits=logits*tf.where(c>0.1,tf.ones_like(c),c)
    return logits, det



def deeplabV3plusSELU(image, is_train, valid_label, num_classes=4, reuse=False, name="deeplab"):
    def resnet_block(x, out_channel, name,strides=2,repeat_num=12):
        if strides==2:
            r0 = conv(x, out_channel[2], 1, name + '_res_0', s=strides)
            r1 = selu(dsconv(x, out_channel[0], 3, name + '_res_1'))
            r2 =selu(dsconv(r1, out_channel[1], 3, name + '_res_2'))
            r3 = selu(dsconv(r2, out_channel[2], 3, name + '_res_3', s=strides))
            x=(r0+r3)
        elif strides==1:
            for i in range(repeat_num):
                r_name = name + '_' + str(i)
                h = selu(dsconv(x, out_channel[0], 3, r_name + '_res_1'))
                h = selu( dsconv(h, out_channel[1], 3, r_name + '_res_2'))
                h = selu(dsconv(h, out_channel[2], 3, r_name + '_res_3'))
                x = (x + h)
        return x
    def atrous_spatial_pyramid_pooling(inputs, is_train, depth=256,DenseMode=False):  # [6,12,18]
        inputs_size = tf.shape(inputs)[1:3]
        conv_1x1 = selu(conv(inputs, depth, 1, 'aspp_1x1', rate=1))
        conv_3x3_1 = selu(dsconv(inputs, depth, 3, 'aspp_3x3_1', rate=6))
        conv_3x3_2 = selu(dsconv(inputs, depth, 3, 'aspp_3x3_2', rate=12))
        conv_3x3_3 = selu(dsconv(inputs, depth, 3, 'aspp_3x3_3', rate=18))

        image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keep_dims=True)
        image_level_features = selu(conv(image_level_features, depth, 1, 'aspp_ilf_1x1'))
        image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')
        net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
        net = selu(conv(net, depth, 1, 'conv_1x1_concat'))
        net = tf.cond(is_train, lambda: tf.nn.dropout(net, 0.9), lambda: net)
        return net

    def AMSoftMaxConv(input, valid_label, is_train, nrof_classes=4):
        m = 0.35
        s = 30
        input = tf.reshape(input, [-1, input.get_shape().as_list()[-1]])
        input_norm = tf.nn.l2_normalize(input, 1, 1e-10, name='embeddings')
        kernel = tf.get_variable(name='kernel', dtype=tf.float32, shape=[input.get_shape().as_list()[-1], nrof_classes], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        kernel_norm = tf.nn.l2_normalize(kernel, 0, 1e-10, name='kernel_norm')
        cos_theta = tf.matmul(input_norm, kernel_norm)
        cos_theta = tf.clip_by_value(cos_theta, -1, 1)
        phi = cos_theta - m
        adjust_theta = tf.cond(is_train, lambda: s * tf.where(tf.equal(tf.one_hot(valid_label, nrof_classes), 1), phi, cos_theta), lambda: s * cos_theta)
        return tf.reshape(adjust_theta, [-1, 1024, 512, 4])
    def z_score(input):
        epsilon=1e-5
        mean, var = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        return (input - mean) / tf.sqrt(var + epsilon)

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        inputs_size = tf.shape(image)[1:3]

        # Block1 512*512*3>>256*256*64

        b0 = selu(conv(z_score(image), 32, 3, 'block1_conv_1', s=2))
        b1 = selu(conv(b0, 64, 3, 'block1_conv_2'))

        # Block2 256*256*64>>128*128*128
        b2 = resnet_block(b1, [128, 128, 128], 'block2')
        low_level_features = selu(conv(b2, 48, 1, 'low_conv'))
        low_level_features_size = tf.shape(low_level_features)[1:3]
        # Block3 128*128*128>>64*64*256
        b3 = resnet_block(b2, [256, 256, 256], 'block3')

        # Block4 64*64*256 (728)>>32*32*1024
        b5 = resnet_block(b3, [256,256,256], 'block4',strides=1,repeat_num= 12)

        b7 = resnet_block(b5, [384, 512, 512], 'block4_exit_conv')

        b8 = selu(dsconv(b7, 512, 3, 'block4_exit_res_4'))
        b9 = selu(dsconv(b8, 768, 3, 'block4_exit_res_5'))
        b10 = selu(dsconv(b9, 1024, 3, 'block4_exit_res_6'))
        with tf.variable_scope('seg_model'):
            # ASPP 32*32*102>>32*32*256
            b11 = atrous_spatial_pyramid_pooling(b10, is_train)
            b12 = tf.image.resize_bilinear(b11, low_level_features_size, name='upsample_1')  # 256*128
            b13 = tf.concat([b12, low_level_features], axis=3, name='concat')
            b14 = selu(dsconv(z_score(b13), 256, 3, 'concat_3x3_1'))
            b15 = selu(dsconv(b14, 128, 3, 'concat_3x3_2'))
            b16 = selu(dsconv(b15, 64, 3, 'concat_3x3_3'))
            b17 = tf.image.resize_bilinear(b16, inputs_size, name='upsample_2')
            logits = AMSoftMaxConv(b17, valid_label, is_train)
        with tf.variable_scope('det_model'):
            # 64*32*512>>32*16*512
            h0 = resnet_block(b7, [512, 512, 512], 'det_block0')
            # h0 = relu(sn(dsconv(b11,512, 3, 'det_conv0', s=2), is_train, 'det_sn0'))
            # 32*16*512>>16*8*1024
            h1 = resnet_block(h0, [512, 512, 512], 'det_block1')
            # h1 = relu(sn(dsconv(h0,1024, 3, 'det_conv1', s=2), is_train, 'det_sn1'))
            # 16*8*1024>>2*1*1024
            h2 = tf.nn.avg_pool(h1, [1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
            # 2*1*1024>>2*1*3
            h3 = conv(h2, 3, 1, 'd_convend')
            det = tf.squeeze(tf.transpose(h3, [0, 3, 1, 2]), axis=3)  # batch*3*2
            # det_guide=tf.nn.softmax(det,dim=-1)
            # a = tf.concat([det_guide, tf.ones_like(det_guide)], axis=1)
            # b =tf.transpose(tf.reshape(tf.slice(a,[0,0,1],[-1,4,1]),[-1,4]),[1,0])
            # c=tf.concat([b[3], b[2], b[1], b[0]], axis=0)

            # logits=logits*tf.where(c>0.1,tf.ones_like(c),c)
    return logits, det

def Nest_Net(image,is_train, valid_label,num_classes=4, reuse=False, name="Nested_UNet++",deep_supervision=False):
    nb_filter = [32,64,128,256,512]
    def standard_unit(input_tensor, stage, nb_filter,is_train,kernel_size=3):
        x = relu(dsconv(input_tensor, nb_filter, kernel_size,'conv' + stage + '_1',use_bias=True))
        x=tf.cond(is_train,lambda:tf.nn.dropout(x,0.5),lambda:x)
        x = relu(dsconv(x, nb_filter, kernel_size,'conv' + stage + '_2',use_bias=True))
        x=tf.cond(is_train,lambda:tf.nn.dropout(x,0.5),lambda:x)
        x = relu(dsconv(x, nb_filter, kernel_size,'conv' + stage + '_3',use_bias=True))
        x=tf.cond(is_train,lambda:tf.nn.dropout(x,0.5),lambda:x)
        return x

    def standard_unit1(input_tensor, stage, nb_filter,is_train,kernel_size=3):
        x=relu(conv(input_tensor,nb_filter,kernel_size,'conv' + stage + '_1',use_bias=True))
        x=tf.cond(is_train,lambda:tf.nn.dropout(x,0.5),lambda:x)
        x=relu(conv(x,nb_filter,kernel_size,'conv' + stage + '_2',use_bias=True))
        x=tf.cond(is_train,lambda:tf.nn.dropout(x,0.5),lambda:x)
        return x
    def pool(input_tensor):
        return tf.nn.max_pool(input_tensor,[1,2,2,1],[1,2,2,1],'SAME')
    #1024*512>>512*256
    conv1_1=standard_unit(image,'11',nb_filter[0],is_train)
    pool1=pool(conv1_1)
    #512*256>>256*128
    conv2_1=standard_unit(pool1,'21',nb_filter[1],is_train)
    pool2=pool(conv2_1)

    up1_2=tf.layers.conv2d_transpose(conv2_1, filters=nb_filter[0], kernel_size=2, strides=2, padding='same', activation=None,name= 'up12')
    conv1_2=tf.concat([up1_2,conv1_1],axis=-1)
    conv1_2 = standard_unit(conv1_2, '12',nb_filter[0],is_train)
    #256*128>>128*64
    conv3_1=standard_unit(pool2,'31',nb_filter[2],is_train)
    pool3=pool(conv3_1)

    up2_2=tf.layers.conv2d_transpose(conv3_1, filters=nb_filter[1], kernel_size=2, strides=2, padding='same', activation=None,name= 'up22')
    conv2_2=tf.concat([up2_2,conv2_1],axis=-1)
    conv2_2 = standard_unit(conv2_2, '22',nb_filter[0],is_train)

    up1_3=tf.layers.conv2d_transpose(conv2_2, filters=nb_filter[0], kernel_size=2, strides=2, padding='same', activation=None,name= 'up13')
    conv1_3=tf.concat([up1_3,conv1_1,conv1_2],axis=-1)
    conv1_3 = standard_unit(conv1_3, '13',nb_filter[0],is_train)
    #128*64>>64*32
    conv4_1=standard_unit(pool3,'41',nb_filter[3],is_train)
    pool4=pool(conv4_1)

    up3_2=tf.layers.conv2d_transpose(conv4_1, filters=nb_filter[2], kernel_size=2, strides=2, padding='same', activation=None,name= 'up32')
    conv3_2=tf.concat([up3_2,conv3_1],axis=-1)
    conv3_2 = standard_unit(conv3_2, '32',nb_filter[2],is_train)

    up2_3=tf.layers.conv2d_transpose(conv3_2, filters=nb_filter[1], kernel_size=2, strides=2, padding='same', activation=None,name= 'up23')
    conv2_3=tf.concat([up2_3,conv2_1,conv2_2],axis=-1)
    conv2_3 = standard_unit(conv2_3, '23',nb_filter[1],is_train)

    up1_4=tf.layers.conv2d_transpose(conv2_3, filters=nb_filter[0], kernel_size=2, strides=2, padding='same', activation=None,name= 'up14')
    conv1_4=tf.concat([up1_4, conv1_1, conv1_2, conv1_3],axis=-1)
    conv1_4 = standard_unit(conv1_4, '14',nb_filter[0],is_train)

    conv5_1=standard_unit(pool4,'51',nb_filter[4],is_train)

    up4_2=tf.layers.conv2d_transpose(conv5_1, filters=nb_filter[3], kernel_size=2, strides=2, padding='same', activation=None,name= 'up42')
    conv4_2 = tf.concat([up4_2, conv4_1],axis=-1)
    conv4_2 = standard_unit(conv4_2,'42', nb_filter[3],is_train)

    up3_3 = tf.layers.conv2d_transpose(conv4_2, filters=nb_filter[2], kernel_size=2, strides=2, padding='same', activation=None, name='up33')
    conv3_3 = tf.concat([up3_3, conv3_1, conv3_2], axis=-1)
    conv3_3 = standard_unit(conv3_3,'33',nb_filter[2],is_train)

    up2_4 = tf.layers.conv2d_transpose(conv3_3, filters=nb_filter[1], kernel_size=2, strides=2, padding='same', activation=None, name='up24')
    conv2_4 = tf.concat([up2_4, conv2_1, conv2_2, conv2_3], axis=-1)
    conv2_4 = standard_unit(conv2_4, '24', nb_filter[1],is_train)

    up1_5=tf.layers.conv2d_transpose(conv2_4, filters=nb_filter[0], kernel_size=2, strides=2, padding='same', activation=None, name='up15')
    conv1_5 = tf.concat([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], axis=-1)
    conv1_5 = standard_unit(conv1_5, '15', nb_filter[0],is_train)

    #nestnet_output_1 = conv(conv1_2,nb_filter,1,'output_1',use_bias=True)
    #nestnet_output_2 = conv(conv1_3,nb_filter,1,'output_2',use_bias=True)
    #nestnet_output_3 = conv(conv1_4,nb_filter,1,'output_3',use_bias=True)
    nestnet_output_4 = conv(conv1_5,4,1,'output_4',use_bias=True)

    #64*32>>32*16
    pool5=pool(conv5_1)
    conv6_1=standard_unit(pool5,'61',nb_filter[4],is_train)
    pool6=pool(conv6_1)
    conv7_1=standard_unit(pool6,'71',nb_filter[4],is_train)
    avgpool = tf.nn.avg_pool(conv7_1, [1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
    # 2*1*1024>>2*1*3
    h = conv(avgpool, 3, 1, 'd_convend')
    det = tf.squeeze(tf.transpose(h, [0, 3, 1, 2]), axis=3)  # batch*3*2
    return nestnet_output_4,det

def FCDenseNet103(image,is_train,num_classes=4,reuse=False,name='fcdense'):
    def BN_ReLU_Conv(inputs, n_filters,is_train,name, filter_size=3, dropout_p = 0.2):
        l = relu(sn(inputs,is_train,name+'sn'))
        l=conv(l,n_filters,filter_size,name,use_bias=True)
        l=tf.cond(is_train, lambda:tf.nn.dropout(l, dropout_p), lambda:l)
        return l
    def TransitionDown(inputs, n_filters,is_train,name):
        l = BN_ReLU_Conv(inputs, n_filters,is_train,name,filter_size=1)
        l = tf.nn.max_pool(l,[1,2,2,1],[1,2,2,1],'SAME')
        return l
    def TransitionUp(skip_connection, block_to_upsample, n_filters_keep):
        # Upsample
        #pad_size=[0,block_to_upsample[0].get_shape().as_list()[1],block_to_upsample[0].get_shape().as_list()[2],0]
        l = tf.concat(block_to_upsample,axis=-1)
        l=tf.layers.conv2d_transpose(l, filters=l.shape[-1], kernel_size=3, strides=2, padding='same')
        # Concatenate with skip connection

        l =  tf.concat([l,skip_connection],axis=-1)
        return l
    n_layers_per_block = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]

    n_filters_first_conv = 48
    n_pool = 5
    growth_rate=16
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        image = tf.image.resize_bilinear(image, (512,256))
        stack=conv(image,48,3,'input_conv',use_bias=True)
        skip_connection_list = []
        n_filters=n_filters_first_conv
        for i in range(n_pool):
            # Dense Block
            for j in range(n_layers_per_block[i]):
                # Compute new feature maps
                l = BN_ReLU_Conv(stack, growth_rate,is_train,'Block_'+str(i)+'_Conv_'+str(j))
                # And stack it : the Tiramisu is growing
                stack =tf.concat([stack, l],axis=-1)
                n_filters += growth_rate
            # At the end of the dense block, the current stack is stored in the skip_connections list
            skip_connection_list.append(stack)
            # Transition Down
            stack = TransitionDown(stack, n_filters, is_train,'Block_'+str(i)+'_TransDown')

        skip_connection_list = skip_connection_list[::-1]
        block_to_upsample = []

        for j in range(n_layers_per_block[n_pool]):
            l = BN_ReLU_Conv(stack, growth_rate, is_train,'Block_5_Conv_'+str(j))
            block_to_upsample.append(l)
            stack = tf.concat([stack, l], axis=-1)

        for i in range(n_pool):
            # Transition Up ( Upsampling + concatenation with the skip connection)
            n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
            stack = TransitionUp(skip_connection_list[i], block_to_upsample, n_filters_keep)

            # Dense Block
            block_to_upsample = []
            for j in range(n_layers_per_block[n_pool + i + 1]):
                l = BN_ReLU_Conv(stack, growth_rate,is_train,'Block_'+str(n_pool+i+1)+'_Conv_'+str(j))
                block_to_upsample.append(l)
                stack = tf.concat([stack, l], axis=-1)
        output=conv(stack, num_classes, 1, 'output_conv', use_bias=True)
        output= tf.image.resize_bilinear(output,(1024,512))
        print(output.get_shape().as_list())
        return output

def FCDenseNet101(image, is_train, num_classes=4, reuse=False, name='fcdense'):
    import gc
    from  io import BytesIO
    import dill
    def BN_ReLU_Conv(inputs, n_filters, is_train, name,scope=None,filter_size=3, dropout_p=0.2):
        with tf.variable_scope(scope, 'xx', [inputs]) as sc:
            l = relu(sn(inputs, is_train, name + 'sn'))
            l = conv(l, n_filters, filter_size, name, use_bias=True)
            l = tf.cond(is_train, lambda: tf.nn.dropout(l, dropout_p), lambda: l)
        return l
    def conv_concat( growth_rate, is_train, name,share,scope=None,block_to_upsample=None):
        share.seek(0)
        inputs = dill.load(share)
        with tf.variable_scope(scope, 'conv_blockx', [inputs]) as sc:
            l = BN_ReLU_Conv(inputs, growth_rate, is_train, name)
            l=tf.concat([inputs,l],axis=-1)
            if block_to_upsample!=None:
                 block_to_upsample.append(l)
            share.seek(0)
            dill.dump(l, share, 0)
            del l
            gc.collect()
    def TransitionDown(inputs, n_filters, is_train, name):
        l = BN_ReLU_Conv(inputs, n_filters, is_train, name, filter_size=1)
        l = tf.nn.max_pool(l, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        return l

    def TransitionUp(skip_connection, block_to_upsample, n_filters_keep):
        # Upsample
        # pad_size=[0,block_to_upsample[0].get_shape().as_list()[1],block_to_upsample[0].get_shape().as_list()[2],0]
        l = tf.concat(block_to_upsample, axis=-1)
        l = tf.layers.conv2d_transpose(l, filters=l.shape[-1], kernel_size=3, strides=2, padding='same')
        # Concatenate with skip connection

        l = tf.concat([l, skip_connection], axis=-1)
        return l

    n_layers_per_block = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
    n_filters_first_conv = 48
    n_pool = 5
    growth_rate = 16
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        stack = conv(image, 48, 3, 'input_conv', use_bias=True)
        skip_connection_list = []
        n_filters = n_filters_first_conv
        for i in range(n_pool):
            # Dense Block
            share = BytesIO()
            dill.dump(stack, share, 0)
            for j in range(n_layers_per_block[i]):
                conv_concat(growth_rate, is_train, 'Conv_' + str(j),share,scope='Block_' + str(i))
                n_filters += growth_rate
            share.seek(0)
            stack = dill.load(share)
            share.close()
            gc.collect()
            del share
            skip_connection_list.append(stack)
            stack = TransitionDown(stack, n_filters, is_train, 'Block_' + str(i) + '_TransDown')

        skip_connection_list = skip_connection_list[::-1]
        block_to_upsample = []

        for j in range(n_layers_per_block[n_pool]):
            l = BN_ReLU_Conv(stack, growth_rate, is_train, 'Block_5_Conv_' + str(j))
            block_to_upsample.append(l)
            stack = tf.concat([stack, l], axis=-1)

        for i in range(n_pool):

            n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
            stack = TransitionUp(skip_connection_list[i], block_to_upsample, n_filters_keep)

            share = BytesIO()
            dill.dump(stack, share, 0)
            block_to_upsample = []
            for j in range(n_layers_per_block[n_pool + i + 1]):
                conv_concat(growth_rate, is_train, 'Conv_' + str(j), share,scope= 'Block_' + str(n_pool + i + 1),block_to_upsample=block_to_upsample )
                print(block_to_upsample)
            share.seek(0)
            stack = dill.load(share)
            share.close()
            gc.collect()
            del share
        output = conv(stack, num_classes, 1, 'output_conv', use_bias=True)
        print(output.get_shape().as_list())
        return output


def deeplabV3plusasd(image,is_train, num_classes=4, reuse=False, name="deeplab"):
    def resnet_block(x, out_channel, is_train, name):
        r0 = sn(conv(x, out_channel, 1, name + '_res_0', s=2), is_train, name + '_res_0_sn')
        r1 = relu(sn(dsconv(x, out_channel, 3, name + '_res_1'), is_train, name + '_res_1_sn'))
        r2 = relu(sn(dsconv(r1, out_channel, 3, name + '_res_2'), is_train, name + '_res_2_sn'))
        r3 = relu(sn(dsconv(r2, out_channel, 3, name + '_res_3', s=2), is_train, name + '_res_3_sn'))
        return r0 + r3
    def resnet_block_repeat(x, out_channel, is_train, name, repeat_num):
        for i in range(repeat_num):
            r_name = name + '_' + str(i)
            h = relu(sn(dsconv(x, out_channel, 3, r_name + '_res_1'), is_train, r_name + '_res_sn_1'))
            h = relu(sn(dsconv(h, out_channel, 3, r_name + '_res_2'), is_train, r_name + '_res_sn_2'))
            h = relu(sn(dsconv(h, out_channel, 3, r_name + '_res_3'), is_train, r_name + '_res_sn_3'))
            x = x + h
        return x
    def atrous_spatial_pyramid_pooling(inputs, is_train, depth=256):  # [6,12,18]
        inputs_size = tf.shape(inputs)[1:3]
        conv_1x1 = conv(inputs, depth, 1, 'aspp_1x1')
        conv_3x3_1 = conv(inputs, depth, 3, 'aspp_3x3_1', rate=6)
        conv_3x3_2 = conv(inputs, depth, 3, 'aspp_3x3_2', rate=12)
        conv_3x3_3 = conv(inputs, depth, 3, 'aspp_3x3_3', rate=18)
        image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keep_dims=True)
        image_level_features = relu(sn(conv(image_level_features, depth, 1, 'aspp_ilf_1x1'), is_train, 'aspp_sn'))
        image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')
        net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
        net = relu(sn(conv(net, depth, 1, 'conv_1x1_concat'), is_train, 'aspp_sn1'))
        return net
    def DenseASPP(input, is_train, depth=256):
        inputs_size = tf.shape(input)[1:3]
        conv_1x1 = conv(input, depth, 1, 'aspp_1x1')
        stack = input
        drate = [3, 6, 12, 18]
        for i in range(4):
            l = conv(stack, depth, 3, 'aspp_3x3_' + str(i + 1), rate=drate[i])
            stack = tf.concat([stack, l], axis=-1)
        image_level_features = tf.reduce_mean(input, [1, 2], name='global_average_pooling', keep_dims=True)
        image_level_features = relu(sn(conv(image_level_features, depth, 1, 'aspp_ilf_1x1'), is_train, 'aspp_sn'))
        image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')
        net = tf.concat([conv_1x1, stack, image_level_features], axis=3, name='concat')
        net = relu(sn(conv(net, depth, 1, 'conv_1x1_concat'), is_train, 'aspp_sn1'))
        return net

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        inputs_size=tf.shape(image)[1:3]

        #Block1 512*512*3>>256*256*64
        b0=relu(sn(conv(image,32,3,'block1_conv_1',s=2),is_train,'sn_1'))
        b1=relu(sn(conv(b0,64,3,'block1_conv_2'),is_train,'sn_2'))

        #Block2 256*256*64>>128*128*128
        b2=resnet_block(b1,128,is_train,'block2')
        low_level_features=relu(sn(conv(b2,48,1,'low_conv'),is_train,'low_conv_sn'))
        low_level_features_size = tf.shape(low_level_features)[1:3]

        #Block3 128*128*128>>64*64*256
        b3=resnet_block(b2,256,is_train,'block3')

        #Block4 64*64*256 (728)>>32*32*1024
        b5=resnet_block_repeat(b3,256,is_train,'block4',6)

        b6_0=sn(conv(b5,512,1,'block4_exit_conv_1',s=2),is_train,'block4_exit_conv_sn0')
        b6_1=sn(dsconv(relu(b5),384,3,'block4_exit_res_1'),is_train,'block4_exit_conv_sn1')
        b6_2=sn(dsconv(relu(b6_1),512,3,'block4_exit_res_2'),is_train,'block4_exit_conv_sn2')
        b6_3=sn(dsconv(relu(b6_2),512,3,'block4_exit_res_3',s=2),is_train,'block4_exit_conv_sn3')
        b7=b6_0+b6_3
        #b8=sn(dsconv(relu(b7),512,3,'block4_exit_res_4'),is_train,'block4_exit_conv_sn4')
        #b9=sn(dsconv(relu(b8),512,3,'block4_exit_res_5'),is_train,'block4_exit_conv_sn5')
        #b10=sn(dsconv(relu(b9),512,3,'block4_exit_res_6'),is_train,'block4_exit_conv_sn6')
        #ASPP 32*32*1024>>32*32*256
        #b11=atrous_spatial_pyramid_pooling(b7,is_train)
        b11=DenseASPP(b7,is_train)
        b12=tf.image.resize_bilinear(b11, low_level_features_size, name='upsample_1')#256*128
        b13=tf.concat([b12,low_level_features],axis=3,name='concat')
        b14=conv(b13,256,3,'concat_3x3_1')
        b15=conv(b14,256,3,'concat_3x3_2')
        b16=conv(b15,num_classes,1,'output')
        logits = tf.image.resize_bilinear(b16, inputs_size, name='upsample_2')
    return logits




















