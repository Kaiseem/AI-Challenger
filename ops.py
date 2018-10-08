import tensorflow as tf

def sn(input,name='Switchable Normalization',train_phase=False,use_bn=False,using_moving_avg=True,epsilon = 1e-5, momentum=0.997):

        beta = tf.Variable(tf.constant(0.0, shape=[input.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[input.shape[-1]]), name='gamma', trainable=True)
        #[B,H,W,C]
        mean_in,var_in=tf.nn.moments(input, axes=[1,2], keep_dims=True)#Instance normalization[H,W]
        mean_ln,var_ln=tf.nn.moments(input, axes=[1,2,3], keep_dims=True)#Layer normalization[H,W,C]
        if use_bn:
            mean_weight = tf.nn.softmax(tf.Variable(tf.constant(1.0, shape=[3]), name='mean_weight', trainable=True))
            var_weight = tf.nn.softmax(tf.Variable(tf.constant(1.0, shape=[3]), name='var_weight', trainable=True))
            ema = tf.train.ExponentialMovingAverage(decay=momentum)
            mean_bn,var_bn=tf.nn.moments(input, axes=[0,1,2], keep_dims=True)#Batch normailzation[B,H,W]
            def mean_var_with_update():
                ema_apply_op = ema.apply([mean_bn, var_bn])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(mean_bn), tf.identity(var_bn)
            if using_moving_avg:
                mean_bn, var_bn = tf.cond(train_phase, mean_var_with_update,
                                    lambda: (ema.average(mean_bn), ema.average(var_bn)))
            mean=mean_weight[0]*mean_in+mean_weight[1]*mean_ln+mean_weight[2]*mean_bn
            var=var_weight[0]*var_in+var_weight[1]*var_ln+var_weight[2]*var_bn
        else:
            mean_weight = tf.nn.softmax(tf.Variable(tf.constant(1.0, shape=[2]), name='mean_weight', trainable=True))
            var_weight = tf.nn.softmax(tf.Variable(tf.constant(1.0, shape=[2]), name='var_weight', trainable=True))
            mean=mean_weight[0]*mean_in+mean_weight[1]*mean_ln
            var=var_weight[0]*var_in+var_weight[1]*var_ln
        normalized = (input - mean) /tf.sqrt(var + epsilon)
        return normalized*gamma+beta

def bn(input,train_phase,name,using_moving_avg=True,epsilon = 1e-5, momentum=0.997):
    beta = tf.Variable(tf.constant(0.0, shape=[input.shape[-1]]), name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[input.shape[-1]]), name='gamma', trainable=True)
    mean_bn, var_bn = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=True)  # Batch normailzation[B,H,W]
    ema = tf.train.ExponentialMovingAverage(decay=momentum)
    def mean_var_with_update():
        ema_apply_op = ema.apply([mean_bn, var_bn])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(mean_bn), tf.identity(var_bn)
    mean_bn, var_bn = tf.cond(train_phase, mean_var_with_update,
                                  lambda: (ema.average(mean_bn), ema.average(var_bn)))
    normalized = (input - mean_bn) / tf.sqrt(var_bn + epsilon)
    return normalized * gamma + beta
def relu(x):
    return tf.nn.relu(x)

def selu(x):#速度 selu 效果bn+relu

    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale*tf.where(x>0.0,x,alpha*tf.nn.elu(x))

def dsconv(x,out_channel,ks,name,rate=1,use_bias=False,s=1):

    return tf.layers.separable_conv2d(x,out_channel, ks, s, padding='SAME', dilation_rate=rate,use_bias=use_bias,name=name)

def dsconv12(x,out_channel,ks,name,rate=1,s=1,wd=0.00004):
    with tf.variable_scope(name):
        shape=x.get_shape().as_list()
        init=tf.truncated_normal_initializer(mean=0,stddev=(1/(shape[1]*shape[2]))**0.5)
        filter=tf.get_variable(name=name+'_depthwise_kernel',shape=[ks,ks,x.get_shape().as_list()[-1],1],initializer=init)
        depth=tf.nn.depthwise_conv2d(x,filter=filter,strides=[1,s,s,1],padding='SAME',rate=[rate,rate],name=name+'_depthwise')
        #regularizer = tf.contrib.layers.l2_regularizer(scale=wd)
        output=tf.layers.conv2d(selu(depth), out_channel, 1, 1, padding='SAME',use_bias=False,kernel_initializer=init,name=name+'_pointwise')
    return output

def conv(x,out_channel,ks,name,use_bias=False,s=1,rate=1):

    return tf.layers.conv2d(x, out_channel, ks, s, padding='SAME',use_bias=use_bias,dilation_rate=rate,name=name)
    #kernel_initializer=tf.variance_scaling_initializer(factor=1.0,model='FAN_IN')

def multi_LSTM(inputs,state):
    input_size=inputs.get_shape().as_list()
    unit_size=input_size[1]*input_size[2]*input_size[3]
    cell_input=tf.reshape(inputs,[-1,1,unit_size])
    BATCH_SIZE=1


    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=unit_size)
    lstm_cell2=tf.nn.rnn_cell.BasicLSTMCell(num_units=unit_size)
    multi_cells=tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_cell,lstm_cell2])

    #batch_size=2
    #init_state=multi_cells.zero_state(batch_size=batc®h_size,dtype=tf.float32)

    cell_outputs,final_state=tf.nn.dynamic_rnn(cell=multi_cells,inputs=cell_input,initial_state=state)
    outputs=tf.reshape(cell_outputs,[-1,input_size[1],input_size[2],input_size[3]])
    return outputs,final_state




