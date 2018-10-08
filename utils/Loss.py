import tensorflow as tf
def focal_loss(valid_logits,valid_label,alpha=0.25,gamma=2.0,name=None,classes_num=4,scope=None):
    """
    logits and onehot_labels must have same shape[batchsize,num_classes] and the same data type(float16 32 64)
    Args:
        onehot_labels: [batchsize,classes]
        logits: Unscaled log probabilities(tensor)
        alpha: The hyperparameter for adjusting biased samples, default is 0.25
        gamma: The hyperparameter for penalizing the easy labeled samples
        name: A name for the operation(optional)
    Returns:
      A 1-D tensor of length batch_size of same type as logits with softmax focal loss
    """
    precise_logits = valid_logits
    onehot_labels= tf.one_hot(valid_label, classes_num)
    onehot_labels = tf.cast(onehot_labels, precise_logits.dtype)
    predictions = tf.nn.softmax(precise_logits)
    predictions_pt = tf.where(tf.equal(onehot_labels,1),predictions,1.-predictions)
    epsilon = 1e-8
    alpha_t = tf.scalar_mul(alpha,tf.ones_like(onehot_labels,dtype=tf.float32))
    alpha_t = tf.where(tf.equal(onehot_labels,1.0),alpha_t,1-alpha_t)
    losses = tf.reduce_mean(-alpha_t*tf.pow(1.-predictions_pt,gamma)*onehot_labels*tf.log(predictions_pt+epsilon),
            name=name)
    #tf.summary.scalar(
    #    scope + '-focal_loss',
    #    losses
    #)
    return losses


def weighted_log_loss( valid_pred, valid_truth, epsilion=1e-6):
    valid_pred = valid_pred / tf.reduce_sum(valid_pred, axis=-1, keep_dims=True)
    valid_pred = tf.clip_by_value(valid_pred, epsilion, 1 - epsilion)
    weight = tf.Variable([1., 1, 5, 5.], dtype=tf.float32)  # 15,110,3360
    loss = valid_truth * tf.log(valid_pred) * weight
    loss = tf.reduce_sum(-tf.reduce_sum(loss, axis=-1))
    return loss


def generalised_dice_loss( valid_logits, valid_label, classes_num):
    valid_pred = tf.nn.softmax(valid_logits)
    valid_truth = tf.one_hot(valid_label, classes_num)
    if valid_pred.shape == valid_truth.shape:
        print('is OJBKK')
    sum_p = tf.reduce_sum(valid_pred, axis=0)
    sum_r = tf.reduce_sum(valid_truth, axis=0)
    sum_pr = tf.reduce_sum(valid_pred * valid_truth, axis=0)
    weights = tf.reciprocal(tf.square(sum_r))
    new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
    weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) * tf.reduce_max(new_weights), weights)
    generalised_dice_numerator = 2 * tf.reduce_sum(tf.multiply(weights, sum_pr))
    generalised_dice_denominator = tf.reduce_sum(tf.multiply(weights, (sum_r + sum_p))) + 1e-6
    generalised_dice_score = generalised_dice_numerator / generalised_dice_denominator
    GDL = 1 - generalised_dice_score
    del sum_p, sum_r, sum_pr, weights
    return GDL  # self.weighted_log_loss(valid_pred,valid_truth)


def dice(y_true, y_pred):
    sum_p = tf.reduce_sum(y_pred, axis=0)
    sum_r = tf.reduce_sum(y_true, axis=0)
    sum_pr = tf.reduce_sum(y_true * y_pred, axis=0)
    dice_numerator = 2 * sum_pr
    dice_denominator = sum_r + sum_p
    dice_score = (dice_numerator + 1e-6) / (dice_denominator + 1e-6)
    return dice_score


def generalised_wasserstein_dice_loss(valid_logits, valid_label, classes_num):
    M_tree_42 = np.array([[0., 1., 1., 1., ],
                          [1., 0., 0.5, 0.5],
                          [1., 0.5, 0., 0.2],
                          [1., 0.5, 0.2, 0.]], dtype=np.float64)
    M_tree_4 = np.array([[0., 1., 1., 1., ],
                         [1., 0., 1., 1.],
                         [1., 1., 0., 1.],
                         [1., 1., 1., 0.]], dtype=np.float64)

    def wasserstein_disagreement_map(prediction, ground_truth, M):
        # pixel-wise Wassertein distance (W) between flat_pred_proba and flat_labels
        # wrt the distance matrix on the label space M
        # unstack_labels = tf.unstack(ground_truth, axis=-1)
        ground_truth = tf.cast(ground_truth, dtype=tf.float32)
        # unstack_pred = tf.unstack(prediction, axis=-1)
        prediction = tf.cast(prediction, dtype=tf.float32)
        # print("shape of M", M.shape, "unstacked labels", unstack_labels,
        #       "unstacked pred" ,unstack_pred)
        # W is a weighting sum of all pairwise correlations (pred_ci x labels_cj)
        pairwise_correlations = []
        for i in range(4):
            for j in range(4):
                pairwise_correlations.append(
                    M[i, j] * tf.multiply(prediction[:, i], ground_truth[:, j]))
        wass_dis_map = tf.add_n(pairwise_correlations)
        return wass_dis_map

    """
    Function to calculate the Generalised Wasserstein Dice Loss defined in
    Fidon, L. et. al. (2017) Generalised Wasserstein Dice Score for Imbalanced
    Multi-class Segmentation using Holistic Convolutional Networks.
    MICCAI 2017 (BrainLes)
    :param prediction: the logits (before softmax)
    :param ground_truth: the segmentation ground_truth
    :param weight_map:
    :return: the loss
    """
    valid_pred = tf.nn.softmax(valid_logits)
    valid_truth = tf.cast(tf.one_hot(valid_label, classes_num), dtype=tf.float32)

    # M = tf.cast(M, dtype=tf.float64)
    # compute disagreement map (delta)
    M = M_tree_4
    # print("M shape is ", M.shape, pred_proba, one_hot)
    delta = wasserstein_disagreement_map(valid_pred, valid_truth, M)
    # compute generalisation of all error for multi-class seg
    all_error = tf.reduce_sum(delta)
    # compute generalisation of true positives for multi-class seg
    true_pos = tf.reduce_sum(
        tf.multiply(tf.constant(M[0, :4], dtype=tf.float32), valid_truth),
        axis=1)
    true_pos = tf.reduce_sum(tf.multiply(true_pos, 1. - delta), axis=0)
    WGDL = 1. - (2. * true_pos) / (2. * true_pos + all_error)

    return tf.cast(WGDL, dtype=tf.float32)