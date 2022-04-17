import tensorflow as tf


def accuracy_constaint(model_out, protected_feature_batch):
    constraint = tf.reduce_mean((protected_feature_batch - tf.reduce_mean(protected_feature_batch)) * model_out)
    return constraint

def p_relu():
    # protected_idx = tf.where(tf.equal(protected_feature_batch, 0))
    # protected_model_outs = tf.gather(model_out, protected_idx)
    pass
