import tensorflow as tf

def diff_abs(x):
    return tf.nn.relu(x) + tf.nn.relu(-x)

def fairness_constaint_bce(model_out, protected_attribute, y_true):

    var = tf.cast(protected_attribute - tf.reduce_mean(protected_attribute), tf.float32)
    dot_product = var* model_out
    constraint = diff_abs(tf.reduce_mean(dot_product))

    loss_fn = tf.keras.losses.BinaryCrossentropy()
    y_true = tf.cast(y_true, tf.float32)
    loss = loss_fn(y_true,model_out)
    
    return loss, constraint

def p_rule(model_out, protected_attribute):
    threshold = 0.5
    epsilon = 1e-5
    model_out = tf.where(model_out>threshold, 1, 0)
    protected_attribute= tf.reshape(protected_attribute, (-1))
    
    protected_idx = tf.where(tf.equal(protected_attribute, 0))
    protected_model_outs = tf.gather(model_out, protected_idx)        
    protected_accepted_per = tf.math.reduce_sum(protected_model_outs)/tf.shape(protected_idx)[0]

    non_protected_idx = tf.where(tf.equal(protected_attribute, 1))
    non_protected_model_outs = tf.gather(model_out, non_protected_idx)
    non_protected_accepted_per = tf.math.reduce_sum(non_protected_model_outs)/tf.shape(non_protected_idx)[0]

    p_value = (protected_accepted_per+epsilon)/(non_protected_accepted_per+epsilon)
    if tf.math.is_nan(p_value):
        print("p_value", p_value)
        print('model_out', model_out)
        print("protected_attribute", protected_attribute)
        print("protected_accepted_per", protected_accepted_per)
        print("non_protected_accepted_per", non_protected_accepted_per)
        print("mean protected att", tf.math.reduce_mean(protected_attribute))
    return p_value
