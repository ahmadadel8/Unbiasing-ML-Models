from baseline_model import * 
from read_data import * 
from datetime import datetime
from loss import *
time_stamp = datetime.now().strftime("%Y-%m-%dT%H.%M")

# load the dataset
ds_train, ds_val = load_dataset("adult_wo_relationship.data", "adult_wo_relationship.test", [8])
model = BaseLine()
# model.compile("adam", loss = fairness_constaint_bce, metrics=[p_rule])

tb_callback = tf.keras.callbacks.TensorBoard("logs/" + time_stamp + "baseline_model" + "/tensorboard/", update_freq=50, profile_batch = 0)
earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
ckpts = tf.keras.callbacks.ModelCheckpoint("logs/" + time_stamp + "baseline_model" +"/checkpoints/cp-{epoch:04d}.ckpt", save_best_only = True)
# tf.keras.backend.set_value(model.optimizer.learning_rate, 0.0005)
# model.fit(X_train,[Y_train, X_proc_train], epochs=500, batch_size = 8, callbacks=[tb_callback, ckpts, earlystop],validation_data=(X_test,[Y_test, X_proc_test]))

train_loss = tf.keras.metrics.Mean()
train_p_rule = tf.keras.metrics.Mean()
train_acc = tf.keras.metrics.BinaryAccuracy()


val_loss = tf.keras.metrics.Mean()
val_p_rule = tf.keras.metrics.Mean()
val_acc = tf.keras.metrics.BinaryAccuracy()

optimizer = tf.keras.optimizers.Adam()

for i in range(100):
    for x, protected_attribute, y in ds_train.batch(64):
        with tf.GradientTape() as tape:
            out = model(x)
            loss = fairness_constaint_bce(out, protected_attribute, y)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_acc(y, out)

    for x, protected_attribute, y in ds_val.batch(64):
        out = model(x)
        loss = fairness_constaint_bce(out, protected_attribute, y)

        val_loss(loss)
        val_acc(y,out)
    for x,p,y in ds_val.batch(50000):
        model_out = model(x)
        p_value = p_rule(model_out, p)

    print(f'Epoch {i} Train Loss: {train_loss.result():=4.4f},  Train accuracy: {train_acc.result():=4.4f}')
    print(f'        Val Loss: {val_loss.result():=4.4f},  Val accuracy: {val_acc.result():=4.4f}, Val prule: {p_value:=4.4f}')

    train_loss.reset_state()
    train_p_rule.reset_state()
    train_acc.reset_state()

    val_loss.reset_state()
    val_p_rule.reset_state()
    val_acc.reset_state()