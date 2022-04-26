import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from baseline_model import * 
from read_data import * 
from datetime import datetime
from loss import *

LAMBDA_CONSTRAINT= 5
PATIENCE = 500
best_p = 0
epoch_best = 0
time_stamp = datetime.now().strftime("%Y-%m-%dT%H.%M")


# load the dataset
ds_train, ds_val = load_dataset("adult_wo_relationship.data", "adult_wo_relationship.test", [8])
model = BaseLine()

tb_callback = tf.keras.callbacks.TensorBoard("logs/" + time_stamp + "baseline_model" + "/tensorboard/", update_freq=50, profile_batch = 0)
earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
ckpts = tf.keras.callbacks.ModelCheckpoint("logs/" + time_stamp + "baseline_model" +"/checkpoints/cp-{epoch:04d}.ckpt", save_best_only = True)

train_loss = tf.keras.metrics.Mean()
train_bce = tf.keras.metrics.Mean()
train_constraint = tf.keras.metrics.Mean()
train_p_rule = tf.keras.metrics.Mean()
train_acc = tf.keras.metrics.BinaryAccuracy()


val_loss = tf.keras.metrics.Mean()
val_bce = tf.keras.metrics.Mean()
val_constraint = tf.keras.metrics.Mean()
val_p_rule = tf.keras.metrics.Mean()
val_acc = tf.keras.metrics.BinaryAccuracy()

optimizer = tf.keras.optimizers.Adam()

def log():
    with open(f'log_{time_stamp}.txt', 'w') as f:
        if LAMBDA_CONSTRAINT == 0:
            f.write(f"baseline unconstrained model\n")
            model_name = f"baseline_model_{time_stamp}.tf"
        else:
            f.write(f"Constrained model\n lambda: {LAMBDA_CONSTRAINT}\n")
            model_name = f"model_lambda_{LAMBDA_CONSTRAINT}_{time_stamp}.tf"
        f.write(f'Epoch {epoch} Train Loss: {mean_train_loss:=4.4f}, accuracy: {mean_train_acc:=4.4f}, bce: {mean_train_bce:=4.4f}, constraint: {mean_train_constraint:=4.4f}\n')
        f.write(f'          Val Loss: {mean_val_loss:=4.4f}, accuracy: {mean_val_acc:=4.4f}, bce: {mean_val_bce:=4.4f}, constraint: {mean_val_constraint:=4.4f}, prule: {p_value:=4.4f}\n')
        model.save(model_name)

for epoch in range(5000):
    try:
        for x, protected_attribute, y in ds_train.batch(64):
            with tf.GradientTape() as tape:
                out = model(x)
                bce, constraint = fairness_constaint_bce(out, protected_attribute, y)
                loss = bce+ LAMBDA_CONSTRAINT*constraint

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss.update_state(loss)
            train_bce.update_state(bce)
            train_constraint.update_state(constraint)
            train_acc.update_state(y, out)

        for x, protected_attribute, y in ds_val.batch(64):
            out = model(x)
            bce, constraint = fairness_constaint_bce(out, protected_attribute, y)
            loss = bce+ LAMBDA_CONSTRAINT*constraint

            val_loss.update_state(loss)
            val_bce.update_state(bce)
            val_constraint.update_state(constraint)
            val_acc.update_state(y,out)
        for x,p,y in ds_val.batch(50000):
            model_out = model(x)
            p_value = p_rule(model_out, p)


        mean_train_loss = train_loss.result()
        mean_train_acc = train_acc.result()
        mean_train_bce = train_bce.result()
        mean_train_constraint = train_constraint.result()

        mean_val_loss = val_loss.result()
        mean_val_acc = val_acc.result()
        mean_val_bce = val_bce.result()
        mean_val_constraint = val_constraint.result()        
        
        if p_value > best_p and epoch > 10 and LAMBDA_CONSTRAINT > 0:
            log()
            best_p = p_value
            epoch_best = epoch
        if epoch_best + PATIENCE < epoch and epoch > 20 and LAMBDA_CONSTRAINT > 0 and mean_val_acc > 0.8:
            print(f"EARLY STOPPING DUE TO THE PASSAGE OF {PATIENCE} EPOCHS WITHOUT IMPROVEMENT IN PRULE")
            break

        print(f'Epoch {epoch} Train Loss: {mean_train_loss:=4.4f}, accuracy: {mean_train_acc:=4.4f}, bce: {mean_train_bce:=4.4f}, constraint: {mean_train_constraint:=4.4f}')
        print(f'          Val Loss: {mean_val_loss:=4.4f}, accuracy: {mean_val_acc:=4.4f}, bce: {mean_val_bce:=4.4f}, constraint: {mean_val_constraint:=4.4f}, prule: {p_value:=4.4f}')

        train_loss.reset_state()
        train_p_rule.reset_state()
        train_acc.reset_state()

        val_loss.reset_state()
        val_p_rule.reset_state()
        val_acc.reset_state()

    except KeyboardInterrupt:
        break
if LAMBDA_CONSTRAINT == 0:
    log()