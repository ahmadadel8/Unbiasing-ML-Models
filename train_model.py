from baseline_model import * 
from read_data import * 
from datetime import datetime
time_stamp = datetime.now().strftime("%Y-%m-%dT%H.%M")

# load the dataset
X_train, X_proc_train, Y_train, X_test, X_proc_test, Y_test = load_dataset("adult.data", "adult.test", [8]) #check if 8 is correct
model = BaseLine()
model.compile("adam", "binary_crossentropy", metrics=['accuracy'])

tb_callback = tf.keras.callbacks.TensorBoard("logs/" + time_stamp + "baseline_model" + "/tensorboard/", update_freq=50, profile_batch = 0)
earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
ckpts = tf.keras.callbacks.ModelCheckpoint("logs/" + time_stamp + "baseline_model" +"/checkpoints/cp-{epoch:04d}.ckpt", save_best_only = True)
# tf.keras.backend.set_value(model.optimizer.learning_rate, 0.0005)
model.fit([X_train, X_proc_train],Y_train, epochs=500, batch_size = 8, callbacks=[tb_callback, ckpts, earlystop],validation_data=([X_test, X_proc_test],Y_test))