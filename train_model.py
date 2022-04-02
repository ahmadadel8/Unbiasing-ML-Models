from baseline_model import * 
from read_data import * 
from datetime import datetime
time_stamp = datetime.now().strftime("%Y-%m-%dT%H.%M")

full_path = 'adult-all.csv'
# load the dataset
X, Y = load_dataset(full_path)
train_split = np.ceil(X.shape[0]*0.8).astype(int)
test_split = X.shape[0]-train_split
X_train,Y_train, X_test, Y_test = X[0:train_split], Y[0:train_split],X[0:test_split], Y[0:test_split]
model = BaseLine()
model.compile("adam", "binary_crossentropy", metrics=['accuracy'])

tb_callback = tf.keras.callbacks.TensorBoard("logs/" + time_stamp + "baseline_model" + "/tensorboard/", update_freq=50, profile_batch = 0)
ckpts = tf.keras.callbacks.ModelCheckpoint("logs/" + time_stamp + "baseline_model" +"/checkpoints/cp-{epoch:04d}.ckpt", save_best_only = True)

model.fit(X_train,Y_train, epochs=20,callbacks=[tb_callback, ckpts],validation_data=(X_train,Y_train))
