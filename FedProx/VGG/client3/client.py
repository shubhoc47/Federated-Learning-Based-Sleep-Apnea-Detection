import argparse
import os
from pathlib import Path
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import flwr as fl
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.callbacks import LearningRateScheduler
from keras.layers import Dropout, MaxPooling1D, Reshape, multiply, Conv1D, GlobalAveragePooling1D, Dense, Flatten
from tensorflow.keras.layers import Input
from keras.regularizers import l2
import keras
import csv
from tensorflow.keras.models import Model, Sequential

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

save = []
save.append(('Round', 'Loss', 'AUC', 'Precision', 'Recall', 'F1score'))

with open('client.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

x_train1 = loaded_data[0]
x_train2 = loaded_data[1]
y_train = loaded_data[2]

cnt = 1

def lr_schedule(epoch):
#        print(epoch)
    lr = 0
    if epoch < 10:
        lr = 0.0001
    elif epoch < 40:
        lr = 0.00001
    else:
        lr = 0.000001
#     print("Learning rate: ", lr)
    return lr

# def lr_schedule(epoch):
# #        print(epoch)
#     lr = 0.000001
#     return lr

def train_step(x_train1, x_train2, y_train, model, loss_fn, optim, train_loss, train_auc, train_precision, train_recall, epoch):
    # Update learning rate using the scheduler
    global cnt
    # print('00000000000000000000000000000000000000000000000000000000')
    # print(cnt)
    # print('00000000000000000000000000000000000000000000000000000000')
    lr = lr_schedule(cnt)
    optim.learning_rate = lr

    with tf.GradientTape() as tape:
        predictions = model([x_train1, x_train2], training=True)
        
#         print(type(predictions), predictions.shape)
#         print(type(y_train), y_train.shape)
#         y_true, y_pred = tf.argmax(y_train, axis=-1), np.argmax(model.predict([x_train1, x_train2, x_train3]), axis=-1)
        loss_value = loss_fn(y_train, predictions)

    gradients = tape.gradient(loss_value, model.trainable_variables)
    optim.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss_value)
    train_auc(y_train, predictions)
    train_precision(y_train, predictions)
    train_recall(y_train, predictions) 

# Define Flower client
class Client(fl.client.NumPyClient):
    def __init__(self, model, input_a, input_b, y_train):
        self.model = model
        self.x_train1 = input_a
        self.x_train2 = input_b
        self.y_train = y_train

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self, config):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")


       
    
    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]



        train_loss = tf.keras.metrics.Mean()
        train_auc = AUC()
        train_precision = Precision()
        train_recall = Recall()
        loss_fn = tf.keras.losses.BinaryCrossentropy()

        optim = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        



        # Train the model using hyperparameters from config
        for epoch in range(epochs):
            print("Epoch", epoch+1)
            train_loss.reset_states()
            train_auc.reset_states()
            train_precision.reset_states()
            train_recall.reset_states()

            
            num_train_batches = len(self.x_train1) // batch_size

            for i in range(num_train_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                x_train1_batch = self.x_train1[start_idx:end_idx]
                x_train2_batch = self.x_train2[start_idx:end_idx]
                y_train_batch = self.y_train[start_idx:end_idx]

                train_step(x_train1_batch, x_train2_batch, y_train_batch, self.model, loss_fn, optim, train_loss, train_auc, train_precision, train_recall, epoch)
                
                # Calculate and print progress
                progress = ((i + 1) / num_train_batches) * 100
                print("Training progress: {:.2f}%".format(progress), end="\r")

            
            
            train_f1 = 2 * (train_precision.result().numpy() * train_recall.result().numpy()) / (train_precision.result().numpy() + train_recall.result().numpy())
            
            
            print()  # Print a newline after each epoch
            
            print('Batch: ', batch_size, end=' ')
            # Get the current learning rate from the optimizer
            current_lr = optim.learning_rate.numpy()
            print("Learning Rate:", current_lr)
            
            print("Train Loss:", train_loss.result().numpy(), end='  ')
            print("Train AUC:", train_auc.result().numpy(), end='  ')
            print("Train Precision:", train_precision.result().numpy(), end='  ')
            print("Train Recall:", train_recall.result().numpy(), end='  ')
            print("Train F1Score:", train_f1)


        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train1)
        results = {
            "loss": float(train_loss.result().numpy()),
            "auc": float(train_auc.result().numpy()),
            "precision": float(train_precision.result().numpy()),
            "recall": float(train_recall.result().numpy()),
            "F1": train_f1,
        }
        return parameters_prime, num_examples_train, results

        
    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)
        batch_size = 128

        optim = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )

        loss_fn = tf.keras.losses.BinaryCrossentropy()
        test_loss = tf.keras.metrics.Mean()
        test_auc = AUC()
        test_precision = Precision()
        test_recall = Recall()
        test_loss.reset_states()
        test_auc.reset_states()
        test_precision.reset_states()
        test_recall.reset_states()
        num_test_batches = len(x_train1) // batch_size

        for i in range(num_test_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size

                x_train1_batch = self.x_train1[start_idx:end_idx]
                x_train2_batch = self.x_train2[start_idx:end_idx]
                y_train_batch = self.y_train[start_idx:end_idx]

                val_step(x_train1_batch, x_train2_batch, y_train_batch, self.model, loss_fn, optim, test_loss, test_auc, test_precision, test_recall)
                progress = ((i + 1) / num_test_batches) * 100
                print("Test progress: {:.2f}%".format(progress), end="\r")

        test_f1 = 2 * (test_precision.result().numpy() * test_recall.result().numpy()) / (test_precision.result().numpy() + test_recall.result().numpy())




        # # Get config values
        # steps: int = config["val_steps"]

        # # Evaluate global model parameters on the local test data and return results
        # loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        # num_examples_test = len(self.x_test)
        global cnt
        global save
        save.append((cnt, float(test_loss.result().numpy()), float(test_auc.result().numpy()), 
            float(test_precision.result().numpy()), float(test_recall.result().numpy()), test_f1))

        cnt += 1
        return [float(test_loss.result().numpy()), len(x_train1), {
            "loss": float(test_loss.result().numpy()),
            "auc": float(test_auc.result().numpy()),
            "precision": float(test_precision.result().numpy()),
            "recall": float(test_recall.result().numpy()),
            "F1": test_f1,
        }]   

def val_step(x_val1, x_val2, y_val, model, loss_fn, optim, val_loss, val_auc, val_precision, val_recall):
    predictions = model([x_val1, x_val2], training=False)
    labels = tf.reshape(y_val, predictions.shape)  # Reshape labels to match predictions shape
    loss_value = loss_fn(y_val, predictions)

    val_loss(loss_value)
    val_auc(y_val, predictions)
    val_precision(y_val, predictions)
    val_recall(y_val, predictions)

def create_model(input_a_shape, input_b_shape, weight=1e-3):
    # Rest of your code
    input1 = Input(shape=input_a_shape)
    x1 = Conv1D(16, kernel_size=11, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(input1)
    x1 = Conv1D(24, kernel_size=11, strides=2, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x1)
    x1 = MaxPooling1D(pool_size=3, padding="same")(x1)
    x1 = Conv1D(32, kernel_size=11, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x1)
    x1 = MaxPooling1D(pool_size=5, padding="same")(x1)

    # SA-CNN-2
    input2 = Input(shape=input_b_shape)
    x2 = Conv1D(16, kernel_size=11, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(input2)
    x2 = Conv1D(24, kernel_size=11, strides=2, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x2)
    x2 = MaxPooling1D(pool_size=3, padding="same")(x2)
    x2 = Conv1D(32, kernel_size=11, strides=3, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x2)


    # Channel-wise attention module
    concat = keras.layers.concatenate([x1, x2], name="Concat_Layer", axis=-1)
    p = Conv1D(96, kernel_size=3, padding='same', activation='relu')(concat)
    
#     print(concat.shape)
    # VGG-like model
    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(concat)
    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    
    x = Dropout(0.5)(x)
    x = Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    
    x = Dropout(0.5)(x)
    x = Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    
    x = Dropout(0.5)(x)
    x = Conv1D(512, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(512, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(512, kernel_size=3, padding='same', activation='relu')(x)
    
    if x.shape[1] >= 2:
        x = MaxPooling1D(pool_size=2, strides=2)(x)

    x = Conv1D(512, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(512, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(512, kernel_size=3, padding='same', activation='relu')(x)
    
    
    if x.shape[1] >= 2:
        x = MaxPooling1D(pool_size=2, strides=2)(x)
    
    x = Dropout(0.2)(x)
    # Flatten the output
#     print(x.shape)
    x = Flatten()(x)
#     print(x.shape)
    # Dense layers
#     x = Dense(4096, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     x = Dense(4096, activation='relu')(x)
#     x = Dropout(0.5)(x)
    excitation = Dense(96, activation='sigmoid')(x)
    excitation = Reshape((1, 96))(excitation)

    scale = multiply([p, excitation])
    x = GlobalAveragePooling1D()(scale)
    dp = Dropout(0.5)(x)
    outputs = Dense(2, activation='softmax', name="Output_Layer")(dp)
    model = Model(inputs=[input1, input2], outputs=outputs)

    return model



def main() -> None:
    # Parse command line argument `partition`
    # parser = argparse.ArgumentParser(description="Flower")
    # parser.add_argument("--partition", type=int, choices=range(0, 10), required=True)
    # args = parser.parse_args()

    # Load and compile Keras model
    model = create_model(x_train1.shape[1:], x_train2.shape[1:])

    client = Client(model, x_train1, x_train2, y_train)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
        # root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )


if __name__ == "__main__":
    main()
    with open(f'metrics.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(save)
