from typing import Dict, Optional, Tuple
from pathlib import Path
import argparse
import pandas as pd
import tensorflow as tf
import flwr as fl
import numpy as np
import pickle
import csv
import keras
import os
from keras.layers import Dropout, MaxPooling1D, Reshape, multiply, Conv1D, GlobalAveragePooling1D, Dense, GRU
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, Sequential
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from scipy.interpolate import splev, splrep
from sklearn.metrics import confusion_matrix, f1_score
import random
from tensorflow.keras.metrics import AUC, Precision, Recall

save = []
cnt=0
save_f1 = 0.50
client_number = 4
save.append(('Round', 'Loss', 'AUC', 'Precision', 'Recall', 'F1score'))
with open('server.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

# print(len(loaded_data))
x_train1 = loaded_data[0]
x_train2 = loaded_data[1]
y_train = loaded_data[2]

def create_model(input_a_shape, input_b_shape, weight=1e-3):
    # SA-CNN-3
    input1 = Input(shape=input_a_shape)
    x1 = Conv1D(64, kernel_size=11, strides=1, padding="same", activation="LeakyReLU", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(input1)
    x1 = Conv1D(128, kernel_size=11, strides=2, padding="same", activation="LeakyReLU", kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x1)
    x1 = MaxPooling1D(pool_size=3, padding="same")(x1)
#     x1 = Dropout(0.3)(x1)
    x1 = Conv1D(32, kernel_size=11, strides=1, padding="same", activation="LeakyReLU", kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x1)
    x1 = MaxPooling1D(pool_size=5, padding="same")(x1)
    gru1 = GRU(64, return_sequences=True)(x1)
    gru1 = Dropout(0.5)(gru1)
    gru1 = GRU(32)(gru1)

    # SA-CNN-2
    input2 = Input(shape=input_b_shape)
    x2 = Conv1D(64, kernel_size=11, strides=1, padding="same", activation="LeakyReLU", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(input2)
    x2 = Conv1D(128, kernel_size=11, strides=2, padding="same", activation="LeakyReLU", kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x2)
    x2 = MaxPooling1D(pool_size=3, padding="same")(x2)
#     x1 = Dropout(0.3)(x1)
    x2 = Conv1D(32, kernel_size=11, strides=3, padding="same", activation="LeakyReLU", kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x2)
    gru2 = GRU(64, return_sequences=True)(x2)
    gru2 = Dropout(0.5)(gru2)
    gru2 = GRU(32)(gru2)

    # Channel-wise attention module
    concat = keras.layers.concatenate([gru1, gru2], name="Concat_Layer", axis=-1)
#     print(concat.shape)

#     squeeze = GlobalAveragePooling1D()(concat)
#     print(squeeze.shape)
    excitation = Dense(128, activation='LeakyReLU')(concat)
    excitation = Dropout(0.3)(excitation)
    excitation = Dense(64, activation='LeakyReLU')(excitation)
    excitation = Reshape((1, 64))(excitation)
    scale = multiply([concat, excitation])
    x = GlobalAveragePooling1D()(scale)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='LeakyReLU')(x)
    
    dp = Dropout(0.3)(x)
    outputs = Dense(2, activation='softmax', name="Output_Layer")(dp)
    model = Model(inputs=[input1, input2], outputs=outputs)
    return model


def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation

    
    # print(x_train1.shape, x_train2.shape, y_train.shape)
    model = create_model(x_train1.shape[1:], x_train2.shape[1:])
    # model = tf.keras.models.load_model("GRU.h5")

    # def save_model_callback(server, num_round):
    #     # Save the global model after each round
    #     model.save(f"global_model_round_{num_round}.h5")
    #     print(f"Global model saved to 'global_model_round_{num_round}.h5'")



    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5, # (0.1 - 0.5) The fraction of clients used for training on each round.
        fraction_evaluate=0.2, # (0.05 - 0.2) The fraction of clients that will be randomly selected to evaluate the global model after each round
        min_fit_clients=4, # (3 - 5) The minimum number of clients that must be available to participate in each round
        min_evaluate_clients=2, # (2 - 3) The minimum number of clients that must be available to evaluate the global model
        min_available_clients=client_number,# (20% - 30%) The minimum number of clients that must be connected and available to participate in each round
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )

    # Start Flower server (SSL-enabled) for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=50),
        strategy=strategy,
        # certificates=(
        #     Path(".cache/certificates/ca.crt").read_bytes(),
        #     Path(".cache/certificates/server.pem").read_bytes(),
        #     Path(".cache/certificates/server.key").read_bytes(),
        # ),
         # Add the save_model_callback to the on_round_end method
        # on_round_end=save_model_callback,
    )

def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    loss = tf.keras.metrics.Mean()
    auc = AUC()
    precision = Precision()
    recall = Recall()
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        global save
        global cnt
        global save_f1
        model.set_weights(parameters)  # Update model with the latest parameters
        
        predictions = model([x_train1, x_train2], training=False)
        labels = tf.reshape(y_train, predictions.shape)  # Reshape labels to match predictions shape
        loss_value = loss_fn(y_train, predictions)

        loss(loss_value)
        auc(y_train, predictions)
        precision(y_train, predictions)
        recall(y_train, predictions)

        f1 = 2 * (precision.result().numpy() * recall.result().numpy()) / (precision.result().numpy() + recall.result().numpy())


        if(f1 > save_f1):
            save_f1 = f1
            model.save('lstm.h5')

        save.append((cnt, loss.result().numpy(), auc.result().numpy(), precision.result().numpy(), recall.result().numpy(), f1))
        cnt+=1

        return float(loss.result().numpy()), {
            "auc": float(auc.result().numpy()),
            "precision": float(precision.result().numpy()),
            "recall": float(recall.result().numpy()),
            "F1": f1,
        }

    return evaluate


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 128,
        # "local_epochs": 1 if server_round < 2 else 2,
        "local_epochs": 5,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to 5 local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 5
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()
    # print(save)
    with open(f'server_{client_number}.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(save)

