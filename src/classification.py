# IMPORTS
print("[INFO]: Importing packages")
# general tools
import os
import pandas as pd
import argparse
# tf tools
import tensorflow as tf
from tensorflow import keras
# image processsing
from tensorflow.keras.preprocessing.image import (ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 VGG16)
# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense,
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model
# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD
# scikit-learn
from sklearn.metrics import classification_report
# for plotting
import numpy as np
import matplotlib.pyplot as plt

# terminal parsing function
def input_parse(): # CONSIDER BATCH SIZE 577 and 2 epochs and rotation range 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True, help="int ID to identify model")
    parser.add_argument("--batch_size", type=int, default=158, help="batch size for model training, default=158")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs, default=10")
    parser.add_argument("--init_lr", type=float, default=0.01, help="initial learning rate to be subjected to exponential decay, default=0.01")
    parser.add_argument("--decay_steps", type=int, default=10000, help="number of steps at which init_lr is multiplied with decay_rate, default=10000")
    parser.add_argument("--decay_rate", type=float, default=0.9, help="decay rate of init_lr, default=0.9")
    parser.add_argument("--horizontal_flip", action="store_true", help="flag, augment images by horizontal flip")
    parser.add_argument("--rotation_range", type=int, default=0, help="degree range for random rotation of images, default=0")
    args = parser.parse_args()

    return(args)

# plotting function
def plot_history(H, epochs, ID):
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    
    plt.savefig(os.path.join("..","models",f"history_plot_{ID}.png"))

def main(ID, batch_size, epochs, init_lr, decay_steps, decay_rate, horizontal_flip, rotation_range):
    
    # load model without classifier layers
    print("[INFO]: Importing VGG16")
    model = VGG16(include_top=False, 
              pooling='avg',
              input_shape=(224, 224, 3))
    
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False

    # add new classifier layers
    print("[INFO]: Building fc network")
    flat1 = Flatten()(model.layers[-1].output)
    bn = BatchNormalization()(flat1)
    class1 = Dense(256, 
                activation='relu')(bn)
    class2 = Dense(128, 
                activation='relu')(class1)
    output = Dense(15,
                activation='softmax')(class2)

    # define new model
    model = Model(inputs=model.inputs, 
                outputs=output)

    # define optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=init_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate,)
    sgd = SGD(learning_rate=lr_schedule)

    # compile
    model.compile(optimizer=sgd,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # prepare data frames
    print("[INFO]: Preparing data")
    df_train = pd.read_json(os.path.join("..","data","train_data.json"), lines=True)
    df_val = pd.read_json(os.path.join("..","data","val_data.json"), lines=True)
    df_test = pd.read_json(os.path.join("..","data","test_data.json"), lines=True)

    # initialize image data generator
    train_generator = ImageDataGenerator(horizontal_flip=horizontal_flip,
                                         rotation_range=rotation_range,
                                         preprocessing_function=preprocess_input)
    
    test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

    # apply generators to data
    data_dir = os.path.join("..","data")
    traingen = train_generator.flow_from_dataframe(df_train,
                                                   directory=data_dir,
                                                   x_col='image_path',
                                                   y_col='class_label',
                                                   target_size=(224, 224),
                                                   class_mode='categorical',
                                                   batch_size=batch_size,
                                                   seed=42)

    valgen = train_generator.flow_from_dataframe(df_val,
                                                 directory=data_dir,
                                                 x_col='image_path',
                                                 y_col='class_label',
                                                 target_size=(224, 224),
                                                 class_mode='categorical',
                                                 batch_size=batch_size,
                                                 seed=42)
    
    testgen = test_generator.flow_from_dataframe(df_test,
                                                 directory=data_dir,
                                                 x_col='image_path',
                                                 y_col='class_label',
                                                 target_size=(224, 224),
                                                 class_mode='categorical',
                                                 batch_size=1,
                                                 shuffle=False,
                                                 seed=42)

    # fit model
    print("[INFO]: Model training")
    H = model.fit(traingen,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=valgen,
                steps_per_epoch=traingen.samples // batch_size,
                validation_steps=valgen.samples // batch_size,
                verbose=1)
    
    # save history plots
    plot_history(H, epochs, ID)

    # save model and model parameters report
    model_path = os.path.join("..","models",f"model_{ID}.SavedModel")
    model.save(model_path)

    txtpath = os.path.join("..", "models", f"model_{ID}.txt")
    with open(txtpath, "w") as file:
        L = [f"Batch size: {batch_size} \n", 
            f"Epochs: {epochs} \n",
            f"Initial learning rate: {init_lr} \n",
            f"Decay steps: {decay_steps} \n",
            f"Decay rate: {decay_rate} \n",
            f"Horizontal flip: {horizontal_flip} \n",
            f"Rotation range: {rotation_range}"]
        file.writelines(L)

    # predict
    print("[INFO]: Predicting")
    predictions = model.predict(testgen)
    
    # classification report
    label_names = list(testgen.class_indices.keys())
    report = classification_report(testgen.classes,
                                   predictions.argmax(axis=1),
                                   target_names=label_names)

    txtpath = os.path.join("..", "reports", f"model_{ID}_classification_report.txt")
    with open(txtpath, "w") as file:
        file.write(report)

if __name__ == "__main__":
    args = input_parse()
    main(args.id, args.batch_size, args.epochs, args.init_lr, args.decay_steps, args.decay_rate, args.horizontal_flip, args.rotation_range)