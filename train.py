# Libraries
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import time
import tensorflow as tf

# Project files
from image_generator_2 import ImageDataGenerator
from network import Net
from utils import getCallbacks


# Data paths
TRAIN_X_DIR = ""
TRAIN_Y_DIR = ""
VALID_X_DIR = ""
VALID_Y_DIR = ""

# Model parameters
LOAD_MODEL = False
BATCH_SIZE = 16
PATCH_SIZE = 256
PATCHES_PER_IMAGE = 1
EPOCHS = 1000
PATIENCE = 10
LEARNING_RATE = 1e-4

PROGRAM_TIME_STAMP = time.strftime("%Y-%m-%d_%H%M%S")


def loadModel(load_pretrained_model=True, model_root="models"):
    if load_pretrained_model:
        latest_model = sorted(glob.glob(model_root + "/*/*.h5"))[-1]
        model = load_model(
            latest_model,
            custom_objects={
                "tf": tf
            })
        print("Loaded model: {}".format(latest_model))
    else:
        model = Net((PATCH_SIZE, PATCH_SIZE, 3))

        # Compile model
        model.compile(
            optimizer=Adam(LEARNING_RATE),
            loss=loss_function,
            metrics=[])
    print("Number of model parameters: {:,}".format(model.count_params()))
    return model


def train():
    # Load model
    model = loadModel(LOAD_MODEL)
    save_root = "models/{}".format(PROGRAM_TIME_STAMP)

    # Load data generators
    train_data_generator = ImageDataGenerator()
    train_batch_generator = train_data_generator.trainAndGtBatchGenerator(
        TRAIN_X_DIR, TRAIN_Y_DIR, BATCH_SIZE, PATCHES_PER_IMAGE, PATCH_SIZE,
        normalize=True)
    number_of_train_batches = train_data_generator.numberOfBatchesPerEpoch(
        TRAIN_X_DIR, BATCH_SIZE, PATCHES_PER_IMAGE)
    valid_data_generator = ImageDataGenerator()
    valid_batch_generator = valid_data_generator.trainAndGtBatchGenerator(
        VALID_X_DIR, VALID_Y_DIR, BATCH_SIZE, PATCHES_PER_IMAGE, PATCH_SIZE,
        normalize=True)
    number_of_valid_batches = valid_data_generator.numberOfBatchesPerEpoch(
        VALID_X_DIR, BATCH_SIZE, PATCHES_PER_IMAGE)

    # Define callbacks
    callbacks = getCallbacks(PATIENCE, save_root, BATCH_SIZE)

    # Start training
    history = model.fit_generator(
        train_batch_generator, steps_per_epoch=number_of_train_batches,
        epochs=EPOCHS, validation_data=valid_batch_generator,
        validation_steps=number_of_valid_batches,
        callbacks=callbacks)
    print("Model saved to: '{}'".format(save_root))


def main():
    train()


main()
