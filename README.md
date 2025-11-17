# The-genetic-algorithn-using-turning-CNN-hyperparamter-for-yoga-pose-estimation.
Yoga pose estimation is essential for fitness, rehabilitation, and posture correction.  2) Convolutional Neural Networks (CNNs) perform well, but manual hyperparameter tuning is slow and    inefficient.  3) Genetic Algorithm (GA) provides an automated, evolutionary approach to optimize CNN hyperparameters for better accuracy and faster convergence.
CODE:
###PYTHON SCRIPT RUN
"""
GA tuner for CNN hyperparameters for yoga-pose classification.
Usage:
 - Place your data in directories: data/train and data/val (each with class subfolders)
 - Adjust DATA_DIR_TRAIN and DATA_DIR_VAL if needed.
 - Run: python ga_cnn_tuner.py
"""

import os
import random
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import accuracy_score

# -------------------- User settings --------------------
DATA_DIR_TRAIN = "data/train"
DATA_DIR_VAL   = "data/val"
IMG_SIZE = (128, 128)      # keep small for fast eval; increase for final runs
NUM_EPOCHS_FOR_FITNESS = 3 # small for GA fitness evaluation
POPULATION_SIZE = 8
GENERATIONS = 6
TOURNAMENT_SIZE = 3
MUTATION_PROB = 0.2
CROSSOVER_PROB = 0.9
SEED = 42
OUTPUT_BEST_JSON = "best_hyperparams.json"
# -------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load dataset (TensorFlow image datasets)
def load_datasets(batch_size):
    train_ds = image_dataset_from_directory(
        DATA_DIR_TRAIN,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=True,
        seed=SEED
    )
    val_ds = image_dataset_from_directory(
        DATA_DIR_VAL,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=False,
        seed=SEED
    )
    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, len(class_names)

# Define hyperparameter search space (discrete choices / ranges)
HYPERSPACE = {
    "num_conv_blocks": [1, 2, 3],
    "filters_base": [16, 32, 48, 64],
    "kernel_size": [3, 5],
    "conv_per_block": [1, 2],
    "dense_units": [64, 128, 256],
    "dropout": [0.0, 0.25, 0.4],
    "learning_rate": [1e-3, 5e-4, 1e-4],
    "optimizer": ["adam", "rmsprop", "sgd"],
    "batch_size": [16, 32, 64]
}

# Individual encoding: dict of hyperparameters
def random_individual():
    return {k: random.choice(v) for k, v in HYPERSPACE.items()}

# Build Keras model from hyperparameters
def build_model(hp, input_shape, num_classes):
    inp = layers.Input(shape=input_shape)
    x = inp
    filters = hp["filters_base"]
    for block in range(hp["num_conv_blocks"]):
        for _ in range(hp["conv_per_block"]):
            x = layers.Conv2D(filters, hp["kernel_size"], padding="same", activation="relu")(x)
            x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        filters = min(filters * 2, 512)
    x = layers.Flatten()(x)
    x = layers.Dense(hp["dense_units"], activation="relu")(x)
    if hp["dropout"] > 0:
        x = layers.Dropout(hp["dropout"])(x)
    out_activation = "softmax" if num_classes > 2 else "sigmoid"
    out_units = num_classes if num_classes > 2 else 1
    outputs = layers.Dense(out_units, activation=out_activation)(x)
    model = models.Model(inputs=inp, outputs=outputs)
    # optimizer
    lr = hp["learning_rate"]
    if hp["optimizer"] == "adam":
        opt = optimizers.Adam(learning_rate=lr)
    elif hp["optimizer"] == "rmsprop":
        opt = optimizers.RMSprop(learning_rate=lr)
    else:
        opt = optimizers.SGD(learning_rate=lr, momentum=0.9)
    loss = "sparse_categorical_crossentropy" if num_classes > 2 else "binary_crossentropy"
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    return model

# Fitness evaluation: trains model for a few epochs and returns validation accuracy
def evaluate_fitness(individual, train_ds, val_ds, num_classes):
    batch_size = individual["batch_size"]
    # we must rebuild datasets with this batch size
    # Instead of recreating datasets every call (costly), we can re-batch existing ones.
    train_ds_b = train_ds.unbatch().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds_b = val_ds.unbatch().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    input_shape = IMG_SIZE + (3,)
    model = build_model(individual, input_shape, num_classes)
    # train briefly
    try:
        history = model.fit(
            train_ds_b,
            validation_data=val_ds_b,
            epochs=NUM_EPOCHS_FOR_FITNESS,
            verbose=0
        )
    except Exception as e:
        print("Training error with individual:", individual, "->", e)
        return 0.0
    val_acc = history.history.get("val_accuracy", [0.0])[-1]
    # free memory
    tf.keras.backend.clear_session()
    return float(val_acc)

# GA operators
def tournament_selection(population, fitnesses, k):
    selected = random.sample(list(range(len(population))), k)
    best = max(selected, key=lambda i: fitnesses[i])
    return population[best].copy()

def crossover(parent1, parent2):
    if random.random() > CROSSOVER_PROB:
        return parent1.copy(), parent2.copy()
    # single-point crossover over keys
    keys = list(parent1.keys())
    pt = random.randint(1, len(keys)-1)
    child1 = {}
    child2 = {}
    for i, key in enumerate(keys):
        if i < pt:
            child1[key] = parent1[key]
            child2[key] = parent2[key]
        else:
            child1[key] = parent2[key]
            child2[key] = parent1[key]
    return child1, child2

def mutate(ind):
    for key in ind.keys():
        if random.random() < MUTATION_PROB:
            ind[key] = random.choice(HYPERSPACE[key])
    return ind

# Main GA loop
def run_ga():
    # initial pop
    population = [random_individual() for _ in range(POPULATION_SIZE)]
    best_overall = None
    best_score = -1.0

    # prepare dataset using a default batch_size (we will re-batch during eval)
    default_batch = 32
    train_ds, val_ds, num_classes = load_datasets(default_batch)

    for gen in range(GENERATIONS):
        print(f"\n=== Generation {gen+1}/{GENERATIONS} ===")
        # evaluate population
        fitnesses = []
        for idx, ind in enumerate(population):
            print(f"Evaluating individual {idx+1}/{len(population)}: {ind}")
            score = evaluate_fitness(ind, train_ds, val_ds, num_classes)
            print(f" -> val accuracy: {score:.4f}")
            fitnesses.append(score)
            if score > best_score:
                best_score = score
                best_overall = ind.copy()
                # save best immediately
                with open(OUTPUT_BEST_JSON, "w") as f:
                    json.dump({"hyperparams": best_overall, "val_acc": best_score}, f, indent=2)
        # new population
        new_pop = []
        while len(new_pop) < POPULATION_SIZE:
            p1 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
            p2 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_pop.append(c1)
            if len(new_pop) < POPULATION_SIZE:
                new_pop.append(c2)
        population = new_pop
        print(f"End of gen {gen+1} best so far: {best_score:.4f} -> {best_overall}")

    print("\n=== GA finished ===")
    print("Best hyperparams:", best_overall)
    print("Best validation accuracy:", best_score)
    print(f"Saved best to {OUTPUT_BEST_JSON}")

if __name__ == "__main__":
    run_ga()





####KOTLIN###
data/
  train/
    tree/
    cobra/
    warrior/
    ...
  val/
    tree/
    cobra/
    warrior/
    ...
  test/
    tree/
    cobra/
    warrior/
    ...


###PYTHON###


import os
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing import image_dataset_from_directory
import cv2

# ---------------- Settings ----------------
DATA_DIR_TRAIN = "data/train"
DATA_DIR_VAL = "data/val"
DATA_DIR_TEST = "data/test"

IMG_SIZE = (128, 128)
POPULATION_SIZE = 6
GENERATIONS = 4
NUM_EPOCHS_FOR_FITNESS = 2
FINAL_TRAIN_EPOCHS = 10
SEED = 42
BEST_HP_FILE = "best_hyperparams.json"
# ------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Hyperparameter search space
HYPERSPACE = {
    "num_conv_blocks": [1, 2, 3],
    "filters_base": [16, 32, 64],
    "kernel_size": [3, 5],
    "dense_units": [64, 128, 256],
    "dropout": [0.0, 0.3, 0.5],
    "learning_rate": [1e-3, 5e-4, 1e-4],
    "optimizer": ["adam", "rmsprop"],
    "batch_size": [16, 32, 64]
}

# Dataset loading
def load_datasets(batch_size):
    train_ds = image_dataset_from_directory(
        DATA_DIR_TRAIN, image_size=IMG_SIZE, batch_size=batch_size
    )
    val_ds = image_dataset_from_directory(
        DATA_DIR_VAL, image_size=IMG_SIZE, batch_size=batch_size
    )
    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, class_names

# Model builder
def build_model(hp, input_shape, num_classes):
    model = models.Sequential()
    filters = hp["filters_base"]
    for b in range(hp["num_conv_blocks"]):
        model.add(layers.Conv2D(filters, hp["kernel_size"], activation='relu', padding='same', input_shape=input_shape if b==0 else None))
        model.add(layers.MaxPooling2D(2))
        filters *= 2
    model.add(layers.Flatten())
    model.add(layers.Dense(hp["dense_units"], activation='relu'))
    if hp["dropout"] > 0:
        model.add(layers.Dropout(hp["dropout"]))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    opt = optimizers.Adam(learning_rate=hp["learning_rate"]) if hp["optimizer"] == "adam" else optimizers.RMSprop(learning_rate=hp["learning_rate"])
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Evaluate fitness
def evaluate_fitness(ind, train_ds, val_ds, num_classes):
    model = build_model(ind, IMG_SIZE+(3,), num_classes)
    try:
        hist = model.fit(train_ds, validation_data=val_ds, epochs=NUM_EPOCHS_FOR_FITNESS, verbose=0)
        val_acc = hist.history['val_accuracy'][-1]
    except Exception:
        val_acc = 0.0
    tf.keras.backend.clear_session()
    return val_acc

# Random individual
def random_individual():
    return {k: random.choice(v) for k, v in HYPERSPACE.items()}

# Selection, crossover, mutation
def tournament_select(pop, fit, k=3):
    sel = random.sample(range(len(pop)), k)
    best = max(sel, key=lambda i: fit[i])
    return pop[best].copy()

def crossover(p1, p2):
    keys = list(p1.keys())
    pt = random.randint(1, len(keys)-1)
    c1, c2 = {}, {}
    for i, k in enumerate(keys):
        c1[k] = p1[k] if i < pt else p2[k]
        c2[k] = p2[k] if i < pt else p1[k]
    return c1, c2

def mutate(ind, rate=0.3):
    for k in ind.keys():
        if random.random() < rate:
            ind[k] = random.choice(HYPERSPACE[k])
    return ind

# GA optimization
def run_ga():
    pop = [random_individual() for _ in range(POPULATION_SIZE)]
    best_ind, best_fit = None, -1
    train_ds, val_ds, class_names = load_datasets(batch_size=32)
    for gen in range(GENERATIONS):
        print(f"\nGeneration {gen+1}/{GENERATIONS}")
        fits = []
        for i, ind in enumerate(pop):
            fit = evaluate_fitness(ind, train_ds, val_ds, len(class_names))
            fits.append(fit)
            print(f"  {i+1}. {ind} => {fit:.4f}")
            if fit > best_fit:
                best_fit, best_ind = fit, ind.copy()
                with open(BEST_HP_FILE, "w") as f:
                    json.dump({"hp": best_ind, "fitness": best_fit}, f, indent=2)
        # Reproduction
        new_pop = []
        while len(new_pop) < POPULATION_SIZE:
            p1, p2 = tournament_select(pop, fits), tournament_select(pop, fits)
            c1, c2 = crossover(p1, p2)
            new_pop.extend([mutate(c1), mutate(c2)])
        pop = new_pop[:POPULATION_SIZE]
    print("\nBest HP:", best_ind, "Fitness:", best_fit)
    return best_ind, class_names

# Final training with best HP
def train_best(best_hp, class_names):
    bs = best_hp["batch_size"]
    train_ds, val_ds, _ = load_datasets(bs)
    model = build_model(best_hp, IMG_SIZE+(3,), len(class_names))
    print("\nTraining final model with best hyperparameters...")
    model.fit(train_ds, validation_data=val_ds, epochs=FINAL_TRAIN_EPOCHS, verbose=1)
    model.save("best_yoga_model.h5")
    print("Model saved as best_yoga_model.h5")
    return model

# Visualization: input + prediction
def visualize_predictions(model, class_names):
    print("\nVisualizing predictions...")
    for root, dirs, files in os.walk(DATA_DIR_TEST):
        for f in random.sample(files, min(5, len(files))):
            img_path = os.path.join(root, f)
            img = cv2.imread(img_path)
            img_resized = cv2.resize(img, IMG_SIZE)
            arr = np.expand_dims(img_resized / 255.0, axis=0)
            preds = model.predict(arr, verbose=0)
            pred_label = class_names[np.argmax(preds)]
            # display
            plt.figure(figsize=(3,3))
            plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
            plt.title(f"Predicted: {pred_label}")
            plt.axis("off")
            plt.show()

# ---------------- Run All ----------------
if __name__ == "__main__":
    best_hp, class_names = run_ga()
    model = train_best(best_hp, class_names)
    visualize_predictions(model, class_names)
