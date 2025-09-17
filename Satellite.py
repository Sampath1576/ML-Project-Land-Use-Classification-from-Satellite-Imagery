# --- Step 1: Project Setup and Data Acquisition ---
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
import random # We will use this for the genetic algorithm

# Load the dataset (same as before)
(ds_train, ds_val, ds_test), info = tfds.load(
    'eurosat/rgb',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True
)

class_names = info.features['label'].names
IMG_SIZE = info.features['image'].shape[0]
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
NUM_CLASSES = info.features['label'].num_classes

# Preprocessing for transfer learning model (Same as before)
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
def preprocess_image_resnet(image, label):
    image = tf.cast(image, tf.float32)
    return resnet50_preprocess_input(image), label

BATCH_SIZE = 32
train_ds_transfer = ds_train.map(preprocess_image_resnet).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds_transfer = ds_val.map(preprocess_image_resnet).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- Step 2: The Genetic Algorithm Core ---

# Define the 'genes' (the hyperparameters we want to optimize)
GENE_OPTIONS = {
    "learning_rate": [1e-3, 1e-4, 1e-5],
    "dense_units": [64, 128, 256],
    "optimizer": ['adam', 'rmsprop']
}

# Define the Fitness Function
def calculate_fitness(genes):
    learning_rate = genes['learning_rate']
    dense_units = genes['dense_units']
    optimizer_name = genes['optimizer']

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(dense_units, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(
        train_ds_transfer,
        validation_data=val_ds_transfer,
        epochs=5,
        verbose=0
    )
    
    fitness = history.history['val_accuracy'][-1]
    return fitness, model

# --- Step 3: The Simplified Genetic Loop ---
def run_genetic_algorithm(generations=3, population_size=5):
    population = []
    
    print("--- Generation 0: Creating initial population ---")
    for _ in range(population_size):
        genes = {gene: random.choice(options) for gene, options in GENE_OPTIONS.items()}
        population.append(genes)
        print(f"  Initial genes: {genes}")

    best_fitness_per_generation = []

    for generation in range(generations):
        print(f"\n--- Generation {generation+1}: Evolving ---")
        
        fitness_scores = []
        for genes in population:
            fitness, _ = calculate_fitness(genes)
            fitness_scores.append(fitness)
            print(f"  Genes: {genes} -> Fitness: {fitness:.4f}")
        
        # Track the best fitness of this generation
        best_fitness_this_gen = max(fitness_scores)
        best_fitness_per_generation.append(best_fitness_this_gen)
        
        print(f"  Best accuracy for this generation: {best_fitness_this_gen:.4f}")
        
        # Selection: Sort by fitness and select the best half
        ranked_population = sorted(zip(fitness_scores, population), key=lambda x: x[0], reverse=True)
        survivors = [p for _, p in ranked_population[:population_size//2]]
        
        # Crossover & Mutation: Create new offspring
        next_generation = survivors[:]
        while len(next_generation) < population_size:
            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)
            
            child_genes = {}
            for gene in GENE_OPTIONS.keys():
                child_genes[gene] = random.choice([parent1[gene], parent2[gene]])
            
            if random.random() < 0.2:
                mutated_gene = random.choice(list(GENE_OPTIONS.keys()))
                child_genes[mutated_gene] = random.choice(GENE_OPTIONS[mutated_gene])
                
            next_generation.append(child_genes)
        
        population = next_generation
        
    print("\n--- Genetic Algorithm Completed ---")
    print(f"Final best accuracy found: {max(best_fitness_per_generation):.4f}")
    
    # --- Plotting the GA's Progress (NEW) ---
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, generations + 1), best_fitness_per_generation, marker='o', linestyle='-')
    plt.title('Genetic Algorithm Accuracy Evolution')
    plt.xlabel('Generation')
    plt.ylabel('Best Validation Accuracy')
    plt.xticks(range(1, generations + 1))
    plt.grid(True)
    plt.show()

# To run this example, simply call the function below
run_genetic_algorithm()
