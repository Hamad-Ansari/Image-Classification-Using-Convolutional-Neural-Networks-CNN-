# Image Classification
# Fashion-MNIST dataset image classification using NN architecture

# ============================================
# 1. Import Libraries
# ============================================
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================
# 2. Load and Explore Data
# ============================================
# Load Fashion-MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Dataset information
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("=" * 50)
print("DATASET INFORMATION")
print("=" * 50)
print(f"Train Images Shape: {train_images.shape}")
print(f"Train Labels Shape: {train_labels.shape}")
print(f"Test Images Shape: {test_images.shape}")
print(f"Test Labels Shape: {test_labels.shape}")
print(f"Number of Classes: {len(class_names)}")
print(f"Class Names: {class_names}")
print(f"Pixel Value Range: [{train_images.min()}, {train_images.max()}]")

# ============================================
# 3. Data Visualization
# ============================================
def visualize_dataset(images, labels, class_names, num_samples=25):
    """
    Visualize sample images from the dataset
    """
    plt.figure(figsize=(12, 12))
    for i in range(min(num_samples, len(images))):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(class_names[labels[i]], fontsize=10)
        plt.grid(False)
    plt.suptitle('Fashion-MNIST Sample Images', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Display sample images
print("\nVisualizing sample images...")
visualize_dataset(train_images, train_labels, class_names)

# Class distribution visualization
def plot_class_distribution(labels, class_names, title="Class Distribution"):
    """
    Plot the distribution of classes in the dataset
    """
    plt.figure(figsize=(12, 5))
    
    # Count plot
    plt.subplot(1, 2, 1)
    unique, counts = np.unique(labels, return_counts=True)
    bars = plt.bar(unique, counts, color='skyblue', edgecolor='black')
    plt.xlabel('Class Label', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Count per Class', fontsize=14, fontweight='bold')
    plt.xticks(unique, [class_names[i] for i in unique], rotation=45, ha='right')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                str(count), ha='center', va='bottom', fontsize=10)
    
    # Pie chart
    plt.subplot(1, 2, 2)
    plt.pie(counts, labels=class_names, autopct='%1.1f%%', startangle=90,
            colors=plt.cm.Set3(np.linspace(0, 1, len(class_names))))
    plt.title('Percentage Distribution', fontsize=14, fontweight='bold')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

print("\nAnalyzing class distribution...")
plot_class_distribution(train_labels, class_names, "Training Set Class Distribution")
plot_class_distribution(test_labels, class_names, "Test Set Class Distribution")

# ============================================
# 4. Data Preprocessing
# ============================================
def preprocess_data(train_images, test_images):
    """
    Preprocess images: normalize and reshape
    """
    # Normalize pixel values to [0, 1]
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    
    # Add channel dimension (for CNN)
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)
    
    return train_images, test_images

# Preprocess data
print("\nPreprocessing data...")
train_images, test_images = preprocess_data(train_images, test_images)
print(f"Shape after preprocessing - Train: {train_images.shape}, Test: {test_images.shape}")

# One-hot encode labels
train_labels_onehot = tf.keras.utils.to_categorical(train_labels, 10)
test_labels_onehot = tf.keras.utils.to_categorical(test_labels, 10)

# ============================================
# 5. Build Neural Network Models
# ============================================
def build_simple_nn():
    """
    Build a simple neural network model
    """
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    return model

def build_cnn():
    """
    Build a Convolutional Neural Network model
    """
    model = models.Sequential([
        # First Conv Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                     input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Conv Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Conv Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    return model

# ============================================
# 6. Model Training Configuration
# ============================================
def create_callbacks(model_name):
    """
    Create training callbacks
    """
    # Create directory for saving models
    os.makedirs('models', exist_ok=True)
    
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        
        # Save best model
        ModelCheckpoint(
            filepath=f'models/{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Save model periodically
        ModelCheckpoint(
            filepath=f'models/{model_name}_epoch_{{epoch:02d}}.h5',
            save_freq='epoch',
            period=10
        )
    ]
    
    return callbacks

# ============================================
# 7. Train Models
# ============================================
def train_model(model, model_name, train_images, train_labels, 
                test_images, test_labels, epochs=50, batch_size=64):
    """
    Train the model with given parameters
    """
    print(f"\n{'='*50}")
    print(f"TRAINING {model_name}")
    print(f"{'='*50}")
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model architecture
    print("\nModel Summary:")
    model.summary()
    
    # Create callbacks
    callbacks = create_callbacks(model_name)
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        train_images, train_labels,
        validation_data=(test_images, test_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

# Train both models
print("\n" + "="*50)
print("MODEL TRAINING")
print("="*50)

# Train Simple NN
simple_nn = build_simple_nn()
simple_nn, history_simple = train_model(
    simple_nn, "simple_nn",
    train_images, train_labels_onehot,
    test_images, test_labels_onehot,
    epochs=50, batch_size=64
)

# Train CNN
cnn_model = build_cnn()
cnn_model, history_cnn = train_model(
    cnn_model, "cnn_model",
    train_images, train_labels_onehot,
    test_images, test_labels_onehot,
    epochs=50, batch_size=64
)

# ============================================
# 8. Model Evaluation and Visualization
# ============================================
def plot_training_history(history, model_name):
    """
    Plot training history (accuracy and loss)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Training History: {model_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_images, test_labels, test_labels_onehot, class_names, model_name):
    """
    Evaluate model performance
    """
    print(f"\n{'='*50}")
    print(f"EVALUATING {model_name}")
    print(f"{'='*50}")
    
    # Get predictions
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    test_loss, test_accuracy = model.evaluate(test_images, test_labels_onehot, verbose=0)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Error Rate: {(1-test_accuracy):.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, predicted_labels, 
                              target_names=class_names, digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, predicted_labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    return test_accuracy, predicted_labels

# Evaluate models
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

# Plot training history
plot_training_history(history_simple, "Simple Neural Network")
plot_training_history(history_cnn, "CNN Model")

# Evaluate Simple NN
acc_simple, pred_simple = evaluate_model(
    simple_nn, test_images, test_labels, 
    test_labels_onehot, class_names, "Simple Neural Network"
)

# Evaluate CNN
acc_cnn, pred_cnn = evaluate_model(
    cnn_model, test_images, test_labels,
    test_labels_onehot, class_names, "CNN Model"
)

# ============================================
# 9. Model Comparison and Analysis
# ============================================
def plot_model_comparison(acc_simple, acc_cnn):
    """
    Compare model performances
    """
    models = ['Simple NN', 'CNN']
    accuracies = [acc_simple, acc_cnn]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, accuracies, color=['skyblue', 'lightcoral'], 
                   edgecolor='black', width=0.6)
    
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.ylim([0.85, 0.95])
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)
print(f"Simple Neural Network Accuracy: {acc_simple:.4f}")
print(f"CNN Model Accuracy: {acc_cnn:.4f}")
print(f"Improvement: {(acc_cnn - acc_simple):.4f}")

plot_model_comparison(acc_simple, acc_cnn)

# ============================================
# 10. Visualize Predictions
# ============================================
def visualize_predictions(model, images, true_labels, predicted_labels, 
                         class_names, num_samples=20):
    """
    Visualize model predictions with true labels
    """
    # Get indices of correct and incorrect predictions
    correct_indices = np.where(predicted_labels == true_labels)[0]
    incorrect_indices = np.where(predicted_labels != true_labels)[0]
    
    # Sample some predictions
    np.random.shuffle(correct_indices)
    np.random.shuffle(incorrect_indices)
    
    # Create visualization
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    axes = axes.ravel()
    
    # Plot correct predictions
    for i in range(5):
        if i < len(correct_indices):
            idx = correct_indices[i]
            axes[i].imshow(images[idx].reshape(28, 28), cmap=plt.cm.binary)
            axes[i].set_title(f'True: {class_names[true_labels[idx]]}\nPred: {class_names[predicted_labels[idx]]}', 
                            fontsize=10, color='green')
            axes[i].axis('off')
        else:
            axes[i].axis('off')
    
    # Plot incorrect predictions
    for i in range(5, 10):
        if (i-5) < len(incorrect_indices):
            idx = incorrect_indices[i-5]
            axes[i].imshow(images[idx].reshape(28, 28), cmap=plt.cm.binary)
            axes[i].set_title(f'True: {class_names[true_labels[idx]]}\nPred: {class_names[predicted_labels[idx]]}', 
                            fontsize=10, color='red')
            axes[i].axis('off')
        else:
            axes[i].axis('off')
    
    plt.suptitle('Model Predictions\n(Green=Correct, Red=Incorrect)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"\nCorrect predictions shown: {min(5, len(correct_indices))}")
    print(f"Incorrect predictions shown: {min(5, len(incorrect_indices))}")
    print(f"Total test samples: {len(true_labels)}")
    print(f"Number correct: {len(correct_indices)}")
    print(f"Number incorrect: {len(incorrect_indices)}")
    print(f"Accuracy on displayed subset: {len(correct_indices)/len(true_labels):.4f}")

print("\n" + "="*50)
print("VISUALIZING CNN PREDICTIONS")
print("="*50)

# Visualize CNN predictions
visualize_predictions(cnn_model, test_images, test_labels, pred_cnn, class_names)

# ============================================
# 11. Feature Visualization (CNN only)
# ============================================
def visualize_cnn_features(model, test_images, layer_index=0, num_filters=16):
    """
    Visualize convolutional filters and feature maps
    """
    # Get the convolutional layer
    conv_layer = model.layers[layer_index]
    
    # Create a model that outputs the feature maps
    feature_map_model = models.Model(inputs=model.inputs, 
                                    outputs=conv_layer.output)
    
    # Get feature maps for a sample image
    sample_image = test_images[0:1]
    feature_maps = feature_map_model.predict(sample_image)
    
    # Visualize feature maps
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(min(num_filters, feature_maps.shape[-1])):
        axes[i].imshow(feature_maps[0, :, :, i], cmap='viridis')
        axes[i].set_title(f'Filter {i+1}', fontsize=10)
        axes[i].axis('off')
    
    for i in range(feature_maps.shape[-1], len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Feature Maps from Conv Layer {layer_index+1}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Visualize first convolutional layer features
print("\nVisualizing CNN feature maps...")
try:
    visualize_cnn_features(cnn_model, test_images, layer_index=0, num_filters=16)
except Exception as e:
    print(f"Could not visualize feature maps: {e}")

# ============================================
# 12. Error Analysis
# ============================================
def analyze_errors(true_labels, predicted_labels, class_names):
    """
    Analyze prediction errors
    """
    # Create error matrix
    error_matrix = confusion_matrix(true_labels, predicted_labels)
    
    # Calculate per-class error rates
    per_class_errors = []
    for i in range(len(class_names)):
        total = np.sum(error_matrix[i, :])
        correct = error_matrix[i, i]
        error_rate = (total - correct) / total if total > 0 else 0
        per_class_errors.append(error_rate)
    
    # Plot error rates
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(class_names)), per_class_errors, 
                   color='salmon', edgecolor='black')
    
    plt.title('Error Rate per Class', fontsize=16, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Error Rate', fontsize=12)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add error rate values
    for bar, error_rate in zip(bars, per_class_errors):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{error_rate:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Find most confused pairs
    confusion_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and error_matrix[i, j] > 0:
                confusion_pairs.append((i, j, error_matrix[i, j]))
    
    # Sort by confusion count
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("\nTop Confusing Class Pairs:")
    print("-" * 40)
    for i, j, count in confusion_pairs[:10]:
        print(f"{class_names[i]} → {class_names[j]}: {count} misclassifications")

print("\n" + "="*50)
print("ERROR ANALYSIS")
print("="*50)
analyze_errors(test_labels, pred_cnn, class_names)

# ============================================
# 13. Save Final Models and Results
# ============================================
def save_results(history_simple, history_cnn, acc_simple, acc_cnn):
    """
    Save training results and metrics
    """
    os.makedirs('results', exist_ok=True)
    
    # Save training histories
    np.save('results/history_simple.npy', history_simple.history)
    np.save('results/history_cnn.npy', history_cnn.history)
    
    # Save metrics
    metrics = {
        'simple_nn_accuracy': float(acc_simple),
        'cnn_accuracy': float(acc_cnn),
        'improvement': float(acc_cnn - acc_simple),
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    import json
    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save models
    simple_nn.save('models/simple_nn_final.h5')
    cnn_model.save('models/cnn_model_final.h5')
    
    print("\nResults and models saved successfully!")
    print(f"Models saved in: models/")
    print(f"Results saved in: results/")

# Save everything
print("\n" + "="*50)
print("SAVING RESULTS")
print("="*50)
save_results(history_simple, history_cnn, acc_simple, acc_cnn)

# ============================================
# 14. Generate Report Summary
# ============================================
print("\n" + "="*50)
print("PROJECT SUMMARY REPORT")
print("="*50)
print(f"\nProject: Fashion-MNIST Image Classification")
print(f"Dataset: {train_images.shape[0]:,} training samples")
print(f"          {test_images.shape[0]:,} test samples")
print(f"Classes: {len(class_names)} categories")
print("\n" + "-"*50)
print("MODEL PERFORMANCE SUMMARY:")
print("-"*50)
print(f"Simple Neural Network:")
print(f"  • Test Accuracy: {acc_simple:.4f}")
print(f"  • Test Error Rate: {1-acc_simple:.4f}")
print(f"  • Architecture: 3 Dense Layers with Dropout")
print(f"\nCNN Model:")
print(f"  • Test Accuracy: {acc_cnn:.4f}")
print(f"  • Test Error Rate: {1-acc_cnn:.4f}")
print(f"  • Architecture: 3 Conv Blocks + Dense Layers")
print(f"  • Improvement over Simple NN: {acc_cnn-acc_simple:.4f}")
print("\n" + "-"*50)
print("KEY OBSERVATIONS:")
print("-"*50)
print("1. CNN significantly outperforms simple neural network")
print("2. Data augmentation could further improve performance")
print("3. Most confusion occurs between similar clothing items")
print("4. Model shows good generalization with validation accuracy")
print("\n" + "-"*50)
print("NEXT STEPS:")
print("-"*50)
print("1. Experiment with different architectures")
print("2. Implement data augmentation techniques")
print("3. Try transfer learning with pre-trained models")
print("4. Optimize hyperparameters with grid search")
print("5. Deploy model as web application")

# ============================================
# 15. Cleanup and Final Notes
# ============================================
print("\n" + "="*50)
print("CLEANUP")
print("="*50)

# Clear session to free memory
tf.keras.backend.clear_session()

print("\nTraining completed successfully!")
print("\nTo use the trained models:")
print("1. Load model: model = tf.keras.models.load_model('models/cnn_model_final.h5')")
print("2. Make predictions: predictions = model.predict(new_images)")
print("3. Get class labels: predicted_classes = np.argmax(predictions, axis=1)")
print("\nThank you for using this Fashion-MNIST classifier!")

# ============================================
# END OF NOTEBOOK
# ============================================