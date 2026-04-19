"""
train_cnn.py
============
Training pipeline for the Mexican Sign Language (LSM) multi-channel CNN classifier.

Architecture:
    Four parallel CNN branches process independent color channels (R, G, B, Grayscale).
    Feature maps are concatenated and passed through fully-connected layers to classify
    16 LSM signs.

Dataset:
    3,750 labeled images (200x200 px) cropped from YOLOv7 hand detections.
    Labels: NoPrediccion, Hola, Como, Estar, Bien, Gracias, Que, Hacer, Tu,
            Tambien, Comer, Trabajar, Mal, Si, No, Adios
"""

import os
import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import (
    Activation,
    AveragePooling1D,
    Add,
    concatenate,
    Conv1D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LeakyReLU,
    MaxPool1D,
    MaxPooling2D,
    ReLU,
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model, Sequence
from keras.activations import sigmoid


# ── Configuration ──────────────────────────────────────────────────────────────
IMAGE_DIR  = 'C:/Users/Imagenes'
LABEL_DIR  = 'C:/Users/Etiquetas'
DIM        = 200       # Image resize dimension (px)
BATCH_SIZE = 32
EPOCHS     = 20
TEST_SIZE  = 0.1
RANDOM_STATE = 42

LABELS = [
    'NoPrediccion', 'Hola', 'Como', 'Estar', 'Bien', 'Gracias',
    'Que', 'Hacer', 'Tu', 'Tambien', 'Comer', 'Trabajar',
    'Mal', 'Si', 'No', 'Adios'
]


# ── Helper functions ───────────────────────────────────────────────────────────

def show_image(X, i):
    """Display the i-th image in array X for visual verification."""
    if i < 0 or i >= X.shape[0]:
        print(f"Error: index {i} out of range. X contains {X.shape[0]} images.")
        return
    img = X[i]
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Image {i}")
    plt.show()


def balanced_train_test_split(X, Y, test_size=0.1, random_state=None):
    """
    Stratified train/test split ensuring equal class representation.

    Parameters
    ----------
    X : np.ndarray  Array of images (n, H, W, C)
    Y : np.ndarray  Array of integer labels (n,)
    test_size : float  Proportion reserved for testing
    random_state : int  Seed for reproducibility

    Returns
    -------
    X_train, X_test, Y_train, Y_test : np.ndarray
    """
    classes = np.unique(Y)
    X_train, X_test, Y_train, Y_test = [], [], [], []

    for c in classes:
        indices = np.where(Y == c)[0]
        X_c, Y_c = X[indices], Y[indices]
        n_test   = max(1, int(len(X_c) * test_size))
        X_tr, X_te, Y_tr, Y_te = train_test_split(
            X_c, Y_c, test_size=n_test, random_state=random_state
        )
        X_train.extend(X_tr); X_test.extend(X_te)
        Y_train.extend(Y_tr); Y_test.extend(Y_te)

    X_train, X_test = np.array(X_train), np.array(X_test)
    Y_train, Y_test = np.array(Y_train), np.array(Y_test)

    rng = np.random.default_rng(random_state)
    for arr_x, arr_y in [(X_train, Y_train), (X_test, Y_test)]:
        idx = np.arange(len(arr_x))
        rng.shuffle(idx)
        arr_x[:] = arr_x[idx]
        arr_y[:] = arr_y[idx]

    return X_train, X_test, Y_train, Y_test


def check_balance(Y_train, Y_test):
    """Print class distribution for train and test sets."""
    classes, train_counts = np.unique(Y_train, return_counts=True)
    _, test_counts         = np.unique(Y_test,  return_counts=True)
    print("Class balance:")
    for c, tr, te in zip(classes, train_counts, test_counts):
        print(f"  Class {c}: train={tr} ({tr/len(Y_train):.1%})  test={te} ({te/len(Y_test):.1%})")


def extract_channels(X):
    """
    Split an RGB image array into R, G, B, and Grayscale channel arrays.

    Parameters
    ----------
    X : np.ndarray  Shape (n, H, W, 3)

    Returns
    -------
    R, G, B, Gray : np.ndarray  Each shape (n, H, W)
    """
    R    = X[:, :, :, 0]
    G    = X[:, :, :, 1]
    B    = X[:, :, :, 2]
    Gray = np.stack([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in X], axis=0)
    return R, G, B, Gray


def data_generator(xr, xg, xb, xbn, Y, batch_size):
    """Infinite generator yielding batches of multi-channel inputs and labels."""
    n = len(Y)
    while True:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            yield (
                [
                    xr [start:end, :, :, np.newaxis],
                    xg [start:end, :, :, np.newaxis],
                    xb [start:end, :, :, np.newaxis],
                    xbn[start:end, :, :, np.newaxis],
                ],
                Y[start:end],
            )


# ── Data loading & preprocessing ──────────────────────────────────────────────

def load_dataset(image_dir, label_dir, dim):
    """
    Load, crop, and resize images using YOLOv7 bounding-box labels.

    Returns
    -------
    X : np.ndarray  Shape (n, dim, dim, 3)
    Y : np.ndarray  Shape (n,)  Integer class labels
    """
    label_names = os.listdir(label_dir)
    image_names = os.listdir(image_dir)
    images_db, labels_db = [], []

    for j, label_file in enumerate(label_names):
        img_array = np.array(Image.open(os.path.join(image_dir, image_names[j])))
        h, w      = img_array.shape[:2]

        label_data = np.loadtxt(os.path.join(label_dir, label_file))
        label_data = np.array(label_data)

        def crop_and_resize(coords, img):
            cx, cw = coords[1] * w, coords[3] * w
            cy, ch = coords[2] * h, coords[4] * h
            x1 = int(cx - cw / 2); y1 = int(cy - ch / 2)
            x2 = int(cx + cw / 2); y2 = int(cy + ch / 2)
            cropped = img[y1:y2, x1:x2]
            return cv2.resize(cropped, (dim, dim)) if cropped.size > 0 else None

        if label_data.size == 5:  # Single detection
            result = crop_and_resize(label_data, img_array)
            if result is not None:
                images_db.append(result)
                labels_db.append(int(label_data[0]))

        elif label_data.size in (10, 15):  # Two or three detections
            for row in label_data:
                result = crop_and_resize(row, img_array)
                if result is not None:
                    images_db.append(result)
                    labels_db.append(int(row[0]))

        if j % 100 == 0:
            print(f"Processed {j}/{len(label_names)} images")

    X = np.array(images_db)
    Y = np.array(labels_db)
    print(f"Dataset loaded: {X.shape[0]} samples, {len(np.unique(Y))} classes")
    return X, Y


# ── Model definition ───────────────────────────────────────────────────────────

def build_cnn_branch(input_tensor, name_prefix):
    """Single CNN branch for one color channel."""
    x = Conv2D(16, kernel_size=3, padding='same', activation=LeakyReLU())(input_tensor)
    x = MaxPooling2D(pool_size=2)(x)
    x = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU())(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Conv2D(8,  kernel_size=3, padding='same', activation=LeakyReLU())(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Flatten()(x)
    return x


def build_model(dim, n_classes):
    """
    Multi-channel CNN with four parallel branches (R, G, B, Grayscale).

    Parameters
    ----------
    dim      : int  Input image dimension (square)
    n_classes: int  Number of output classes

    Returns
    -------
    model : tf.keras.Model
    """
    entrada_R  = Input(shape=(dim, dim, 1), name='input_R')
    entrada_G  = Input(shape=(dim, dim, 1), name='input_G')
    entrada_B  = Input(shape=(dim, dim, 1), name='input_B')
    entrada_BN = Input(shape=(dim, dim, 1), name='input_Gray')

    branch_R  = build_cnn_branch(entrada_R,  'R')
    branch_G  = build_cnn_branch(entrada_G,  'G')
    branch_B  = build_cnn_branch(entrada_B,  'B')
    branch_BN = build_cnn_branch(entrada_BN, 'Gray')

    merged  = concatenate([branch_R, branch_G, branch_B, branch_BN])
    x       = Dense(500, activation='sigmoid')(merged)
    x       = Dropout(0.25)(x)
    x       = Dense(750, activation=LeakyReLU())(x)
    output  = Dense(n_classes, activation='softmax')(x)

    model = Model(
        inputs=[entrada_R, entrada_G, entrada_B, entrada_BN],
        outputs=output,
        name='LSM_MultiChannel_CNN'
    )
    return model


# ── Evaluation & visualization ─────────────────────────────────────────────────

def evaluate_model(model, xrp, xgp, xbp, xbnp, Y_test):
    """Print metrics and plot confusion matrix and per-class bar chart."""
    loss, accuracy, mse, mae, mape, cos_sim = model.evaluate(
        [xrp, xgp, xbp, xbnp], Y_test
    )
    print(f"\nTest loss:          {loss:.4f}")
    print(f"Test accuracy:      {accuracy:.4f}")
    print(f"MSE:                {mse:.4f}")
    print(f"MAE:                {mae:.4f}")
    print(f"MAPE:               {mape:.4f}")
    print(f"Cosine similarity:  {cos_sim:.4f}")

    y_pred    = model.predict([xrp, xgp, xbp, xbnp])
    pred_labels = np.argmax(y_pred, axis=1).reshape(-1, 1)

    # Confusion matrix
    cm = confusion_matrix(Y_test, pred_labels)
    ConfusionMatrixDisplay(cm).plot()
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Per-class metrics
    fp  = (cm.sum(axis=0) - np.diag(cm)).astype(float)
    fn  = (cm.sum(axis=1) - np.diag(cm)).astype(float)
    tp  = np.diag(cm).astype(float)
    tn  = (cm.sum() - (fp + fn + tp)).astype(float)

    recall      = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision   = tp / (tp + fp)
    f1          = 2 * precision * recall / (precision + recall + 1e-8)
    f2          = fbeta_score(Y_test, pred_labels, beta=2, average='micro')

    print(f"\nMicro F1:  {f1_score(Y_test, pred_labels, average='micro'):.4f}")
    print(f"Micro F2:  {f2:.4f}")

    # Bar chart — per-class metrics
    sign_labels = LABELS[1:]  # Skip 'NoPrediccion'
    bw  = 0.10
    br1 = np.arange(len(sign_labels))

    plt.figure(figsize=(16, 10), dpi=150)
    for offset, values, color, label in [
        (0,    precision,   '#46185f', 'Precision'),
        (bw,   specificity, '#1B6B93', 'Specificity'),
        (2*bw, recall,      '#82CD47', 'Recall'),
        (3*bw, f1,          '#FCE22A', 'F1 score'),
    ]:
        plt.bar(br1 + offset, values, color=color, width=bw, edgecolor='grey', label=label)

    plt.xlabel('Sign', fontweight='bold', fontsize=14)
    plt.ylabel('Score', fontweight='bold', fontsize=14)
    plt.title('Per-class Performance Metrics', fontsize=16)
    plt.xticks(br1 + 1.5 * bw, sign_labels, rotation=45, ha='right', fontsize=12)
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    # Training history
    return pred_labels


def plot_history(history):
    """Plot accuracy and loss curves from training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['accuracy'], label='Train accuracy')
    axes[0].plot(history.history.get('val_accuracy', []), label='Val accuracy')
    axes[0].set_title('Accuracy per epoch')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    axes[1].plot(history.history['loss'], label='Train loss')
    axes[1].plot(history.history.get('val_loss', []), label='Val loss')
    axes[1].set_title('Loss per epoch')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # 1. Load dataset
    X, Y = load_dataset(IMAGE_DIR, LABEL_DIR, DIM)

    # 2. Train / test split
    X_train, X_test, Y_train, Y_test = balanced_train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    check_balance(Y_train, Y_test)

    # 3. Extract and normalize channels
    xrt, xgt, xbt, xbnt = extract_channels(X_train)
    xrp, xgp, xbp, xbnp = extract_channels(X_test)

    xrt,  xgt,  xbt,  xbnt  = xrt/255,  xgt/255,  xbt/255,  xbnt/255
    xrp,  xgp,  xbp,  xbnp  = xrp/255,  xgp/255,  xbp/255,  xbnp/255

    # 4. Build and compile model
    model = build_model(dim=DIM, n_classes=len(LABELS))
    model.summary()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'mse', 'mae', 'mape', 'cosine_similarity'],
    )

    # 5. Train
    steps_per_epoch  = len(Y_train) // BATCH_SIZE
    validation_steps = len(Y_test)  // BATCH_SIZE

    train_gen = data_generator(xrt, xgt, xbt, xbnt, Y_train, BATCH_SIZE)
    val_gen   = data_generator(xrp, xgp, xbp, xbnp, Y_test,  BATCH_SIZE)

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
    )

    # 6. Evaluate
    evaluate_model(model, xrp, xgp, xbp, xbnp, Y_test)
    plot_history(history)

    # 7. Save model
    model.save('lsm_cnn_model.h5')
    print("Model saved to lsm_cnn_model.h5")
