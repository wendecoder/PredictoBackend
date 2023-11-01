import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.regularizers import l2  # Importing l2 regularization
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler
from keras.metrics import Metric

# # Connect to TPU
# tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()

# # Instantiate a distribution strategy
# tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# with tpu_strategy.scope():
# Load labels from the CSV file
labels_df = pd.read_csv(r'D:\train.csv')  # Update with your file path

# Image dimensions for preprocessing
img_width, img_height = 331, 331

# Define the number of classes
num_classes = 5

def parse_image(image_path, label):
    img = load_img(image_path, target_size=(img_width, img_height))
    img_array = img_to_array(img)

    # Convert the image array to 8-bit unsigned integer data type
    img_array = img_array.astype(np.uint8)

    # Apply image enhancement preprocessing using OpenCV
    img_array_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    img_array_eq = cv2.equalizeHist(img_array_gray)  # Apply histogram equalization
    img_array_eq_rgb = cv2.cvtColor(img_array_eq, cv2.COLOR_GRAY2RGB)  # Convert back to RGB

    # Ensure pixel values are within the expected range (0-255)
    img_array_eq_rgb = np.clip(img_array_eq_rgb, 0, 255)

    normalized = img_array_eq_rgb / 255.0

    return normalized, label

sample_size = 3662
selected_indices = np.random.choice(labels_df.index, size=sample_size, replace=False)

X_train = []
Y_train = []
i = 0

for index in selected_indices:
    row = labels_df.iloc[index]
    img_path = os.path.join(r'D:\train_images', row['id_code'] + '.png')
    img_normalized, label = parse_image(img_path, row['diagnosis'])

    X_train.append(img_normalized)
    Y_train.append(label)
    print("converted", i, "images")
    i += 1

X_train = np.array(X_train)
Y_train = to_categorical(Y_train, num_classes=num_classes)

# Save the processed labels to a file

np.save(r'D:\X_train.npy', X_train)
np.save(r'D:\Y_train.npy', Y_train)

# X_train = np.load('X_train.npy')
# Y_train = np.load('Y_train.npy')

# Split data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# Apply oversampling to balance classes
oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_train, Y_train = oversampler.fit_resample(X_train.reshape(-1, img_width * img_height * 3), Y_train)

# Create the model
def create_model(input_shape, num_classes, dropout_rate, l2_lambda):
    model = Sequential()

    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=input_shape, kernel_regularizer=l2(l2_lambda)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (5, 5), activation='relu', kernel_regularizer=l2(l2_lambda)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (5, 5), activation='relu', kernel_regularizer=l2(l2_lambda)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())

#     New Conv2D and MaxPooling2D layers
#     model.add(Conv2D(512, (5, 5), activation='relu', kernel_regularizer=l2(l2_lambda)))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Dropout(dropout_rate))
#     model.add(BatchNormalization())

#     model.add(Conv2D(1024, (5, 5), activation='relu', kernel_regularizer=l2(l2_lambda)))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Dropout(dropout_rate))
#     model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

# Hyperparameters for regularization and dropout
dropout_rate = 0.3  # Adjust as needed
l2_lambda = 0.001  # Adjust as needed

# Create the model
model = create_model(input_shape=(img_width, img_height, 3), num_classes=num_classes, dropout_rate=dropout_rate, l2_lambda=l2_lambda)


# Define learning rate scheduler
def lr_scheduler(epoch):
    lr = 1e-4 * (0.9 ** epoch)  # Starting from 1e-4, decrease by 10% every epoch
    return lr

def custom_preprocessing(image):
    # Apply the random saturation adjustment here
    modified = tf.image.random_saturation(image, 0, 2)
    return modified

# Define data generators with moderate data augmentation
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    preprocessing_function=custom_preprocessing
#     channel_shift_range=20,
#     brightness_range=[0.8, 1.2]
)

# Load and preprocess training data
train_generator = train_datagen.flow(
    X_train.reshape(-1, img_width, img_height, 3), Y_train,
    batch_size=32,
    shuffle=True
)

class F1ScoreMetric(Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1ScoreMetric, self).__init__(name=name, **kwargs)
        self.f1_score = self.add_weight(name='f1_score', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, y_pred.dtype)  # Cast y_true to the same dtype as y_pred
        tp = tf.reduce_sum(y_true * y_pred)
        total_true = tf.reduce_sum(y_true)
        total_pred = tf.reduce_sum(y_pred)
        epsilon = tf.keras.backend.epsilon()
        f1 = 2 * tp / (total_true + total_pred + epsilon)
        self.f1_score.assign_add(f1)

    def result(self):
        return self.f1_score

# Create an instance of the F1ScoreMetric
f1_metric = F1ScoreMetric()

# Compile the model with Adam optimizer and learning rate scheduler
optimizer = Adam(learning_rate=1e-4)  # Starting learning rate
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', f1_metric])

# Callbacks for model training
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_decay = LearningRateScheduler(lr_scheduler)

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // train_generator.batch_size,
    epochs=30,
    validation_data=(X_val, Y_val),
    callbacks=[lr_decay, checkpoint]
)

# Load the best model
best_model = create_model(input_shape=(img_width, img_height, 3), num_classes=num_classes, dropout_rate=dropout_rate, l2_lambda=l2_lambda)
best_model.load_weights('best_model.h5')

# Save the best model based on validation F1 score
best_model.save('best_f1_model.h5')
print('Best F1 score model saved as best_f1_model.h5')