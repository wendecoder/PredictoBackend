import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

# Load labels from the CSV file
labels_df = pd.read_csv(r'D:\train.csv')  # Update with your file path

# Image dimensions for preprocessing
img_width, img_height = 224, 224

# Define the number of classes
num_classes = 5

# Load and preprocess images
X_train = []
Y_train = []

for index, row in labels_df.iterrows():
    img_path = os.path.join(r'D:\train_images', row['id_code'] + '.png')
    img = load_img(img_path, target_size=(img_width, img_height))
    img_array = img_to_array(img)
    X_train.append(img_array)
    Y_train.append(row['diagnosis'])

X_train = np.array(X_train)
Y_train = to_categorical(Y_train, num_classes=num_classes)  # Convert to categorical format

# Save the processed labels to a file
np.save('X_train.npy', X_train)
np.save('Y_train.npy', Y_train)

# Split data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# Create the model
def create_model(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Create the model
model = create_model(input_shape=(img_width, img_height, 3), num_classes=num_classes)

# Define data generators with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load and preprocess training data
train_generator = train_datagen.flow(
    X_train, Y_train,
    batch_size=32,
    shuffle=True
)

# Define callbacks
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // train_generator.batch_size,
    epochs=30,  # Adjust as needed
    validation_data=(X_val, Y_val),
    callbacks=[checkpoint]
)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, Y_val)
print(f'Validation accuracy: {val_accuracy:.2f}')

# Load the best model
best_model = create_model(input_shape=(img_width, img_height, 3), num_classes=num_classes)
best_model.load_weights('best_model.h5')

# Save the best model
best_model.save('final_model.h5')
print('Model saved as final_model.h5')
