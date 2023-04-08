import os
import random
import shutil
import tensorflow as tf

# Define the image and annotation directories
image_dir = 'D:/Downloads/archive/images'
annot_dir = 'D:/Downloads/archive/annotations'

# Define the directories for the training and validation sets
train_image_dir = 'D:/Downloads/archive/train/images'
train_annot_dir = 'D:/Downloads/archive/train/annotations'
val_image_dir = 'D:/Downloads/archive/val/images'
val_annot_dir = 'D:/Downloads/archive/val/annotations'

# Define the validation set percentage
val_percent = 0.3

# Get all the subdirectories (dog breeds)
subdirs = [subdir for subdir in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, subdir))]

# Loop through each dog breed
for subdir in subdirs:
    print(f"Processing {subdir}...")

    # Create the corresponding directories for the current dog breed in the training and validation sets
    os.makedirs(os.path.join(train_image_dir, subdir), exist_ok=True)
    os.makedirs(os.path.join(train_annot_dir, subdir), exist_ok=True)
    os.makedirs(os.path.join(val_image_dir, subdir), exist_ok=True)
    os.makedirs(os.path.join(val_annot_dir, subdir), exist_ok=True)

    # Get all the image and annotation files for the current dog breed
    image_files = [os.path.join(image_dir, subdir, filename) for filename in
                   os.listdir(os.path.join(image_dir, subdir))]
    annot_files = [os.path.join(annot_dir, subdir, filename) for filename in
                   os.listdir(os.path.join(annot_dir, subdir))]

    # Get the total number of images and annotations
    num_files = len(image_files)
    assert num_files == len(
        annot_files), f"Number of image files ({num_files}) does not match number of annotation files ({len(annot_files)}) for {subdir}"

    # Shuffle the image and annotation files
    random.seed(42)  # for reproducibility
    random.shuffle(image_files)
    random.shuffle(annot_files)

    # Split the files into training and validation sets
    val_size = int(num_files * val_percent)
    train_image_files = image_files[val_size:]
    train_annot_files = annot_files[val_size:]
    val_image_files = image_files[:val_size]
    val_annot_files = annot_files[:val_size]

    # Copy the training and validation image and annotation files to their corresponding directories
    for src_image, src_annot in zip(train_image_files, train_annot_files):
        dst_image = os.path.join(train_image_dir, subdir, os.path.basename(src_image))
        dst_annot = os.path.join(train_annot_dir, subdir, os.path.basename(src_annot))
        shutil.copy(src_image, dst_image)
        shutil.copy(src_annot, dst_annot)

    for src_image, src_annot in zip(val_image_files, val_annot_files):
        dst_image = os.path.join(val_image_dir, subdir, os.path.basename(src_image))
        dst_annot = os.path.join(val_annot_dir, subdir, os.path.basename(src_annot))
        shutil.copy(src_image, dst_image)
        shutil.copy(src_annot, dst_annot)


# Get all the subdirectories in the image directory (dog breeds)
train_subdirs = [os.path.join(train_image_dir, subdir) for subdir in os.listdir(train_image_dir) if
                 os.path.isdir(os.path.join(train_image_dir, subdir))]
val_subdirs = [os.path.join(val_image_dir, subdir) for subdir in os.listdir(val_image_dir) if
               os.path.isdir(os.path.join(val_image_dir, subdir))]

# Create a dataset of image filenames and their corresponding labels
train_image_files = []
train_labels = []
val_image_files = []
val_labels = []

for i, subdir in enumerate(train_subdirs):
    breed_images = tf.data.Dataset.list_files(os.path.join(subdir, '*.jpg'))
    for filename in breed_images:
        train_image_files.append(filename)
        train_labels.append(i)

for i, subdir in enumerate(val_subdirs):
    breed_images = tf.data.Dataset.list_files(os.path.join(subdir, '*.jpg'))
    for filename in breed_images:
        val_image_files.append(filename)
        val_labels.append(i)

# print(train_image_files)

train_dataset = tf.data.Dataset.from_tensor_slices((train_image_files, train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val_image_files, val_labels))


# Define a function to load the image and its corresponding annotation
def load_image_and_label(image_filename, label):
    image = tf.io.read_file(image_filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, size=[256, 256])
    return image, label


# Map the function to each element in the dataset
train_dataset = train_dataset.map(load_image_and_label)
val_dataset = val_dataset.map(load_image_and_label)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(32, (3, 3), 1, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(16, (3, 3), 1, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(len(train_subdirs), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

batch_size = 32
train_dataset = train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

# Train the model
epochs = 20
history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)

# Print final accuracy and loss
train_loss, train_acc = model.evaluate(train_dataset, verbose=2)
val_loss, val_acc = model.evaluate(val_dataset, verbose=2)
print('Training loss:', train_loss)
print('Training accuracy:', train_acc)
print('Validation loss:', val_loss)
print('Validation accuracy:', val_acc)
