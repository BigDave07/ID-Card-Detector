import tensorflow as tf
from tensorflow.keras import layers


# Access dataset
train_dir = r"C:\Users\dell\OneDrive\Desktop\Marathon\Day 1\dataset\Train"
test_dir = r"C:\Users\dell\OneDrive\Desktop\Marathon\Day 1\dataset\Test"

# Set image and batch size
Img_size = (224,224)  #Resize all images to this pixel
Batch_size = 32

# Data Augumentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomZoom(0.1),
    layers.RandomRotation(0.1)
])

# Load Training
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split = 0.2,
    subset = 'training',
    seed = 123,
    image_size = Img_size,
    batch_size = Batch_size,
    shuffle = True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split = 0.2,
    subset = "validation",
    seed = 123,
    image_size = Img_size,
    batch_size = Batch_size,
    shuffle = True
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size = Img_size,
    batch_size = Batch_size,
    shuffle = False
)

# Normalization
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x,y: (normalization_layer(x), y))

# Improve Performance with Caching and Prefetching
train_ds = train_ds.cache().prefetch(buffer_size = tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

# Use Transfer Learning (Pretrained Model)
base_model = tf.keras.applications.MobileNetV2(
    input_shape = (224,224,3),
    include_top = False,
    weights = 'imagenet'
)

base_model.trainable=  False  # Freeze model for now

# Build Classifier
model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])

# Compile and Train 
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit(train_ds, validation_data =val_ds , epochs = 10)

model.save('Id Dectector.h5')