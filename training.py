import os
import shutil
import zipfile

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

root = 'emotions/'
local_zip = 'KDEF_and_AKDEF.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall()
zip_ref.close()

files = []
dir = []
basepath = 'KDEF_and_AKDEF/KDEF'
for entry in os.listdir(basepath):
    sub = []
    dir.append(basepath + '/' + entry + '/')
    for file in os.listdir(basepath + '/' + entry):
        sub.append(file)
    files.append(sub)

emotion_type = ['AF', 'AN', 'DI', 'HA', 'NE', 'SA', 'SU']

for sublist in range(len(files)):
    for file in files[sublist]:
        for i in range(len(emotion_type)):
            if file[4:6] == emotion_type[i]:
                try:
                    os.makedirs(root + emotion_type[i], exist_ok=True)
                except OSError as error:
                    print(error)
                # print(dir[sublist]+file)
                shutil.copy(dir[sublist] + file, root + emotion_type[i])
                break

"""basepath2 = '/content/Thesis_Emodet/KDEF_and_AKDEF/AKDEF'
files_AKDEF = []
for entry in os.listdir(basepath2):
    files_AKDEF.append(entry)
for file in files_AKDEF:
    for i in range (len(emotion_type)):
        if file[1:3] == emotion_type[i]:
            try:
                os.makedirs(root+emotion_type[i], exist_ok=True)
            except OSError as error:
                print(error)
                #print(root+emotion_type[i]+'/'+file)
            shutil.copy(basepath2+'/'+file, root+emotion_type[i])
            break"""

checkpoint_path = 'test cp/cp.ckpt'
try:
    os.makedirs(checkpoint_path, exist_ok=True)
except OSError as error:
    print(error)
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period=3
                                                 )

EarlyStoppingCallback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)


# Call_back function for stopping training after a certain level
class MyCallback(tf.keras.callbacks.Callback):  # Your Code
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc') > .95):
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True


AccuracyLimiterCallback = MyCallback()

# For data split into test & validation
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True,
                                   rotation_range=30,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   fill_mode='nearest',
                                   validation_split=0.15)  # set validation split

dataset_devidation_train = 'dataset_devidation_train/'
dataset_devidation_validation = 'dataset_devidation_validation/'
try:
    os.makedirs(dataset_devidation_train, exist_ok=True)
    os.makedirs(dataset_devidation_validation, exist_ok=True)
except OSError as error:
    print(error)

batch_size = 10
img_height = 400
img_width = 300
train_data_dir = 'emotions/'  # source directory for training images
train_generator = train_datagen.flow_from_directory(
    directory=train_data_dir,
    color_mode='grayscale',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    #save_to_dir=dataset_devidation_train,
    # save_prefix='',
    # save_format='jpeg',
    subset='training')  # training data

validation_generator = train_datagen.flow_from_directory(
    directory=train_data_dir,  # same directory as training data
    color_mode='grayscale',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    #save_to_dir=dataset_devidation_validation,
    #save_prefix='',
    #save_format='jpeg',
    subset='validation')  # validation data

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(400, 300, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fifth convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.4),
    # 1024 neuron hidden layer
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=250,
    callbacks=[cp_callback, AccuracyLimiterCallback],
    verbose=1)

model.save('model_with_gray_image.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))  # Get number of epochs

# ------------------------------------------------
# Plot training and validation accuracy per epoch
# ------------------------------------------------
try:
    os.makedirs('Thesis_Fig', exist_ok=True)
except OSError as error:
    print(error)

plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

plt.savefig('Thesis_Fig/Training and validation accuracy.png')
plt.savefig('Thesis_Fig/Training and validation accuracy.pdf')
# ------------------------------------------------
# Plot training and validation loss per epoch
# ------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.title('Training and validation Loss')
plt.figure()
plt.savefig('Thesis_Fig/Training and validation Loss.png')
plt.savefig('Thesis_Fig/Training and validation Loss.pdf')
