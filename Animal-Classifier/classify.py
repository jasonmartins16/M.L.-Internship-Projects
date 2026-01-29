import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

#Load Data into Split train into train/val
train_data = ImageDataGenerator(rescale = 1./255, validation_split = 0.1)

train_generator = train_data.flow_from_directory(
    'C:\\Animal Classifier\\dataset\\train',
    target_size = (224, 224),
    batch_size = 32,
    class_mode = 'categorical',
    subset = 'training')

val_generator = train_data.flow_from_directory(
    'C:\\Animal Classifier\dataset\\train',
    target_size=(224, 224),
    batch_size = 32,
    class_mode = 'categorical',
    subset = 'validation')

test_data = ImageDataGenerator(rescale = 1./255)

test_generator = test_data.flow_from_directory(
    'C:\\Animal Classifier\\dataset\\test',
    target_size = (224, 224),
    batch_size = 32,
    class_mode = 'categorical',
    shuffle = False)

#Build transfer learning model

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation = 'relu')(x)
outputs = Dense(train_generator.num_classes, activation = 'softmax')(x)

model = Model(inputs = base_model.input, outputs = outputs)

#Compile and Train
optimizer = Adam(learning_rate=0.001)

model.compile(optimizer = optimizer, 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])

model.fit(train_generator, epochs = 10, validation_data = val_generator)
model.save('MobileNetV2.h5')
print("Model saved as .h5")

#Testing
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.2f}")
