import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import json
from dataset_loader import load_data, generate_set

with open("settings.json", "r", encoding='utf-8') as f:
    data = json.loads(f.read())

train_data, train_labels, test_data, test_labels = load_data(data["dataset_dir"])
train_data, train_labels = generate_set(train_data, train_labels, data["train_set_length"])
test_data, test_labels = generate_set(test_data, test_labels, data["test_set_length"])

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 64, 6)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2))
model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_data, train_labels, epochs=10,
                    validation_data=(test_data, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)