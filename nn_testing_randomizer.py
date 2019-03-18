import tensorflow as tf
import numpy as np
import input_randomizer as ir

mnist = tf.keras.datasets.mnist.load_data()

#create dictionary of numbers with position of their pixelated sample
noToPlace = {}
for no in range(10):
    for counter, label in enumerate(mnist[0][1]):
        if label == no:
            noToPlace[no] = counter
            break
#create x_train with samples of numbers - list of images from 0 to 9
x_train = np.array([mnist[0][0][noToPlace[i]] for i in range(10)])
#labels
y_train = np.array([i for i in range(10)])
#testing cases
x_test = mnist[0][0]
y_test = mnist[0][1]

#create randomized input
input_set = ir.input_randomizer(x_train, y_train)
input_set.set_init()
input_set.image_horizontal_randomizer()
input_set.image_vertical_randomizer()
input_set.general_randomizer(ir.input_randomizer.image_horizontal_randomizer,ir.input_randomizer.image_vertical_randomizer)
input_set.image_pixel_shuffling()
input_set.set_normalizer()
x_set, y_set = input_set.randomized_set()

#NN model to test if randomized input will be successfull at training NN to recognize real numbers
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(3000, activation=tf.nn.relu))#, input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(2000, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))
model.compile(optimizer='adadelta', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_set/255, y_set, epochs=10)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

