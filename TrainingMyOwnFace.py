import keras.layers
import keras.utils
import keras.optimizers
import keras.losses
import keras
import keras_vggface


dataset = keras.utils.image_dataset_from_directory(directory='./training/personA', shuffle=True, batch_size=8, image_size=(224, 224), labels=None)

dataAugmentation = keras.Sequential([keras.layers.RandomFlip('horizontal'), keras.layers.RandomRotation(0.2)],)

teste = keras_vggface.VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))


nb_class = 2
teste.trainable = False
last_layer = teste.get_layer('avg_pool').output

inputs = keras.Input(shape=(224, 224, 3))
x = dataAugmentation(inputs)
x = teste(x)
x = keras.layers.Flatten(name='flatten')(x)
out = keras.layers.Dense(nb_class, name='classifier')(x)

custom = keras.Model(inputs, out)


baseLearn = 0.0001

custom.compile(optimizer=keras.optimizers.Adam(learning_rate=baseLearn), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])


history = custom.fit(dataset, epochs=20)