### Model 1 python file for training and, takes processed data
from src.dependencies import *
import src.data as data

## demonstrate structure of the data
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

MODEL_ID = 5
if MODEL_ID == 0: MODEL_ID = int(input('Model id: '))

(train_imgs, train_labels), (test_imgs, test_labels) = train_data, test_data = data.load_fashion_MNIST(MODEL_ID)

def visualise_data(predictions = []):
    plt.figure(figsize=(10,10))
    for i in range(10):
        plt.subplot(5,5,2*i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        j = np.random.randint(0,train_imgs.shape[0] - 10)
        if len(predictions) > 0:
            plt.xlabel('Model: '+class_names[train_labels[i+j]]+' ('+class_names[predictions[i+j]]+')')

        else:
            plt.xlabel(class_names[train_labels[i+j]])
        plt.imshow(train_imgs[i+j])
    plt.show()

### QUESTIONS
# metrics keyword
# predict method
# evaluate method
# Onehot encoding
# Import layers
# From logits without softmax == not from logits with?


class NET():
    # Define model architectures

    def Make_Model(i=1):
        model = None
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, LeakyReLU
        if i == 1:
            model = tf.keras.Sequential([
                Flatten(input_shape=(28, 28)),
                Dense(128, activation='relu'),
                Dense(10)
            ])
            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )

        elif i == 2:
            model = tf.keras.Sequential([
                Conv2D(16, input_shape=(28, 28, 1), kernel_size=(3, 3), padding='same'),
                LeakyReLU(0.1),
                MaxPooling2D(pool_size=(2, 2), padding='valid'),
                Conv2D(32, kernel_size=(3, 3), padding='same'),
                LeakyReLU(0.1),
                MaxPooling2D(pool_size=(2, 2), padding='valid'),
                Flatten(),
                Dense(18),
                Dropout(0.15),
                LeakyReLU(0.1),
                Dense(10),
                LeakyReLU(0.1),
                Activation('softmax')
            ])
            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                metrics=['accuracy']
            )


        elif i == 3:

            model = tf.keras.Sequential([
                hub.KerasLayer('https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5',
                               trainable=False),
                Dense(10, activation='softmax')
            ])
            model.build([None, 224, 224, 3])
            print(model.summary())


        elif i == 4:
            vgg_model = tf.keras.applications.VGG19(
                include_top=False,
                weights='imagenet',
                input_tensor=None,
                input_shape=(224, 224, 3),
                pooling=None,
                classes=1000,
                classifier_activation='softmax'
            )
            print(vgg_model.summary())
            # Copy vgg_model architechture to sequential model
            model = tf.keras.models.Sequential()
            for layer in vgg_model.layers:
                print(layer)
                model.add(layer)

            # Add final layers for classification
            [model.add(layer) for layer in [
                Dense(600),
                LeakyReLU(0.1),
                Dense(200),
                LeakyReLU(0.1),
                Dense(10),
                LeakyReLU(0.1),
                Activation('softmax')
            ]]


            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                metrics=['accuracy']
            )
            model.build((None, 224, 224, 3))

        elif i == 5:
            autoencoder_id=1
            code_length=10
            encoder = tf.keras.models.load_model(f'../Models/Autoencoders/model{autoencoder_id}_size{code_length}/saved_encoder',
                                                 compile=False)
            model = tf.keras.models.Sequential(encoder)
            # Freeze loaded models weights for transfer learning
            for layer in model.layers: layer.trainable = False

            [model.add(layer) for layer in [
                Dense(10),
                LeakyReLU(0.1),
                Activation('softmax')
            ]]
            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                metrics=['accuracy']
            )

        print('Model successfully built...')
        return model

    def __init__(self, MODEL_ID):
        self.MODEL_ID = MODEL_ID
        self.model = NET.Make_Model(i=MODEL_ID)
        self.Load_Data()

        self.last_finished_epoch = 0
        self.save_dir = '../Models/FashionClassifier/model{}'.format(str(MODEL_ID))
        cp_path = self.save_dir+'/Checkpoints/cp.ckpt'
        self.cp_path = cp_path
        self.cp_dir = os.path.dirname(cp_path)
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cp_path,
                                                              save_weights_only=True)
        print('Set up checkpoint path for:', self.cp_dir)

        print('Network(model id {}) successfully built'.format(MODEL_ID))

    def Load_Data(self):
        dataset = data.load_fashion_MNIST(self.MODEL_ID)
        (self.train_imgs, self.train_labels), (self.test_imgs, self.test_labels) = self.train_data, self.test_data = dataset
        self.N_train = int(self.train_imgs.shape[0])
        return dataset

    def Train(self):

        hist = self.model.fit(train_imgs, train_labels,
                       epochs=30,
                       validation_data=(self.test_imgs, self.test_labels),
                       initial_epoch=self.last_finished_epoch,
                       callbacks=[self.cp_callback])
        print(self.model.summary())

        test_loss, test_acc = self.model.evaluate(test_imgs, test_labels, verbose=2)
        print('\nTest accuracy acheived:', test_acc)
        self.model.save(self.save_dir+'/saved_model')


        predictions = np.argmax(self.model.predict(train_imgs), axis=1)

        if self.MODEL_ID == 1: visualise_data(predictions)

    def Reload_Weights(self):
        latest_cp = tf.train.latest_checkpoint(checkpoint_dir=self.cp_dir)
        if latest_cp == None: print('No saved weights found, using random weights...')
        else:
            print('Latest saved weights found in:', latest_cp)
            self.model.load_weights(latest_cp)


net = NET(MODEL_ID=MODEL_ID)

# net.Reload_Weights()
net.Train()


#saved_model_loaded = tf.keras.models.load_model('../Models/FashionClassifier/model2/saved_model')
#saved_model_loaded.summary()