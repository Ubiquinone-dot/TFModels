from src.dependencies import *


class NET:
    # Contains:
    # Checkpoint generation during construction
    # .Train() for training model on dataset
    # .Load_Data() for storing and returning dataset
    # Reload_Weights() for reloading checkpointed weights


    def __init__(self, MODEL_ID=1, MODEL_CLASS='Autoencoders', code_length=10):
        # Empty and assigned attributes
        self.input_shape = []
        self.N_train = 0
        self.N_test = 0
        self.last_finished_epoch = 0
        self.MODEL_ID, self.MODEL_CLASS, self.code_length = MODEL_ID, MODEL_CLASS, code_length

        self.Create_Checkpoints()
        self.Load_Data()
        self.model, self.encoder, self.decoder = NET.Make_Autoencoder(code_length, INPUT_SHAPE=self.input_shape)

        print(self.model.summary())
        print(f'Network (model id {MODEL_ID}, of class {MODEL_CLASS}) successfully built.\n\n')


    # Define model architectures in Make_Model
    @staticmethod  # it should be possible to make the autoencoder without an instance of the network
    def Make_Autoencoder(CODE_LENGTH, INPUT_SHAPE):
        from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, \
            LeakyReLU, Reshape, Input
        def Make_Coders(CODE_LENGTH, INPUT_SHAPE):
            encoder = tf.keras.models.Sequential([
                InputLayer(INPUT_SHAPE),
                Flatten(),
                Dense(CODE_LENGTH)
            ])
            decoder = tf.keras.models.Sequential([
                InputLayer((CODE_LENGTH,)),
                Dense(np.prod(INPUT_SHAPE)),
                Reshape(INPUT_SHAPE)
            ])
            return encoder, decoder

        encoder, decoder = Make_Coders(CODE_LENGTH, INPUT_SHAPE)
        inp = Input(shape=INPUT_SHAPE)
        code = encoder(inp)
        reconstruction = decoder(code)
        autoencoder = tf.keras.models.Model(inputs=inp, outputs=reconstruction)
        autoencoder.compile(
            loss='mse',
            optimizer='adamax'
        )
        print(autoencoder.summary())
        return autoencoder, encoder, decoder


    def Create_Checkpoints(self):
        self.last_finished_epoch = 0
        self.save_dir = f'../Models/{self.MODEL_CLASS}/model{self.MODEL_ID}_size{self.code_length}'
        cp_path = self.save_dir + '/Checkpoints/cp.ckpt'
        self.cp_path = cp_path
        self.cp_dir = os.path.dirname(cp_path)
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cp_path,
                                                              save_weights_only=True)
        print('Set up checkpoint path for:', self.cp_dir)


    def Load_Data(self):
        dataset = tf.keras.datasets.fashion_mnist.load_data()
        (self.train_x, self.train_labels), (self.test_x, self.test_labels) = self.train_data, self.test_data = dataset
        self.train_x = self.train_x / 255
        self.test_x = self.test_x / 255

        self.N_train = int(self.train_x.shape[0])
        self.input_shape = self.train_x.shape[1:]
        return dataset

    def Draw_Histogram(self):
        pass

    def Train(self):

        hist = self.model.fit(
            x=self.train_x,
            y=self.train_x,
            epochs=15,
            validation_data=(self.test_x, self.test_x),
            initial_epoch=self.last_finished_epoch,
            callbacks=[self.cp_callback]
        )

        self.model.save(self.save_dir+'/saved_model')
        self.encoder.save(self.save_dir+'/saved_encoder')


    def Reload_Weights(self):   # Returns bool True if weights reloaded
        latest_cp = tf.train.latest_checkpoint(checkpoint_dir=self.cp_dir)
        if latest_cp == None:
            print('No saved weights found, using initialized weights...')
            return False
        else:
            print('Latest saved weights found in:', latest_cp)
            self.model.load_weights(latest_cp)
            return True

    def Show_Reconstruction(self, i):
        img = self.train_x[i]
        code = self.encoder.predict(np.expand_dims(img,axis=0))
        reco = self.decoder.predict(code)[0]

        def cleanfig():
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)

        plt.figure(figsize=(10, 10))
        cleanfig()
        # Add the first image (true)
        plt.subplot(1, 3, 1)
        plt.title('Original')
        cleanfig()
        plt.imshow(img)

        # Add code
        plt.subplot(1,3,2)
        plt.title('Code')
        cleanfig()
        plt.imshow(code.reshape(code.shape[-1]//2, -1))

        # Add reconstruction
        plt.subplot(1, 3, 3)
        plt.title('reconstruction')
        cleanfig()
        plt.imshow(reco)

        plt.show()


nn = NET(code_length=10)
#if not nn.Reload_Weights(): nn.Train()
nn.Train()
for i in range(10, 20):
    continue
    # nn.Show_Reconstruction(i)


