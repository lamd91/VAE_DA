import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import math
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

# Define global constants
BATCH_SIZE = 32
LATENT_DIM = 2
EPOCHS = 100

def map_image(image):
    '''returns a reshaped tensor from a given image'''
    image = tf.cast(image, dtype=tf.float32)
    image = tf.reshape(image, shape=(50, 500, 1,))

    return image, image

def get_datasets(map_fn, test_size):
    """Loads and prepares the dataset from a 2D array loaded from a text file."""
    dataset = np.transpose(np.loadtxt('data/iniMPSimEns_1000.txt'))
    num_examples = dataset.shape[0]
    train_dataset = tf.data.Dataset.from_tensor_slices(dataset[0:int(num_examples*(1-test_size))])
    val_dataset = tf.data.Dataset.from_tensor_slices(dataset[int(num_examples*(1-test_size)):])

    train_dataset = train_dataset.map(map_fn)
    val_dataset = val_dataset.map(map_fn)

    train_dataset = train_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.shuffle(1024).batch(BATCH_SIZE)

    return train_dataset, val_dataset, num_examples

def display_three_train_images(train_dataset):
    """Display 3 images from the training dataset"""
    plt.figure(figsize=(10, 14))
    for input_images, images in train_dataset.take(1):
        for i in range(3):
            plt.subplot(3, 1, i+1)
            plt.imshow(np.squeeze(images[i].numpy()).astype('uint8'), cmap='gray')
    plt.show()


# Load and prepare image dataset for training
train_dataset, val_dataset, num_examples = get_datasets(map_image, test_size=0)
print(f"Num of examples: {num_examples}")
display_three_train_images(train_dataset)

# Get image dimensions
for input_images, images in train_dataset.take(1):
    batch_size, height, width = images.shape[0], images.shape[1], images.shape[2]

# Define decision variables for adding Cropping2D layers in decoder layers
topcrop_after_upsampling1 = (round(height/2) % 2 != 0)
leftcrop_after_upsampling1 = (round(width/2) % 2 != 0)
topcrop_after_upsampling2 = (height % 2 != 0)
leftcrop_after_upsampling2 = (width % 2 != 0)


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        """Generates a random sample and combines with the encoder output

        Args:
          inputs -- output tensor from the encoder

        Returns:
          `inputs` tensors combined with a random sample
        """

        # unpack the output of the encoder
        mu, sigma = inputs

        # get the size and dimensions of the batch
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]

        # generate a random tensor
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        # combine the inputs and noise
        return mu + tf.exp(0.5 * sigma) * epsilon

def encoder_layers(inputs, latent_dim):
    """Defines the encoder's layers.
    Args:
      inputs -- batch from the dataset
      latent_dim -- dimensionality of the latent space

    Returns:
      mu -- learned mean
      sigma -- learned standard deviation
      batch_2.shape -- shape of the features before flattening
    """

    # add the Conv2D layers followed by BatchNormalization
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same", activation='relu',
                               name="encode_conv1")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu',
                               name="encode_conv2")(x)

    # assign to a different variable so you can extract the shape later
    batch_2 = tf.keras.layers.BatchNormalization()(x)

    # flatten the features and feed into the Dense network
    x = tf.keras.layers.Flatten(name="encode_flatten")(batch_2)

    # we arbitrarily used 20 units here but feel free to change
    x = tf.keras.layers.Dense(20, activation='relu', name="encode_dense")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # add output Dense networks for mu and sigma, units equal to the declared latent_dim.
    mu = tf.keras.layers.Dense(latent_dim, name='latent_mu')(x)
    sigma = tf.keras.layers.Dense(latent_dim, name='latent_sigma')(x)

    return mu, sigma, batch_2.shape

def encoder_model(latent_dim, input_shape):
    """Defines the encoder model with the Sampling layer
    Args:
      latent_dim -- dimensionality of the latent space
      input_shape -- shape of the dataset batch

    Returns:
      model -- the encoder model
      conv_shape -- shape of the features before flattening
    """

    # declare the inputs tensor with the given shape
    inputs = tf.keras.layers.Input(shape=input_shape)

    # get the output of the encoder_layers() function
    mu, sigma, conv_shape = encoder_layers(inputs, latent_dim=LATENT_DIM)

    # feed mu and sigma to the Sampling layer
    z = Sampling()((mu, sigma))

    # build the whole encoder model
    model = tf.keras.Model(inputs, outputs=[mu, sigma, z])

    return model, conv_shape

def decoder_layers(inputs, conv_shape, topcrop_after_upsampling1, leftcrop_after_upsampling1,
                   topcrop_after_upsampling2, leftcrop_after_upsampling2):
    """Defines the decoder layers.
    Args:
      inputs -- output of the encoder
      conv_shape -- shape of the features before flattening

    Returns:
      tensor containing the decoded output
    """

    # feed to a Dense network with units computed from the conv_shape dimensions
    units = conv_shape[1] * conv_shape[2] * conv_shape[3]
    x = tf.keras.layers.Dense(units, activation='relu', name="decode_dense1")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    # reshape output using the conv_shape dimensions
    x = tf.keras.layers.Reshape((conv_shape[1], conv_shape[2], conv_shape[3]), name="decode_reshape")(x)

    # upsample the features back to the original dimensions
    # for that, make sure to add Cropping2D layers after upsampling when needed
    x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu',
                                        name="decode_conv2d_2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if topcrop_after_upsampling1:
        x = tf.keras.layers.Cropping2D(cropping=((1, 0), (0, 0)))(x)
    if leftcrop_after_upsampling1:
        x = tf.keras.layers.Cropping2D(cropping=((0, 0), (1, 0)))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu',
                                        name="decode_conv2d_3")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if topcrop_after_upsampling2:
        x = tf.keras.layers.Cropping2D(cropping=((1, 0), (0, 0)))(x)
    if leftcrop_after_upsampling2:
        x = tf.keras.layers.Cropping2D(cropping=((0, 0), (1, 0)))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same', activation='sigmoid',
                                        name="decode_final")(x)

    return x

def decoder_model(latent_dim, conv_shape):
    """Defines the decoder model.
    Args:
      latent_dim -- dimensionality of the latent space
      conv_shape -- shape of the features before flattening

    Returns:
      model -- the decoder model
    """

    # set the inputs to the shape of the latent space
    inputs = tf.keras.layers.Input(shape=(latent_dim,))

    # get the output of the decoder layers
    outputs = decoder_layers(inputs, conv_shape, topcrop_after_upsampling1, leftcrop_after_upsampling1,
                   topcrop_after_upsampling2, leftcrop_after_upsampling2)

    # declare the inputs and outputs of the model
    model = tf.keras.Model(inputs, outputs)

    return model

def kl_reconstruction_loss(inputs, outputs, mu, sigma):
    """ Computes the Kullback-Leibler Divergence (KLD)
    Args:
      inputs -- batch from the dataset
      outputs -- output of the Sampling layer
      mu -- mean
      sigma -- standard deviation

    Returns:
      KLD loss
    """
    kl_loss = 1 + sigma - tf.square(mu) - tf.math.exp(sigma)
    kl_loss = tf.reduce_mean(kl_loss) * -0.5

    return kl_loss

def vae_model(encoder, decoder, input_shape):
    """Defines the VAE model
    Args:
      encoder -- the encoder model
      decoder -- the decoder model
      input_shape -- shape of the dataset batch

    Returns:
      the complete VAE model
    """
    # set the inputs
    inputs = tf.keras.layers.Input(shape=input_shape)

    # get mu, sigma, and z from the encoder output
    mu, sigma, z = encoder(inputs)

    # get reconstructed output from the decoder
    reconstructed = decoder(z)

    # define the inputs and outputs of the VAE
    model = tf.keras.Model(inputs=inputs, outputs=reconstructed)

    # add the KL loss
    loss = kl_reconstruction_loss(inputs, z, mu, sigma)
    model.add_loss(loss)

    return model

def get_models(input_shape, latent_dim):
    """Returns the encoder, decoder, and vae models"""
    encoder, conv_shape = encoder_model(latent_dim=latent_dim, input_shape=input_shape)
    decoder = decoder_model(latent_dim=latent_dim, conv_shape=conv_shape)
    vae = vae_model(encoder, decoder, input_shape=input_shape)
    return encoder, decoder, vae

# Define a VAE class via model subclassing
loss_metrics = tf.keras.metrics.Mean()

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, variational_autoencoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vae = variational_autoencoder

    # override train_step method
    def train_step(self, images):
        if isinstance(images, tuple):
            input_images = images[0]
        with tf.GradientTape() as tape:
            # feed a batch to the VAE model
            reconstructed = self.vae(input_images)
            # compute reconstruction loss
            flattened_inputs = tf.reshape(input_images, [-1])
            flattened_outputs = tf.reshape(reconstructed, [-1])
            loss = self.compiled_loss(flattened_inputs, flattened_outputs) \
                   * input_images.shape[1] * input_images.shape[2]
            # add KLD regularization loss
            loss += sum(self.vae.losses)

        # compute the gradients and update the model weights
        grads = tape.gradient(loss, self.vae.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.vae.trainable_weights))

        # update metrics
        loss_metrics.update_state(loss)

        # return a dict mapping metrics names to current value
        return {'loss': loss_metrics.result()}

    # override test_step method
    def test_step(self, images):
        if isinstance(images, tuple):
            input_images = images[0]
        # compute predictions
        reconstructed = self.vae(input_images)
        # compute loss
        flattened_inputs = tf.reshape(input_images, [-1])
        flattened_outputs = tf.reshape(reconstructed, [-1])
        loss = self.compiled_loss(flattened_inputs, flattened_outputs) \
               * input_images.shape[1] * input_images.shape[2]
        # add KLD regularization loss
        loss += sum(self.vae.losses)
        # update metrics
        loss_metrics.update_state(loss)
        return {'loss': loss_metrics.result()}

def generate_and_save_images(model, epoch, step, test_input):
    """Helper function to plot our 8 images

    Args:

    model -- the decoder model
    epoch -- current epoch number during training
    step -- current step number during training
    test_input -- random tensor with shape (8, LATENT_DIM)
    """

    # generate images from the test input
    predictions = model.predict(test_input)

    # plot the results
    fig = plt.figure(figsize=(12, 14))

    for i in range(predictions.shape[0]):
        plt.subplot(8, 1, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    fig.suptitle("epoch: {}, step: {}".format(epoch, step))
    plt.savefig('image_at_epoch_{:04d}_step{:04d}.png'.format(epoch, step))
    #plt.show()

# Create a callback that saves the model's weights every few epochs during training
checkpoint_path = 'checkpoint/cp-cp{epoch:04d}.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_path,
    verbose = 1,
    save_weights_only = True,
    save_freq = math.ceil(num_examples/BATCH_SIZE) * 10
)

# Create custom callback to display outputs (via helper function) at the end of each epoch of training
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        # Generate random vector as test input to the decoder
        random_vector_for_generation = tf.random.normal(shape=[8, LATENT_DIM])
        # Generate and save images
        generate_and_save_images(decoder, epoch, math.ceil(num_examples/BATCH_SIZE), random_vector_for_generation)
        print('End of epoch {} - mean loss = {}'.format(epoch, logs[keys[0]]))

# Get the encoder, decoder and 'master' model (called vae)
encoder, decoder, var_autoencoder = get_models(input_shape=(50, 500, 1,), latent_dim=LATENT_DIM)

## Instantiate VAE class
#vae = VAE(encoder, decoder, var_autoencoder)
#
# # Compile model
# vae.compile(
#     optimizer = tf.keras.optimizers.Adam(),
#     loss = tf.keras.losses.BinaryCrossentropy()
# )
#
# # Generate random vector as test input to the decoder
# random_vector_for_generation = tf.random.normal(shape=[8, LATENT_DIM])
#
# # Initialize the helper function to display outputs from an untrained model
# generate_and_save_images(decoder, 0, 0, random_vector_for_generation)
#
# # Training loop
# vae.fit(train_dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=[cp_callback, CustomCallback()])

# Create new instance of VAE class
new_vae = VAE(encoder, decoder, var_autoencoder)

# Load weights from last checkpoint
checkpoint_dir = 'checkpoint'
latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)
new_vae.load_weights(latest)
new_vae.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss = tf.keras.losses.BinaryCrossentropy()
)
new_vae.evaluate(train_dataset, verbose=1)

# Resume training
new_vae.fit(train_dataset, epochs=20, batch_size=BATCH_SIZE,  verbose=1, callbacks=[cp_callback, CustomCallback()])