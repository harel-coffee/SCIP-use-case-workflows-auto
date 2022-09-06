import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
import operator
import click
from pathlib import Path
from datetime import datetime

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

from libtiff import TIFF
import libtiff
libtiff.libtiff_ctypes.suppress_warnings()


class Denoise(Model):
    def __init__(self, latent_dim, input_shape):
        super(Denoise, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
            layers.Flatten(),
            layers.Dense(latent_dim)
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(7*7*8, activation="relu"),
            layers.Reshape((7,7,8)),
            layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(input_shape[-1], kernel_size=(3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def cropND(img, bounding):
  
    # compute possibly necessary padding widths
    padding = tuple(map(lambda a,b: abs(min(0, b-a)), bounding, img.shape))
    if sum(padding) > 0:
      
        # split padding into before and after part
        before_after = tuple(map(lambda a: (a//2, (a//2)+(a%2)), padding))
        img = np.pad(
            array=img,
            pad_width=before_after,
            mode="edge",
        )
        
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


def process_path(paths):
    img = np.empty(shape=(28,28, len(paths)), dtype=np.float32)
    
    for i, path in enumerate(paths):
        x = TIFF.open(path)
        x = x.read_image()
        x = x / 2**12
        img[..., i] = cropND(x, (28, 28))
    
    return img

@click.command(name="Train autoencoder")
@click.argument("output", type=click.Path(file_okay=False, exists=True))
def main(output):

    output = Path(output) / datetime.today().strftime("%d%m%Y%H%M%S")
    output.mkdir()

    data_dir = Path("/home/maximl/data/EhV_infection/images/")

    x_train = list(map(
        process_path, 
        zip(
            data_dir.glob("H1_T[7,8]/*Ch1.*"), 
            data_dir.glob("H1_T[7,8]/*Ch6.*"), 
            data_dir.glob("H1_T[7,8]/*Ch7.*")
        )
    ))
    x_test = list(map(
        process_path, 
        zip(
            data_dir.glob("H1_T9/*Ch1.*"), 
            data_dir.glob("H1_T9/*Ch6.*"), 
            data_dir.glob("H1_T9/*Ch7.*")
        )
    ))
    
    x_train = tf.stack(x_train)
    x_test = tf.stack(x_test)
    
    noise_factor = 0.05
    x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape, stddev=0.05) 
    x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape, stddev=0.05)

    x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
    x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)

    autoencoder = Denoise(128, input_shape=(28, 28, 3))
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    history = autoencoder.fit(x_train_noisy, x_train,
                    batch_size=32,
                    epochs=100,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test))

    autoencoder.save(str(output / "model"))

    _, ax = plt.subplots(1, 1)
    ax.plot(history.epoch, history.history["loss"])
    ax.plot(history.epoch, history.history["val_loss"])
    plt.savefig(str(output / "learning_curves.pdf"))

    encoded_imgs = autoencoder.encoder(x_test).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    n = 10
    for j in range(3):
        _, axes = plt.subplots(3, n, figsize=(20, 6))
        for i in range(n):
            
            # display original
            ax = axes[0, i]
            ax.set_title("original")
            ax.imshow(tf.squeeze(x_test[i, ..., j]), cmap="gray")
            ax.set_axis_off()

            # display original + noise
            ax = axes[1, i]
            ax.set_title("original + noise")
            ax.imshow(tf.squeeze(x_test_noisy[i, ..., j]), cmap="gray")
            ax.set_axis_off()

            # display reconstruction
            ax = axes[2, i]
            ax.set_title("reconstructed")
            ax.imshow(tf.squeeze(decoded_imgs[i, ..., j]), cmap="gray")
            ax.set_axis_off()
            
        plt.savefig(str(output / f"example_images_{j}.pdf"))

    paths = data_dir.glob("H1_T9/*Ch1.*")
    objectnumbers = list(int(p.stem.split("_")[0]) for p in paths)
    df = pd.DataFrame(index=objectnumbers, data=encoded_imgs)
    df.columns = [str(c) for c in df.columns]
    df.to_parquet(str(output / "representation.parquet"))


if __name__ == "__main__":
    main()
