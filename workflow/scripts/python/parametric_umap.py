import numpy
import pyarrow.parquet as pq
from pathlib import Path
from umap.parametric_umap import ParametricUMAP
from tensorflow import keras
from sklearn.model_selection import train_test_split


DATA_DIR = Path("/home/maximl/scratch/data/cd7/800/results/scip/202203221745/")
WIDTH = 28
HEIGHT = 28


def main():
    idx = numpy.load(DATA_DIR / "neutrophils.npy", allow_pickle=True)
    print(len(idx))
    df = pq.read_table(DATA_DIR / "features.parquet").to_pandas().set_index(
        ["meta_panel", "meta_replicate", "meta_P", "meta_id"]
    ).loc["D"]
    df = df.sort_index()
    df = df.loc[idx]

    X = numpy.load(DATA_DIR / "neutrophil_images_scale_mnist.npy")
    X = numpy.swapaxes(X, 1, -1)

    # select channels
    X = X[..., 0]

    X = X.reshape(X.shape[0], -1)

    dims = (WIDTH, HEIGHT, 1)
    n_components = 2

    encoder = keras.Sequential([
        keras.layers.InputLayer(input_shape=dims),
        keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=(2, 2), activation="relu", padding="same"
        ),
        keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=(2, 2), activation="relu", padding="same"
        ),
        keras.layers.Flatten(),
        keras.layers.Dense(units=256, activation="relu"),
        keras.layers.Dense(units=256, activation="relu"),
        keras.layers.Dense(units=n_components),
    ])
    decoder = keras.Sequential([
        keras.layers.InputLayer(input_shape=(n_components)),
        keras.layers.Dense(units=256, activation="relu"),
        keras.layers.Dense(units=7 * 7 * 256, activation="relu"),
        keras.layers.Reshape(target_shape=(7, 7, 256)),
        keras.layers.Conv2DTranspose(
        filters=128, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
        ),
        keras.layers.Conv2DTranspose(
            filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
        ),
        keras.layers.Conv2DTranspose(
            filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation="sigmoid"
        )
    ])

    train_idx, test_idx = train_test_split(numpy.arange(len(X)), test_size=0.1)

    embedder = ParametricUMAP(
        encoder=encoder,
        decoder=decoder,
        dims=dims,
        n_components=n_components,
        parametric_reconstruction=True,
        reconstruction_validation=X[test_idx],
        autoencoder_loss=True,
        verbose=True,
        n_training_epochs=10
    )
    embedding = embedder.fit_transform(X[train_idx])

    output_dir = DATA_DIR / "embeddings/param_umap/202203241023_scalemnist_example"
    output_dir.mkdir(parents=True, exist_ok=True)
    numpy.save(output_dir / "embedding.npy", embedding)
    embedder.save(output_dir / "model")


if __name__ == "__main__":
    main()
