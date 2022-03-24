import numpy
import pyarrow.parquet as pq
from pathlib import Path
from umap.parametric_umap import ParametricUMAP
from tensorflow import keras
from tensorflow.keras import layers


DATA_DIR = Path("/home/maximl/scratch/data/cd7/800/results/scip/202203221745/")
WIDTH=109
HEIGHT=112


def main():
    idx = numpy.load(DATA_DIR / "neutrophils.npy", allow_pickle=True)
    print(len(idx))
    df = pq.read_table(DATA_DIR / "features.parquet").to_pandas().set_index(
        ["meta_panel", "meta_replicate", "meta_P", "meta_id"]
    ).loc["D"]
    df = df.sort_index()
    df = df.loc[idx]

    X = numpy.load(DATA_DIR / "neutrophil_images.npy")
    X = numpy.swapaxes(X, 1, -1)
    X = X.reshape(X.shape[0], -1)

    dims = (WIDTH, HEIGHT, 3)
    n_components = 2

    encoder = keras.Sequential([
        layers.Input(shape=dims),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
        layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
        layers.Flatten(),
        layers.Dense(512),
        layers.Dense(512),
        layers.Dense(n_components)
    ])

    embedder = ParametricUMAP(
        encoder=encoder,
        dims=dims,
        n_components=n_components,
        verbose=True,
        n_training_epochs=2
    )
    embedding = embedder.fit_transform(X)

    output_dir = DATA_DIR / "embeddings/param_umap/202203241023"
    output_dir.mkdir(parents=True, exist_ok=True)
    numpy.save(output_dir / "embedding.npy", embedding)
    embedder.save(output_dir / "model")


if __name__ == "__main__":
    main()
