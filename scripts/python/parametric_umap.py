import numpy
import pyarrow.parquet as pq
from pathlib import Path
from umap.parametric_umap import ParametricUMAP
from tensorflow import keras
from tensorflow.keras import layers


DATA_DIR = Path("/home/maximl/scratch/data/cd7/800/results/scip/202203221745/")
WIDTH = 109
HEIGHT = 112
SCALE = 2


def main():
    idx = numpy.load(DATA_DIR / "neutrophils.npy", allow_pickle=True)
    print(len(idx))
    df = pq.read_table(DATA_DIR / "features.parquet").to_pandas().set_index(
        ["meta_panel", "meta_replicate", "meta_P", "meta_id"]
    ).loc["D"]
    df = df.sort_index()
    df = df.loc[idx]

    X = numpy.load(DATA_DIR / "neutrophil_images_scale2.npy")
    X = numpy.swapaxes(X, 1, -1)
    X = X.reshape(X.shape[0], -1)

    dims = (WIDTH // SCALE, HEIGHT // SCALE, 3)
    n_components = 2

    base_model = keras.applications.MobileNetV2(
        input_shape=dims, weights = 'imagenet', include_top = False)
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(n_components)(x)

    encoder = keras.Model(inputs=base_model.input, outputs=x)

    embedder = ParametricUMAP(
        encoder=encoder,
        dims=dims,
        n_components=n_components,
        verbose=True,
        n_training_epochs=1
    )
    embedding = embedder.fit_transform(X)

    output_dir = DATA_DIR / "embeddings/param_umap/202203241023_scale2_mobilenetv2"
    output_dir.mkdir(parents=True, exist_ok=True)
    numpy.save(output_dir / "embedding.npy", embedding)
    embedder.save(output_dir / "model")


if __name__ == "__main__":
    main()
