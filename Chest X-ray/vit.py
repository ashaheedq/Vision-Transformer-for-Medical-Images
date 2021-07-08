import tensorflow as tf
from utils import AttrDict


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


class ViTClassfier(tf.keras.models.Model):
    CONFIG = {
        "patch_size": 6,
        "embedding_dim": 32,
        "n_heads": 2,
        "n_transformers": 1,
        "transformer_units": [64, 32],
        "transformer_dropout": 0.1,
        "mlp_units": [2048, 1024],
        "mlp_dropout": 0.1,
    }

    def __init__(self, input_shape, n_classes, config={}):
        super(ViTClassfier, self).__init__()

        self._config = AttrDict(
            **{
                **self.CONFIG,
                **config,
                "input_shape": input_shape,
                "n_classes": n_classes,
            }
        )
        self._construct_model()

    def call(self, x):
        return self._model(x)

    def _construct_model(self):
        config = self._config

        num_patches = (config.input_shape[0] // config.patch_size) ** 2

        inputs = tf.keras.layers.Input(shape=config.input_shape)

        patches = Patches(config.patch_size)(inputs)
        # Encode patches.
        encoded_patches = PatchEncoder(num_patches, config.embedding_dim)(patches)

        # Create multipletf.keras.layers of the Transformer block.
        for _ in range(config.n_transformers):
            # Layer normalization 1.
            x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=config.n_heads, key_dim=config.embedding_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            for units in config.transformer_units:
                x3 = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x3)
                x3 = tf.keras.layers.Dropout(config.transformer_dropout)(x3)

            # Skip connection 2.
            encoded_patches = tf.keras.layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        features = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        features = tf.keras.layers.Flatten()(features)
        features = tf.keras.layers.Dropout(0.5)(features)
        # Add MLP.
        for units in config.mlp_units:
            features = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(features)
            features = tf.keras.layers.Dropout(config.mlp_dropout)(features)
        # Classify outputs.
        logits = tf.keras.layers.Dense(config.n_classes)(features)
        # Create the Keras model.
        self._model = tf.keras.models.Model(inputs=inputs, outputs=logits)
