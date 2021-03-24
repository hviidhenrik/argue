class NetworkBlock(Model):
    """
    Doesn't work with the encoder/decoder in trying to extract their hidden activations. Use
    functional API for these. This can perhaps be used with the alarm and gating network,
    where activations aren't needed from.

    """
    def __init__(self, units_in_layers: List[int], activation="selu", **kwargs):
        super().__init__(**kwargs)
        self.hidden_layers = {}

        for layer_number, units in enumerate(units_in_layers):
            name = f"hidden{layer_number+1}"
            self.hidden_layers[name] = tf.keras.layers.Dense(units=units, activation=activation, name=name)

    def call(self, inputs):
        x = inputs
        for _, layer in self.hidden_layers.items():
            x = layer(x)
        return x


def build_encoder(inputs, latent_dim: int = 2, latent_activation: str = "selu", **kwargs):
    encoder_block = network_block(inputs, **kwargs)
    latent_layer = Dense(latent_dim, activation=latent_activation)(encoder_block)
    return latent_layer


def build_decoder(inputs, feature_dim: int, output_activation: str = "linear" ,**kwargs):
    decoder_block = network_block(inputs, **kwargs)
    reconstruction_layer = Dense(feature_dim, activation=output_activation)(decoder_block)


# hidden = network_block(inputs, [500, 250, 125], "selu")
# latent = Dense(2, name="latent")(hidden)
# hidden = network_block(latent, [125, 250, 500], "selu")
# outputs = Dense(3, name="output")(hidden)
#
# model = Model(inputs=inputs, outputs=outputs)



model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss=tf.keras.losses.MeanSquaredError())
model.fit(x=df, y=df, batch_size=256, epochs=20)
activations = extract_activations(model, "activations")

activation_model = Model(inputs=inputs, outputs=activations)

# inputs = self.encoder.keras_model.input
# x = self.encoder.activation_model(inputs)
# outputs = self.alarm.keras_model(x)
# self.input_to_alarm = Model(inputs, outputs, name="x->encoder->expert->alarm->y")