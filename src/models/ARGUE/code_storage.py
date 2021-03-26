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


vprint(verbose, "\n=== Phase 2: training alarm network ===")
for epoch in range(epochs):
    vprint(verbose, f"\n>> Epoch {epoch}")
    epoch_loss = []
    epoch_metric = []
    for step, (x_batch_train, true_gating) in enumerate(alarm_gating_train_dataset):
        vprint(step % 20 == 0, f"\nStep: {step}")
        with tf.GradientTape(persistent=True) as tape:
            for name, model in self.input_to_alarm_dict.items():
                predicted_alarm = model(x_batch_train, training=True)
                true_alarm = (1 - true_gating.numpy())[:, int(name[-1])].reshape((-1, 1))
                loss_value = alarm_loss(true_alarm, predicted_alarm)
                vprint(step % 20 == 0, f"Alarm model {name} batch loss: {float(loss_value)}")
                loss_value += loss_value

        epoch_loss.append(float(loss_value))
        gradients = tape.gradient(loss_value, self.alarm.keras_model.trainable_weights)
        alarm_optimizer.apply_gradients(zip(gradients, self.alarm.keras_model.trainable_weights))
        alarm_metric.update_state(true_alarm, predicted_alarm)
        error_metric = alarm_metric.result()
        epoch_metric.append(error_metric)
        alarm_metric.reset_states()

        if step % 40 == 0 and verbose > 1:
            print(f"Batch {step} training loss: {float(loss_value):.4f}, ")

    vprint(verbose, f"Alarm epoch loss: {np.mean(epoch_loss):.4f}, "
                    f"Binary accuracy: {np.mean(epoch_metric):.4f}")