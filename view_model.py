from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("rps_model.h5")

# View model architecture
model.summary()

# View the model's layers
for layer in model.layers:
    print(layer.name, layer.output_shape)
