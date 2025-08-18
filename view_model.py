from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# === Load the model ===
model = load_model("rps_model.h5")

# === 1. Print standard Keras summary ===
print("\n--- Model Summary ---\n")
model.summary()

# === 2. Make a clean DataFrame of layers ===
layer_data = []
for layer in model.layers:
    try:
        output_shape = layer.output.shape
    except AttributeError:
        try:
            output_shape = layer.get_output_shape_at(0)
        except:
            output_shape = "N/A"

    layer_data.append({
        "Layer Name": layer.name,
        "Layer Type": layer.__class__.__name__,
        "Output Shape": str(output_shape),
        "Parameters": layer.count_params()
    })

df = pd.DataFrame(layer_data)
print("\n--- Layer Table ---\n")
print(df)

# Save to CSV
df.to_csv("model_layers.csv", index=False)
print("\n✅ Layer details saved to model_layers.csv")

# === 3. Generate and save architecture diagram ===
plot_model(model, to_file="model_architecture.png", show_shapes=True, show_layer_names=True)
print("✅ Model diagram saved as model_architecture.png")

# === 4. Display the architecture inline (if in Jupyter/Colab) ===
try:
    img = mpimg.imread("model_architecture.png")
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.axis("off")
    plt.show()
except Exception as e:
    print("⚠️ Could not display image inline:", e)
