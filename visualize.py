import pandas as pd
import matplotlib.pyplot as plt

# Load the data from metrics.csv
file_path = "lightning_logs/version_23/metrics.csv"  # Update this path if the file is located elsewhere
data = pd.read_csv(file_path)

# Check if required columns are in the data
required_columns = ["epoch", "train_loss_step", "validation_loss_step"]
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"The file must contain the following columns: {', '.join(required_columns)}")

# Extract data for plotting
epochs = data["step"]
train_loss = data["train_loss_step"]
validation_loss = data["validation_loss_step"]

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label="Train Loss", color="blue", linewidth=2, marker="o")
plt.plot(epochs, validation_loss, label="Validation Loss", color="orange", linewidth=2, marker="x")

# Add titles and labels
plt.title("Training and Validation Loss per Epoch", fontsize=16)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Show the plot
plt.tight_layout()
plt.show()
