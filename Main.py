from train_model import train
from predict import predict

# Step 1: Train model
train()

# Step 2: Test prediction
test_signal = 0.85  # simulate human presence
result = predict(test_signal)

print("Prediction:", result)
