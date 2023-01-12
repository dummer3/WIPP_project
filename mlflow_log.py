import numpy as np
import torch
import mlflow.pytorch

class LinearNNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

def gen_data():
    # Example linear model modified to use y = 2x
    # from https://github.com/hunkim/PyTorchZeroToAll
    # X training data, y labels
    X = torch.arange(1.0, 25.0).view(-1, 1)
    y = torch.from_numpy(np.array([x * 2 for x in X])).view(-1, 1)
    return X, y

# Define model, loss, and optimizer
model = LinearNNModel()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Training loop
epochs = 250
X, y = gen_data()
for epoch in range(epochs):
    # Forward pass: Compute predicted y by passing X to the model
    y_pred = model(X)

    # Compute the loss
    loss = criterion(y_pred, y)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Save PyTorch models to current working directory
with mlflow.start_run() as run:
    mlflow.pytorch.save_model(model, "model")

    # Convert to a scripted model and save it
    scripted_pytorch_model = torch.jit.script(model)
    mlflow.pytorch.save_model(scripted_pytorch_model, "scripted_model")

# Load each saved model for inference
for model_path in ["model", "scripted_model"]:
    model_uri = "{}/{}".format(os.getcwd(), model_path)
    loaded_model = mlflow.pytorch.load_model(model_uri)
    print("Loaded {}:".format(model_path))
    for x in [6.0, 8.0, 12.0, 30.0]:
        X = torch.Tensor([[x]])
        y_pred = loaded_model(X)
        print("predict X: {}, y_pred: {:.2f}".format(x, y_pred.data.item()))
    print("--")