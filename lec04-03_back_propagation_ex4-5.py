import torch
import pdb

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
#wx=[1.0,1.0,1.0]
w2 = torch.tensor([1.0], requires_grad=True)
w1 = torch.tensor([1.0], requires_grad=True)
w0 = torch.tensor([1.0], requires_grad=True)

# our model forward pass
def forward(x):
    return (x*x * w2)+(x * w1)*w0

# Loss function
def loss(y_pred, y_val):
    return (y_pred - y_val) ** 2

# Before training
print("Prediction (before training)",  4, forward(4).item())

# Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        y_pred = forward(x_val) # 1) Forward pass
        l = loss(y_pred, y_val) # 2) Compute loss
        l.backward() # 3) Back propagation to update weights
        print("\tgrad: ", x_val, y_val, w2.grad.item(),w1.grad.item(),w0.grad.item())
        w0.data = w0.data - 0.01 * w0.grad.item()
        w1.data = w1.data - 0.01 * w1.grad.item()
        w2.data = w2.data - 0.01 * w2.grad.item()

        # Manually zero the gradients after updating weights
        w0.grad.data.zero_()
        w1.grad.data.zero_()
        w2.grad.data.zero_()

    print(f"Epoch: {epoch} | Loss: {l.item()}")

# After training
print("Prediction (after training)",  4, forward(4).item())
