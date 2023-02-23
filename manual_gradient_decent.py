# Training Data
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w=[1.0,1.0,1.0]

# our model forward pass
def forward(x):
    return (x*x * w[2])+(x * w[1])*w[0]


# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

#compute subtask
def gradi(x,y):
    return 2*((x*x * w[2])+(x * w[1])*w[0]-y)
# compute gradient
def gradient(x, y, k):  # d_loss/d_w
    if k==2:
        return x*x*gradi(x,y)
    elif k==1:
        return x*gradi(x,y)
    elif k==0:
        return gradi(x,y)


# Before training
print("Prediction (before training)",  4, forward(4))

# Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        # Compute derivative w.r.t to the learned weights
        # Update the weights
        # Compute the loss and print progress
        for k in range(3):
            grad = gradient(x_val, y_val,k)
            w[k] = w[k] - 0.01 * grad
       # print("\tgrad: ", x_val, y_val, round(grad, 2))
        l = loss(x_val, y_val)
    print("progress:", epoch, "w2=", round(w[2], 2),"w1=", round(w[1], 2),"w0=", round(w[0], 2), "loss=", round(l, 2))

# After training
print("Predicted score (after training)",  "4 hours of studying: ", forward(4))
