import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim #we can use Adam, RMSprop etc
import matplotlib.pyplot as plt

class SimpleNN(nn.Module):
    def __init__(self, in_features, out_features): #DEFINE LAYERS
        #for LLM it should be (self, embedding_dim, ffn_dim)
        super().__init__()
        self.linear_layer = nn.Linear(in_features, out_features)

        #For LLMs, we would have more layers like:
        #self.layer1 = nn.Linear(embedding_dim, ffn_dim)
        #self.activation = nn.ReLU()  # or nn.GELU()
        #self.layer2 = nn.Linear(ffn_dim, embedding_dim)

    def forward(self, x): #CONNECT LAYERS
        #and here for LLMs:
        #x = self.layer1(x)
        #x = self.activation(x)
        #x = self.layer2(x)
        #return x
        return self.linear_layer(x)

model = SimpleNN(in_features=1, out_features=1)
#print("Model architecture:")
#print(model)

"""STEPS
1. Forward pass (y_hat = model(x))
2. Compute loss (loss = loss_function(y_hat, y_true))
3. The three line mantra:
  a. optimizer.zero_grad()  # Zero the gradients
  b. loss.backward()        # Backpropagation
  c. optimizer.step()       # Update the weights"""

learning_rate = 0.01
# Using Adam optimizer
#we pass model.parameters() to tell it which tensor to manage
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

loss_function = nn.MSELoss() #Mean Squared Error Loss for regression tasks
#for LLMs, we might use CrossEntropyLoss for classification tasks

epochs = 250
x = torch.randn(500, 1)
y_true = 3 * x + 2 + 0.1 * torch.randn(500, 1)  # y = 3x + 2 + noise
#print(f"y_true: {y_true[:3]}")


#print(f"Layer's Weight (W): {model.linear_layer.weight}\n, Bias (b): {model.linear_layer.bias}")

for epoch in range(epochs):
    # Forward pass
    y_hat = model(x)
    #print(f"Predicted y (y_hat): {y_hat[:3]}")

    # Compute loss
    loss = loss_function(y_hat, y_true)

    # Backpropagation and optimization
    optimizer.zero_grad()  # Zero the gradients
    loss.backward()        # Backpropagation
    optimizer.step()       # Update the weights

    #if (epoch+1) % 10 == 0:
     #   print(f'Epoch {epoch+1:02d}, Loss: {loss.item():.4f}')

plt.scatter(y_true.detach().numpy(), y_hat.detach().numpy())
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()