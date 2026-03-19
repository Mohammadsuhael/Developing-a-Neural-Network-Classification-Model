# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="1013" height="683" alt="image" src="https://github.com/user-attachments/assets/2d1aba28-3c2a-4d76-962f-87d08487b4e6" />

## DESIGN STEPS
### STEP 1: 
Data Collection and Understanding – Load the dataset, inspect features, and identify the target variable.

### STEP 2: 
 Data Cleaning and Encoding – Handle missing values and convert categorical data and labels into numerical form.

### STEP 3: 
Feature Scaling and Data Splitting – Normalize features and split data into training and testing sets.


### STEP 4: 
Model Architecture Design – Define the neural network layers, neurons, and activation functions.


### STEP 5: 
 Model Training and Optimization – Train the model using a loss function and optimizer through backpropagation.


### STEP 6: 

Model Evaluation and Prediction – Evaluate performance using metrics and make predictions on unseen data.

## PROGRAM

### Name: Mohammad Suhael

### Register Number: 212224230164

```python
# Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1=nn.Linear(input_size,32)
        self.fc2=nn.Linear(32,16)
        self.fc3=nn.Linear(16,8)
        self.fc4=nn.Linear(8,4)

    def forward(self, x):
      x=F.relu(self.fc1(x))
      x=F.relu(self.fc2(x))
      x=F.relu(self.fc3(x))
      x=self.fc4(x)
      return x

# Initialize the Model, Loss Function, and Optimizer


# Training Loop
def train_model(model, train_loader, criterion, optimizer, epochs):
  model.train()
  for epoch in range(epochs):
    for inputs, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Initialize model
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
train_model(model,train_loader,criterion,optimizer,epochs=100)

```

### Dataset Information
<img width="1168" height="259" alt="image" src="https://github.com/user-attachments/assets/e0ec71c0-a0ca-4ad3-9794-cce3e8868f86" />

### OUTPUT

## Confusion Matrix

<img width="692" height="568" alt="image" src="https://github.com/user-attachments/assets/c3b4e403-efb8-464e-8ed9-609b5cb7fbb2" />

## Classification Report
<img width="596" height="444" alt="image" src="https://github.com/user-attachments/assets/7d17d30c-425f-4e6e-92fb-9e429bd51122" />

### New Sample Data Prediction
<img width="407" height="110" alt="image" src="https://github.com/user-attachments/assets/03fc4443-df17-4ab9-92a1-74e43332a166" />

## RESULT
Neural network classification model for the given dataset is successfully developed.
