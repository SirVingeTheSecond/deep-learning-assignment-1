import medmnist
from medmnist import BloodMNIST
import matplotlib.pyplot as plt
import random
import numpy as np


trainDataset = BloodMNIST(split="train", download=True,size=28)
valDataset = BloodMNIST(split="val", download=True,size=28)
testDataset = BloodMNIST(split="test", download=True,size=28)

trainImages,trainLabels,trainInfo = trainDataset.__dict__['imgs'],trainDataset.__dict__['labels'],trainDataset.__dict__['info']['label']


print('Træningsdata:')
print(f'Billeder: {trainImages.shape}, Labels: {trainLabels.shape}')


valImages,valLabels = valDataset.__dict__['imgs'],valDataset.__dict__['labels']
print('Valideringsdata:')
print(f'Billeder: {valImages.shape}, Labels: {valLabels.shape}')

testImages,testLabels = testDataset.__dict__['imgs'],testDataset.__dict__['labels']
print('Testdata:')
print(f'Billeder: {testImages.shape}, Labels: {testLabels.shape}')

print('\n')
print('plot nogle eksempler:')

random.seed(42)
fig, axes = plt.subplots(5, len(trainInfo), figsize=(15, 5))

for class_,name in trainInfo.items():
    print(f'klasse: {class_}, klassenavn: {name}. Antal træning samples: {len(trainLabels[trainLabels==int(class_)])}')
    # Get indices of all images belonging to class i
    class_indices = [idx for idx, label in enumerate(trainLabels) if int(class_) == label]
    # Randomly select 5 indices
    selected_indices = random.sample(class_indices, 5)
    for j, idx in enumerate(selected_indices):
        image, label = trainImages[idx],trainLabels[idx]
        axes[j, int(class_)].imshow(image, cmap='gray')
        axes[j, int(class_)].axis('off')
        if j == 0:
            axes[j, int(class_)].set_title(f'{name[:5]}: {class_}')

plt.tight_layout()
plt.show()



#set random seed for reproducibility
np.random.seed(0)

#Shuffle the data
randomize = np.arange(trainImages.shape[0])
np.random.shuffle(randomize)

X_train = trainImages[randomize]
y_train = trainLabels[randomize].flatten()

randomizeVal = np.arange(valImages.shape[0])
X_val = valImages[randomizeVal]
y_val = valLabels[randomizeVal].flatten()


# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = list(range(num_training))

X_train = X_train[mask]
y_train = y_train[mask]

print('first 10 examples in train: ',y_train[:10])

num_val = 500
mask = list(range(num_val))
X_val = X_val[mask]
y_val = y_val[mask]

print('first 10 examples in val: ',y_val[:10])

# Reshape the image data into rows for effecient distance calculation
#(vi tager billedet med dimensioner 28x28x3 og strækker det ud til en vektor med længden 28*28*3 = 2352)
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
print(f'New train shape: {X_train.shape}')
print(f'New val shape: {X_val.shape}')

# Preprocessing: subtract the mean image
# first: compute the image mean based on the training data
mean_image = np.mean(X_train, axis=0)
print(mean_image[:10]) # print a few of the elements
plt.figure(figsize=(4,4))
plt.imshow(mean_image.reshape((28,28,3)).astype('uint8')) # visualize the mean image
plt.show()

# second: subtract the mean image from train and test data
X_train = X_train.astype(np.float32)-mean_image
X_val = X_val.astype(np.float32)-mean_image

print(X_train.min(),X_train.max())

print(X_train.shape, X_val.shape)

X_train /= 255.
X_val /= 255.


class FullyConnectedNN:
    def __init__(self, layers, reg_strength=0.01, loss='softmax', seed=42):
        """
        layers: List where each element represents the number of nodes in that layer.
        reg_strength: L2 regularization strength
        loss: 'softmax' or 'hinge'
        """
        np.random.seed(seed)
        self.layers = layers
        self.reg_strength = reg_strength
        self.loss_type = loss
        self.params = self._initialize_weights(layers)

    def _initialize_weights(self, layers):
        """
        Initialize weights and biases for each layer
        """
        params = {}
        for i in range(1, len(layers)):
            params['W' + str(i)] = np.random.randn(layers[i-1], layers[i]) * 0.01
            params['b' + str(i)] = np.zeros((1, layers[i]))
        return params

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return Z > 0

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def softmax_loss(self, A, y):
        m = y.shape[0]
        p = self.softmax(A)
        log_likelihood = -np.log(p[range(m), y])
        loss = np.sum(log_likelihood) / m
        return loss

    def hinge_loss(self, A, y):
        m = y.shape[0]
        correct_class_scores = A[range(m), y].reshape(-1, 1)
        margins = np.maximum(0, A - correct_class_scores + 1)
        margins[range(m), y] = 0
        loss = np.sum(margins) / m
        return loss

    def compute_loss(self, A, y):
        if self.loss_type == 'softmax':
            return self.softmax_loss(A, y) + self._l2_regularization()
        elif self.loss_type == 'hinge':
            return self.hinge_loss(A, y) + self._l2_regularization()

    def _l2_regularization(self):
        reg_loss = 0
        for i in range(1, len(self.layers)):
            reg_loss += np.sum(np.square(self.params['W' + str(i)]))
        return self.reg_strength * reg_loss / 2

    def forward(self, X):
        cache = {'A0': X}
        A = X
        for i in range(1, len(self.layers)):
            W, b = self.params['W' + str(i)], self.params['b' + str(i)]
            Z = np.dot(A, W) + b
            if i != len(self.layers) - 1:
                A = self.relu(Z)
            else:
                A = Z  # No activation in the output layer (raw scores for loss)
            cache['Z' + str(i)] = Z
            cache['A' + str(i)] = A
        return A, cache

    def backward(self, cache, y):
        grads = {}
        m = y.shape[0]
        A_last = cache['A' + str(len(self.layers) - 1)]
        if self.loss_type == 'softmax':
            dA = self.softmax(A_last)
            dA[range(m), y] -= 1
            dA /= m
        elif self.loss_type == 'hinge':
            margins = (A_last - A_last[range(m), y].reshape(-1, 1) + 1) > 0
            margins[range(m), y] = 0
            dA = np.where(margins, 1, 0)
            dA /= m

        for i in reversed(range(1, len(self.layers))):
            dZ = dA
            A_prev = cache['A' + str(i - 1)]
            grads['W' + str(i)] = np.dot(A_prev.T, dZ) + self.reg_strength * self.params['W' + str(i)]
            grads['b' + str(i)] = np.sum(dZ, axis=0, keepdims=True)
            if i > 1:
                dA = np.dot(dZ, self.params['W' + str(i)].T) * self.relu_derivative(cache['Z' + str(i - 1)])

        return grads

    def update_params(self, grads, learning_rate, v=None, beta1=0.9, beta2=0.999, t=1, optimizer='sgd'):
        """
        Updates parameters with chosen optimization method.
        If optimizer is 'momentum' or 'adam', it requires v for velocity and also t for time step in adam.
        """
        for i in range(1, len(self.layers)):
            if optimizer == 'sgd':
                self.params['W' + str(i)] -= learning_rate * grads['W' + str(i)]
                self.params['b' + str(i)] -= learning_rate * grads['b' + str(i)]
            elif optimizer == 'momentum':
                v['dW' + str(i)] = beta1 * v['dW' + str(i)] + (1 - beta1) * grads['W' + str(i)]
                v['db' + str(i)] = beta1 * v['db' + str(i)] + (1 - beta1) * grads['b' + str(i)]
                self.params['W' + str(i)] -= learning_rate * v['dW' + str(i)]
                self.params['b' + str(i)] -= learning_rate * v['db' + str(i)]
            elif optimizer == 'adam':
                v['mW' + str(i)] = beta1 * v['mW' + str(i)] + (1 - beta1) * grads['W' + str(i)]
                v['vW' + str(i)] = beta2 * v['vW' + str(i)] + (1 - beta2) * np.square(grads['W' + str(i)])
                mW_hat = v['mW' + str(i)] / (1 - beta1**t)
                vW_hat = v['vW' + str(i)] / (1 - beta2**t)
                self.params['W' + str(i)] -= learning_rate * mW_hat / (np.sqrt(vW_hat) + 1e-8)

    def fit(self, X, y, epochs=100, batch_size=64, learning_rate=0.01, optimizer='sgd'):
        """
        Trains the model using the chosen optimizer.
        """
        v = None
        if optimizer in ['momentum', 'adam']:
            v = {}
            for i in range(1, len(self.layers)):
                v['dW' + str(i)] = np.zeros_like(self.params['W' + str(i)])
                v['db' + str(i)] = np.zeros_like(self.params['b' + str(i)])
                if optimizer == 'adam':
                    v['mW' + str(i)] = np.zeros_like(self.params['W' + str(i)])
                    v['vW' + str(i)] = np.zeros_like(self.params['W' + str(i)])

        for epoch in range(epochs):
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                A_last, cache = self.forward(X_batch)
                loss = self.compute_loss(A_last, y_batch)
                grads = self.backward(cache, y_batch)
                self.update_params(grads, learning_rate, v=v, optimizer=optimizer)

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        A_last, _ = self.forward(X)
        if self.loss_type == 'softmax':
            return np.argmax(A_last, axis=1)
        elif self.loss_type == 'hinge':
            return np.argmax(A_last, axis=1)
        
    # Example usage:
    # To-lags neuralt netværk med input lag med antal neuroner svarende til input
    # størrelsen, ét "hidden" lag med 200 neuroner, og et output lag
    # med antal neuroner = antallet af klasser i datasættet.

nn = FullyConnectedNN(layers=[X_train.shape[1], 500, 8], loss='softmax')

# Train the neural network on the subsampled training data for 200 epochs
# using stochastic gradient decent
nn.fit(X_train, y_train,learning_rate=0.01, epochs=200,optimizer='sgd')

    #classify training data and val data using the trained model
preds = nn.predict(X_train)

print(f'Training accuracy={np.mean(preds==y_train)}')


val_preds = nn.predict(X_val)

print(f'Validation accuracy={np.mean(val_preds==y_val)}')

