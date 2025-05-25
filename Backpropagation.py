import numpy as np
import itertools
def gerar_entradas(n):
    return np.array(list(itertools.product([0, 1], repeat=n)))

def gerar_saidas(X, funcao):
    if funcao == "AND":
        return np.all(X == 1, axis=1).astype(int)
    elif funcao == "OR":
        return np.any(X == 1, axis=1).astype(int)
    elif funcao == "XOR" and X.shape[1] == 2:
        return np.logical_xor(X[:,0], X[:,1]).astype(int)
    else:
        raise ValueError("Função não suportada para esse número de entradas.")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2

class MLP:
    def __init__(self, n_inputs, n_hidden, n_outputs, lr=0.1, activation='sigmoid'):
        self.lr = lr
        self.weights1 = np.random.randn(n_inputs + 1, n_hidden)
        self.weights2 = np.random.randn(n_hidden + 1, n_outputs)
        if activation == 'sigmoid':
            self.act = sigmoid
            self.act_deriv = sigmoid_deriv
        elif activation == 'tanh':
            self.act = tanh
            self.act_deriv = tanh_deriv

    def forward(self, x):
        x = np.insert(x, 0, 1)
        self.z1 = np.dot(x, self.weights1)
        self.a1 = self.act(self.z1)
        self.a1 = np.insert(self.a1, 0, 1) 
        self.z2 = np.dot(self.a1, self.weights2)
        self.a2 = self.act(self.z2)
        return self.a2

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            for i in range(len(X)):
                x = X[i]
                target = y[i]
                out = self.forward(x)
                error = target - out
                delta2 = error * self.act_deriv(self.z2)
                delta1 = self.act_deriv(self.z1) * (self.weights2[1:, 0] * delta2)
                self.weights2 += self.lr * np.outer(self.a1, delta2)
                x_bias = np.insert(x, 0, 1)
                self.weights1 += self.lr * np.outer(x_bias, delta1)

    def predict(self, x):
        return int(self.forward(x) > 0.5)


n_inputs = 2
funcao = "OR"
activation = "tanh"

X = gerar_entradas(n_inputs)
y = gerar_saidas(X, funcao)

mlp = MLP(n_inputs=n_inputs, n_hidden=2, n_outputs=1, lr=0.1, activation=activation)
mlp.train(X, y, epochs=10000)
print(f"\nResultados para {funcao} com função de ativação {activation}:")
for xi, yi in zip(X, y):
    print(f"Entrada: {xi}, Esperado: {yi}, MLP: {mlp.predict(xi)}")
