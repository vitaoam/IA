import numpy as np
import matplotlib.pyplot as plt
import itertools

class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1, n_epochs=10):
        self.n_inputs = n_inputs
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.weights = np.zeros(n_inputs + 1)  # +1 para bias

    def predict(self, x):
        x = np.insert(x, 0, 1)  # adiciona bias
        return 1 if np.dot(self.weights, x) >= 0 else 0

    def train(self, X, y, plot=False):
        if plot and self.n_inputs == 2:
            plt.figure(figsize=(10, self.n_epochs*2))
        for epoch in range(self.n_epochs):
            for i in range(len(X)):
                x_i = np.insert(X[i], 0, 1)
                y_pred = 1 if np.dot(self.weights, x_i) >= 0 else 0
                self.weights += self.lr * (y[i] - y_pred) * x_i
            if plot and self.n_inputs == 2:
                plt.subplot(self.n_epochs, 2, epoch*2+1)
                self.plot_decision_boundary(X, y, title=f"Epoch {epoch+1}")
        if plot and self.n_inputs == 2:
            plt.tight_layout()
            plt.show()

    def plot_decision_boundary(self, X, y, title=""):
        for idx, label in enumerate(np.unique(y)):
            plt.scatter(X[y==label][:, 0], X[y==label][:, 1], label=f"Classe {label}")
        x1_vals = np.linspace(-0.2, 1.2, 10)
        if self.weights[2] != 0:
            x2_vals = -(self.weights[0] + self.weights[1]*x1_vals) / self.weights[2]
            plt.plot(x1_vals, x2_vals, 'k--')
        plt.xlim(-0.2, 1.2)
        plt.ylim(-0.2, 1.2)
        plt.title(title)
        plt.legend()

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

def test_model(model, X, y, name):
    print(f"\n{name}")
    for xi, yi in zip(X, y):
        pred = model.predict(xi)
        print(f"Entrada: {xi}, Saída esperada: {yi}, Saída perceptron: {pred}")

# =========================
# Troque aqui para AND, OR ou XOR:
funcao = "XOR"  # escolha: "AND", "OR" ou "XOR"
# =========================

n_inputs = 2  # para visualizar o gráfico, mantenha 2

X = gerar_entradas(n_inputs)
y = gerar_saidas(X, funcao)

print(f"\nTreinando Perceptron para {funcao}")
p = Perceptron(n_inputs=n_inputs, learning_rate=0.1, n_epochs=10)
p.train(X, y, plot=True)
test_model(p, X, y, funcao)
