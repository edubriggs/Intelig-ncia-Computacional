import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Função para converter valores para float
def convert_to_float(value):
    if isinstance(value, str):
        return float(value.replace('.', '').replace(',', '.'))
    return float(value)

# Carregar os dados
dados_dias = pd.read_csv('csv_dias.csv', delimiter=',', decimal=',')
dados_meses = pd.read_csv('csv_meses.csv', delimiter=',', decimal=',')

# Converter os valores das colunas 'Último' para float corretamente
dados_dias['Último'] = dados_dias['Último'].apply(convert_to_float)
dados_meses['Último'] = dados_meses['Último'].apply(convert_to_float)

# Ordenar os dados por data
dados_dias = dados_dias.sort_values(by='Data')
dados_meses = dados_meses.sort_values(by='Data')

# Verificar se os dados foram carregados corretamente
print(dados_dias.head())
print(dados_dias.tail())

# Extrair o valor de fechamento ('Último') para treinamento e validação
total_days = len(dados_dias)
train_data = dados_dias['Último'][:total_days - 7].values  # Dados de treino (tudo menos os últimos 7 dias)
validation_data = dados_dias['Último'][total_days - 7:].values  # Dados de validação (últimos 7 dias)

# Verificar se os dados de validação foram extraídos corretamente
print(validation_data)

# Certificar-se de que validation_data não está vazio
if validation_data.size == 0:
    raise ValueError("Dados de validação estão vazios. Verifique o intervalo de linhas selecionadas.")

# Normalizar os dados de treino e validação com base nos dados de treino
train_min, train_max = np.min(train_data), np.max(train_data)
train_data = (train_data - train_min) / (train_max - train_min)
validation_data = (validation_data - train_min) / (train_max - train_min)

# Preparar os dados de entrada e saída
X_train = np.array([train_data[i] for i in range(len(train_data) - 1)]).reshape(-1, 1)
y_train = np.array([train_data[i + 1] for i in range(len(train_data) - 1)]).reshape(-1, 1)
X_val = np.array([validation_data[i] for i in range(len(validation_data) - 1)]).reshape(-1, 1)
y_val = np.array([validation_data[i + 1] for i in range(len(validation_data) - 1)]).reshape(-1, 1)

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Inicializar pesos e bias
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = self.sigmoid(self.z2)
        return self.output

    def backward(self, X, y):
        self.loss = y - self.output
        d_output = self.loss * self.sigmoid_derivative(self.output)
        d_hidden = np.dot(d_output, self.W2.T) * self.sigmoid_derivative(self.a1)

        # Atualizar pesos e bias
        self.W2 += np.dot(self.a1.T, d_output) * self.learning_rate
        self.b2 += self.learning_rate * np.sum(d_output, axis=0, keepdims=True)
        self.W1 += np.dot(X.T, d_hidden) * self.learning_rate
        self.b1 += self.learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - self.output))
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        return self.forward(X)

# Definir os tamanhos de entrada, oculto e saída
input_size = X_train.shape[1]
hidden_size = 10
output_size = 1

# Criar e treinar a rede neural
mlp = MLP(input_size, hidden_size, output_size, learning_rate=0.01)
mlp.train(X_train, y_train, epochs=1000)

# Predizer os dados de validação
predictions = mlp.predict(X_val)

# Desnormalizar os dados para comparação
predictions = predictions * (train_max - train_min) + train_min
y_val = y_val * (train_max - train_min) + train_min

# Calcular o erro médio absoluto
mae = np.mean(np.abs(predictions - y_val))
print(f"Mean Absolute Error: {mae}")

# Plotar os resultados
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_val)), y_val, label='Valor Real')
plt.plot(range(len(predictions)), predictions, label='Predição')
plt.legend()
plt.xlabel('Dias')
plt.ylabel('Valor de Fechamento')
plt.title('Predição de Valor de Fechamento da Ação')
plt.show()
