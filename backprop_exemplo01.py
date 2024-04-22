import numpy as np

# Função de ativação sigmóide e sua derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define os inputs da rede
inputs = np.array([0.5, 0.9, -0.3])

# Define os outputs esperados da rede
expected_output = np.array([0.9, 0.3])

# Pesos das camadas, fornecidos pelo diagrama
weights_hidden = np.array([[1.0, -2.0, 2.0],   # Pesos para H1
                           [2.0, 1.0, -4.0],   # Pesos para H2
                           [1.0, -1.0, 0.0]])  # Pesos para H3

weights_output = np.array([[-3.0, 1.0, -3.0],  # Pesos para O1
                           [0.0, 1.0, 2.0]])   # Pesos para O2

# Define a taxa de aprendizagem
learning_rate = 0.1

# Define o número de épocas
epochs = 1000

# Treina a rede neural
for epoch in range(epochs):
    # Passo forward
    hidden_layer_activation = np.dot(weights_hidden, inputs)
    hidden_layer_output = sigmoid(hidden_layer_activation)
    
    final_output = sigmoid(np.dot(weights_output, hidden_layer_output))
    
    # Calcula o erro
    error = expected_output - final_output
    
    # Inicia backpropagation
    d_final_output = error * sigmoid_derivative(final_output)
    
    error_hidden_layer = d_final_output.dot(weights_output)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Atualiza os pesos
    weights_output += learning_rate * np.outer(d_final_output, hidden_layer_output)
    weights_hidden += learning_rate * np.outer(d_hidden_layer, inputs)
    
    # Imprime o erro a cada 10% do número de épocas decorridas
    if epoch % (epochs/10) == 0:
        loss = np.mean(np.abs(error))
        print(f'Epoch {epoch}, Loss: {loss}')

# Resultado após o treino

print(f'\n\nResultados após o treino com {epochs} épocas: ')
print(f'\nPesos finais da camada oculta: \n{weights_hidden}')
print(f'\nPesos finais da camada de saída: \n{weights_output}')
print(f'\n\nValores finais esperados (output): {expected_output}')
print(f'Valores finais obtidos (output): {final_output}')
print(f'\n')


