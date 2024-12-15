import numpy as np

# Função de aptidão
def calcular_aptidao(matriz_horario):
    aptidao = 0
    for i in range(matriz_horario.shape[0]):
        for j in range(matriz_horario.shape[1]):
            if matriz_horario[i, j] != -1:  # Aulas diferentes
                if np.sum(matriz_horario[i] == matriz_horario[i, j]) <= 2:
                    aptidao += 1  # Bônus por aulas repetidas no máximo duas vezes
                if np.sum(matriz_horario[:, j] == matriz_horario[i, j]) > 1:
                    aptidao -= 1  # Penalização por choque de horários
    return aptidao

# Gerar população inicial
def gerar_populacao(tamanho_populacao, num_professores, num_horarios):
    populacao = []
    for _ in range(tamanho_populacao):
        matriz_horario = np.random.randint(0, num_professores, size=(num_horarios, num_horarios))
        populacao.append(matriz_horario)
    return np.array(populacao)

# Seleção por roleta
def selecao_roleta(populacao, aptidao):
    soma_aptidao = np.sum(aptidao)
    prob = aptidao / soma_aptidao
    indices = np.random.choice(np.arange(len(populacao)), size=len(populacao), p=prob)
    return populacao[indices]

# Seleção por torneio
def selecao_torneio(populacao, aptidao, tamanho_torneio):
    nova_populacao = []
    for _ in range(len(populacao)):
        competidores = np.random.choice(np.arange(len(populacao)), size=tamanho_torneio, replace=False)
        vencedor = competidores[np.argmax(aptidao[competidores])]
        nova_populacao.append(populacao[vencedor])
    return np.array(nova_populacao)

# Cruzamento
def cruzamento(populacao, tipo_cruzamento):
    filhos = []
    for i in range(0, len(populacao), 2):
        pai = populacao[i]
        mae = populacao[i + 1]
        if tipo_cruzamento == 1:
            ponto_corte = np.random.randint(1, pai.shape[0])
            filho1 = np.vstack((pai[:ponto_corte], mae[ponto_corte:]))
            filho2 = np.vstack((mae[:ponto_corte], pai[ponto_corte:]))
        elif tipo_cruzamento == 2:
            ponto_corte1 = np.random.randint(1, pai.shape[0])
            ponto_corte2 = np.random.randint(ponto_corte1 + 1, pai.shape[0])
            filho1 = np.vstack((pai[:ponto_corte1], mae[ponto_corte1:ponto_corte2], pai[ponto_corte2:]))
            filho2 = np.vstack((mae[:ponto_corte1], pai[ponto_corte1:ponto_corte2], mae[ponto_corte2:]))
        filhos.extend([filho1, filho2])
    return np.array(filhos)

# Mutação
def mutacao(populacao, prob_mutacao):
    for i in range(len(populacao)):
        if np.random.rand() < prob_mutacao:
            linha_mutacao = np.random.randint(0, populacao[i].shape[0])
            coluna_mutacao = np.random.randint(0, populacao[i].shape[1])
            novo_professor = np.random.randint(0, np.max(populacao) + 1)
            populacao[i][linha_mutacao, coluna_mutacao] = novo_professor
    return populacao

# Parâmetros do algoritmo
num_professores = 5
num_horarios = 5
tamanho_populacao = 10
num_geracoes = 50
prob_cruzamento = 0.8
prob_mutacao = 0.1
tipo_cruzamento = 1  # 1 ponto ou 2 pontos
metodo_selecao = "torneio"  # roleta ou torneio
tamanho_torneio = 3

# Algoritmo Genético
populacao = gerar_populacao(tamanho_populacao, num_professores, num_horarios)
for _ in range(num_geracoes):
    aptidao = np.array([calcular_aptidao(individuo) for individuo in populacao])
    if metodo_selecao == "roleta":
        populacao = selecao_roleta(populacao, aptidao)
    elif metodo_selecao == "torneio":
        populacao = selecao_torneio(populacao, aptidao, tamanho_torneio)
    filhos = cruzamento(populacao, tipo_cruzamento)
    populacao = mutacao(filhos, prob_mutacao)

# Resultado final
aptidao_final = np.array([calcular_aptidao(individuo) for individuo in populacao])
melhor_indice = np.argmax(aptidao_final)
melhor_horario = populacao[melhor_indice]
print(f"Melhor horário dos professores:\n{melhor_horario}")
print(f"Aptidão do melhor horário: {aptidao_final[melhor_indice]}")
