import numpy as np

# Função a ser minimizada
def funcao(x):
    return -abs(x * np.sin(np.sqrt(abs(x))))

# Gerar uma população inicial de elementos decimais
def gerarElementos(tamanho_populacao, intervalo):
    return np.random.uniform(intervalo[0], intervalo[1], tamanho_populacao)

# Converter elementos decimais em binários
def gerarElementosBinarios(popDec, bits):
    return np.array([np.binary_repr(int(element), width=bits) for element in popDec])

# Calcular a imagem da função para cada elemento da população
def gerarImagem(popDec):
    return np.array([funcao(element) for element in popDec])

# Calcular a probabilidade da roleta baseada na imagem da função
def gerarProbabilidade(imagemFuncao):
    fitness = imagemFuncao - np.min(imagemFuncao)
    soma_fitness = np.sum(fitness)
    if soma_fitness == 0:
        probabilidade = np.ones(len(fitness)) / len(fitness)
    else:
        probabilidade = fitness / soma_fitness
    return probabilidade

# Selecionar os melhores indivíduos com base na roleta
def girarRoleta(probabRolet, popBin, quantidade):
    indices = np.random.choice(np.arange(len(popBin)), size=quantidade, p=probabRolet)
    return popBin[indices]

# Sortear casais para o cruzamento
def sortearCasais(sMelhores):
    np.random.shuffle(sMelhores)
    return [(sMelhores[i], sMelhores[i+1]) for i in range(0, len(sMelhores)-1, 2)]

# Gerar um ponto de corte aleatório para o cruzamento
def gerarPontoDeCorte(bits):
    return np.random.randint(1, bits)

# Recombinar para criar filhos
def recombinar(pontoCorte, combinacao, prob_cruzamento):
    s_filhos = []
    for pai, mae in combinacao:
        if np.random.rand() < prob_cruzamento:
            filho1 = pai[:pontoCorte] + mae[pontoCorte:]
            filho2 = mae[:pontoCorte] + pai[pontoCorte:]
            s_filhos.extend([filho1, filho2])
        else:
            s_filhos.extend([pai, mae])
    return np.array(s_filhos)

# Efetuar mutação nos filhos
def efetuarMutacao(s_filhos, prob_mutacao):
    for i in range(len(s_filhos)):
        if np.random.rand() < prob_mutacao:
            ponto_mutacao = np.random.randint(0, len(s_filhos[i]))
            s_filhos[i] = s_filhos[i][:ponto_mutacao] + str(1 - int(s_filhos[i][ponto_mutacao])) + s_filhos[i][ponto_mutacao+1:]
    return s_filhos

# Misturar novos filhos e antigos para criar a nova população
def misturarNovosEAntigos(sMelhores, popBin, s_filhos):
    return np.concatenate((sMelhores, s_filhos))

# Converter elementos binários em decimais
def novosValoresDecimais(popBin):
    return np.array([int(element, 2) for element in popBin])

# Parâmetros do algoritmo
tamanho_populacao = 20
intervalo = [0, 512]
bits = 10
geracoes = 100
prob_cruzamento = 0.7
prob_mutacao = 0.01
quantidade_melhores = 10

# Algoritmo Genético
popDec = gerarElementos(tamanho_populacao, intervalo)
for _ in range(geracoes):
    popBin = gerarElementosBinarios(popDec, bits)
    imagemFuncao = gerarImagem(popDec)
    probabRolet = gerarProbabilidade(imagemFuncao)
    sMelhores = girarRoleta(probabRolet, popBin, quantidade_melhores)
    combinacao = sortearCasais(sMelhores)
    pontoCorte = gerarPontoDeCorte(bits)
    s_filhos = recombinar(pontoCorte, combinacao, prob_cruzamento)
    s_filhos = efetuarMutacao(s_filhos, prob_mutacao)
    popBin = misturarNovosEAntigos(sMelhores, popBin, s_filhos)
    popDec = novosValoresDecimais(popBin)

# Resultado final
imagemFuncao_final = gerarImagem(popDec)
x_min = popDec[np.argmin(imagemFuncao_final)]
print(f"O valor de x que minimiza a função é: {x_min}")
print(f"Valor mínimo da função: {funcao(x_min)}")
