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

# Selecionar os melhores indivíduos com base no torneio
def torneio(popBin, imagemFuncao, tamanho_torneio):
    sMelhores = []
    for _ in range(len(popBin)):
        competidores = np.random.choice(np.arange(len(popBin)), size=tamanho_torneio, replace=False)
        vencedor = competidores[np.argmin(imagemFuncao[competidores])]
        sMelhores.append(popBin[vencedor])
    return np.array(sMelhores)

# Sortear casais para o cruzamento
def sortearCasais(sMelhores):
    np.random.shuffle(sMelhores)
    return [(sMelhores[i], sMelhores[i+1]) for i in range(0, len(sMelhores)-1, 2)]

# Gerar um ou dois pontos de corte aleatórios para o cruzamento
def gerarPontosDeCorte(bits, tipo_cruzamento):
    if tipo_cruzamento == 1:
        return [np.random.randint(1, bits)]
    elif tipo_cruzamento == 2:
        ponto1 = np.random.randint(1, bits)
        ponto2 = np.random.randint(1, bits)
        while ponto2 == ponto1:
            ponto2 = np.random.randint(1, bits)
        return sorted([ponto1, ponto2])

# Recombinar para criar filhos
def recombinar(pontosCorte, combinacao, prob_cruzamento, tipo_cruzamento):
    s_filhos = []
    for pai, mae in combinacao:
        if np.random.rand() < prob_cruzamento:
            if tipo_cruzamento == 1:
                pontoCorte = pontosCorte[0]
                filho1 = pai[:pontoCorte] + mae[pontoCorte:]
                filho2 = mae[:pontoCorte] + pai[pontoCorte:]
            elif tipo_cruzamento == 2:
                ponto1, ponto2 = pontosCorte
                filho1 = pai[:ponto1] + mae[ponto1:ponto2] + pai[ponto2:]
                filho2 = mae[:ponto1] + pai[ponto1:ponto2] + mae[ponto2:]
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
def misturarNovosEAntigos(popBin, s_filhos):
    return s_filhos

# Converter elementos binários em decimais
def novosValoresDecimais(popBin):
    return np.array([int(element, 2) for element in popBin])

# Parâmetros do algoritmo solicitados ao usuário
tamanho_cromossomo = int(input("Tamanho do cromossomo (em bits): "))
tamanho_populacao = int(input("Tamanho da população: "))
quantidade_geracoes = int(input("Quantidade máxima de gerações: "))
porcentagem_cruzamento = float(input("Porcentagem de integrantes da população para cruzamento (0-1): "))
prob_mutacao = float(input("Probabilidade de mutação (0-1): "))
metodo_selecao = input("Método de seleção (roleta/torneio): ")
if metodo_selecao.lower() == 'torneio':
    tamanho_torneio = int(input("Tamanho do torneio: "))
tipo_cruzamento = int(input("Cruzamento (1 ou 2 pontos): "))

# Probabilidade de cruzamento
prob_cruzamento = porcentagem_cruzamento

# Algoritmo Genético
popDec = gerarElementos(tamanho_populacao, [0, 512])
for _ in range(quantidade_geracoes):
    popBin = gerarElementosBinarios(popDec, tamanho_cromossomo)
    imagemFuncao = gerarImagem(popDec)
    if metodo_selecao.lower() == 'roleta':
        probabRolet = gerarProbabilidade(imagemFuncao)
        sMelhores = girarRoleta(probabRolet, popBin, tamanho_populacao)
    elif metodo_selecao.lower() == 'torneio':
        sMelhores = torneio(popBin, imagemFuncao, tamanho_torneio)
    combinacao = sortearCasais(sMelhores)
    pontosCorte = gerarPontosDeCorte(tamanho_cromossomo, tipo_cruzamento)
    s_filhos = recombinar(pontosCorte, combinacao, prob_cruzamento, tipo_cruzamento)
    s_filhos = efetuarMutacao(s_filhos, prob_mutacao)
    popBin = misturarNovosEAntigos(popBin, s_filhos)
    popDec = novosValoresDecimais(popBin)

# Resultado final
imagemFuncao_final = gerarImagem(popDec)
x_min = popDec[np.argmin(imagemFuncao_final)]
print(f"O valor de x que minimiza a função é: {x_min}")
print(f"Valor mínimo da função: {funcao(x_min)}")
