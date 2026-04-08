import numpy as np
import pandas as pd


def calcular_volatilidade_historica(precos_historicos, janela=20):


    retornos_log = np.log(precos_historicos / precos_historicos.shift(1))

    # 2. Variância/Desvio Padrão amostral da taxa de variação (Equação 3.2.2 do TCC)
    # O parâmetro ddof=1 divide por (n-1), garantindo um estimador não viesado
    s = retornos_log.rolling(window=janela).std(ddof=1)

    # 3. Anualização da volatilidade histórica (Equação 3.2.3 do TCC)
    # Assume-se a existência de 252 dias de negociação por ano
    sigma_hist = s * np.sqrt(252)

    return sigma_hist