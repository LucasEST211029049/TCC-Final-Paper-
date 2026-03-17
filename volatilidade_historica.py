import numpy as np
import pandas as pd


def calcular_volatilidade_historica(precos_historicos, janela=20):
    """
    Calcula a volatilidade histórica anualizada conforme a metodologia de Hull (2018).

    Parâmetros:
    precos_historicos (pd.Series): Série de preços diários de fechamento do ativo-objeto.
    janela (int): Número de dias para a janela de volatilidade (padrão de 20 dias, ref: Ke e Yang).

    Retorna:
    pd.Series: Volatilidade histórica anualizada correspondente a cada dia.
    """

    # 1. Cálculo dos retornos logarítmicos contínuos (Equação 3.2.1 do TCC)
    retornos_log = np.log(precos_historicos / precos_historicos.shift(1))

    # 2. Variância/Desvio Padrão amostral da taxa de variação (Equação 3.2.2 do TCC)
    # O parâmetro ddof=1 divide por (n-1), garantindo um estimador não viesado
    s = retornos_log.rolling(window=janela).std(ddof=1)

    # 3. Anualização da volatilidade histórica (Equação 3.2.3 do TCC)
    # Assume-se a existência de 252 dias de negociação por ano
    sigma_hist = s * np.sqrt(252)

    return sigma_hist