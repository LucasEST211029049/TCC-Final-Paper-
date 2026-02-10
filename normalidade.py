import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from scipy.stats import skew, kurtosis, jarque_bera
# 1. Carregue seus dados históricos da AÇÃO (não das opções)
# Supondo que você tenha um csv com datas e preços de fechamento
# df_acao = pd.read_csv('historico_petr4.csv')
# Ou baixe agora se não tiver:
import yfinance as yf
df_acao = yf.download('PETR4.SA', start='2025-01-01', end='2025-12-31')['Close']

# 2. Calcular os Retornos Logarítmicos (Conforme Eq 3.2.1 do seu PDF)
# u_i = ln(S_i / S_{i-1})
log_returns = np.log(df_acao / df_acao.shift(1)).dropna()

# 3. Visualização Gráfica (Histograma vs Curva Normal)
plt.figure(figsize=(10, 6))
sns.histplot(log_returns, kde=True, stat="density", label="Dados Reais", color="blue")

# Gerar curva normal teórica com mesma média e desvio padrão
mu, std = stats.norm.fit(log_returns)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
plt.plot(x, p, 'r', linewidth=2, label="Distribuição Normal (Teórica)")

plt.title("Distribuição dos Retornos Logarítmicos vs Normal Teórica")
plt.legend()
plt.show()


# 5. Testes Estatísticos Formais
shapiro_stat, shapiro_p = stats.shapiro(log_returns)
jarque_bera_stat, jarque_bera_p = stats.jarque_bera(log_returns)
ks_stat, ks_p = stats.kstest(log_returns, 'norm', args=(mu, std))

print(f"--- Resultados dos Testes de Normalidade ---")
print(f"Shapiro-Wilk: p-valor = {shapiro_p:.5f} (H0: É Normal)")
print(f"Jarque-Bera: p-valor = {jarque_bera_p:.5f} (H0: É Normal)")


if shapiro_p < 0.05:
    print("\nCONCLUSÃO: Rejeitamos a hipótese nula. Os dados NÃO seguem uma distribuição Normal.")
    print("Isso justifica o uso de Redes Neurais, que não exigem essa premissa!")
else:
    print("\--- Resultados dos Testes de Normalidade ---")


