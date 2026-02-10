import pandas as pd
import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    """
    Preço de uma opção de compra europeia (Black-Scholes)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

df = pd.read_csv("base_black_scholes_petr4_2025.csv")
df = df[df["tipo"] == "CALL"].copy()

# Taxa livre de risco (exemplo: 10% a.a.)
r = 0.10

# Volatilidade constante (exemplo: 30% a.a.)
sigma_bs = 0.30
df_test["C_BS"] = black_scholes_call(
    S=df_test["S"].values,
    K=df_test["K"].values,
    T=df_test["T"].values,
    r=r,
    sigma=sigma_bs
)

from sklearn.metrics import mean_squared_error, mean_absolute_error

mse_bs = mean_squared_error(df["C"], df["C_BS"])
mae_bs = mean_absolute_error(df["C"], df["C_BS"])

print("Black-Scholes MSE:", mse_bs)
print("Black-Scholes MAE:", mae_bs)
