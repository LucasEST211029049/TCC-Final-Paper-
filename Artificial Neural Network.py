# ============================================================
# 0. IMPORTS
# ============================================================

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from scipy.stats import norm
import matplotlib.pyplot as plt
# ============================================================
# 1. LEITURA DA BASE
# ============================================================

df = pd.read_csv("base_black_scholes_petr4_2025.csv")

# Apenas opções de compra
df = df[df["tipo"] == "CALL"].reset_index(drop=True)
# ============================================================
# 2. CLASSIFICAÇÃO OTM / ATM / ITM (CALL)
# ============================================================

def classify_moneyness(row, eps=0.02):
    if abs(row["S"] - row["K"]) / row["K"] <= eps:
        return "ATM"
    elif row["S"] > row["K"]:
        return "ITM"
    else:
        return "OTM"

df["moneyness"] = df.apply(classify_moneyness, axis=1)
# ============================================================
# 3. ONE-HOT ENCODING DA MONEINESS
# ============================================================

df = pd.get_dummies(df, columns=["moneyness"])

# ============================================================
# 4. DEFINIÇÃO DAS VARIÁVEIS
# ============================================================

# Variáveis contínuas
X_cont = df[["S", "K", "T"]].values

# Variáveis categóricas (não escalar!)
X_cat = df[
    ["moneyness_ITM", "moneyness_ATM", "moneyness_OTM"]
].values

# Padronização apenas das contínuas
scaler_X = StandardScaler()
X_cont_scaled = scaler_X.fit_transform(X_cont)

# Matriz final de entrada
X = np.hstack([X_cont_scaled, X_cat])

# Variável resposta
y = df["C"].values.reshape(-1, 1)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)
# ============================================================
# 5. DIVISÃO TREINO / VAL / TEST (COM ÍNDICES)
# ============================================================

indices = np.arange(len(df))

X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
    X, y_scaled, indices, test_size=0.2, random_state=42
)

X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
    X_temp, y_temp, idx_temp, test_size=0.5, random_state=42
)

df_test = df.iloc[idx_test].copy()
# ============================================================
# 6. CONVERSÃO PARA TENSORES
# ============================================================

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
# ============================================================
# 7. DEFINIÇÃO DA RNA (MLP COM TANH)
# ============================================================

class OptionPricingNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = OptionPricingNN()
# ============================================================
# 8. TREINAMENTO
# ============================================================

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 500

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)

        print(f"Epoch {epoch} | Train MSE: {loss.item():.4f} | Val MSE: {val_loss.item():.4f}")
# ============================================================
# 9. AVALIAÇÃO DA RNA (ESCALA ORIGINAL)
# ============================================================

model.eval()
with torch.no_grad():
    test_pred_scaled = model(X_test)

test_pred = scaler_y.inverse_transform(test_pred_scaled.numpy())
y_test_real = scaler_y.inverse_transform(y_test.numpy())

mse_rna = mean_squared_error(y_test_real, test_pred)
mae_rna = mean_absolute_error(y_test_real, test_pred)

print("\nRNA RESULTS")
print("MSE:", mse_rna)
print("MAE:", mae_rna)
# ============================================================
# 10. BLACK-SCHOLES (MESMO CONJUNTO DE TESTE)
# ============================================================

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

r = 0.10
sigma_bs = 0.30

df_test["C_BS"] = black_scholes_call(
    df_test["S"].values,
    df_test["K"].values,
    df_test["T"].values,
    r,
    sigma_bs
)

mse_bs = mean_squared_error(df_test["C"], df_test["C_BS"])
mae_bs = mean_absolute_error(df_test["C"], df_test["C_BS"])

print("\nBLACK-SCHOLES RESULTS")
print("MSE:", mse_bs)
print("MAE:", mae_bs)
# ============================================================
# 11. GRÁFICOS DIAGNÓSTICOS POR MONEINESS
# ============================================================

df_test["C_RNA"] = test_pred.flatten()
df_test["erro_abs"] = np.abs(df_test["C"] - df_test["C_RNA"])
plt.figure()

for label in ["ITM", "ATM", "OTM"]:
    subset = df_test[df_test[f"moneyness_{label}"] == 1]
    plt.scatter(subset["K"], subset["erro_abs"], label=label, alpha=0.6)

plt.xlabel("Strike (K)")
plt.ylabel("Erro absoluto |C - Ĉ|")
plt.title("Erro da RNA por moneyness")
plt.legend()
plt.show()
plt.figure()

for label in ["ITM", "ATM", "OTM"]:
    subset = df_test[df_test[f"moneyness_{label}"] == 1]
    plt.scatter(subset["C"], subset["C_RNA"], label=label, alpha=0.6)

min_c = df_test["C"].min()
max_c = df_test["C"].max()
plt.plot([min_c, max_c], [min_c, max_c])

plt.xlabel("Preço real da opção")
plt.ylabel("Preço estimado pela RNA")
plt.title("RNA: Preço real vs estimado por moneyness")
plt.legend()
plt.show()
plt.figure()

for label in ["ITM", "ATM", "OTM"]:
    subset = df_test[df_test[f"moneyness_{label}"] == 1]
    plt.scatter(subset["T"], subset["erro_abs"], label=label, alpha=0.6)

plt.xlabel("Tempo até vencimento (T)")
plt.ylabel("Erro absoluto")
plt.title("Erro da RNA × Tempo até vencimento")
plt.legend()
plt.show()
