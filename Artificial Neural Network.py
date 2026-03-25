# ============================================================
# 0. IMPORTS E CONFIGURAÇÕES
# ============================================================

import numpy as np
import pandas as pd
import yfinance as yf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset  # [SUGESTÃO 5f] Mini-batches

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error  # [SUGESTÃO 5b]
from sklearn.linear_model import LinearRegression  # [SUGESTÃO 5a] Benchmark simples

from scipy.stats import norm
import matplotlib.pyplot as plt
import copy

from volatilidade_historica import calcular_volatilidade_historica

# ============================================================
# 1. LEITURA DA BASE E CÁLCULO DA VOLATILIDADE
# ============================================================

df = pd.read_csv("base_black_scholes_petr4_2025.csv")
df = df[df["tipo"] == "CALL"].reset_index(drop=True)
df['data'] = pd.to_datetime(df['data'])
df = df.sort_values(by='data').reset_index(drop=True)

print("Baixando histórico da PETR4 para cálculo da volatilidade...")
petr4 = yf.download("PETR4.SA", start="2024-11-01", end="2025-12-31", progress=False)['Close']
petr4 = petr4.reset_index()

if isinstance(petr4.columns, pd.MultiIndex):
    petr4.columns = petr4.columns.get_level_values(0)

petr4.columns = ['data', 'S_hist']
petr4['data'] = pd.to_datetime(petr4['data']).dt.tz_localize(None)

petr4['sigma_hist'] = calcular_volatilidade_historica(petr4['S_hist'], janela=20)

df = pd.merge(df, petr4[['data', 'sigma_hist']], on='data', how='left')
df['sigma_hist'] = df['sigma_hist'].ffill()


# ============================================================
# 2 E 3. CLASSIFICAÇÃO OTM / ATM / ITM E ONE-HOT ENCODING
# ============================================================

def classify_moneyness(row, eps=0.02):
    if abs(row["S"] - row["K"]) / row["K"] <= eps:
        return "ATM"
    elif row["S"] > row["K"]:
        return "ITM"
    else:
        return "OTM"


df["moneyness"] = df.apply(classify_moneyness, axis=1)
df = pd.get_dummies(df, columns=["moneyness"])

# ============================================================
# 4. DEFINIÇÃO DAS VARIÁVEIS
# ============================================================

df['r'] = 0.10
colunas_cont = ["S", "K", "T", "sigma_hist", "r"]
X_cont = df[colunas_cont].values
colunas_cat = ["moneyness_ITM", "moneyness_ATM", "moneyness_OTM"]
X_cat = df[colunas_cat].astype(float).values
y = df["C"].values.reshape(-1, 1)

# ============================================================
# 5. DIVISÃO TEMPORAL E PADRONIZAÇÃO
# ============================================================

indices = np.arange(len(df))
idx_train, idx_temp = train_test_split(indices, test_size=0.2, shuffle=False)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, shuffle=False)

df_test = df.iloc[idx_test].copy()

X_cont_train, X_cont_val, X_cont_test = X_cont[idx_train], X_cont[idx_val], X_cont[idx_test]
X_cat_train, X_cat_val, X_cat_test = X_cat[idx_train], X_cat[idx_val], X_cat[idx_test]
y_train_raw, y_val_raw, y_test_raw = y[idx_train], y[idx_val], y[idx_test]

scaler_X = StandardScaler()
X_cont_train_scaled = scaler_X.fit_transform(X_cont_train)
X_cont_val_scaled = scaler_X.transform(X_cont_val)
X_cont_test_scaled = scaler_X.transform(X_cont_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train_raw)
y_val_scaled = scaler_y.transform(y_val_raw)
y_test_scaled = scaler_y.transform(y_test_raw)

X_train_final = np.hstack([X_cont_train_scaled, X_cat_train])
X_val_final = np.hstack([X_cont_val_scaled, X_cat_val])
X_test_final = np.hstack([X_cont_test_scaled, X_cat_test])

# ============================================================
# 6. CONVERSÃO PARA TENSORES E DATALOADERS [SUGESTÃO 5f]
# ============================================================

X_train_tensor = torch.tensor(X_train_final, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_final, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_final, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# Usando DataLoader para processamento em lotes (Batch Size de 64)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# ============================================================
# 7. ARQUITETURA DA RNA (COM REGULARIZAÇÃO) [SUGESTÃO 5d]
# ============================================================

class OptionPricingNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


model = OptionPricingNN()

# ============================================================
# 8. TREINAMENTO COM EARLY STOPPING E L2 PENALTY [SUGESTÃO 5d]
# ============================================================

criterion = nn.MSELoss()
# Adicionado weight_decay para Regularização L2
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

epochs = 200
patience = 30  # Early stopping patience
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_wts = copy.deepcopy(model.state_dict())

print("\nIniciando treinamento da RNA...")
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_X.size(0)

    train_loss /= len(train_loader.dataset)

    # Validação
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_tensor)
        val_loss = criterion(val_pred, y_val_tensor).item()

    # Early Stopping Logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epoch % 50 == 0 or epochs_no_improve == patience:
        print(f"Epoch {epoch:3d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

    if epochs_no_improve == patience:
        print(f"Early stopping acionado na época {epoch}!")
        break

# Carrega os melhores pesos encontrados na validação
model.load_state_dict(best_model_wts)


# ============================================================
# 9. RESULTADOS GLOBAIS: BENCHMARKS (RNA vs Regressão vs BS)
# ============================================================

def safe_mape(y_true, y_pred):
    # MAPE seguro que lida com zeros colocando um epsilon mínimo
    return mean_absolute_percentage_error(y_true + 1e-8, y_pred) * 100


# PREDIÇÕES DA RNA
model.eval()
with torch.no_grad():
    test_pred_scaled = model(X_test_tensor)
test_pred_rna = scaler_y.inverse_transform(test_pred_scaled.numpy()).flatten()

# PREDIÇÕES DA REGRESSÃO LINEAR [SUGESTÃO 5a]
lr_model = LinearRegression()
lr_model.fit(X_train_final, y_train_raw.flatten())
test_pred_lr = lr_model.predict(X_test_final)


# PREDIÇÕES DO BLACK-SCHOLES
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


df_test["C_BS"] = black_scholes_call(
    df_test["S"].values, df_test["K"].values, df_test["T"].values,
    df_test["r"].values, df_test["sigma_hist"].values
)
test_pred_bs = df_test["C_BS"].values
y_true = df_test["C"].values

# CALCULANDO MÉTRICAS GLOBAIS [SUGESTÃO 5b]
print("\n=== COMPARAÇÃO DE MODELOS (CONJUNTO DE TESTE) ===")
for name, preds in [("Rede Neural", test_pred_rna), ("Reg. Linear Múltipla", test_pred_lr),
                    ("Black-Scholes", test_pred_bs)]:
    mse = mean_squared_error(y_true, preds)
    mae = mean_absolute_error(y_true, preds)
    mape = safe_mape(y_true, preds)
    print(f"[{name}] MSE: {mse:.4f} | MAE: R$ {mae:.4f} | MAPE: {mape:.2f}%")

df_test["C_RNA"] = test_pred_rna


# ============================================================
# 10. SEGMENTAÇÃO POR MONEYNESS E MATURIDADE [SUGESTÃO 5c]
# ============================================================

def calc_metrics(df_subset, true_col, pred_col):
    if len(df_subset) == 0: return np.nan, np.nan, np.nan
    y_t = df_subset[true_col].values
    y_p = df_subset[pred_col].values
    return mean_squared_error(y_t, y_p), mean_absolute_error(y_t, y_p), safe_mape(y_t, y_p)


# Segmentação de Maturidade (Exemplo: Curta < 0.25 anos, Media, Longa > 0.75 anos)
df_test['maturidade'] = pd.cut(df_test['T'], bins=[-np.inf, 0.25, 0.75, np.inf], labels=['Curta', 'Média', 'Longa'])

print("\n=== DESEMPENHO DA RNA POR MONEYNESS E MATURIDADE ===")
for moneyness in ["ITM", "ATM", "OTM"]:
    for mat in ["Curta", "Média", "Longa"]:
        subset = df_test[(df_test[f"moneyness_{moneyness}"] == 1) & (df_test['maturidade'] == mat)]
        mse, mae, mape = calc_metrics(subset, "C", "C_RNA")
        if not np.isnan(mse):
            print(
                f"[{moneyness} | {mat:<5}] N={len(subset):<4} | MSE: {mse:.4f} | MAE: R$ {mae:.4f} | MAPE: {mape:.2f}%")

# ============================================================
# 11. ANÁLISE DE SENSIBILIDADE (AS GREGAS DA REDE) [SUGESTÃO 5e]
# ============================================================

print("\n--- CALCULANDO A ANÁLISE DE SENSIBILIDADE (DELTA EMPÍRICO) ---")

# Vamos fixar todas as variáveis nas suas medianas do conjunto de teste
median_K = np.median(df_test["K"])
median_T = np.median(df_test["T"])
median_sigma = np.median(df_test["sigma_hist"])
median_r = 0.10

# Variar o preço da Ação (S) artificialmente de 50% do Strike até 150% do Strike
S_range = np.linspace(median_K * 0.5, median_K * 1.5, 100)

sens_df = pd.DataFrame({
    "S": S_range,
    "K": median_K,
    "T": median_T,
    "sigma_hist": median_sigma,
    "r": median_r
})

# Recalcular One-Hot para o dataframe de sensibilidade
sens_df["moneyness"] = sens_df.apply(classify_moneyness, axis=1)
sens_df = pd.get_dummies(sens_df, columns=["moneyness"])
for col in ["moneyness_ITM", "moneyness_ATM", "moneyness_OTM"]:
    if col not in sens_df: sens_df[col] = 0

# Padronizar o vetor
X_cont_sens = scaler_X.transform(sens_df[["S", "K", "T", "sigma_hist", "r"]].values)
X_cat_sens = sens_df[["moneyness_ITM", "moneyness_ATM", "moneyness_OTM"]].astype(float).values
X_sens_final = np.hstack([X_cont_sens, X_cat_sens])

X_sens_tensor = torch.tensor(X_sens_final, dtype=torch.float32)

model.eval()
with torch.no_grad():
    C_pred_scaled = model(X_sens_tensor)

C_pred_sens = scaler_y.inverse_transform(C_pred_scaled.numpy()).flatten()

# Custo Intrínseco Teórico: max(S - K, 0)
C_teorico = np.maximum(S_range - median_K, 0)

plt.figure(figsize=(8, 5))
plt.plot(S_range, C_pred_sens, label="Preço Estimado RNA", color="blue", linewidth=2)
plt.plot(S_range, C_teorico, label="Valor Intrínseco (S - K)", color="black", linestyle="--")
plt.axvline(x=median_K, color='red', linestyle=':', label="Preço de Exercício (ATM)")
plt.xlabel("Preço da Ação (S)")
plt.ylabel("Preço Estimado da Opção (C)")
plt.title("Análise de Sensibilidade da RNA (Preço da Ação vs Preço da Opção)")
plt.legend()
plt.grid(alpha=0.4)
plt.show()

# ============================================================
# 12. GRÁFICOS DIAGNÓSTICOS (IGUAL AO ANTERIOR)
# ============================================================
# (Coloque aqui os seus plot.scatter de erros que já estavam no código!)

# ============================================================
# 11. GRÁFICOS DIAGNÓSTICOS POR MONEINESS
# ============================================================


# ============================================================
# 12. GRÁFICOS DIAGNÓSTICOS POR MONEYNESS E TEMPO
# ============================================================

# --- LINHAS CORRIGIDAS: Garantindo que a coluna exista ---
df_test["C_RNA"] = test_pred_rna
df_test["erro_abs"] = np.abs(df_test["C"] - df_test["C_RNA"])
# ---------------------------------------------------------

# Gráfico 1: Erro absoluto x Strike
plt.figure(figsize=(8, 5))
for label in ["ITM", "ATM", "OTM"]:
    subset = df_test[df_test[f"moneyness_{label}"] == 1]
    plt.scatter(subset["K"], subset["erro_abs"], label=label, alpha=0.6)
plt.xlabel("Strike (K)")
plt.ylabel("Erro absoluto |C - Ĉ|")
plt.title("Erro da RNA por Moneyness e Strike")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Gráfico 2: Preço Real vs Preço Estimado
plt.figure(figsize=(8, 5))
for label in ["ITM", "ATM", "OTM"]:
    subset = df_test[df_test[f"moneyness_{label}"] == 1]
    plt.scatter(subset["C"], subset["C_RNA"], label=label, alpha=0.6)
min_c, max_c = df_test["C"].min(), df_test["C"].max()
plt.plot([min_c, max_c], [min_c, max_c], color='black', linestyle='--', label="Linha Ideal")
plt.xlabel("Preço real da opção (C)")
plt.ylabel("Preço estimado pela RNA (Ĉ)")
plt.title("RNA: Preço Real vs Estimado")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Gráfico 3: Erro absoluto x Tempo
plt.figure(figsize=(8, 5))
for label in ["ITM", "ATM", "OTM"]:
    subset = df_test[df_test[f"moneyness_{label}"] == 1]
    plt.scatter(subset["T"], subset["erro_abs"], label=label, alpha=0.6)
plt.xlabel("Tempo até vencimento (T)")
plt.ylabel("Erro absoluto |C - Ĉ|")
plt.title("Erro da RNA vs Tempo até Vencimento")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
# ============================================================
# 12. IMPORTÂNCIA DAS VARIÁVEIS (PERMUTATION IMPORTANCE)
# ============================================================
import copy

print("\n--- CALCULANDO A IMPORTÂNCIA DAS VARIÁVEIS ---")

# Nomes das 8 variáveis na mesma ordem em que foram empilhadas no X
feature_names = [
    "Preço Ação (S)",
    "Strike (K)",
    "Tempo (T)",
    "Volatilidade (sigma_hist)",
    "Taxa Juros (r)",
    "ITM",
    "ATM",
    "OTM"
]


def calculate_permutation_importance(model, X_tensor, y_tensor, criterion, feature_names):
    model.eval()

    # 1. Calcula o erro base (sem embaralhar nada)
    with torch.no_grad():
        base_pred = model(X_tensor)
        base_loss = criterion(base_pred, y_tensor).item()

    importances = []

    # 2. Itera sobre cada coluna (feature)
    for i in range(X_tensor.shape[1]):
        X_permuted = X_tensor.clone()

        # Embaralha apenas a coluna 'i'
        indices_embaralhados = torch.randperm(X_tensor.shape[0])
        X_permuted[:, i] = X_tensor[indices_embaralhados, i]

        # 3. Calcula o erro com a coluna embaralhada
        with torch.no_grad():
            permuted_pred = model(X_permuted)
            permuted_loss = criterion(permuted_pred, y_tensor).item()

        # A importância é o quanto o erro aumentou em relação ao erro base
        # (Permuted Loss / Base Loss). Quanto maior, mais importante.
        aumento_erro = permuted_loss / base_loss
        importances.append(aumento_erro)

    return importances


# Vamos calcular a importância usando o conjunto de VALIDAÇÃO ou TESTE
importances = calculate_permutation_importance(model, X_test_tensor, y_test_tensor, criterion, feature_names)

# Organizar os resultados do maior para o menor
importance_df = pd.DataFrame({
    'Variavel': feature_names,
    'Aumento_MSE': importances
}).sort_values(by='Aumento_MSE', ascending=True)

# Plotando o gráfico de importância
plt.figure(figsize=(10, 6))
bars = plt.barh(importance_df['Variavel'], importance_df['Aumento_MSE'], color='skyblue', edgecolor='black')

# Adicionando uma linha de base (1.0 = o erro não mudou)
plt.axvline(x=1.0, color='red', linestyle='--', label='Nenhuma mudança no Erro')

plt.xlabel("Aumento relativo do MSE (x vezes)")
plt.ylabel("Variável de Entrada")
plt.title("Importância das Variáveis (Permutation Importance)")
plt.legend()
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Adicionando os valores nas barras para facilitar a leitura no TCC
for bar in bars:
    plt.text(
        bar.get_width() + 0.05,
        bar.get_y() + bar.get_height() / 2,
        f"{bar.get_width():.2f}x",
        va='center'
    )

plt.tight_layout()
plt.show()