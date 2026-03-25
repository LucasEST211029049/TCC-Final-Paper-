import pandas as pd

df = pd.read_parquet("b3_2025.parquet")

# manter só registros válidos
df = df[df["tipo_registro"] == 1]
# tipo mercado 70 = opções
opcoes = df[df["tipo_mercado"] == 70].copy()

# apenas Petrobras
petr_opcoes = opcoes[opcoes["ticker"].str.startswith("PETR")].copy()

petr_opcoes["serie"] = petr_opcoes["ticker"].str[4]

calls_letters = list("ABCDEFGHIJKL")

petr_opcoes["tipo"] = petr_opcoes["serie"].apply(
    lambda x: "CALL" if x in calls_letters else "PUT"
)
petr_opcoes["C"] = petr_opcoes["preco_ultimo"]
petr_opcoes["K"] = petr_opcoes["preco_exercicio"]

spot = df[df["ticker"] == "PETR4"][["data","preco_ultimo"]].copy()
spot.rename(columns={"preco_ultimo":"S"}, inplace=True)

petr_opcoes = petr_opcoes.merge(spot, on="data", how="left")

petr_opcoes["T"] = (
    (petr_opcoes["data_vencimento"] - petr_opcoes["data"])
    .dt.days / 252
)

petr_opcoes = petr_opcoes[petr_opcoes["T"] > 0]

r = 0.105  # ~10,5% a.a. (SELIC média 2025 aproximada)

petr_opcoes = petr_opcoes[
    (petr_opcoes["C"] > 0.01) &
    (petr_opcoes["S"] > 1) &
    (petr_opcoes["K"] > 1)
]

petr_opcoes[["ticker","data","tipo","S","K","T","C"]].to_csv(
    "base_black_scholes_petr4_2025.csv",
    index=False
)
