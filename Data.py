import pandas as pd

arquivo = "COTAHIST_A2025.TXT"
# layout oficial B3 COTAHIST

colspecs = [
    (0, 2),
    (2, 10),
    (10, 12),
    (12, 24),
    (24, 27),
    (27, 39),
    (39, 49),
    (49, 52),
    (52, 56),
    (56, 69),
    (69, 82),
    (82, 95),
    (95, 108),
    (108, 121),
    (121, 134),
    (134, 147),
    (147, 152),
    (152, 170),
    (170, 188),
    (188, 201),
    (201, 202),
    (202, 210),
    (210, 217),
    (217, 230),
    (230, 242),
    (242, 245),
]

cols = [
    "tipo_registro",
    "data",
    "cod_bdi",
    "ticker",
    "tipo_mercado",
    "nome_empresa",
    "especificacao",
    "prazo_dias",
    "moeda",
    "preco_abertura",
    "preco_max",
    "preco_min",
    "preco_medio",
    "preco_ultimo",
    "preco_melhor_compra",
    "preco_melhor_venda",
    "num_negocios",
    "quantidade",
    "volume",
    "preco_exercicio",
    "indicador_correcao",
    "data_vencimento",
    "fator_cotacao",
    "preco_exercicio_pontos",
    "cod_isin",
    "distribuicao"
]

df = pd.read_fwf(
    arquivo,
    colspecs=colspecs,
    names=cols,
    encoding="latin1"
)

# remove header e trailer
df = df[df["tipo_registro"] == 1]

# datas
df["data"] = pd.to_datetime(df["data"], format="%Y%m%d")
df["data_vencimento"] = pd.to_datetime(df["data_vencimento"], format="%Y%m%d", errors="coerce")

# preços (B3 multiplica por 100)
for c in ["preco_ultimo","preco_abertura","preco_max","preco_min","preco_exercicio"]:
    df[c] = df[c] / 100

df.to_parquet("b3_2025.parquet")

