import pandas as pd

df = pd.read_csv("results/prb.csv")  # já tem colunas: cod6, municipio, year, prob

# Pegue o valor médio de risco por município (média das probabilidades em todos anos)
medias = df.groupby("cod6")["prob"].mean().reset_index()

# Defina as faixas (bins)
bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
labels = ["muito_baixo", "baixo", "medio", "alto", "muito_alto"]

# Classifique cada município
medias["faixa"] = pd.cut(medias["prob"], bins=bins, labels=labels, include_lowest=True)

# Conte quantos municípios por faixa
contagem = medias["faixa"].value_counts(sort=False).reset_index()
contagem.columns = ["faixa", "n_municipios"]

# Salve resultado
contagem.to_csv("results/faixa.csv", index=False)

print(contagem)