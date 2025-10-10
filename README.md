# Análise Preditiva e Espacial do Risco de Dengue no Brasil

## 1. Visão Geral

Este repositório contém os dados e códigos para uma análise abrangente sobre o risco de dengue nos municípios brasileiros. O projeto integra dados epidemiológicos, climáticos e socioeconômicos para construir modelos preditivos, identificar padrões espaciais e caracterizar perfis de risco municipais.

A metodologia é dividida em três etapas principais:

-   **Modelagem Preditiva**: Utilização de redes neurais **Long Short-Term Memory (LSTM)** para prever o risco anual de dengue.
-   **Análise Espacial**: Investigação de clusters espaciais de risco com as técnicas de **I de Moran** e **LISA**.
-   **Clusterização**: Agrupamento de municípios com perfis de risco similares utilizando o algoritmo **K-Means**.

## 2. Estrutura do Repositório

O projeto está organizado nas seguintes pastas:

```
├── .venv/
├── code/
│   ├── Python/
│   └── R/
├── data/
│   ├── Python/
│   └── R/
├── flow/
└── results/
```

### Detalhamento das Pastas

📂 **`.venv/`**: Diretório do ambiente virtual Python. Devido ao tamanho, não foi incluído no repositório; o autor pode disponibilizá-lo mediante solicitação.

📂 **`code/`**: Contém todos os scripts utilizados na análise.
-   `code/Python/`: Scripts em Python para pré-processamento, treinamento do modelo LSTM e clusterização.
-   `code/R/`: Scripts em R para análise espacial e visualização de mapas.

📂 **`data/`**: Contém os conjuntos de dados brutos e intermediários.
-   `data/Python/`: Datasets primários utilizados na modelagem.
-   `data/R/`: Datasets secundários, gerados pela modelagem e utilizados na análise espacial.

📂 **`flow/`**: Contém os fluxogramas e diagramas que ilustram a metodologia do projeto.

📂 **`results/`**: Contém todos os resultados gerados, como tabelas, métricas de modelo e figuras.

## 3. Conjunto de Dados (`/data`)

#### 🔹 `/data/Python`
Este diretório armazena os dados primários, organizados por município.

-   `ds1.csv`: **Dados Epidemiológicos**. Contém as séries históricas de casos de dengue por município.
-   `ds2.csv`: **Dados Climáticos e Geográficos**. Inclui variáveis como temperatura, precipitação e classificação climática de Köppen.
-   `ds3.csv`: **Dados Socioeconômicos e de Infraestrutura**. Engloba indicadores como PIB per capita, saneamento básico, coleta de lixo, acesso à informação e adequação de moradia.

#### 🔹 `/data/R`
Este diretório contém os outputs da fase de modelagem, que servem de input para a análise espacial.

-   `clu.csv`: **Resultados da Clusterização**. Contém o ID (`cod6`) de cada município e o cluster de risco ao qual foi associado pelo algoritmo K-Means.
-   `prb.csv`: **Probabilidades Previstas**. Arquivo com as probabilidades de alto risco para cada município, geradas anualmente pelo modelo LSTM.

## 4. Scripts (`/code`)

#### 🔹 `/code/Python`
Scripts para a modelagem preditiva e clusterização.

-   `code1`: **Pré-processamento e Modelagem**. Script principal que realiza a carga, limpeza e junção dos dados (`ds1`, `ds2`, `ds3`). Implementa o treinamento do modelo LSTM, a seleção de variáveis com SHAP, a validação cruzada (k=5) e a geração das probabilidades de risco.
-   `code2`: **Análises Anuais e de Cluster**. Script que utiliza os resultados do `code1` para análises mais aprofundadas. Gera a "pirâmide de AUC" (distribuição da performance do modelo para cada ano) e realiza a análise dos perfis de risco por cluster.
-   `code3`: **Classificação de Risco**. Script auxiliar que classifica os municípios em faixas de risco (e.g., "muito baixo", "alto") com base na probabilidade média prevista, gerando um sumário quantitativo.

#### 🔹 `/code/R`
Scripts focados na análise geoespacial e na criação de mapas temáticos.

-   `clusters.R`: Gera os mapas que ilustram a distribuição geográfica dos clusters de risco identificados pelo K-Means.
-   `lisa.R`: Executa a análise de autocorrelação espacial. Calcula o I de Moran global e o LISA para identificar hotspots e coldspots de risco de dengue.
-   `maps.R`: Cria os mapas coropléticos que exibem o nível de risco previsto para cada município no Brasil.
-   `panel_southeast.R`: Script específico para a região Sudeste, que identifica as variáveis mais correlacionadas com o risco nos municípios de maior criticidade.
-   `southeast.R`: Gera os arquivos de dados detalhados para a análise da região Sudeste.
-   `southeast_maps.R`: Produz os mapas de risco detalhados para cada estado da região Sudeste (MG, ES, RJ, SP).
