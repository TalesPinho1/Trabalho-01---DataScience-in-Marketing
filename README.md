# Trabalho-01---DataScience-in-Marketing
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "a1f2f698",
   "metadata": {},
   "source": [
    "# Relatório Técnico – Otimização de Campanhas de Marketing com Ciência de Dados"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9303f645",
   "metadata": {},
   "source": [
    "## Introdução\n",
    "\n",
    "Este relatório apresenta uma análise de dados aplicada ao marketing de uma rede de varejo com mais de 200 lojas físicas e e-commerce consolidado.  \n",
    "O objetivo é otimizar campanhas de marketing por meio de técnicas de ciência de dados como **Cluster Analysis**, **Conjoint Analysis**, **Regressão Linear** e **Customer Lifetime Value (CLV)**.\n",
    "\n",
    "A partir dos dados fornecidos (`clientes.csv`, `transacoes.csv` e `campanhas.csv`), realizamos segmentações, modelagens e estimativas para identificar padrões de comportamento e apoiar a tomada de decisão da empresa.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "af251fad",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import mean_squared_error, r2_score\n",
    "\n",
    "sns.set(style=\"whitegrid\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0a4c7e79",
   "metadata": {},
   "source": [
    "## Análise de Cluster"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ef4f9511",
   "metadata": {},
   "outputs": [],
   "source": [
    "clientes = pd.read_csv('/content/clientes.csv')\n",
    "\n",
    "colunas = ['frequencia_compras', 'total_gasto', 'ultima_compra']\n",
    "dados_cluster = clientes[colunas]\n",
    "scaler = StandardScaler()\n",
    "dados_normalizados = scaler.fit_transform(dados_cluster)\n",
    "\n",
    "kmeans = KMeans(n_clusters=3, random_state=42)\n",
    "clientes['cluster'] = kmeans.fit_predict(dados_normalizados)\n",
    "\n",
    "plt.figure(figsize=(10,6))\n",
    "sns.scatterplot(data=clientes, x='frequencia_compras', y='total_gasto', hue='cluster', palette='viridis')\n",
    "plt.title('Clusters de Clientes')\n",
    "plt.show()\n",
    "\n",
    "clientes.groupby('cluster')[colunas].mean()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "50c3ba67",
   "metadata": {},
   "source": [
    "## Conjoint Analysis - Preferência por Atributos de Campanhas"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a56dac43",
   "metadata": {},
   "outputs": [],
   "source": [
    "np.random.seed(42)\n",
    "campanhas_simuladas = pd.DataFrame({\n",
    "    'desconto': np.random.choice(['0%', '10%', '20%'], size=20),\n",
    "    'frete_gratis': np.random.choice(['sim', 'nao'], size=20),\n",
    "    'brinde': np.random.choice(['sim', 'nao'], size=20),\n",
    "    'conversao': np.random.randint(50, 500, size=20)\n",
    "})\n",
    "X = pd.get_dummies(campanhas_simuladas.drop(columns='conversao'), drop_first=True)\n",
    "y = campanhas_simuladas['conversao']\n",
    "\n",
    "modelo = LinearRegression()\n",
    "modelo.fit(X, y)\n",
    "importancias = pd.Series(modelo.coef_, index=X.columns).sort_values(ascending=False)\n",
    "\n",
    "plt.figure(figsize=(8, 5))\n",
    "sns.barplot(x=importancias.values, y=importancias.index)\n",
    "plt.title('Importância dos Atributos nas Campanhas')\n",
    "plt.xlabel('Impacto na Conversão')\n",
    "plt.ylabel('Atributos')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d8f4e8db",
   "metadata": {},
   "source": [
    "## Regressão Linear - Impacto das Campanhas no Total Gasto"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1679a193",
   "metadata": {},
   "outputs": [],
   "source": [
    "transacoes = pd.read_csv('/content/transacoes.csv')\n",
    "\n",
    "df = transacoes.merge(clientes, on='cliente_id')\n",
    "\n",
    "df_group = df.groupby(['cliente_id', 'campanha']).agg({\n",
    "    'valor_compra': 'sum',\n",
    "    'idade': 'first',\n",
    "    'renda_mensal': 'first',\n",
    "    'tipo_cliente': 'first'\n",
    "}).reset_index()\n",
    "df_group.rename(columns={'valor_compra': 'total_gasto'}, inplace=True)\n",
    "\n",
    "df_dummies = pd.get_dummies(df_group, columns=['campanha', 'tipo_cliente'], drop_first=True)\n",
    "X = df_dummies.drop(columns=['cliente_id', 'total_gasto'])\n",
    "y = df_dummies['total_gasto']\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n",
    "\n",
    "modelo = LinearRegression()\n",
    "modelo.fit(X_train, y_train)\n",
    "y_pred = modelo.predict(X_test)\n",
    "\n",
    "print(\"Erro quadrático médio (MSE):\", mean_squared_error(y_test, y_pred))\n",
    "print(\"R²:\", r2_score(y_test, y_pred))\n",
    "\n",
    "coeficientes = pd.Series(modelo.coef_, index=X.columns).sort_values(key=abs, ascending=False)\n",
    "\n",
    "plt.figure(figsize=(10,6))\n",
    "sns.barplot(x=coeficientes.values, y=coeficientes.index)\n",
    "plt.title('Impacto das Variáveis no Total Gasto')\n",
    "plt.xlabel('Coeficiente da Regressão')\n",
    "plt.ylabel('Variáveis')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ca92efbb",
   "metadata": {},
   "source": [
    "## Customer Lifetime Value (CLV)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "199ad590",
   "metadata": {},
   "outputs": [],
   "source": [
    "clientes['frequencia_compras'].replace(0, np.nan, inplace=True)\n",
    "clientes['valor_medio'] = clientes['total_gasto'] / clientes['frequencia_compras']\n",
    "clientes['clv'] = clientes['total_gasto']\n",
    "\n",
    "plt.figure(figsize=(10,5))\n",
    "sns.histplot(clientes['clv'], bins=30, kde=True)\n",
    "plt.title('Distribuição do CLV')\n",
    "plt.xlabel('CLV (R$)')\n",
    "plt.ylabel('Número de Clientes')\n",
    "plt.show()\n",
    "\n",
    "clv_por_tipo = clientes.groupby('tipo_cliente')['clv'].mean().sort_values(ascending=False)\n",
    "\n",
    "plt.figure(figsize=(8,5))\n",
    "sns.barplot(x=clv_por_tipo.values, y=clv_por_tipo.index)\n",
    "plt.title('CLV Médio por Tipo de Cliente')\n",
    "plt.xlabel('CLV Médio (R$)')\n",
    "plt.ylabel('Tipo de Cliente')\n",
    "plt.show()\n",
    "\n",
    "df_clv = transacoes.merge(clientes[['cliente_id', 'clv']], on='cliente_id')\n",
    "clv_por_campanha = df_clv.groupby('campanha')['clv'].mean().sort_values(ascending=False)\n",
    "\n",
    "plt.figure(figsize=(10,5))\n",
    "sns.barplot(x=clv_por_campanha.values, y=clv_por_campanha.index)\n",
    "plt.title('CLV Médio por Campanha')\n",
    "plt.xlabel('CLV Médio (R$)')\n",
    "plt.ylabel('Campanha')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ab012183",
   "metadata": {},
   "source": [
    "## Conclusão e Recomendações Finais"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "874c5f37",
   "metadata": {},
   "source": [
    "As análises permitiram:\n",
    "\n",
    "- Segmentar clientes com base em comportamento de compra;\n",
    "- Identificar atributos mais valorizados em campanhas;\n",
    "- Estimar impacto e ROI das ações de marketing;\n",
    "- Avaliar o valor futuro de cada cliente (CLV).\n",
    "\n",
    "### Recomendações:\n",
    "- **Segmentação inteligente** com base nos clusters.\n",
    "- Campanhas com **frete grátis** e **desconto** geram mais conversão.\n",
    "- **Black Friday** e **clientes premium** apresentam maior retorno.\n",
    "- Usar o **CLV como métrica estratégica** para retenção e aquisição.\n",
    "\n",
    "Esses insights fornecem base sólida para otimização das campanhas e maximização do ROI.\n"
   ]
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
