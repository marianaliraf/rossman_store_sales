
# 🧠 Rossmann Sales Forecasting

Este repositório contém um pipeline completo de Ciência de Dados, que resolve o problema de previsão de vendas diárias para lojas da rede Rossmann, com base no desafio do Kaggle ["Rossmann Store Sales"](https://www.kaggle.com/competitions/rossmann-store-sales).

A solução abrange desde o pré-processamento e modelagem até a disponibilização do modelo como serviço em produção via API Flask com deploy no Heroku.

---

## 🗂️ Estrutura do Projeto

```
📁 projeto_rossmann/
├── api/                 # Código da API em Flask (handler, classe Rossman)
├── datasets/            # Dados brutos e tratados
├── hossman_api/         # Projeto pronto para subir no Heroku (inclui Procfile, runtime.txt etc.)
├── model/               # Modelo treinado (model_rossman.pkl)
├── notebooks/           # Notebooks Jupyter com EDA, modelagem, etc.
├── parameter/           # Scalers serializados (.pkl) usados na preparação dos dados
```

---

## 🚀 Visão Geral

A rede Rossmann possui mais de 3.000 lojas em 7 países europeus. Gerentes precisam prever vendas com semanas de antecedência, considerando:

- Promoções
- Concorrência
- Feriados escolares e estaduais
- Sazonalidade
- Localização

Este projeto automatiza esse processo por meio de machine learning, permitindo previsões mais confiáveis, otimizando recursos e melhorando a tomada de decisão nas lojas.

---

## ⚙️ Tecnologias Utilizadas

- **Linguagem**: Python 3.10+
- **Bibliotecas**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Pickle
- **API**: Flask
- **Deploy**: Heroku

---

## 🧪 Etapas do Projeto

1. **Análise Exploratória (EDA)**  
   Visualização e entendimento dos principais padrões de vendas.

2. **Engenharia de Atributos**  
   - Criação de variáveis cíclicas (mês, dia, semana)
   - Conversão de feriados, promoções, competidores
   - Encoding e tratamento de nulos

3. **Preparação dos Dados**  
   - Normalização com `RobustScaler`, `MinMaxScaler` e `OneHotEncoding`

4. **Treinamento do Modelo**  
   - Modelo: `XGBoostRegressor`
   - Avaliação com validação cruzada e métricas MAE e MAPE

5. **Serialização**  
   - Modelo e scalers salvos com `pickle`

6. **Criação da API (Flask)**  
   - Recebe JSON via endpoint `/rossman/predict`
   - Retorna previsão em formato JSON

7. **Deploy (Heroku)**  
   - Estrutura compatível em `hossman_api/`

---

## ▶️ Como Executar Localmente

1. Clone o repositório:

```bash
git clone https://github.com/seu-usuario/projeto_rossmann.git
cd projeto_rossmann/api
```

2. Crie o ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

4. Execute a API:

```bash
python handler.py
```

A API estará disponível em: `http://localhost:5000/rossman/predict`

---

## 🔁 Exemplo de Requisição

### Requisição `POST`

```bash
POST /rossman/predict
Content-Type: application/json
```

### JSON de Entrada

```json
[
  {
    "Store": 1,
    "DayOfWeek": 5,
    "Date": "2015-07-31",
    "Sales": 5263,
    "Customers": 555,
    "Open": 1,
    "Promo": 1,
    "StateHoliday": "0",
    "SchoolHoliday": 1,
    "StoreType": "c",
    "Assortment": "a",
    "CompetitionDistance": 1270.0,
    "CompetitionOpenSinceMonth": 9,
    "CompetitionOpenSinceYear": 2008,
    "Promo2": 1,
    "Promo2SinceWeek": 13,
    "Promo2SinceYear": 2010,
    "PromoInterval": "Jan,Apr,Jul,Oct"
  }
]
```

### Resposta Esperada

```json
[
  {
    "Store": 1,
    "Date": "2015-07-31T00:00:00.000Z",
    "prediction": 5423.7
  }
]
```

---

## ☁️ Acesse o Deploy (Heroku)

Você pode testar a API diretamente no ar pelo link:

🔗 [https://hossman-model-prediction-315aed2b5d3b.herokuapp.com/rossman/predict](https://hossman-model-prediction-315aed2b5d3b.herokuapp.com/rossman/predict)

---

## 🧾 Licença

Este projeto é de uso educacional, sem fins comerciais.
