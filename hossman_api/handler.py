from flask import Flask, request, Response
import pandas as pd
from rossman.Rossman import Rossman
import os
import pickle
import traceback

#load_model
model_path ='model/model_rossman.pkl'
model = pickle.load(open(model_path, 'rb'))

#initialize API
app = Flask(__name__)

@app.route('/rossman/predict', methods=['POST'])
def rossman_predict():
    try:
        test_json = request.get_json()
        #print("JSON recebido:", test_json)

        if test_json:
            if isinstance(test_json, dict):  # Exemplo único
                test_raw = pd.DataFrame(test_json, index=[0])
            elif isinstance(test_json, list) and isinstance(test_json[0], dict):
                test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
            else:
                return Response("Formato de entrada inválido. Esperado dict ou lista de dicts.", status=400)

            # Instancia a classe
            pipeline = Rossman()

            # Pipeline de processamento
            df1 = pipeline.data_cleaning(test_raw)
            df2 = pipeline.feature_engineering(df1)
            df3 = pipeline.data_preparation(df2)

            # Geração da predição
            df_response = pipeline.get_prediction(model, test_raw, df3)

            return df_response

        else:
            return Response("JSON vazio ou inválido recebido.", status=400, mimetype='application/json')

    except Exception as e:
        # Captura stack trace para log ou depuração
        error_msg = f"Erro interno no servidor: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)  # Loga no terminal
        return Response(f"Erro interno: {str(e)}", status=500, mimetype='text/plain')

if __name__ == '__main__':
  port = os.environ.get('PORT', 5000)
  app.run(host='0.0.0.0', port=port, debug=True)