import os
import mlflow
import numpy as np

os.environ['MLFLOW_TRACKING_USERNAME'] = 'AndyGoncalves'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '1d7f8459c2cdd075a0317f26fc322caf353acdf4'
mlflow.set_tracking_uri('https://dagshub.com/AndyGoncalves/mlops-ead.mlflow')

"""## Criando um client para comunicar com o registro no DagsHub"""

client = mlflow.MlflowClient('https://dagshub.com/AndyGoncalves/mlops-ead.mlflow')

"""## Recebendo o modelo registrado e suas versões"""

registered_model = client.get_registered_model('fetal_health')

#registered_model.latest_versions

"""## Obtendo o id da execução do modelo"""

run_id = registered_model.latest_versions[-1].run_id

"""## Carregando o modelo"""

logged_model = f'runs:/{run_id}/fetal_health'

loaded_model = mlflow.pyfunc.load_model(logged_model)

#loaded_model = mlflow.sklearn.load_model(logged_model)

#loaded_model = mlflow.pytorch.load_model(logged_model)

#loaded_model = mlflow.tensorflow.load_model(logged_model)

"""## Fazendo uma predição com o modelo carregadO"""

accelerations = 0
fetal_movement = 0
uterine_contractions = 0
severe_decelerations = 0

received_data = np.array([
        accelerations,
        fetal_movement,
        uterine_contractions,
        severe_decelerations,
    ]).reshape(1, -1)

loaded_model.predict(received_data)

