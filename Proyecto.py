import boto3
import pandas as pd
from io import StringIO, BytesIO
from datetime import datetime, timedelta
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import s3fs
from sklearn.preprocessing import MinMaxScaler

class ETLPadre:
    def __init__(self, bucket_name, target_bucket_name):
        self.s3 = boto3.resource('s3')
        self.bucket = self.s3.Bucket(bucket_name)
        self.target_bucket = self.s3.Bucket(target_bucket_name)

class ETL(ETLPadre):
    
    def extract(self, arg_date):
        arg_date_dt = datetime.strptime(arg_date, '%Y-%m-%d').date() - timedelta(days=1)
        objects = [obj for obj in self.bucket.objects.all() 
                   if datetime.strptime(obj.key.split("/")[0], '%Y-%m-%d').date() == arg_date_dt]
        df_all = self.read_csv_to_df(objects)
        return df_all

    def transform_report(self, df_all, arg_date):
        df_all['opening_price'] = df_all.sort_values(by=['Time']).groupby(['ISIN', 'Date'])['StartPrice'].transform('first')
        df_all['closing_price'] = df_all.sort_values(by=['Time']).groupby(['ISIN', 'Date'])['EndPrice'].transform('last')

        df_all = df_all.query('"08:00" < Time < "12:00"').groupby(['ISIN', 'Date'], as_index=False).agg(opening_price_eur=('opening_price', 'min'), closing_price_eur=('closing_price', 'min'), minimum_price_eur=('MinPrice', 'min'), maximum_price_eur=('MaxPrice', 'max'), daily_traded_volume=('TradedVolume', 'sum'))

        df_all['PesoM'] = df_all['closing_price_eur'] * 20    
        df_all.dropna(inplace=True)

        df_all['Desviacion'] = df_all[['opening_price_eur', 'closing_price_eur']].std(axis=1)

        return df_all

    def load(self, df_all, key):
        out_buffer = BytesIO()
        df_all.to_parquet(out_buffer, index=False)
        self.target_bucket.put_object(Body=out_buffer.getvalue(), Key=key)

    def read_csv_to_df(self, objects):
        csv_obj_init = self.bucket.Object(key=objects[0].key).get().get('Body').read().decode('utf-8')
        data = StringIO(csv_obj_init)
        df_init = pd.read_csv(data, delimiter=',')

        df_all = pd.DataFrame(columns=df_init.columns)

        for obj in objects:
            csv_obj = self.bucket.Object(key=obj.key).get().get('Body').read().decode('utf-8')
            data = StringIO(csv_obj)
            df = pd.read_csv(data, delimiter=',')
            df_all = pd.concat([df, df_all], ignore_index=True)

        return df_all
    
    def etl_report(self, key):
        arg_date = key.split('/')[0]
        df_all = self.extract(arg_date)
        df_all_transformed = self.transform_report(df_all, arg_date)
        self.load(df_all_transformed, key)

etl = ETL('xetra-1234', 'xetra-vgv')
df = etl.extract('2022-12-30')
df_report = etl.transform_report(df, '2022-03-26')
etl.load(df_report, 'xetra-vgv20230310_073013.parquet')

print(df_report)

s3 = boto3.resource('s3')
bucket_target = s3.Bucket('xetra-vgv')
prq_obj = bucket_target.Object(key='xetra-vgv20230228_103751.parquet').get().get('Body').read()
data = BytesIO(prq_obj)
df_report = pd.read_parquet(data)

df_report

s3 = boto3.resource('s3')
bucket_target = s3.Bucket('xetra-vgv')
prq_obj = bucket_target.Object(key='xetra-vgv20230228_103751.parquet').get().get('Body').read()
data = BytesIO(prq_obj)
df_report = pd.read_parquet(data)

# Guardar el DataFrame como archivo CSV
df_report.to_csv('xetra-vgv20230228_103751.csv', index=False)

# Cargar el archivo CSV para verificar que se guardó correctamente
df_csv = pd.read_csv('xetra-vgv20230228_103751.csv')
print(df_csv.head())

df = pd.read_csv('xetra-vgv20230228_103751.csv')


# Seleccionar las características que se utilizarán para la predicción
X = df[['opening_price_eur', 'closing_price_eur']].values

# Seleccionar la variable a predecir (EndPrice)
y = df[['closing_price_eur']].values

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalizar los datos
sc = MinMaxScaler(feature_range=(0, 1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Construir el modelo de red neuronal
model = Sequential()
model.add(Dense(64, input_shape=(2,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))

print(X_train.shape)
print(X_test.shape)

# Entrenar el modelo de red neuronal
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Realizar la predicción con una proyección de una semana
# Cargar los datos del nuevo día que se quiere predecir
new_data = pd.read_csv('C:/Users/virid/Desktop/xetra-vgv20230228_103751.csv')
new_X = new_data[['opening_price_eur', 'closing_price_eur']].values
new_X = sc.transform(new_X)
prediction = model.predict(new_X)

# Imprimir la predicción
print("La predicción para el EndPrice en una semana es:", prediction)

# Guardar el modelo entrenado
model.save('ruta/al/modelo_entrenado.h5')

