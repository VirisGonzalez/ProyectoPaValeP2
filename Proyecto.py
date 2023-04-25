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