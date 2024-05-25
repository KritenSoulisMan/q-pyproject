import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.config import PROCESSED_DATA_PATH, PROCESSED_LABELS_PATH, MODEL_SAVE_PATH

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    return mse, mae

# Загрузка данных
data = np.load(PROCESSED_DATA_PATH)
labels = np.load(PROCESSED_LABELS_PATH)

# Разделение данных (только для тестирования)
_, X_test, _, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Загрузка модели
model = load_model(MODEL_SAVE_PATH)

# Оценка модели
evaluate_model(model, X_test, y_test)
