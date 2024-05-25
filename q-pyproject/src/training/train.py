import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from src.config import (PROCESSED_DATA_PATH, PROCESSED_LABELS_PATH, 
                        MODEL_SAVE_PATH, INPUT_SHAPE, EPOCHS, BATCH_SIZE, 
                        TEST_SIZE, RANDOM_STATE, VALIDATION_SPLIT)

def create_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def main():
    # Загрузка данных
    data = np.load(PROCESSED_DATA_PATH)
    labels = np.load(PROCESSED_LABELS_PATH)

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Создание модели
    model = create_model(INPUT_SHAPE)

    # Обучение модели
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT)

    # Сохранение модели
    model.save(MODEL_SAVE_PATH)

    # Оценка модели
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test MAE: {mae}")

if __name__ == '__main__':
    main()
