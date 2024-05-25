import unittest
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.config import (PROCESSED_DATA_PATH, PROCESSED_LABELS_PATH, MODEL_SAVE_PATH, TEST_SIZE, RANDOM_STATE)
from src.evaluation.evaluate import evaluate_model

class TestNeuralNetwork(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Загрузка данных и модели перед запуском всех тестов
        cls.data = np.load(PROCESSED_DATA_PATH)
        cls.labels = np.load(PROCESSED_LABELS_PATH)
        cls.model = load_model(MODEL_SAVE_PATH)
        
        # Разделение данных (только для тестирования)
        from sklearn.model_selection import train_test_split
        _, cls.X_test, _, cls.y_test = train_test_split(cls.data, cls.labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    def test_model_loading(self):
        # Проверка, что модель загружается правильно
        self.assertIsNotNone(self.model)
        
    def test_model_predictions(self):
        # Проверка, что модель выдает предсказания
        predictions = self.model.predict(self.X_test)
        self.assertEqual(predictions.shape[0], self.X_test.shape[0])
        
    def test_evaluate_model(self):
        # Проверка, что функция оценки модели работает правильно
        mse, mae = evaluate_model(self.model, self.X_test, self.y_test)
        self.assertGreaterEqual(mse, 0)
        self.assertGreaterEqual(mae, 0)

if __name__ == '__main__':
    unittest.main()
