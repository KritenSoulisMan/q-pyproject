import unittest
import numpy as np
from tensorflow.keras.models import load_model
from src.training.train import create_model
from src.config import (PROCESSED_DATA_PATH, PROCESSED_LABELS_PATH, 
                        TEST_SIZE, RANDOM_STATE, INPUT_SHAPE, 
                        EPOCHS, BATCH_SIZE, VALIDATION_SPLIT)

class TestTrain(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Загрузка данных перед запуском всех тестов
        cls.data = np.load(PROCESSED_DATA_PATH)
        cls.labels = np.load(PROCESSED_LABELS_PATH)
        
    def test_create_model(self):
        # Проверка, что модель создается правильно
        model = create_model(INPUT_SHAPE)
        self.assertEqual(len(model.layers), 3)
        self.assertEqual(model.layers[0].output_shape[1], 64)
    
    def test_model_training(self):
        from sklearn.model_selection import train_test_split
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            self.data, self.labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        
        # Создание и обучение модели
        model = create_model(INPUT_SHAPE)
        history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT)
        
        # Проверка, что модель обучилась и история обучения содержит данные
        self.assertIn('loss', history.history)
        self.assertIn('mae', history.history)
        self.assertTrue(len(history.history['loss']) > 0)

if __name__ == '__main__':
    unittest.main()
