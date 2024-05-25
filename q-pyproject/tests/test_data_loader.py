import unittest
from src.data.data_loader import load_data, preprocess_data

class TestDataLoader(unittest.TestCase):

    def test_load_data(self):
        data = load_data('data/raw/sample.csv')
        self.assertIsNotNone(data)

    def test_preprocess_data(self):
        data = load_data('data/raw/sample.csv')
        processed_data = preprocess_data(data)
        self.assertEqual(processed_data.shape, data.shape)

if __name__ == '__main__':
    unittest.main()