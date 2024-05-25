# Путь к данным
DATA_PATH = 'data/raw/dataset.csv'
PROCESSED_DATA_PATH = 'data/processed/data.npy'
PROCESSED_LABELS_PATH = 'data/processed/labels.npy'

# Настройки модели
MODEL_SAVE_PATH = 'models/saved_model.h5'
INPUT_SHAPE = (10,)  # Пример, используйте фактическое количество признаков в ваших данных
EPOCHS = 50
BATCH_SIZE = 32

# Настройки разделения данных
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Другие параметры
VALIDATION_SPLIT = 0.2
