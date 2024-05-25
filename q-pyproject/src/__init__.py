from src.data import load_data, preprocess_data
from src.models import create_model
from src.training import main as train_model
from src.evaluation import evaluate_model
from src.utils import create_directory

# Использование импортированных функций
data = load_data('data/raw/dataset.csv')
processed_data = preprocess_data(data)
