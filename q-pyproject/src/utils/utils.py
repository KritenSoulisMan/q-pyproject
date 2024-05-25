import os

def create_directory(path):
    """
    Создание директории, если она не существует
    """
    if not os.path.exists(path):
        os.makedirs(path)
