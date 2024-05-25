from setuptools import setup, find_packages

setup(
    name='my_neural_network_project',  # Имя вашего пакета
    version='0.1.0',  # Версия вашего пакета
    packages=find_packages(),  # Автоматическое нахождение всех пакетов
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'tensorflow',  # или 'torch', если используете PyTorch
        'matplotlib',
        'jupyter',
        'seaborn',
    ],  # Список зависимостей
    entry_points={
        'console_scripts': [
            'train_model=src.training.train:main',  # Создание консольной команды для запуска тренировки модели
        ],
    },
    author='Ваше Имя',  # Ваше имя
    author_email='your.email@example.com',  # Ваш email
    description='Проект для создания и обучения нейросетей',  # Краткое описание проекта
    url='https://github.com/yourusername/my_neural_network_project',  # URL репозитория проекта
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
