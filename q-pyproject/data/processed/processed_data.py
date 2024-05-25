import numpy as np

# Пример данных после предварительной обработки
processed_data = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9]
])

np.save('processed_data.npy', processed_data)
