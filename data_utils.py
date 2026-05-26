import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def get_prepared_data():
    X, y = make_classification(n_samples = 500, n_features = 2,
    n_redundant = 0, n_informative = 2, random_state = 42, n_clusters_per_class = 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    X_test = (X_test - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    
    return [X_train, X_test, y_train, y_test]

# Бонус - генерация кастомного датасета (linear/XOR/circle)
def generate_custom_data(type='linear', n_samples=500, noise=0.0):
    """
    type: 'linear', 'xor', 'circle'
    n_samples: количество точек в датасете
    noise: вероятность переворота метки (от 0.0 до 1.0)
    """

    np.random.seed(42) # задаем сид, чтобы каждый раз были одни и те же числа

    if type == 'linear':
        n = n_samples // 2
        
        X0 = np.random.multivariate_normal([2, 2], np.eye(2), size=n)
        y0 = np.zeros(n)
        
        X1 = np.random.multivariate_normal([-2, -2], np.eye(2), size=n)
        y1 = np.ones(n)
        
        X = np.vstack((X0, X1))
        y = np.hstack((y0, y1))

        
    elif type == 'xor':
        n = n_samples // 4
        X0_a = np.random.multivariate_normal([2, 2], np.eye(2)*0.5, size = n)
        y0_a = np.zeros(n)

        X0_b = np.random.multivariate_normal([-2, -2], np.eye(2)*0.5, size = n)
        y0_b = np.zeros(n)

        X1_a = np.random.multivariate_normal([2, -2], np.eye(2)*0.5, size = n)
        y1_a = np.ones(n)
        
        X1_b = np.random.multivariate_normal([-2, 2], np.eye(2)*0.5, size = n)
        y1_b = np.ones(n)

        X = np.vstack((X0_a, X0_b, X1_a, X1_b))
        y = np.hstack((y0_a, y0_b, y1_a, y1_b))
        
    elif type == 'circle':
        n = n_samples // 2
        
        # Класс 0: точки в центре (маленький радиус)
        r0 = np.random.uniform(0, 1.5, size=n)
        phi0 = np.random.uniform(0, 2*np.pi, size=n)
        X0 = np.column_stack((r0 * np.cos(phi0), r0 * np.sin(phi0))) # Перевод в декартовы X, Y
        y0 = np.zeros(n)
        
        # Класс 1: кольцо снаружи (большой радиус)
        r1 = np.random.uniform(2.5, 4.0, size=n)
        phi1 = np.random.uniform(0, 2*np.pi, size=n)
        X1 = np.column_stack((r1 * np.cos(phi1), r1 * np.sin(phi1)))
        y1 = np.ones(n)
        
        X = np.vstack((X0, X1))
        y = np.hstack((y0, y1))

    if noise > 0.0:
        # Вычисляем, скольким точкам нужно испортить метки
        n_noise = int(n_samples * noise)
        
        # Выбираем случайные индексы этих точек
        noise_indexes = np.random.choice(n_samples, n_noise, replace=False)
        
        # Инвертируем метки (0 становится 1, а 1 становится 0)
        y[noise_indexes] = 1 - y[noise_indexes]
        
    return X, y