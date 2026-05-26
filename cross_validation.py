import numpy as np
from sklearn.model_selection import KFold
from perceptron import Perceptron

# Бонус - кросс-валидация с параметрами
def run_cross_validation(X, y, k=5, lr=0.1, batch_size=32, epochs=50, verbose=True):
    if verbose:
        print(f"\n--- Запуск {k}-Fold Кросс-валидации ---")
        print(f"Гиперпараметры: epochs={epochs}, lr={lr}, batch_size={batch_size}, gamma=0.9")
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []
    
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        # 1. Получаем срезы данных
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # 2. Нормализуем
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        # Предотвращаем деление на 0
        std = np.where(std == 0, 1.0, std)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        
        # 3. Создаем модель
        model = Perceptron(gamma=0.9)
        
        # 4. Обучаем
        model.fit(X_train, y_train, epochs=epochs, lr=lr, batch_size=batch_size)
        
        # 5. Оцениваем
        test_preds = model.predict(X_test)
        acc = np.mean(test_preds == y_test) * 100
        
        # 6. Сохраняем результат
        if verbose:
            print(f"Фолд {fold + 1}: Точность = {acc:.2f}%")
        fold_accuracies.append(acc)
        
    # 7. Средний результат
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    if verbose:
        print(f"Средняя точность кросс-валидации: {mean_acc:.2f}% (±{std_acc:.2f}%)\n")
    return mean_acc, std_acc

# Бонус - подбор гиперпараметров (Grid Search) по сетке
def grid_search_cross_validation(X, y, k=5):
    print("\n==================================================")
    print("=== ПОИСК ПО СЕТКЕ (GRID SEARCH) КРОСС-ВАЛИДАЦИИ ===")
    print("==================================================")
    
    lrs = [0.01, 0.05, 0.1]
    batch_sizes = [16, 32, 64]
    
    results = []
    best_acc = -1.0
    best_params = None
    
    print(f"{'Learning Rate':<15} | {'Batch Size':<10} | {'Mean Accuracy':<15} | {'Std':<10}")
    print("-" * 60)
    
    # Чтобы результаты были стабильными при кросс-валидации
    np.random.seed(42)
    
    for lr in lrs:
        for bs in batch_sizes:
            mean_acc, std_acc = run_cross_validation(
                X, y, k=k, lr=lr, batch_size=bs, epochs=50, verbose=False
            )
            # Делим на 100 для соответствия формату долей в таблице
            mean_acc_val = mean_acc / 100.0
            std_acc_val = std_acc / 100.0
            print(f"{lr:<15} | {bs:<10} | {mean_acc_val:<15.4f} | {std_acc_val:<10.4f}")
            results.append((lr, bs, mean_acc_val, std_acc_val))
            
            if mean_acc_val > best_acc:
                best_acc = mean_acc_val
                best_params = (lr, bs, mean_acc_val, std_acc_val)
                
    print("-" * 60)
    print(f"Лучшие параметры: LR = {best_params[0]}, Batch Size = {best_params[1]}")
    print(f"Ср. точность: {best_params[2]:.4f} (±{best_params[3]:.4f})")
    print("==================================================\n")
    return best_params