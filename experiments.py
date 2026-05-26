from perceptron import Perceptron
import matplotlib.pyplot as plt
import numpy as np

def run_experiments(X_train, y_train, X_test, y_test):
    print("\n--- ЭКСПЕРИМЕНТ 1: Влияние скорости обучения (LR) ---")
    lrs = [0.001, 0.01, 0.1, 0.5, 1.0]
    
    plt.figure(figsize=(12, 6))
    for lr in lrs:
        model = Perceptron()
        train_losses, _ = model.fit(X_train, y_train, epochs=50, lr=lr, batch_size=32)
        
        # Считаем итоговую точность
        acc = np.mean(model.predict(X_test) == y_test) * 100
        print(f"LR = {lr:<5} | Точность: {acc:.2f}% | Финальный Loss: {train_losses[-1]:.4f}")
        
        # Рисуем график для этого LR
        plt.plot(train_losses, label=f'LR = {lr}')
        
    plt.title('Влияние скорости обучения (Learning Rate)')
    plt.xlabel('Эпоха')
    plt.ylabel('Train Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n--- ЭКСПЕРИМЕНТ 2: Влияние размера батча ---")
    batches = [1, 16, 32, 64, 256]
    
    plt.figure(figsize=(12, 6))
    for b in batches:
        model = Perceptron()
        train_losses, _ = model.fit(X_train, y_train, epochs=50, lr=0.1, batch_size=b)
        
        acc = np.mean(model.predict(X_test) == y_test) * 100
        print(f"Batch = {b:<3} | Точность: {acc:.2f}% | Финальный Loss: {train_losses[-1]:.4f}")
        plt.plot(train_losses, label=f'Batch = {b}')
        
    plt.title('Влияние размера батча (Batch Size)')
    plt.xlabel('Эпоха')
    plt.ylabel('Train Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n--- ЭКСПЕРИМЕНТ 3: Влияние L2-регуляризации ---")
    alphas = [0.0, 0.001, 0.01, 0.1, 1.0]
    for alpha in alphas:
        # Устанавливаем seed для воспроизводимости
        np.random.seed(42)
        model = Perceptron(loss_type='cross_entropy', alpha=alpha)
        model.fit(X_train, y_train, epochs=50, lr=0.1, batch_size=32)
        acc = np.mean(model.predict(X_test) == y_test) * 100
        w_norm = np.linalg.norm(model.w)
        print(f"L2 alpha = {alpha:<5} | Точность: {acc:.2f}% | Норма весов ||w||_2: {w_norm:.4f}")

    print("\n--- ЭКСПЕРИМЕНТ 4: Сравнение Binary Cross-Entropy и Hinge Loss ---")
    losses = ['cross_entropy', 'hinge']
    for loss in losses:
        np.random.seed(42)
        model = Perceptron(loss_type=loss)
        model.fit(X_train, y_train, epochs=50, lr=0.1, batch_size=32)
        acc = np.mean(model.predict(X_test) == y_test) * 100
        print(f"Loss Type = {loss:<15} | Точность: {acc:.2f}%")

    print("\n--- ЭКСПЕРИМЕНТ 5: Сравнение SGD и Momentum ---")
    gammas = [0.0, 0.5, 0.9, 0.99]
    plt.figure(figsize=(12, 6))
    for g in gammas:
        np.random.seed(42)
        model = Perceptron(gamma=g)
        train_losses, _ = model.fit(X_train, y_train, epochs=100, lr=0.1, batch_size=32)
        acc = np.mean(model.predict(X_test) == y_test) * 100
        label_text = 'SGD (gamma = 0.0)' if g == 0.0 else f'Momentum (gamma = {g})'
        print(f"{label_text:<25} | Точность: {acc:.2f}% | Финальный Loss: {train_losses[-1]:.4f}")
        plt.plot(train_losses, label=label_text)
        
    plt.title('Сравнение SGD и Momentum (сходимость)')
    plt.xlabel('Эпоха')
    plt.ylabel('Train Loss')
    plt.legend()
    plt.grid(True)
    plt.show()