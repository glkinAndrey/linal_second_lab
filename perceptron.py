import numpy as np

class Perceptron:
    def __init__(self, alpha=0.0, gamma=0.9, loss_type='cross_entropy', weight_init='small_random'):
        # Веса (w) и смещение (b) инициализируем в методе fit
        self.w = None
        self.b = None
        self.alpha = alpha # Бонус - параметр для L2 регуляризации
        self.gamma = gamma # Бонус - параметр для ускорения сходимости (Momentum)
        self.loss_type = loss_type # Бонус - тип функции потерь ('cross_entropy' или 'hinge')
        self.weight_init = weight_init # Бонус - тип инициализации весов ('small_random', 'zero', 'large_random')

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        z = np.dot(X, self.w) + self.b
        return self.sigmoid(z)

    def compute_loss(self, y_true, y_pred, X=None):
        if self.loss_type == 'hinge':
            # Кодируем метки как {-1, 1}
            y_mapped = 2 * y_true - 1
            if X is not None:
                z = np.dot(X, self.w) + self.b
            else:
                y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
                z = -np.log(1 / y_pred_clipped - 1)
            hinge_loss = np.mean(np.maximum(0, 1 - y_mapped * z)) + self.alpha * np.sum(self.w**2)
            return hinge_loss
        else:
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            log_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) + self.alpha * np.sum(self.w**2)
            return log_loss

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)    

    def fit(self, X, y, X_val = None, y_val = None,
            epochs=100, lr=0.1, batch_size=32):

        d = X.shape[1]
        
        # Выбираем инициализацию весов
        if self.weight_init == 'zero':
            self.w = np.zeros(d)
        elif self.weight_init == 'large_random':
            self.w = np.random.randn(d) * 10.0
        else: # 'small_random' (по умолчанию)
            self.w = np.random.randn(d) * 0.01
            
        self.b = 0.0
        v_w = np.zeros(d)
        v_b = 0.0
        train_losses = []
        test_losses = []

        for epoch in range(epochs):
            permutations = np.random.permutation(X.shape[0])
            X_shuffled = X[permutations]
            y_shuffled = y[permutations]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                m = X_batch.shape[0]

                if self.loss_type == 'hinge':
                    z_batch = np.dot(X_batch, self.w) + self.b
                    y_batch_mapped = 2 * y_batch - 1
                    margin = 1 - y_batch_mapped * z_batch
                    mask = (margin > 0).astype(float)
                    
                    dw = (1/m) * np.dot(X_batch.T, -mask * y_batch_mapped) + self.alpha * self.w * 2
                    db = (1/m) * np.sum(-mask * y_batch_mapped)
                else:
                    y_pred = self.forward(X_batch)
                    error = y_pred - y_batch
                    dw = (1/m) * np.dot(X_batch.T, error) + self.alpha * self.w * 2
                    db = (1/m) * np.sum(error)

                # Бонус - моментум
                v_w = self.gamma * v_w + lr * dw
                v_b = self.gamma * v_b + lr * db

                self.w -= v_w
                self.b -= v_b

            train_losses.append(self.compute_loss(y, self.forward(X), X))
            
            if X_val is not None and y_val is not None:
                epoch_val_loss = self.compute_loss(y_val, self.forward(X_val), X_val)
                test_losses.append(epoch_val_loss)

        # Возвращаем списки, когда все эпохи закончились
        return train_losses, test_losses