#加上Fast weight 的 echo state network 实现 正弦波预测任务

import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ==========================================
# 1. 环境配置
# ==========================================
bm.set_environment(x64=True, mode=bm.batching_mode)  # 高精度 + 批处理模式
bm.set_platform('cpu')  # 或 'gpu'
print(f"BrainPy Version: {bp.__version__}")

# ==========================================
# 2. 数据生成（正弦波）
# ==========================================
dt = 0.1
num_steps = 25000
total_time = num_steps * dt

# 时间轴
ts = bm.arange(0, total_time, dt)

# 正弦波数据 y = sin(0.2*t)
xs = bm.sin(ts * 0.2)

# 增加特征维度 (Time, 1)，符合 ESN 输入要求
xs = xs[:, None]

# ==========================================
# 3. 数据切分函数
# ==========================================
def get_data(t_warm, t_forecast, t_train, sample_rate=1):
    """
    Prepare warmup, training, and testing datasets for ESN.
    """
    warmup = int(t_warm / dt)
    forecast = int(t_forecast / dt)
    train_length = int(t_train / dt)

    X_warm = bm.expand_dims(xs[:warmup:sample_rate], 0)
    X_train = bm.expand_dims(xs[warmup:warmup+train_length:sample_rate], 0)
    Y_train = bm.expand_dims(xs[warmup+forecast:warmup+train_length+forecast:sample_rate], 0)
    X_test = bm.expand_dims(xs[warmup+train_length:-forecast:sample_rate], 0)
    Y_test = bm.expand_dims(xs[warmup+train_length+forecast::sample_rate], 0)

    return X_warm, X_train, Y_train, X_test, Y_test

# 获取数据
x_warm, x_train, y_train, x_test, y_test = get_data(t_warm=100, t_forecast=1, t_train=2000)

# 可视化训练数据片段
plt.figure(figsize=(15, 3))
plt.plot(bm.as_numpy(x_train[0, :1000]), label="Training Input (Sine)")
plt.plot(bm.as_numpy(y_train[0, :1000]), label="Training Target (Shifted Sine)")
plt.title("Sine Wave Data Sample")
plt.legend()
plt.show()

# ==========================================
# 4. 定义 Mamba-Like Reservoir
# ==========================================
class MambaLikeReservoir(bp.DynamicalSystem):
    """
    Reservoir with dynamic (fast) weights.
    """
    def __init__(self, num_in, num_hidden, sr=0.8, leaky_rate=0.3,
                 lambda_dec=0.8, eta=0.1, Win_initializer=bp.init.Uniform(-0.1, 0.1)):
        super().__init__()
        self.num_hidden = num_hidden
        self.leaky_rate = leaky_rate
        self.lambda_dec = lambda_dec
        self.eta = eta

        # 输入权重
        self.Win = bm.Variable(Win_initializer((num_in, num_hidden)))

        # 静态循环权重 W（经过谱半径调整）
        w_init = bm.random.normal(size=(num_hidden, num_hidden))
        eig_vals = bm.linalg.eigvals(w_init)
        spectral_radius = bm.max(bm.abs(eig_vals))
        self.W = bm.Variable((w_init / spectral_radius) * sr)

        # 动态权重矩阵 F
        self.F = bm.Variable(bm.zeros((num_hidden, num_hidden)))

        # Key-Value 向量
        self.Fv = bm.Variable(bm.random.normal(size=(num_hidden,)))
        self.Fk = bm.Variable(bm.random.normal(size=(num_hidden,)))

        # 隐藏状态
        self.state = bm.Variable(bm.zeros((1, num_hidden)))

    def update(self, x):
        """
        单步更新函数
        """
        # Key-Value 外积
        vec_v = x * self.Fv
        vec_k = x * self.Fk
        batch_size = x.shape[0]
        delta_F = bm.einsum('bi,bj->ij', vec_v, vec_k) / batch_size

        # 更新动态权重 F
        self.F.value = self.lambda_dec * self.F + self.eta * delta_F

        # 计算总权重
        W_total = self.W + self.F

        # Reservoir 状态更新
        in_signal = x @ self.Win
        rec_signal = self.state @ W_total.T
        h_new = (1 - self.leaky_rate) * self.state + self.leaky_rate * bm.tanh(in_signal + rec_signal)
        self.state.value = h_new

        return h_new

    def reset_state(self, batch_size):
        """
        Reset hidden state and dynamic weights.
        """
        self.state.value = bm.zeros((batch_size, self.num_hidden))
        self.F.value = bm.zeros((self.num_hidden, self.num_hidden))


class MambaESN(bp.DynamicalSystem):
    """
    ESN with Mamba-Like Reservoir.
    """
    def __init__(self, num_in, num_hidden, num_out, **kwargs):
        super().__init__()
        self.r = MambaLikeReservoir(num_in, num_hidden, **kwargs)
        self.o = bp.dnn.Dense(num_hidden, num_out, mode=bm.training_mode)

    def update(self, x):
        return x >> self.r >> self.o

    def reset(self, batch_size):
        self.r.reset_state(batch_size)

# ==========================================
# 5. 初始化模型并训练
# ==========================================
model = MambaESN(num_in=1, num_hidden=100, num_out=1,
                 sr=0.8, leaky_rate=0.3, lambda_dec=0.8, eta=0.1)
model.reset(1)

trainer = bp.RidgeTrainer(model, alpha=1e-6)

print("Status: Warmup...")
_ = trainer.predict(x_warm)

print("Status: Training...")
_ = trainer.fit([x_train, y_train])

print("Status: Predicting...")
ys_predict = trainer.predict(x_test)

# 计算均方误差
mse = bp.losses.mean_squared_error(ys_predict, y_test)
print(f"Prediction MSE: {mse:.6e}")

# 可视化预测
start, end = 500, 1000
plt.figure(figsize=(15, 6))
plt.plot(bm.as_numpy(ys_predict)[0, start:end, 0], lw=2, label="Mamba-ESN Prediction")
plt.plot(bm.as_numpy(y_test)[0, start:end, 0], linestyle="--", lw=2, alpha=0.7, label="Ground Truth")
plt.title(f'Sine Wave Prediction (Mamba-ESN)\nMSE: {mse:.6e}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ==========================================
# 6. PCA 降维分析
# ==========================================
inputs = bm.moveaxis(x_test, 1, 0)

def run_step(x_t):
    model.update(x_t)
    return model.r.state

print("Running simulation (collecting hidden states)...")
model.reset(1)
states = bm.for_loop(run_step, inputs, progress_bar=True)

# 转换为 NumPy 并调整维度 (Batch, Time, Hidden)
hidden_states_np = bm.as_numpy(bm.moveaxis(states, 0, 1))
print(f"Hidden States Shape: {hidden_states_np.shape}")

# 截取前 limit 步进行 PCA
time_steps = 5000
limit = min(time_steps, hidden_states_np.shape[1])
trajectory_high_dim = hidden_states_np[0, :limit, :]

# PCA 降到 3 维
pca = PCA(n_components=3)
trajectory_low_dim = pca.fit_transform(trajectory_high_dim)
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

# ==========================================
# 7. PCA 可视化
# ==========================================
figure = plt.figure(figsize=(16, 7))

# --- 2D 投影 ---
ax1 = figure.add_subplot(121)
colors = plt.cm.viridis(bm.linspace(0, 1, limit))
ax1.scatter(trajectory_low_dim[:, 0], trajectory_low_dim[:, 1],
            c=colors, s=1, alpha=0.6)
ax1.set_title('2D PCA Projection (Reservoir Dynamics)')
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')
ax1.grid(True, alpha=0.3)

# --- 3D 投影 ---
ax2 = figure.add_subplot(122, projection='3d')
sc = ax2.scatter(trajectory_low_dim[:, 0],
                 trajectory_low_dim[:, 1],
                 trajectory_low_dim[:, 2],
                 c=colors, s=1, alpha=0.6)
ax2.set_title('3D Attractor Reconstruction (Limit Cycle)')
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
ax2.set_zlabel('PC 3')

cbar = plt.colorbar(sc, ax=[ax1, ax2], shrink=0.8)
cbar.set_label('Time Evolution')

plt.show()

