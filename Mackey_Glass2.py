##加上Fast weight 的 echo state network 实现 prediction of Mackey-Glass time series的任务

import brainpy as bp
import brainpy.math as bm
import brainpy_datasets as bd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -------------------------------
# 1. 环境配置
# -------------------------------
# 启用高精度计算（x64），开启批处理模式
bm.set_environment(x64=True, mode=bm.batching_mode)

# 设置计算平台（CPU/GPU）
bm.set_platform('cpu')

# -------------------------------
# 2. 数据生成
# -------------------------------
dt = 0.1  # 时间步长
mg_data = bd.chaos.MackeyGlassEq(
    25000, dt=dt, tau=17, beta=0.2, gamma=0.1, n=10
)
xs = mg_data.xs  # 系统状态序列

# -------------------------------
# 3. 数据预处理函数
# -------------------------------
def get_data(t_warm, t_forcast, t_train, sample_rate=1):
    """
    Prepare warmup, training, and testing data for the ESN.

    Args:
        t_warm: Warm-up duration (ms)
        t_forcast: Forecast horizon (ms)
        t_train: Training duration (ms)
        sample_rate: Downsampling rate

    Returns:
        X_warm, X_train, Y_train, X_test, Y_test
    """
    warmup = int(t_warm / dt)
    forecast = int(t_forcast / dt)
    train_length = int(t_train / dt)

    # Warmup
    X_warm = bm.expand_dims(xs[:warmup:sample_rate], 0)

    # Training data
    X_train = bm.expand_dims(xs[warmup:warmup+train_length:sample_rate], 0)
    Y_train = bm.expand_dims(xs[warmup+forecast:warmup+train_length+forecast:sample_rate], 0)

    # Testing data
    X_test = bm.expand_dims(xs[warmup+train_length:-forecast:sample_rate], 0)
    Y_test = bm.expand_dims(xs[warmup+train_length+forecast::sample_rate], 0)

    return X_warm, X_train, Y_train, X_test, Y_test


# 获取数据
x_warm, x_train, y_train, x_test, y_test = get_data(100, 1, 20000)

# 可视化训练数据样例
plt.figure(figsize=(15, 3))
plt.plot(x_train[0, :2000], label="Training Input")
plt.plot(y_train[0, :2000], label="Training Target (Next Step)")
plt.title("Mackey-Glass Data Sample")
plt.legend()
plt.show()

# -------------------------------
# 4. 定义 Mamba-ESN Reservoir
# -------------------------------
class MambaLikeReservoir(bp.DynamicalSystem):
    """
    A reservoir with input-driven dynamic weights (Mamba-like mechanism).
    """
    def __init__(self, num_in, num_hidden, sr=0.8, leaky_rate=0.3,
                 lambda_dec=0.8, eta=0.05,
                 Win_initializer=bp.init.Uniform(-0.1, 0.1)):
        super(MambaLikeReservoir, self).__init__()

        self.num_hidden = num_hidden
        self.leaky_rate = leaky_rate  # 漏斗率
        self.lambda_dec = lambda_dec  # 权重衰减系数
        self.eta = eta                # 权重更新学习率

        # 输入权重
        self.Win = bm.Variable(Win_initializer((num_in, num_hidden)))

        # 随机初始化递归权重并归一化到谱半径 sr
        w_init = bm.random.normal(size=(num_hidden, num_hidden))
        eig_vals = bm.linalg.eigvals(w_init)
        spectral_radius = bm.max(bm.abs(eig_vals))
        self.W = bm.Variable((w_init / spectral_radius) * sr)

        # 可塑性矩阵
        self.F = bm.Variable(bm.zeros((num_hidden, num_hidden)))

        # 用于动态更新的向量
        self.Fv = bm.Variable(bm.random.normal(size=(num_hidden,)))
        self.Fk = bm.Variable(bm.random.normal(size=(num_hidden,)))

        # 初始状态
        self.state = bm.Variable(bm.zeros((1, num_hidden)))

    def update(self, x):
        """
        Update the reservoir state with input x.
        """
        # 动态权重更新
        vec_v = x * self.Fv
        vec_k = x * self.Fk
        batch_size = x.shape[0]
        delta_F = bm.einsum('bi,bj->ij', vec_v, vec_k) / batch_size
        self.F.value = self.lambda_dec * self.F + self.eta * delta_F

        # 总权重
        W_total = self.W + self.F

        # 信号计算
        in_signal = x @ self.Win
        rec_signal = self.state @ W_total.transpose()

        # 状态更新（leaky tanh）
        h_new = (1 - self.leaky_rate) * self.state + self.leaky_rate * bm.tanh(in_signal + rec_signal)
        self.state.value = h_new
        return h_new

    def reset_state(self, batch_size=1):
        """Reset the reservoir state and dynamic weights"""
        self.state.value = bm.zeros((batch_size, self.num_hidden))
        self.F.value = bm.zeros((self.num_hidden, self.num_hidden))


# -------------------------------
# 5. 定义 Mamba-ESN 模型
# -------------------------------
class MambaESN(bp.DynamicalSystem):
    """
    ESN using Mamba-like dynamic reservoir + linear readout.
    """
    def __init__(self, num_in, num_hidden, num_out, **kwargs):
        super(MambaESN, self).__init__()
        self.r = MambaLikeReservoir(num_in, num_hidden, **kwargs)
        self.o = bp.dnn.Dense(num_hidden, num_out, mode=bm.training_mode)

    def update(self, x):
        """Forward pass: input -> reservoir -> output"""
        return x >> self.r >> self.o

    def reset(self, batch_size=1):
        """Reset reservoir state"""
        self.r.reset_state(batch_size)

# -------------------------------
# 6. 初始化模型与训练
# -------------------------------
model = MambaESN(
    num_in=1, num_hidden=100, num_out=1,
    sr=0.8, leaky_rate=0.3, lambda_dec=0.8, eta=0.05
)

# 重置模型状态
model.reset(batch_size=1)

# Ridge 回归训练输出层
trainer = bp.RidgeTrainer(model, alpha=1e-6)

# Warmup
print("Status: Warmup...")
_ = trainer.predict(x_warm)

# 训练
print("Status: Training...")
_ = trainer.fit([x_train, y_train])

# 预测
print("Status: Predicting...")
ys_predict = trainer.predict(x_test)

# 计算均方误差
mse = bp.losses.mean_squared_error(ys_predict, y_test)
print(f"Prediction MSE: {mse}")

# 可视化预测结果
start, end = 1000, 4000
plt.figure(figsize=(15, 6))
plt.plot(bm.as_numpy(ys_predict)[0, start:end, 0], lw=2, label="Mamba-ESN Prediction")
plt.plot(bm.as_numpy(y_test)[0, start:end, 0], linestyle="--", lw=2, alpha=0.7, label="Ground Truth")
plt.title(f'Mackey-Glass Prediction (Input-Driven Dynamic Weights)\nMSE: {mse:.6f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# -------------------------------
# 7. 隐状态收集与 PCA 降维
# -------------------------------
inputs = bm.moveaxis(x_test, 1, 0)  # 时间步在前

def run_step(x_t):
    """一次时间步更新，返回 reservoir 状态"""
    model.update(x_t)
    return model.r.state

print("Running simulation (collecting hidden states)...")
model.reset(1)

# 收集所有时间步隐藏状态
states = bm.for_loop(run_step, inputs, progress_bar=True)

# 转换为 NumPy 数组，恢复 batch 维度在前
hidden_states_np = bm.as_numpy(bm.moveaxis(states, 0, 1))
print(f"Hidden States Shape: {hidden_states_np.shape}")

# 提取第 0 个样本的前 5000 步
time_steps = 5000
limit = min(time_steps, hidden_states_np.shape[1])
trajectory_high_dim = hidden_states_np[0, :limit, :]

# PCA 降维到 3D
pca = PCA(n_components=3)
trajectory_low_dim = pca.fit_transform(trajectory_high_dim)
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

# -------------------------------
# 8. 可视化高维动态
# -------------------------------
fig = plt.figure(figsize=(16, 7))

# --- 2D 投影 ---
ax1 = fig.add_subplot(121)
colors = plt.cm.viridis(bm.linspace(0, 1, limit))
ax1.scatter(trajectory_low_dim[:, 0], trajectory_low_dim[:, 1],
            c=colors, s=1, alpha=0.6)
ax1.set_title('2D PCA Projection (Reservoir Dynamics)')
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')
ax1.grid(True, alpha=0.3)

# --- 3D 投影 ---
ax2 = fig.add_subplot(122, projection='3d')
sc = ax2.scatter(trajectory_low_dim[:, 0],
                 trajectory_low_dim[:, 1],
                 trajectory_low_dim[:, 2],
                 c=colors, s=1, alpha=0.6)
ax2.set_title('3D Attractor Reconstruction')
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
ax2.set_zlabel('PC 3')

cbar = plt.colorbar(sc, ax=[ax1, ax2], shrink=0.8)
cbar.set_label('Time Evolution')

plt.show()








