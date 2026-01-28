#加上Fast weight 的 echo state network 实现 小球双摆 预测任务

import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.decomposition import PCA

# -------------------------------
# 1. 环境配置
# -------------------------------
bm.set_environment(x64=True, mode=bm.batching_mode)
bm.set_platform('cpu')

# -------------------------------
# 2. 数据生成：双摆系统 (Double Pendulum)
# -------------------------------
#

def double_pendulum_ode(state, t, m1, m2, l1, l2, g):
    theta1, z1, theta2, z2 = state
    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

    theta1dot = z1
    z1dot = (m2*g*np.sin(theta2)*c - m2*s*(l1*z1**2*c + l2*z2**2) - (m1+m2)*g*np.sin(theta1)) / (l1 * (m1 + m2*s**2))
    theta2dot = z2
    z2dot = ((m1+m2)*(l1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + m2*l2*z2**2*s*c) / (l2 * (m1 + m2*s**2))

    return theta1dot, z1dot, theta2dot, z2dot

def generate_data(steps, dt):
    # 物理参数
    m1, m2, l1, l2, g = 1.0, 1.0, 1.0, 1.0, 9.8
    # 初始状态
    init_state = [np.pi/2, 0, np.pi/2, 0]

    t = np.arange(0, steps * dt, dt)
    y = odeint(double_pendulum_ode, init_state, t, args=(m1, m2, l1, l2, g))

    theta1, theta2 = y[:, 0], y[:, 2]

    # 转为笛卡尔坐标
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)

    # 拼接数据 (Steps, 4) -> [x1, y1, x2, y2]
    return np.stack([x1, y1, x2, y2], axis=1)

dt = 0.05
total_steps = 25000
raw_data = generate_data(total_steps, dt)
raw_data = bm.asarray(raw_data) # 转为 BrainPy 数组

print(f"原始数据形状: {raw_data.shape} (Time, Features=4)")

# -------------------------------
# 3. 数据预处理
# -------------------------------
def get_data(data, t_warm, t_forecast, t_train, sample_rate=1):
    warmup = int(t_warm / dt)
    forecast = int(t_forecast / dt)
    train_length = int(t_train / dt)

    # 调整维度为 (Batch, Time, Feature)
    # Warmup
    X_warm = bm.expand_dims(data[:warmup:sample_rate], 0)

    # Train
    X_train = bm.expand_dims(data[warmup:warmup+train_length:sample_rate], 0)
    Y_train = bm.expand_dims(data[warmup+forecast:warmup+train_length+forecast:sample_rate], 0)

    # Test
    X_test = bm.expand_dims(data[warmup+train_length:-forecast:sample_rate], 0)
    Y_test = bm.expand_dims(data[warmup+train_length+forecast::sample_rate], 0)

    return X_warm, X_train, Y_train, X_test, Y_test

# 这里的 forecast 设置为 1 (单步预测)，因为双摆是混沌系统，长时预测极难
x_warm, x_train, y_train, x_test, y_test = get_data(raw_data, 100, dt, 800)

print(f"训练输入形状: {x_train.shape}")
print(f"训练目标形状: {y_train.shape}")

# -------------------------------
# 4. 定义 Mamba-ESN Reservoir (适配多维输入)
# -------------------------------
class MambaLikeReservoir(bp.DynamicalSystem):
    def __init__(self, num_in, num_hidden, sr=0.8, leaky_rate=0.3,
                 lambda_dec=0.8, eta=0.01,
                 Win_initializer=bp.init.Uniform(-0.1, 0.1)):
        super(MambaLikeReservoir, self).__init__()

        self.num_hidden = num_hidden
        self.leaky_rate = leaky_rate
        self.lambda_dec = lambda_dec
        self.eta = eta

        # 1. 静态输入权重
        self.Win = bm.Variable(Win_initializer((num_in, num_hidden)))

        # 2. 动态权重控制层 (将输入投影到 Hidden 维度，用于计算 V 和 K)
        # 这是为了适配多维输入 (4维) 到动态矩阵计算
        self.W_control = bm.Variable(bp.init.XavierNormal()((num_in, num_hidden)))

        # 3. 递归权重 (固定部分)
        w_init = bm.random.normal(size=(num_hidden, num_hidden))
        eig_vals = bm.linalg.eigvals(w_init)
        spectral_radius = bm.max(bm.abs(eig_vals))
        self.W = bm.Variable((w_init / spectral_radius) * sr)

        # 4. 可塑性矩阵 (Fast Weights)
        self.F = bm.Variable(bm.zeros((num_hidden, num_hidden)))

        # 5. 动态更新因子
        self.Fv = bm.Variable(bm.random.normal(size=(num_hidden,)))
        self.Fk = bm.Variable(bm.random.normal(size=(num_hidden,)))

        self.state = bm.Variable(bm.zeros((1, num_hidden)))

    def update(self, x):
        # x shape: (Batch, num_in)

        # --- Mamba-like 动态权重更新 ---
        # 首先将输入投影到控制空间: (Batch, 4) @ (4, 100) -> (Batch, 100)
        u_control = x @ self.W_control

        # 逐元素乘法生成 Key 和 Value 向量
        vec_v = u_control * self.Fv
        vec_k = u_control * self.Fk

        batch_size = x.shape[0]
        # 计算外积作为权重增量 delta_F
        delta_F = bm.einsum('bi,bj->ij', vec_v, vec_k) / batch_size

        # 更新 Fast Weight 矩阵 F
        self.F.value = self.lambda_dec * self.F + self.eta * delta_F

        # --- ESN 状态更新 ---
        W_total = self.W + self.F

        in_signal = x @ self.Win
        rec_signal = self.state @ W_total.transpose()

        h_new = (1 - self.leaky_rate) * self.state + self.leaky_rate * bm.tanh(in_signal + rec_signal)
        self.state.value = h_new
        return h_new

    def reset_state(self, batch_size=1):
        self.state.value = bm.zeros((batch_size, self.num_hidden))
        self.F.value = bm.zeros((self.num_hidden, self.num_hidden))

# -------------------------------
# 5. 模型定义与训练
# -------------------------------
class MambaESN(bp.DynamicalSystem):
    def __init__(self, num_in, num_hidden, num_out, **kwargs):
        super(MambaESN, self).__init__()
        self.r = MambaLikeReservoir(num_in, num_hidden, **kwargs)
        self.o = bp.dnn.Dense(num_hidden, num_out, mode=bm.training_mode)

    def update(self, x):
        return x >> self.r >> self.o

    def reset(self, batch_size=1):
        self.r.reset_state(batch_size)

# 输入输出都是4维 (x1, y1, x2, y2)
model = MambaESN(
    num_in=4, num_hidden=300, num_out=4,
    sr=1.0, leaky_rate=0.5, lambda_dec=0.9, eta=0.005
)

model.reset(batch_size=1)
trainer = bp.RidgeTrainer(model, alpha=1e-5)

print("Status: Warmup...")
_ = trainer.predict(x_warm)

print("Status: Training...")
_ = trainer.fit([x_train, y_train])

print("Status: Predicting...")
ys_predict = trainer.predict(x_test)

mse = bp.losses.mean_squared_error(ys_predict, y_test)
print(f"Prediction MSE: {mse}")

# -------------------------------
# 6. 结果可视化
# -------------------------------
ys_np = bm.as_numpy(ys_predict)
yt_np = bm.as_numpy(y_test)

start, end = 0, 1000  # 展示前1000步

fig = plt.figure(figsize=(15, 8))

# --- 子图1: 时序对比 (以第二个球的X坐标为例) ---
ax1 = fig.add_subplot(211)
ax1.plot(ys_np[0, start:end, 2], lw=2, label="Prediction (Ball 2 X)")
ax1.plot(yt_np[0, start:end, 2], linestyle="--", lw=1.5, color='k', alpha=0.7, label="Ground Truth")
ax1.set_title(f'Double Pendulum Time Series (Ball 2 X-coord) | MSE: {mse:.6f}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- 子图2: 空间轨迹对比 (XY平面) ---
ax2 = fig.add_subplot(212)
# 绘制真实轨迹
ax2.plot(yt_np[0, start:end, 2], yt_np[0, start:end, 3], 'k--', alpha=0.5, label='True Trajectory (Ball 2)')
# 绘制预测轨迹
ax2.plot(ys_np[0, start:end, 2], ys_np[0, start:end, 3], 'r-', alpha=0.8, label='Predicted Trajectory (Ball 2)')
ax2.set_title("Spatial Trajectory (X vs Y)")
ax2.set_xlabel("X Position")
ax2.set_ylabel("Y Position")
ax2.axis('equal') # 保持比例一致
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# -------------------------------
# 7. 隐状态 PCA 分析
# -------------------------------
inputs = bm.moveaxis(x_test, 1, 0)

def run_step(x_t):
    model.update(x_t)
    return model.r.state

print("Collecting hidden states for PCA...")
model.reset(1)
states = bm.for_loop(run_step, inputs, progress_bar=True)
hidden_states_np = bm.as_numpy(bm.moveaxis(states, 0, 1))

time_steps = 3000
limit = min(time_steps, hidden_states_np.shape[1])
trajectory_high_dim = hidden_states_np[0, :limit, :]

pca = PCA(n_components=3)
trajectory_low_dim = pca.fit_transform(trajectory_high_dim)
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

# PCA 可视化
fig = plt.figure(figsize=(16, 7))

# 2D PCA
ax1 = fig.add_subplot(121)
colors = plt.cm.plasma(bm.linspace(0, 1, limit))
ax1.scatter(trajectory_low_dim[:, 0], trajectory_low_dim[:, 1],
            c=colors, s=2, alpha=0.6)
ax1.set_title('Reservoir Dynamics (PCA 2D)\nColor = Time')
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')

# 3D PCA
ax2 = fig.add_subplot(122, projection='3d')
sc = ax2.scatter(trajectory_low_dim[:, 0],
                 trajectory_low_dim[:, 1],
                 trajectory_low_dim[:, 2],
                 c=colors, s=2, alpha=0.6)
ax2.set_title('Reservoir Dynamics (PCA 3D)')
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
ax2.set_zlabel('PC 3')

plt.colorbar(sc, ax=[ax1, ax2], label='Time Evolution', shrink=0.8)
plt.show()
