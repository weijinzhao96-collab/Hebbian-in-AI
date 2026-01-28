#用一般的 echo state network 实现小球双摆预测任务

import brainpy.math as bm
bm.set_platform('gpu') # 如果没有GPU，BrainPy会自动回退到CPU
import brainpy as bp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# enable x64 computation for higher precision in chaos
bm.set_environment(x64=True, mode=bm.batching_mode)

print(f"BrainPy 运行平台: {bp.math.get_platform()}")

# ==========================================
# 1. 数据生成: 小球双摆 (Double Pendulum)
# ==========================================

def double_pendulum_derivs(state, t, m1, m2, l1, l2, g):
    theta1, z1, theta2, z2 = state
    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

    theta1dot = z1
    z1dot = (m2*g*np.sin(theta2)*c - m2*s*(l1*z1**2*c + l2*z2**2) - (m1+m2)*g*np.sin(theta1)) / (l1 * (m1 + m2*s**2))
    theta2dot = z2
    z2dot = ((m1+m2)*(l1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + m2*l2*z2**2*s*c) / (l2 * (m1 + m2*s**2))

    return theta1dot, z1dot, theta2dot, z2dot

def generate_double_pendulum(steps, dt):
    # 物理参数
    m1, m2 = 1.0, 1.0
    l1, l2 = 1.0, 1.0
    g = 9.8

    # 初始状态 (角度1, 角速度1, 角度2, 角速度2)
    init_state = [np.pi/2, 0, np.pi/2, 0]

    t = np.arange(0, steps * dt, dt)

    # 积分求解
    y = odeint(double_pendulum_derivs, init_state, t, args=(m1, m2, l1, l2, g))

    theta1, theta2 = y[:, 0], y[:, 2]

    # 转换为笛卡尔坐标 (x1, y1) -> 第一个球, (x2, y2) -> 第二个球
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)

    # 拼接数据: shape (steps, 4) -> [x1, y1, x2, y2]
    data = np.stack([x1, y1, x2, y2], axis=1)
    return data

# 设置参数
dt = 0.05  # 时间步长
total_steps = 25000
raw_data = generate_double_pendulum(total_steps, dt)

# 转换为 BrainPy 数组
raw_data = bm.asarray(raw_data)

print(f"原始数据维度: {raw_data.shape} (Time, Features=4)")

# ==========================================
# 2. 数据处理与切分
# ==========================================

def get_data(data, t_warm, t_forecast, t_train, sample_rate=1):
    # data shape: (Time, Features)
    warmup = int(t_warm / dt)
    forecast = int(t_forecast / dt)
    train_length = int(t_train / dt)

    # 这里的维度调整很重要: (Batch, Time, Features)
    # 我们的 Batch = 1

    # Warmup: 给 Reservoir 预热的状态
    X_warm = data[:warmup:sample_rate]
    X_warm = bm.expand_dims(X_warm, 0)

    # Training Input
    X_train = data[warmup: warmup+train_length: sample_rate]
    X_train = bm.expand_dims(X_train, 0)

    # Training Target (预测未来 forecast 步)
    Y_train = data[warmup+forecast: warmup+train_length+forecast: sample_rate]
    Y_train = bm.expand_dims(Y_train, 0)

    # Test Input
    X_test = data[warmup + train_length: -forecast: sample_rate]
    X_test = bm.expand_dims(X_test, 0)

    # Test Target
    Y_test = data[warmup + train_length + forecast::sample_rate]
    Y_test = bm.expand_dims(Y_test, 0)

    return X_warm, X_train, Y_train, X_test, Y_test

# 准备数据：预热 100s, 预测步长 1 (即dt), 训练时长 800s
x_warm, x_train, y_train, x_test, y_test = get_data(raw_data, 100, dt, 800)

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)

# 可视化部分训练数据 (只看第二个球的 X 坐标)
sample = 1000
plt.figure(figsize=(15, 4))
plt.plot(x_train[0, :sample, 2], label="Ball 2 - X (Input)")
plt.plot(y_train[0, :sample, 2], label="Ball 2 - X (Target)")
plt.title("Double Pendulum Training Data Sample (Ball 2 X-coord)")
plt.legend()
plt.show()

# ==========================================
# 3. 构建 ESN 模型
# ==========================================

class ESN(bp.DynamicalSystem):
  def __init__(self, num_in, num_hidden, num_out, sr=1.0, leaky_rate=0.3,
               Win_initializer=bp.init.Uniform(-0.2, 0.2)):
    super(ESN, self).__init__()
    # Reservoir
    self.r = bp.dyn.Reservoir(
        num_in, num_hidden,
        Win_initializer=Win_initializer,
        spectral_radius=sr,
        leaky_rate=leaky_rate,
    )
    # Readout (Trainable)
    self.o = bp.dnn.Dense(num_hidden, num_out, mode=bm.training_mode)

  def update(self, x):
    return x >> self.r >> self.o

# 特征数为4 (x1, y1, x2, y2)
num_features = 4
num_hidden = 500 # 增加隐藏层神经元以捕捉更复杂的动力学

model = ESN(num_in=num_features, num_hidden=num_hidden, num_out=num_features,
            sr=1.1, leaky_rate=0.5)

model.reset(1) # Reset for batch size 1

print(f"Win shape: {model.r.Win.shape}")
print(f"Wrec shape: {model.r.Wrec.shape}")
print(f"Wout shape: {model.o.W.shape}")


# ==========================================
# 4. 训练 (Ridge Regression)
# ==========================================

trainer = bp.RidgeTrainer(model, alpha=1e-5)

print("Training...")
# warmup
_ = trainer.predict(x_warm)
# train
_ = trainer.fit([x_train, y_train])

print("Predicting...")
ys_predict = trainer.predict(x_test)

# ==========================================
# 5. 结果验证与可视化
# ==========================================

start, end = 0, 1000
test_len = ys_predict.shape[1]
limit = min(start+end, test_len)

ys_np = bm.as_numpy(ys_predict)
yt_np = bm.as_numpy(y_test)

# --- 图 1: 时序对比 (Ball 2 X坐标) ---
plt.figure(figsize=(15, 6))
plt.plot(ys_np[0, start:limit, 2], lw=2, label="ESN Prediction")
plt.plot(yt_np[0, start:limit, 2], linestyle="--", lw=1.5, label="Ground Truth")
plt.title(f'Double Pendulum Prediction (Ball 2 X-Axis) - MSE: {bp.losses.mean_squared_error(ys_predict, y_test):.6f}')
plt.legend()
plt.show()

# --- 图 2: 空间轨迹对比 (XY 平面) ---
plt.figure(figsize=(8, 8))
# 绘制真实轨迹
plt.plot(yt_np[0, start:limit, 2], yt_np[0, start:limit, 3], 'k--', alpha=0.6, label='True Trajectory')
# 绘制预测轨迹
plt.plot(ys_np[0, start:limit, 2], ys_np[0, start:limit, 3], 'r-', alpha=0.8, label='Predicted Trajectory')
plt.title("Ball 2 Trajectory in Space (X vs Y)")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# 6. Reservoir 状态 PCA 分析
# ==========================================

from sklearn.decomposition import PCA

# 准备输入数据进行状态采集
inputs = bm.moveaxis(x_test, 1, 0) # (Time, Batch, Features)

def run_step(x_t):
    model.update(x_t)
    return model.r.state

print("Collecting hidden states for PCA...")
model.reset(1)
# 重新运行模型以收集内部状态
states = bm.for_loop(run_step, inputs, progress_bar=True)

# 转换形状: (Batch, Time, Hidden) -> (1, Time, Hidden)
hidden_states_np = bm.as_numpy(bm.moveaxis(states, 0, 1))

time_steps_pca = 3000
limit_pca = min(time_steps_pca, hidden_states_np.shape[1])
trajectory_high_dim = hidden_states_np[0, :limit_pca, :]

# PCA 计算
pca = PCA(n_components=3)
trajectory_low_dim = pca.fit_transform(trajectory_high_dim)

print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

# 可视化 PCA
fig = plt.figure(figsize=(16, 7))

# --- 2D 投影 ---
ax1 = fig.add_subplot(121)
colors = plt.cm.viridis(bm.linspace(0, 1, limit_pca))
ax1.scatter(trajectory_low_dim[:, 0], trajectory_low_dim[:, 1],
            c=colors, s=2, alpha=0.6)
ax1.set_title('Reservoir Dynamics (PCA 2D)\nColor represents Time')
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')
ax1.grid(True, alpha=0.3)

# --- 3D 投影 ---
ax2 = fig.add_subplot(122, projection='3d')
sc = ax2.scatter(trajectory_low_dim[:, 0],
                 trajectory_low_dim[:, 1],
                 trajectory_low_dim[:, 2],
                 c=colors, s=2, alpha=0.6)
ax2.set_title('Reservoir Dynamics (PCA 3D)')
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
ax2.set_zlabel('PC 3')

cbar = plt.colorbar(sc, ax=[ax1, ax2], shrink=0.8)
cbar.set_label('Time Evolution')

plt.show()

