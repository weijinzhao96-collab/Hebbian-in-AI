#加上Fast weight 的 echo state network 实现 小球双摆 任务
#添加 BP 的训练wout，fv，fk，batch_size=32

import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.decomposition import PCA
import time

# -------------------------------
# 1. 环境与双摆数据生成
# -------------------------------
bm.set_environment(x64=True, mode=bm.training_mode)
bm.set_platform('cpu')

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
    init_state = [np.pi/2, 0, np.pi/2, 0] # 初始水平放置

    t = np.arange(0, steps * dt, dt)
    y = odeint(double_pendulum_ode, init_state, t, args=(m1, m2, l1, l2, g))

    theta1, theta2 = y[:, 0], y[:, 2]

    # 转换为笛卡尔坐标 (x1, y1, x2, y2)
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)

    # 归一化：双摆最大半径为 L1+L2=2，除以 2 将数据缩放到 [-1, 1] 区间，利于神经网络训练
    return np.stack([x1, y1, x2, y2], axis=1) / 2.0

# 生成数据
dt = 0.05
total_steps = 25000
raw_data = generate_data(total_steps, dt)  # Shape: (25000, 4)

# 构建 Batch 数据
BATCH_SIZE = 32
TIME_STEPS = 500
total_len = BATCH_SIZE * TIME_STEPS

data_subset = raw_data[:total_len + 1]
# X: 当前时刻状态 (x1, y1, x2, y2)
X_train = data_subset[:total_len].reshape(BATCH_SIZE, TIME_STEPS, 4)
# Y: 下一时刻状态
Y_train = data_subset[1:total_len+1].reshape(BATCH_SIZE, TIME_STEPS, 4)

X_train = bm.asarray(X_train)
Y_train = bm.asarray(Y_train)

print(f"训练数据形状: {X_train.shape}")

# -------------------------------
# 2. 定义支持多维输入的 Hebbian ESN
# -------------------------------
class IndependentHebbianESN(bp.DynamicalSystem):
    def __init__(self, num_in, num_hidden, num_out,
                 sr=0.95, leaky_rate=0.4, lambda_dec=0.9, eta=0.005):
        super(IndependentHebbianESN, self).__init__()
        self.num_hidden = num_hidden
        self.leaky_rate = leaky_rate
        self.lambda_dec = lambda_dec
        self.eta = eta

        # --- 静态权重 ---
        w_init = bm.random.normal(size=(num_hidden, num_hidden))
        eig_vals = bm.linalg.eigvals(w_init)
        spectral_radius = bm.max(bm.abs(eig_vals))
        self.W = bm.Variable((w_init / spectral_radius) * sr)

        # 输入权重
        self.Win = bm.Variable(bm.random.uniform(-0.2, 0.2, (num_hidden, num_in)))

        # --- Hebbian 控制层 (新增) ---
        # 用于将多维输入映射到 Hidden 维度，以便计算 Hebbian 更新
        self.W_control = bm.Variable(bm.random.normal(0, 0.1, (num_in, num_hidden)))

        # Hebbian 动态因子
        self.Fv = bm.Variable(bm.random.normal(size=(num_hidden,)))
        self.Fk = bm.Variable(bm.random.normal(size=(num_hidden,)))

        # 输出层 (Trainable)
        self.w_ro = bm.TrainVar(bm.random.uniform(-0.1, 0.1, (num_hidden, num_out)))
        self.b_ro = bm.TrainVar(bm.zeros((num_out,)))

        # --- 动态状态 ---
        self.h = bm.Variable(bm.zeros((1, num_hidden)), batch_axis=0)
        self.F = bm.Variable(bm.zeros((1, num_hidden, num_hidden)), batch_axis=0)

    def reset_state(self, batch_size):
        self.h.value = bm.zeros((batch_size, self.num_hidden))
        self.F.value = bm.zeros((batch_size, self.num_hidden, self.num_hidden))

    def update(self, x):
        """
        x: (batch, 4)
        """
        # --- Hebbian 动态权重更新 ---
        # 1. 将输入投影到控制向量 u
        u = x @ self.W_control # (batch, hidden)

        # 2. 利用控制向量计算 Key 和 Value
        v = u * self.Fv  # (batch, hidden)
        k = u * self.Fk  # (batch, hidden)

        # 3. 计算外积并更新快速权重 F
        delta_F = bm.einsum('bi,bj->bij', v, k)
        self.F.value = self.lambda_dec * self.F + self.eta * delta_F

        # --- 状态更新 ---
        W_total = self.W + self.F
        h_exp = bm.expand_dims(self.h, 1)             # (batch, 1, hidden)
        W_total_T = bm.transpose(W_total, (0, 2, 1))    # (batch, hidden, hidden)

        # 递归输入
        rec_term = bm.matmul(h_exp, W_total_T)[:, 0, :] # (batch, hidden)
        # 外部输入
        in_term = x @ self.Win.transpose()            # (batch, hidden)

        self.h.value = (1 - self.leaky_rate) * self.h + self.leaky_rate * bm.tanh(in_term + rec_term)

        return self.h @ self.w_ro + self.b_ro

    def predict_sequence(self, xs):
        xs_T = bm.moveaxis(xs, 1, 0)
        preds_T = bm.for_loop(self.update, xs_T)
        return bm.moveaxis(preds_T, 0, 1)

# -------------------------------
# 3. 训练模型
# -------------------------------
# 输入输出均为 4 维 (x1, y1, x2, y2)
model = IndependentHebbianESN(num_in=4, num_hidden=200, num_out=4, eta=0.01)
optimizer = bp.optim.Adam(lr=0.005, train_vars=model.train_vars().unique())

def loss_fn(xs, ys):
    batch_size = xs.shape[0]
    model.reset_state(batch_size)
    preds = model.predict_sequence(xs)
    warmup = 50
    # 计算 MSE Loss
    return bm.mean(bm.square(preds[:, warmup:, :] - ys[:, warmup:, :]))

@bm.jit
def train_step(xs, ys):
    grads, loss = bm.grad(loss_fn, grad_vars=model.train_vars().unique(), return_value=True)(xs, ys)
    optimizer.update(grads)
    return loss

print("Starting Double Pendulum Training...")
start_time = time.time()
losses = []

for epoch in range(100):
    loss = train_step(X_train, Y_train)
    losses.append(float(loss))
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {losses[-1]:.6f}")

print(f"Training finished in {time.time() - start_time:.2f}s")

# -------------------------------
# 4. 验证与空间可视化
# -------------------------------
warmup_len = 200
test_len = 1000
start_idx = 20000

# 准备测试数据 (Batch=1)
X_test_seq = raw_data[start_idx - warmup_len : start_idx + test_len].reshape(1, -1, 4)
Y_test_true = raw_data[start_idx + 1 : start_idx + test_len + 1].reshape(1, -1, 4)

model.reset_state(1)
# 转换为 BrainPy 数组
X_test_bm = bm.asarray(X_test_seq)
Y_pred_seq = model.predict_sequence(X_test_bm)

# 转换为 Numpy 并切除 Warmup
preds = bm.as_numpy(Y_pred_seq)[0, warmup_len:, :]
trues = Y_test_true[0, warmup_len:, :]

# --- 可视化 ---
fig = plt.figure(figsize=(15, 6))

# 图1: 时序对比 (第二个球的 X 坐标)
ax1 = fig.add_subplot(121)
ax1.plot(trues[:, 2], 'k--', alpha=0.6, label='True (Ball 2 X)')
ax1.plot(preds[:, 2], 'r', alpha=0.8, label='Pred (Ball 2 X)')
ax1.set_title("Time Series Prediction")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图2: 空间轨迹对比 (Ball 2 的 XY 平面)
ax2 = fig.add_subplot(122)
ax2.plot(trues[:, 2], trues[:, 3], 'k--', alpha=0.4, label='True Trajectory')
ax2.plot(preds[:, 2], preds[:, 3], 'r-', alpha=0.8, linewidth=1.5, label='Predicted Trajectory')
ax2.set_title("Spatial Trajectory (Ball 2)")
ax2.set_xlabel("X Position")
ax2.set_ylabel("Y Position")
ax2.axis('equal')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# -------------------------------
# 5. 隐状态 PCA 分析
# -------------------------------
inputs = bm.moveaxis(X_test_bm, 1, 0)

def run_step(x_t):
    model.update(x_t)
    return model.h

print("Collecting hidden states...")
model.reset_state(1)
states = bm.for_loop(run_step, inputs, progress_bar=False)
hidden_states_np = bm.as_numpy(bm.moveaxis(states, 0, 1))

# 分析
trajectory_high_dim = hidden_states_np[0, warmup_len:, :]
pca = PCA(n_components=3)
trajectory_low_dim = pca.fit_transform(trajectory_high_dim)
print(f"Explained Variance: {pca.explained_variance_ratio_}")

# PCA 可视化
fig = plt.figure(figsize=(16, 7))
times = np.arange(trajectory_low_dim.shape[0])
colors = plt.cm.plasma(np.linspace(0, 1, len(times)))

# 2D
ax1 = fig.add_subplot(121)
ax1.scatter(trajectory_low_dim[:, 0], trajectory_low_dim[:, 1], c=colors, s=2, alpha=0.7)
ax1.set_title('Reservoir Dynamics (PCA 2D)\nColor represents Time')
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')

# 3D
ax2 = fig.add_subplot(122, projection='3d')
sc = ax2.scatter(trajectory_low_dim[:, 0], trajectory_low_dim[:, 1], trajectory_low_dim[:, 2],
                 c=colors, s=2, alpha=0.7)
ax2.set_title('Reservoir Dynamics (PCA 3D)')
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
ax2.set_zlabel('PC 3')

plt.colorbar(sc, ax=[ax1, ax2], label='Time Evolution', shrink=0.8)
plt.show()

