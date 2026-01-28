#加上Fast weight 的 echo state network 实现 正弦波预测任务
#添加 BP 的训练wout,fk,fv，batch_size=20

import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ==========================================
# 1. 环境配置
# ==========================================
bm.set_environment(x64=True, mode=bm.training_mode)
bm.set_platform('cpu')  # 或 'gpu'

# ==========================================
# 2. 数据生成（正弦波）
# ==========================================
dt = 0.1
num_steps = 25000
ts = bm.arange(0, num_steps * dt, dt)
xs_data = bm.sin(ts * 0.2)  # 正弦波，频率适中
xs_data = xs_data[:, None]  # 增加特征维度 (Time, 1)

# ==========================================
# 3. 构造批量训练数据
# ==========================================
BATCH_SIZE = 20
TIME_STEPS = 1000
total_len = BATCH_SIZE * TIME_STEPS

data_subset = xs_data[:total_len + 1]
X_train = data_subset[:total_len].reshape(BATCH_SIZE, TIME_STEPS, 1)
Y_train = data_subset[1:total_len + 1].reshape(BATCH_SIZE, TIME_STEPS, 1)

X_train = bm.asarray(X_train)
Y_train = bm.asarray(Y_train)

# ==========================================
# 4. 定义独立 Hebbian ESN
# ==========================================
class IndependentHebbianESN(bp.DynamicalSystem):
    def __init__(self, num_in, num_hidden, num_out,
                 sr=0.9, leaky_rate=0.3, lambda_dec=0.95, eta=0.0001):
        super().__init__()
        self.num_hidden = num_hidden
        self.leaky_rate = leaky_rate
        self.lambda_dec = lambda_dec
        self.eta = eta

        # --- 静态权重 ---
        w_init = bm.random.normal(size=(num_hidden, num_hidden))
        sr_current = bm.max(bm.abs(bm.linalg.eigvals(w_init)))
        self.W = bm.Variable((w_init / sr_current) * sr)
        self.Win = bm.Variable(bm.random.uniform(-0.1, 0.1, (num_hidden, num_in)))

        # --- Hebbian 动态权重 ---
        self.Fv = bm.TrainVar(bm.random.normal(size=(num_hidden,)))
        self.Fk = bm.TrainVar(bm.random.normal(size=(num_hidden,)))
        self.F = bm.Variable(bm.zeros((1, num_hidden, num_hidden)), batch_axis=0)

        # --- 输出权重 ---
        self.w_ro = bm.TrainVar(bm.random.uniform(-0.1, 0.1, (num_hidden, num_out)))
        self.b_ro = bm.TrainVar(bm.zeros((num_out,)))

        # --- 隐藏状态 ---
        self.h = bm.Variable(bm.zeros((1, num_hidden)), batch_axis=0)

    def reset_state(self, batch_size):
        """重置隐藏状态和动态权重"""
        self.h.value = bm.zeros((batch_size, self.num_hidden))
        self.F.value = bm.zeros((batch_size, self.num_hidden, self.num_hidden))

    def update(self, x):
        """单步更新"""
        # x: (batch, 1)
        v = x * self.Fv
        k = x * self.Fk
        delta_F = bm.einsum('bi,bj->bij', v, k)
        self.F.value = self.lambda_dec * self.F + self.eta * delta_F

        W_total = self.W + self.F
        h_exp = bm.expand_dims(self.h, 1)
        W_total_T = bm.transpose(W_total, (0, 2, 1))
        rec_term = bm.matmul(h_exp, W_total_T)[:, 0, :]
        in_term = x @ self.Win.T
        self.h.value = (1 - self.leaky_rate) * self.h + self.leaky_rate * bm.tanh(in_term + rec_term)

        return self.h @ self.w_ro + self.b_ro

    def predict_sequence(self, xs):
        xs_T = bm.moveaxis(xs, 1, 0)
        preds_T = bm.for_loop(self.update, xs_T)
        return bm.moveaxis(preds_T, 0, 1)

# ==========================================
# 5. 训练流程
# ==========================================
model = IndependentHebbianESN(num_in=1, num_hidden=100, num_out=1, eta=0.005)
optimizer = bp.optim.Adam(lr=0.01, train_vars=model.train_vars().unique())

def loss_fn(xs, ys):
    model.reset_state(xs.shape[0])
    preds = model.predict_sequence(xs)
    warmup = 100
    return bm.mean(bm.square(preds[:, warmup:, :] - ys[:, warmup:, :]))

@bm.jit
def train_step(xs, ys):
    grads, loss = bm.grad(loss_fn, grad_vars=model.train_vars().unique(), return_value=True)(xs, ys)
    optimizer.update(grads)
    return loss

# 训练
print("Training Independent Hebbian ESN on Sine Wave...")
import time
start_time = time.time()
losses = []

for epoch in range(50):
    loss = train_step(X_train, Y_train)
    losses.append(float(loss))
    if epoch % 5 == 0:
        print(f"Epoch {epoch} | Loss: {losses[-1]:.8f}")

print(f"Training finished in {time.time() - start_time:.2f}s")

# ==========================================
# 6. 验证与预测
# ==========================================
warmup_len = 1000
test_len = 2000
start_idx = 20000

total_input = xs_data[start_idx - warmup_len : start_idx + test_len]
total_input = total_input.reshape(1, -1, 1)
Y_true = xs_data[start_idx + 1 : start_idx + test_len + 1].reshape(1, -1, 1)

model.reset_state(1)
total_pred = model.predict_sequence(bm.asarray(total_input))
Y_pred_plot = total_pred[:, warmup_len:, :]

# 绘图
pred_np = bm.as_numpy(Y_pred_plot)[0, :, 0]
true_np = bm.as_numpy(Y_true)[0, :, 0]

plt.figure(figsize=(12, 5))
plt.plot(true_np, label='True (Sine)', linestyle='--', linewidth=2)
plt.plot(pred_np, label='Pred (Hebbian ESN)', alpha=0.9, linewidth=2)
plt.title(f"Sine Wave Prediction (MSE: {np.mean((pred_np - true_np)**2):.2e})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ==========================================
# 7. PCA 隐藏状态分析
# ==========================================
inputs = bm.moveaxis(total_input, 1, 0)

def run_step_for_pca(x_t):
    model.update(x_t)
    return model.h

print("Collecting hidden states for PCA...")
model.reset_state(1)
states = bm.for_loop(run_step_for_pca, inputs, progress_bar=True)

hidden_states_np = bm.as_numpy(bm.moveaxis(states, 0, 1))  # (Batch, Time, Hidden)
print(f"Hidden States Shape: {hidden_states_np.shape}")

# 选择分析区间
batch_idx = 0
full_trajectory = hidden_states_np[batch_idx]
warmup_steps = 500
analyze_steps = 3000
start_idx_pca = warmup_steps
end_idx_pca = min(start_idx_pca + analyze_steps, full_trajectory.shape[0])
trajectory_high_dim = full_trajectory[start_idx_pca:end_idx_pca, :]

# PCA
pca = PCA(n_components=3)
trajectory_low_dim = pca.fit_transform(trajectory_high_dim)
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

# ==========================================
# 8. PCA 可视化
# ==========================================
figure = plt.figure(figsize=(16, 7))
times = np.arange(trajectory_low_dim.shape[0])
colors = plt.cm.viridis(np.linspace(0, 1, len(times)))

# --- 2D 投影 ---
ax1 = figure.add_subplot(121)
ax1.plot(trajectory_low_dim[:, 0], trajectory_low_dim[:, 1],
         color='gray', alpha=0.3, linewidth=0.8)
ax1.scatter(trajectory_low_dim[:, 0], trajectory_low_dim[:, 1],
            c=colors, s=3, alpha=0.8)
ax1.set_title('2D PCA Projection (Sine Limit Cycle)')
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')
ax1.grid(True, alpha=0.3)

# --- 3D 投影 ---
ax2 = figure.add_subplot(122, projection='3d')
ax2.plot(trajectory_low_dim[:, 0], trajectory_low_dim[:, 1], trajectory_low_dim[:, 2],
         color='gray', alpha=0.3, linewidth=0.5)
sc2 = ax2.scatter(trajectory_low_dim[:, 0],
                  trajectory_low_dim[:, 1],
                  trajectory_low_dim[:, 2],
                  c=colors, s=3, alpha=0.8)
ax2.set_title('3D Attractor Reconstruction')
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
ax2.set_zlabel('PC 3')

cbar = plt.colorbar(sc2, ax=[ax1, ax2], shrink=0.8, pad=0.05)
cbar.set_label('Time Evolution')

plt.suptitle("Neural State Space Trajectory Analysis (Sine Task)", fontsize=14, y=0.95)
plt.show()
