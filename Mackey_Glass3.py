#加上Fast weight 的 echo state network 实现 prediction of Mackey-Glass time series的任务
#添加 BP 的训练wout，batch_size=20
import brainpy as bp
import brainpy.math as bm
import brainpy_datasets as bd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import time

# -------------------------------
# 1. 环境与数据准备
# -------------------------------
# 启用 x64 精度计算，训练模式
bm.set_environment(x64=True, mode=bm.training_mode)
bm.set_platform('cpu')

# Mackey-Glass 时间序列
dt = 0.1
mg_data = bd.chaos.MackeyGlassEq(25000, dt=dt, tau=17, beta=0.2, gamma=0.1, n=10)
xs_data = mg_data.xs

# 构建 Batch 数据
BATCH_SIZE = 20
TIME_STEPS = 1000
total_len = BATCH_SIZE * TIME_STEPS

data_subset = xs_data[:total_len + 1]
X_train = data_subset[:total_len].reshape(BATCH_SIZE, TIME_STEPS, 1)
Y_train = data_subset[1:total_len+1].reshape(BATCH_SIZE, TIME_STEPS, 1)

# 转换为 BrainPy 数组
X_train = bm.asarray(X_train)
Y_train = bm.asarray(Y_train)

# -------------------------------
# 2. 定义独立 Hebbian ESN
# -------------------------------
class IndependentHebbianESN(bp.DynamicalSystem):
    """
    Echo State Network with independent Hebbian plasticity for batch inputs.
    """
    def __init__(self, num_in, num_hidden, num_out,
                 sr=0.9, leaky_rate=0.3, lambda_dec=0.95, eta=0.0001):
        super(IndependentHebbianESN, self).__init__()
        self.num_hidden = num_hidden
        self.leaky_rate = leaky_rate
        self.lambda_dec = lambda_dec
        self.eta = eta

        # --- 静态权重 ---
        w_init = bm.random.normal(size=(num_hidden, num_hidden))
        current_sr = bm.max(bm.abs(bm.linalg.eigvals(w_init)))
        self.W = bm.Variable((w_init / current_sr) * sr)
        self.Win = bm.Variable(bm.random.uniform(-0.1, 0.1, (num_hidden, num_in)))

        # Hebbian 动态权重参数
        self.Fv = bm.Variable(bm.random.normal(size=(num_hidden,)))
        self.Fk = bm.Variable(bm.random.normal(size=(num_hidden,)))

        # 输出层权重
        self.w_ro = bm.TrainVar(bm.random.uniform(-0.1, 0.1, (num_hidden, num_out)))
        self.b_ro = bm.TrainVar(bm.zeros((num_out,)))

        # --- 动态状态 ---
        # h: 隐状态，batch_axis=0 支持 Batch Size > 1
        self.h = bm.Variable(bm.zeros((1, num_hidden)), batch_axis=0)
        self.F = bm.Variable(bm.zeros((1, num_hidden, num_hidden)), batch_axis=0)

    def reset_state(self, batch_size):
        """重置隐状态和动态权重"""
        self.h.value = bm.zeros((batch_size, self.num_hidden))
        self.F.value = bm.zeros((batch_size, self.num_hidden, self.num_hidden))

    def update(self, x):
        """
        单步更新：输入 x -> 更新 F -> 更新隐状态 -> 输出预测
        x: (batch, 1)
        """
        # --- Hebbian 动态权重更新 ---
        v = x * self.Fv  # (batch, hidden)
        k = x * self.Fk  # (batch, hidden)
        delta_F = bm.einsum('bi,bj->bij', v, k)  # (batch, hidden, hidden)
        self.F.value = self.lambda_dec * self.F + self.eta * delta_F

        # --- 状态更新 ---
        W_total = self.W + self.F
        h_exp = bm.expand_dims(self.h, 1)           # (batch, 1, hidden)
        W_total_T = bm.transpose(W_total, (0, 2, 1))  # (batch, hidden, hidden)
        rec_term = bm.matmul(h_exp, W_total_T)[:, 0, :]  # (batch, hidden)
        in_term = x @ self.Win.transpose()          # (batch, hidden)

        self.h.value = (1 - self.leaky_rate) * self.h + self.leaky_rate * bm.tanh(in_term + rec_term)

        # 输出层计算
        return self.h @ self.w_ro + self.b_ro

    def predict_sequence(self, xs):
        """
        对整个时间序列进行预测
        xs: (batch, time, feature)
        """
        xs_T = bm.moveaxis(xs, 1, 0)  # 时间步在前
        preds_T = bm.for_loop(self.update, xs_T)
        return bm.moveaxis(preds_T, 0, 1)

# -------------------------------
# 3. 训练模型
# -------------------------------
model = IndependentHebbianESN(num_in=1, num_hidden=100, num_out=1, eta=0.005)
optimizer = bp.optim.Adam(lr=0.01, train_vars=model.train_vars().unique())

def loss_fn(xs, ys):
    """计算批次 MSE（丢弃 warmup 部分）"""
    batch_size = xs.shape[0]
    model.reset_state(batch_size)
    preds = model.predict_sequence(xs)
    warmup = 100
    return bm.mean(bm.square(preds[:, warmup:, :] - ys[:, warmup:, :]))

@bm.jit
def train_step(xs, ys):
    grads, loss = bm.grad(loss_fn, grad_vars=model.train_vars().unique(), return_value=True)(xs, ys)
    optimizer.update(grads)
    return loss

print("Starting Independent Batch Training...")
start = time.time()
losses = []

for epoch in range(50):
    loss = train_step(X_train, Y_train)
    losses.append(float(loss))
    if epoch % 5 == 0:
        print(f"Epoch {epoch} | Loss: {losses[-1]:.6f}")

print(f"Training finished in {time.time() - start:.2f}s")

# -------------------------------
# 4. 验证预测
# -------------------------------
warmup_len = 1000
test_len = 2000
start_idx = 20000

# 准备输入序列（包含预热段 + 测试段）
total_input = xs_data[start_idx - warmup_len : start_idx + test_len].reshape(1, -1, 1)
Y_true = xs_data[start_idx + 1 : start_idx + test_len + 1].reshape(1, -1, 1)

model.reset_state(1)
total_pred = model.predict_sequence(bm.asarray(total_input))

# 丢弃预热部分
Y_pred_plot = total_pred[:, warmup_len:, :]

# 可视化预测
pred_np = bm.as_numpy(Y_pred_plot)[0, :, 0]
true_np = bm.as_numpy(Y_true)[0, :, 0]

plt.figure(figsize=(12, 5))
plt.plot(true_np, label='True', linestyle='--', linewidth=2)
plt.plot(pred_np, label='Pred', alpha=0.9, linewidth=2)
plt.title(f"Prediction with Warmup (First {warmup_len} steps discarded)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# -------------------------------
# 5. 隐状态收集与 PCA 分析
# -------------------------------
inputs = bm.moveaxis(total_input, 1, 0)  # (Time, Batch, Feature)

def run_step(x_t):
    """单步返回隐藏状态"""
    model.update(x_t)
    return model.h

print("Running simulation (collecting hidden states)...")
model.reset_state(1)
states = bm.for_loop(run_step, inputs, progress_bar=True)
hidden_states_np = bm.as_numpy(bm.moveaxis(states, 0, 1))  # (Batch, Time, Hidden)
print(f"Collected Hidden States Shape: {hidden_states_np.shape}")

# 只分析第 0 个样本，丢弃预热
warmup_steps = 500
analyze_steps = 3000
trajectory_high_dim = hidden_states_np[0, warmup_steps:warmup_steps+analyze_steps, :]
print(f"Analyzing shape (after warmup): {trajectory_high_dim.shape}")

# PCA 降维
pca = PCA(n_components=3)
trajectory_low_dim = pca.fit_transform(trajectory_high_dim)
print(f"Explained Variance Ratio (PC1, PC2, PC3): {pca.explained_variance_ratio_}")
print(f"Total Variance Explained: {np.sum(pca.explained_variance_ratio_):.2%}")

# -------------------------------
# 6. 可视化 PCA 投影
# -------------------------------
figure = plt.figure(figsize=(16, 7))
times = np.arange(trajectory_low_dim.shape[0])
colors = plt.cm.viridis(np.linspace(0, 1, len(times)))

# 2D 投影 (PC1 vs PC2)
ax1 = figure.add_subplot(121)
ax1.plot(trajectory_low_dim[:, 0], trajectory_low_dim[:, 1], color='gray', alpha=0.3, linewidth=0.8)
ax1.scatter(trajectory_low_dim[:, 0], trajectory_low_dim[:, 1], c=colors, s=3, alpha=0.8)
ax1.set_title('2D PCA Projection (Mackey-Glass Attractor in ESN)', fontsize=12)
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')
ax1.grid(True, alpha=0.3)

# 3D 投影 (PC1 vs PC2 vs PC3)
ax2 = figure.add_subplot(122, projection='3d')
ax2.plot(trajectory_low_dim[:, 0], trajectory_low_dim[:, 1], trajectory_low_dim[:, 2],
         color='gray', alpha=0.3, linewidth=0.5)
ax2.scatter(trajectory_low_dim[:, 0], trajectory_low_dim[:, 1], trajectory_low_dim[:, 2],
            c=colors, s=3, alpha=0.8)
ax2.set_title('3D Attractor Reconstruction', fontsize=12)
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
ax2.set_zlabel('PC 3')

cbar = plt.colorbar(ax2.collections[0], ax=[ax1, ax2], shrink=0.8, pad=0.05)
cbar.set_label('Time Evolution (Start -> End)')

plt.suptitle("Neural State Space Trajectory Analysis", fontsize=14, y=0.95)
plt.show()
