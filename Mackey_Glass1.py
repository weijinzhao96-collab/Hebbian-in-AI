#用一般的 echo state network 实现prediction of Mackey-Glass time series的任务

# import brainpy.math as bm
# import brainpy as bp
# import brainpy_datasets as bd
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
#
# # 设置计算平台为 GPU
# bm.set_platform('gpu')
#
# # 启用 x64 精度计算，并开启批处理模式
# bm.set_environment(x64=True, mode=bm.batching_mode)
#
# # 查看 BrainPy 版本
# print("BrainPy version:", bp.__version__)
#
# # -------------------------------
# # 1. 生成 Mackey-Glass 数据
# # -------------------------------
# dt = 0.1
# mg_data = bd.chaos.MackeyGlassEq(25000, dt=dt, tau=17, beta=0.2, gamma=0.1, n=10)
# ts = mg_data.ts        # 时间序列
# xs = mg_data.xs        # 系统状态
# ys = mg_data.ys        # 输出（同 xs）
#
# # -------------------------------
# # 2. 数据预处理函数
# # -------------------------------
# def get_data(t_warm, t_forcast, t_train, sample_rate=1):
#     """
#     Prepare warmup, training, and testing data for ESN.
#
#     Args:
#         t_warm: Warm-up duration (ms)
#         t_forcast: Forecast horizon (ms)
#         t_train: Training duration (ms)
#         sample_rate: Downsampling rate
#
#     Returns:
#         X_warm, X_train, Y_train, X_test, Y_test
#     """
#     warmup = int(t_warm / dt)       # warmup steps
#     forecast = int(t_forcast / dt)  # steps ahead to predict
#     train_length = int(t_train / dt)
#
#     # Warmup data
#     X_warm = bm.expand_dims(xs[:warmup:sample_rate], 0)
#
#     # Training data
#     X_train = bm.expand_dims(xs[warmup: warmup+train_length: sample_rate], 0)
#     Y_train = bm.expand_dims(xs[warmup+forecast: warmup+train_length+forecast: sample_rate], 0)
#
#     # Testing data
#     X_test = bm.expand_dims(xs[warmup + train_length: -forecast: sample_rate], 0)
#     Y_test = bm.expand_dims(xs[warmup + train_length + forecast::sample_rate], 0)
#
#     return X_warm, X_train, Y_train, X_test, Y_test
#
# # 获取训练/测试数据
# x_warm, x_train, y_train, x_test, y_test = get_data(100, 1, 20000)
#
# # -------------------------------
# # 3. 可视化训练数据与预测目标
# # -------------------------------
# sample = 3000
# plt.figure(figsize=(15, 5))
# plt.plot(x_train[0, :sample], label="Training data")
# plt.plot(y_train[0, :sample], label="True prediction")
# plt.legend()
# plt.title("Training Data and Target Prediction")
# plt.show()
#
# # -------------------------------
# # 4. 定义 Echo State Network (ESN)
# # -------------------------------
# class ESN(bp.DynamicalSystem):
#     def __init__(self, num_in, num_hidden, num_out, sr=1., leaky_rate=0.3,
#                  Win_initializer=bp.init.Uniform(0, 0.2)):
#         """
#         Echo State Network with a reservoir and a linear readout.
#
#         Args:
#             num_in: 输入维度
#             num_hidden: 隐状态维度
#             num_out: 输出维度
#             sr: 谱半径（spectral radius）
#             leaky_rate: 漏斗率
#             Win_initializer: 输入权重初始化
#         """
#         super(ESN, self).__init__()
#         # Reservoir layer
#         self.r = bp.dyn.Reservoir(
#             num_in, num_hidden,
#             Win_initializer=Win_initializer,
#             spectral_radius=sr,
#             leaky_rate=leaky_rate
#         )
#         # Linear readout layer
#         self.o = bp.dnn.Dense(num_hidden, num_out, mode=bm.training_mode)
#
#     def update(self, x):
#         """Forward pass: input -> reservoir -> output"""
#         return x >> self.r >> self.o
#
# # 初始化模型
# model = ESN(num_in=1, num_hidden=100, num_out=1)
# model.reset(1)  # 重置模型状态
#
# # 查看权重形状
# print("Win shape:", model.r.Win.shape)
# print("Wrec shape:", model.r.Wrec.shape)
# print("Wout shape:", model.o.W.shape)
#
# # -------------------------------
# # 5. 训练 ESN
# # -------------------------------
# trainer = bp.RidgeTrainer(model, alpha=1e-6)
#
# # 先用 warmup 数据更新 reservoir 状态
# _ = trainer.predict(x_warm)
#
# # 再训练模型
# _ = trainer.fit([x_train, y_train])
#
# # 测试预测
# ys_predict = trainer.predict(x_test)
#
# # 可视化预测结果
# start, end = 1000, 6000
# plt.figure(figsize=(15, 7))
# plt.subplot(211)
# plt.plot(bm.as_numpy(ys_predict)[0, start:end, 0], lw=3, label="ESN prediction")
# plt.plot(bm.as_numpy(y_test)[0, start:end, 0], linestyle="--", lw=2, label="True value")
# plt.title(f'Mean Square Error: {bp.losses.mean_squared_error(ys_predict, y_test):.6f}')
# plt.legend()
# plt.show()
#
# # -------------------------------
# # 6. 收集隐藏状态并进行 PCA 分析
# # -------------------------------
# # 转换输入维度为时间步在前
# inputs = bm.moveaxis(x_test, 1, 0)
#
# def run_step(x_t):
#     """一次时间步更新，返回 reservoir 状态"""
#     model.update(x_t)
#     return model.r.state
#
# print("Running simulation (collecting hidden states)...")
# model.reset(1)
# states = bm.for_loop(run_step, inputs, progress_bar=True)
#
# # 转换为 NumPy 并恢复 batch 维度到前面
# hidden_states_np = bm.as_numpy(bm.moveaxis(states, 0, 1))
# print(f"Hidden States Shape: {hidden_states_np.shape}")
#
# # 提取前 time_steps 个时间步进行 PCA
# time_steps = 5000
# limit = min(time_steps, hidden_states_np.shape[1])
# trajectory_high_dim = hidden_states_np[0, :limit, :]
#
# # PCA 降维到 3D
# pca = PCA(n_components=3)
# trajectory_low_dim = pca.fit_transform(trajectory_high_dim)
# print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
#
# # -------------------------------
# # 7. 可视化高维动态
# # -------------------------------
# fig = plt.figure(figsize=(16, 7))
#
# # --- 2D PCA 投影 ---
# ax1 = fig.add_subplot(121)
# colors = plt.cm.viridis(bm.linspace(0, 1, limit))
# ax1.scatter(trajectory_low_dim[:, 0], trajectory_low_dim[:, 1],
#             c=colors, s=1, alpha=0.6)
# ax1.set_title('2D PCA Projection (Reservoir Dynamics)')
# ax1.set_xlabel('PC 1')
# ax1.set_ylabel('PC 2')
# ax1.grid(True, alpha=0.3)
#
# # --- 3D PCA 投影 ---
# ax2 = fig.add_subplot(122, projection='3d')
# sc = ax2.scatter(trajectory_low_dim[:, 0],
#                  trajectory_low_dim[:, 1],
#                  trajectory_low_dim[:, 2],
#                  c=colors, s=1, alpha=0.6)
# ax2.set_title('3D Attractor Reconstruction')
# ax2.set_xlabel('PC 1')
# ax2.set_ylabel('PC 2')
# ax2.set_zlabel('PC 3')
#
# cbar = plt.colorbar(sc, ax=[ax1, ax2], shrink=0.8)
# cbar.set_label('Time Evolution')
#
# plt.show()

import brainpy as bp
import brainpy.math as bm
import brainpy_datasets as bd
import matplotlib.pyplot as plt
import numpy as np

# 1. 设置环境
bm.set_platform('gpu') # 如果没有GPU，改为 'cpu'
bm.set_environment(mode=bm.TrainingMode(batch_size=20))

# 2. 数据准备
dt = 0.1
# 生成较长的数据以便切片
mg_data = bd.chaos.MackeyGlassEq(50000, dt=dt, tau=17, beta=0.2, gamma=0.1, n=10)
data = mg_data.xs

def create_dataset(data, batch_size, seq_len):
    data_len = len(data)
    xs, ys = [], []
    rng = np.random.RandomState(42)
    start_indices = rng.randint(0, data_len - seq_len - 1, batch_size)

    for start in start_indices:
        xs.append(data[start : start + seq_len])
        ys.append(data[start + 1 : start + seq_len + 1])

    xs = np.array(xs)[..., None]
    ys = np.array(ys)[..., None]
    return bm.asarray(xs), bm.asarray(ys)

SEQ_LEN = 200
BATCH_SIZE = 20
X_batch, Y_batch = create_dataset(data, BATCH_SIZE, SEQ_LEN)

# 3. 模型定义 (修正版)
class ESN_BP(bp.DynamicalSystem):
  def __init__(self, num_in, num_hidden, num_out):
    super(ESN_BP, self).__init__()

    self.r = bp.dyn.Reservoir(
        num_in, num_hidden,
        spectral_radius=1.5,
        leaky_rate=0.3,
        Win_initializer=bp.init.Uniform(-0.1, 0.1)
    )

    self.o = bp.dnn.Dense(num_hidden, num_out)

  def update(self, x):
    return x >> self.r >> self.o

  @bm.cls_jit
  def predict(self, xs):
    xs = bm.moveaxis(xs, 1, 0)

    # >>> 关键修正 >>>
    # 使用 .value 赋值，并显式指定全零矩阵的物理形状 (Batch, Hidden)
    batch_size = xs.shape[1]
    hidden_size = self.r.state.shape[-1]
    self.r.state.value = bm.zeros((batch_size, hidden_size))
    # <<< 关键修正 <<<

    outs = bm.for_loop(self.update, xs)
    return bm.moveaxis(outs, 0, 1)

model = ESN_BP(num_in=1, num_hidden=256, num_out=1)

# 4. 训练设置
def loss_func(xs, ys):
    preds = model.predict(xs)
    # 忽略前20步 (Washout)
    warmup_steps = 20
    valid_preds = preds[:, warmup_steps:, :]
    valid_ys = ys[:, warmup_steps:, :]
    return bp.losses.mean_squared_error(valid_preds, valid_ys)

# 只优化输出层 model.o
optimizer = bp.optim.Adam(lr=0.01, train_vars=model.o.train_vars().unique())
grad_f = bm.grad(loss_func, grad_vars=model.o.train_vars().unique(), return_value=True)

@bm.jit
def train_step(xs, ys):
    grads, loss = grad_f(xs, ys)
    optimizer.update(grads)
    return loss

# 5. 训练循环
losses = []
print("Start Training...")

for epoch in range(200):
    loss = train_step(X_batch, Y_batch)
    losses.append(loss)

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# 6. 简单的结果验证
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("Training Loss")

plt.subplot(1, 2, 2)
preds = model.predict(X_batch)
plt.plot(bm.as_numpy(Y_batch[0, :, 0]), 'k--', label='Target')
plt.plot(bm.as_numpy(preds[0, :, 0]), 'r', label='Prediction')
plt.title("Sample Prediction")
plt.legend()
plt.show()
