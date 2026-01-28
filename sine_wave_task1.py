#用一般的 echo state network 实现正弦波预测任务

import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -------------------------------
# 1. 环境配置
# -------------------------------
bm.set_platform('gpu')  # 如果没有 GPU，请改为 'cpu'
bm.set_environment(x64=True, mode=bm.batching_mode)  # 高精度 + 批处理模式

print(f"BrainPy Version: {bp.__version__}")

# -------------------------------
# 2. 数据生成 (正弦波)
# -------------------------------
dt = 0.1
num_steps = 25000
ts = bm.arange(0, num_steps * dt, dt)
xs = bm.sin(ts * 0.2)  # 正弦波

# 增加特征维度 (Time, 1)
xs = xs[:, None]
ys = xs.copy()

# -------------------------------
# 2.1 数据切分函数
# -------------------------------
def get_data(t_warm, t_forecast, t_train, sample_rate=1):
    """
    Prepare ESN warmup, training, and testing data.

    Args:
        t_warm: warmup duration (ms)
        t_forecast: forecast horizon (ms)
        t_train: training duration (ms)
        sample_rate: downsampling rate
    Returns:
        X_warm, X_train, Y_train, X_test, Y_test
    """
    warmup = int(t_warm / dt)
    forecast = int(t_forecast / dt)
    train_length = int(t_train / dt)

    # Warmup 输入
    X_warm = bm.expand_dims(xs[:warmup:sample_rate], 0)

    # 训练数据
    X_train = bm.expand_dims(xs[warmup:warmup + train_length:sample_rate], 0)
    Y_train = bm.expand_dims(xs[warmup + forecast:warmup + train_length + forecast:sample_rate], 0)

    # 测试数据
    X_test = bm.expand_dims(xs[warmup + train_length:-forecast:sample_rate], 0)
    Y_test = bm.expand_dims(xs[warmup + train_length + forecast::sample_rate], 0)

    return X_warm, X_train, Y_train, X_test, Y_test

# 获取数据
x_warm, x_train, y_train, x_test, y_test = get_data(t_warm=100, t_forecast=1, t_train=2000)

# -------------------------------
# 2.2 可视化训练数据片段
# -------------------------------
sample = 1000
plt.figure(figsize=(15, 5))
plt.plot(bm.as_numpy(x_train[0, :sample]), label="Training Input (Sine)")
plt.plot(bm.as_numpy(y_train[0, :sample]), label="Target (Shifted Sine)")
plt.title("Data Preview")
plt.legend()
plt.show()

# -------------------------------
# 3. 定义 ESN 模型
# -------------------------------
class ESN(bp.DynamicalSystem):
    """
    Standard Echo State Network (No feedback).
    """
    def __init__(self, num_in, num_hidden, num_out, sr=1.0, leaky_rate=0.3,
                 Win_initializer=bp.init.Uniform(0, 0.2)):
        super(ESN, self).__init__()
        # Reservoir 层
        self.r = bp.dyn.Reservoir(
            num_in, num_hidden,
            Win_initializer=Win_initializer,
            spectral_radius=sr,
            leaky_rate=leaky_rate
        )
        # 输出层
        self.o = bp.dnn.Dense(num_hidden, num_out, mode=bm.training_mode)

    def update(self, x):
        # Forward: Input -> Reservoir -> Output
        return x >> self.r >> self.o

# 初始化模型
model = ESN(num_in=1, num_hidden=100, num_out=1)
model.reset(1)  # Batch=1

# -------------------------------
# 4. ESN 训练
# -------------------------------
trainer = bp.RidgeTrainer(model, alpha=1e-6)

# Warmup
print("Warming up...")
_ = trainer.predict(x_warm)

# 训练
print("Training...")
_ = trainer.fit([x_train, y_train])

# 预测
print("Predicting...")
ys_predict = trainer.predict(x_test)

# 可视化预测
start, end = 500, 1000
plt.figure(figsize=(15, 7))
plt.plot(bm.as_numpy(ys_predict)[0, start:end, 0], lw=3, label="ESN Prediction")
plt.plot(bm.as_numpy(y_test)[0, start:end, 0], linestyle="--", lw=2, label="True Value")
plt.title(f'MSE: {bp.losses.mean_squared_error(ys_predict, y_test):.6f}')
plt.legend()
plt.show()

# -------------------------------
# 5. PCA 状态分析
# -------------------------------
# 将时间维度移到前面
inputs = bm.moveaxis(x_test, 1, 0)

def run_step(x_t):
    """一次时间步更新，返回隐藏状态"""
    model.update(x_t)
    return model.r.state

print("Running simulation (collecting hidden states)...")
model.reset(1)
states = bm.for_loop(run_step, inputs, progress_bar=True)

# 转换为 NumPy 并恢复 Batch 维度
hidden_states_np = bm.as_numpy(bm.moveaxis(states, 0, 1))
print(f"Hidden States Shape: {hidden_states_np.shape}")

# 选取时间步进行分析
time_steps = 5000
limit = min(time_steps, hidden_states_np.shape[1])
trajectory_high_dim = hidden_states_np[0, :limit, :]

# PCA 降维
pca = PCA(n_components=3)
trajectory_low_dim = pca.fit_transform(trajectory_high_dim)
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

# -------------------------------
# 6. PCA 可视化
# -------------------------------
figure = plt.figure(figsize=(16, 7))

# 2D 投影
ax1 = figure.add_subplot(121)
colors = plt.cm.viridis(bm.linspace(0, 1, limit))
ax1.scatter(trajectory_low_dim[:, 0], trajectory_low_dim[:, 1],
            c=colors, s=1, alpha=0.6)
ax1.set_title('2D PCA Projection (Sine Wave Dynamics)')
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')
ax1.grid(True, alpha=0.3)

# 3D 投影
ax2 = figure.add_subplot(122, projection='3d')
sc = ax2.scatter(trajectory_low_dim[:, 0],
                 trajectory_low_dim[:, 1],
                 trajectory_low_dim[:, 2],
                 c=colors, s=1, alpha=0.6)
ax2.set_title('3D Attractor Reconstruction (Limit Cycle)')
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
ax2.set_zlabel('PC 3')

# 颜色条表示时间演化
cbar = plt.colorbar(sc, ax=[ax1, ax2], shrink=0.8)
cbar.set_label('Time Evolution')

plt.show()
