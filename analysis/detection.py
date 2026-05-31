import numpy as np
from sklearn.linear_model import Ridge
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


def train_ocsvm_from_thetas(theta_scaled_list, nu=0.05, gamma="scale"):
    """
    theta_scaled_list: List[np.ndarray], 每个元素是某个文件的 theta_scaled, shape=(K_i, D)
    返回: scaler, svm
    """
    X_train = np.vstack(theta_scaled_list)  # (sum_K, D)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    svm = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    svm.fit(X_train_scaled)

    return scaler, svm

def predict_ocsvm(theta_scaled, scaler, svm):
    """
    theta_scaled: (K, D)
    返回: labels (K,), 1正常/-1异常
    """
    X = scaler.transform(theta_scaled)
    return svm.predict(X)


def fit_dynamic_model(features):
    """
    Step 3.5: 局部动态模型拟合
    输入: 降维后的特征序列 (K, T, d)
    输出: 模型向量集合 theta (K, D_flat)
    """
    K, T, d = features.shape
    theta_list = []

    # 岭回归参数 (lambda)
    ridge_alpha = 1.0
    model = Ridge(alpha=ridge_alpha)

    print(f"开始拟合动态模型 (Ridge Regression)，共 {K} 个窗口...")

    for k in range(K):
        # 取出第 k 个窗口的特征序列: (T, d)
        # u_t
        u = features[k]

        # 构造输入输出对用于训练线性模型: u_{t+1} = M * u_t + c
        # X: 0 到 T-2 (时刻 t)
        # Y: 1 到 T-1 (时刻 t+1)
        X_train = u[:-1, :]
        Y_train = u[1:, :]

        # 拟合 M 和 c
        model.fit(X_train, Y_train)

        # 提取参数
        # M: model.coef_ (d, d)
        # c: model.intercept_ (d,)
        M_flat = model.coef_.flatten()
        c_flat = model.intercept_

        # Step 3.6: 打包成 θk [cite: 60-61]
        # 拼接 M 和 c
        theta_k = np.concatenate([M_flat, c_flat])
        theta_list.append(theta_k)

    return np.array(theta_list)


def detect_anomalies(theta_list):
    """
    Step 5: 异常检测 (One-Class SVM)
    输入: 模型向量 theta (K, dim)
    输出: 异常分数 score, 预测标签 label (-1为异常, 1为正常)
    """
    print("开始训练异常检测器 (One-Class SVM)...")

    # 1. 数据标准化 (对 SVM 很重要)
    scaler = StandardScaler()
    theta_scaled = scaler.fit_transform(theta_list)

    # 2. 定义模型 [cite: 85]
    # nu: 异常比例的估计上限，比如假设路面有 5%-10% 是坏的
    svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.10)

    # 3. 训练与预测
    svm.fit(theta_scaled)

    # label: 1 (正常), -1 (异常)
    labels = svm.predict(theta_scaled)

    # score: 距离超平面的距离，越小(负数)越异常
    scores = svm.decision_function(theta_scaled)

    return labels, scores