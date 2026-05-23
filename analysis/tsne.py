import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import glob
import json
import numpy as np

# Resolve dataset paths relative to this file so the script works from any cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
MEMORY_ATTACK_DIR = os.path.join(_HERE, "..", "MA", "agent_graph_dataset", "memory_attack")
# H5_PATH = os.path.join(_HERE, "..", "MA", "agent_graph_dataset", "acts_4d.h5")  # 改成你的 h5 文件

def load_tensor_from_pt(file_path, key=None):
    """
    从 .pt 文件加载 tensor
    
    Args:
        file_path: .pt 文件路径
        key: 如果是字典，指定要取的 key；None 则返回整个内容
    
    Returns:
        tensor 或字典
    """
    data = torch.load(file_path, map_location='cpu')

    if isinstance(data, list):
        # 先把每轮的 agents stack 起来，并转成 float32
        data = [torch.stack([act.float() for act in round_acts]) for round_acts in data]
        data = torch.stack(data)  # (R, A, L, H)
    
    # 确保是 float32
    if isinstance(data, torch.Tensor):
        data = data.float()
    
    if key is not None:
        return data[key]
    return data


# # 使用示例
# # 情况1: 文件直接存的是 tensor
# tensor = load_tensor_from_pt("model.pt")

# # 情况2: 文件存的是字典 {"activations": tensor, "labels": tensor}
# activations = load_tensor_from_pt("model.pt", key="activations")

# # 情况3: 查看文件里有什么
# data = load_tensor_from_pt("model.pt")
# if isinstance(data, dict):
#     print("Keys:", data.keys())
#     for k, v in data.items():
#         if isinstance(v, torch.Tensor):
#             print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
# else:
#     print(f"Tensor: shape={data.shape}, dtype={data.dtype}")


# 从pt或h5文件加载数据
def single_file(X):
    # X: (R,A,L,H) 四维数组
    R, A, L, H = X.shape
    N = R * A
    print("Loaded shape:", X.shape, "N points:", N)

    # ===== 1) 每次回答 -> 一个向量点 =====
    # 推荐：只取最后一层 (H,)
    V = X[:, :, -1, :]         # (R,A,H)    # (N,H)



    return V

   


if __name__ == "__main__":
    # with h5py.File(PATH, "r") as f:
    #     X = f["activations"][:]   # (R,A,L,H)
    X = []
    labels = []
    for s in range(2,11,2):
        for a in range(1,4):
            folder = f"train_n6_s0{s}_a{a}" if s < 10 else f"train_n6_s{s}_a{a}"
            path = os.path.join(MEMORY_ATTACK_DIR, folder)
            path_activation = os.path.join(path,"activations")
            # todo: 找json文件获取label，放进labels列表，要知道每个sample对应的label
            json_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.json')]
            path_labels = json_files[0] # 只有一个json文件
            with open(path_labels, "r") as f:
                data = json.load(f)
                for task in data:
                    attackers = task["attacker_idxes"]
                    label = [0]*6
                    for idx in attackers:
                        label[idx] = 1
                    label_3d = np.tile(label, (4, 1)).reshape(4, 6, 1) # (R,A,1)
                    labels.append(label_3d)
            # 读取tensor
            for task_id in range(0,20):  
                file_path = os.path.join(path_activation, f"sample_{task_id:04d}.pt")
                x = load_tensor_from_pt(file_path)  # (R,A,L,H)
                # todo: append x to X
                x = x[:, :, -1, :]  # (R,A,H) 
                if isinstance(x, torch.Tensor):
                    x = x.numpy()
                X.append(x)

X = np.stack(X, axis=0)  # (N, R, A, H)
labels = np.stack(labels, axis=0)  # (N, R, A, 1)

print(f"X shape: {X.shape}")
print(f"labels shape: {labels.shape}")

for round_n in range(4):  # 选择第几轮
    N, R, A, H = X.shape
    V_R1 = X[:, round_n, :, :]  # (N, A, H)
    V = (V_R1 - V_R1.mean(axis=0)) / (V_R1.std(axis=0) + 1e-6) # (N, A, H)
    V = V.reshape(N * A, H)  # (N * A, H)
    Label_R1 = labels[:, round_n, :, :]  # (N, A)

    colors = ['blue' if l == 0 else 'red' for l in Label_R1.flatten()]

        # # ===== 2) PCA -> 50 维（t-SNE更稳） =====
    n_samples = V.shape[0] 
    n_features = V.shape[1]
    pca_dim = min(50, n_features, n_samples - 1)
    V_pca = PCA(n_components=pca_dim, random_state=0).fit_transform(V) # (N * A, 50)

        # # ===== 3) t-SNE -> 2D =====
    tsne = TSNE(
        n_components=2,
        perplexity=8,          
        init="pca",
        learning_rate="auto",
        random_state=0,
    )
    Z = tsne.fit_transform(V_pca)   # (N * A,2)

        # # ===== 4) 标签：round/agent =====
        # round_id = np.repeat(np.arange(1, R + 1), A)
    # agent_id = np.tile(np.arange(A), R)

        # ===== 5) 画图 + 标注 =====
    plt.figure()
    plt.scatter(Z[:, 0], Z[:, 1], c=colors, alpha=0.6, s=10)


    plt.legend()
    plt.title(f't-SNE Visualization of Activations (Round {round_n + 1})')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig(f'tsne_result_round_{round_n + 1}.png', dpi=150)
