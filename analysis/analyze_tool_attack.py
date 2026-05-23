import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(_HERE, "..", "TA", "agent_graph_dataset", "tool_attack")
OUTPUT_DIR = os.path.join(_HERE, "..", "Figure", "tool_attack")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPARSITIES = [2, 4, 6, 8, 10]
ATTACKER_COUNTS = [1, 2, 3, 4]
NUM_AGENTS = 8
NUM_SAMPLES = 20


def load_activations(file_path):
    """Load .pt file -> (R, A, L, H) tensor"""
    data = torch.load(file_path, map_location='cpu')
    if isinstance(data, list):
        data = [torch.stack([act.float() for act in round_acts]) for round_acts in data]
        data = torch.stack(data)  # (R, A, L, H)
    return data.float()


def load_directory(dir_path, num_attackers):
    """
    Load all samples from one directory.
    Returns:
        X: (N_samples, R, A, H) — last layer only
        labels: (N_samples, R, A) — 0 normal, 1 attacker
    """
    json_files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
    assert len(json_files) == 1, f"Expected 1 json, got {len(json_files)}"
    with open(os.path.join(dir_path, json_files[0])) as f:
        meta = json.load(f)

    act_dir = os.path.join(dir_path, "activations")
    X_list, label_list = [], []

    for task_id in range(NUM_SAMPLES):
        file_path = os.path.join(act_dir, f"sample_{task_id:04d}.pt")
        if not os.path.exists(file_path):
            print(f"  Missing: {file_path}")
            continue

        act = load_activations(file_path)  # (R, A, L, H)
        x = act[:, :, -1, :].numpy()       # (R, A, H) — last layer

        attacker_idxes = set(meta[task_id]['attacker_idxes'])
        R, A, H = x.shape
        label = np.zeros((R, A), dtype=int)
        for idx in attacker_idxes:
            label[:, idx] = 1

        X_list.append(x)
        label_list.append(label)

    X = np.stack(X_list, axis=0)       # (N, R, A, H)
    labels = np.stack(label_list, axis=0)  # (N, R, A)
    return X, labels


def tsne_plot(X_all, labels_all, title, save_path):
    """
    X_all: (M, H)
    labels_all: (M,) — 0 or 1
    """
    # Normalize
    X_norm = (X_all - X_all.mean(axis=0)) / (X_all.std(axis=0) + 1e-6)

    # PCA -> 50D
    pca_dim = min(50, X_norm.shape[1], X_norm.shape[0] - 1)
    V_pca = PCA(n_components=pca_dim, random_state=0).fit_transform(X_norm)

    # t-SNE -> 2D
    perp = min(30, V_pca.shape[0] // 4)
    Z = TSNE(n_components=2, perplexity=perp, init="pca",
             learning_rate="auto", random_state=0).fit_transform(V_pca)

    colors = ['steelblue' if l == 0 else 'tomato' for l in labels_all]
    plt.figure(figsize=(6, 5))
    for label, color, name in [(0, 'steelblue', 'Normal'), (1, 'tomato', 'Attacker')]:
        mask = labels_all == label
        plt.scatter(Z[mask, 0], Z[mask, 1], c=color, alpha=0.6, s=12, label=name)
    plt.legend()
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def centroid_distance_plot(X_normal, X_attacker, title, save_path):
    """
    Compute centroid from normal samples, plot distance distribution.
    X_normal, X_attacker: (N, H)
    """
    centroid = X_normal.mean(axis=0)
    dist_normal = np.linalg.norm(X_normal - centroid, axis=1)
    dist_attacker = np.linalg.norm(X_attacker - centroid, axis=1)

    plt.figure(figsize=(6, 4))
    bins = np.linspace(0, max(dist_normal.max(), dist_attacker.max()) * 1.05, 40)
    plt.hist(dist_normal, bins=bins, alpha=0.6, color='steelblue', label='Normal')
    plt.hist(dist_attacker, bins=bins, alpha=0.6, color='tomato', label='Attacker')
    plt.xlabel('Distance to Normal Centroid')
    plt.ylabel('Count')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")

    return dist_normal, dist_attacker


# ==================== Main ====================

if __name__ == "__main__":

    # --- 1. Overall: merge all configs, plot one big t-SNE ---
    print("=== Loading all data ===")
    X_all, labels_all = [], []

    for s in SPARSITIES:
        for a in ATTACKER_COUNTS:
            dir_name = f"train_n8_s{s:02d}_a{a}"
            dir_path = os.path.join(BASE_DIR, dir_name)
            print(f"  Loading {dir_name}...")
            X, labels = load_directory(dir_path, num_attackers=a)
            # Use round 0 only for overall plot
            X_all.append(X[:, 0, :, :].reshape(-1, X.shape[-1]))     # (N*A, H)
            labels_all.append(labels[:, 0, :].flatten())              # (N*A,)

    X_all = np.concatenate(X_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    print(f"Total points: {X_all.shape[0]} (normal: {(labels_all==0).sum()}, attacker: {(labels_all==1).sum()})")

    # Overall t-SNE
    print("Running overall t-SNE...")
    tsne_plot(X_all, labels_all,
              title="t-SNE: All Tool Attack Data (Round 1)",
              save_path=os.path.join(OUTPUT_DIR, "tsne_all.png"))

    # Overall centroid distance
    print("Computing overall centroid distances...")
    centroid_distance_plot(
        X_all[labels_all == 0], X_all[labels_all == 1],
        title="Centroid Distance Distribution (All)",
        save_path=os.path.join(OUTPUT_DIR, "dist_all.png")
    )

    # --- 2. Per-attacker-count: one subplot grid ---
    print("\n=== Per attacker count analysis ===")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, a in enumerate(ATTACKER_COUNTS):
        X_a, labels_a = [], []
        for s in SPARSITIES:
            dir_name = f"train_n8_s{s:02d}_a{a}"
            dir_path = os.path.join(BASE_DIR, dir_name)
            X, labels = load_directory(dir_path, num_attackers=a)
            X_a.append(X[:, 0, :, :].reshape(-1, X.shape[-1]))
            labels_a.append(labels[:, 0, :].flatten())

        X_a = np.concatenate(X_a, axis=0)
        labels_a = np.concatenate(labels_a, axis=0)

        # PCA for subplot
        X_norm = (X_a - X_a.mean(axis=0)) / (X_a.std(axis=0) + 1e-6)
        pca_dim = min(50, X_norm.shape[1], X_norm.shape[0] - 1)
        V_pca = PCA(n_components=pca_dim, random_state=0).fit_transform(X_norm)
        perp = min(30, V_pca.shape[0] // 4)
        Z = TSNE(n_components=2, perplexity=perp, init="pca",
                 learning_rate="auto", random_state=0).fit_transform(V_pca)

        ax = axes[i]
        for label, color, name in [(0, 'steelblue', 'Normal'), (1, 'tomato', 'Attacker')]:
            mask = labels_a == label
            ax.scatter(Z[mask, 0], Z[mask, 1], c=color, alpha=0.6, s=10, label=name)
        ax.set_title(f'{a} Attacker(s)')
        ax.legend(fontsize=8)

    fig.suptitle('t-SNE by Attacker Count (Round 1)')
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "tsne_by_attacker_count.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")

    # --- 3. Centroid distance summary: mean separation per config ---
    print("\n=== Centroid distance summary ===")
    results = {}
    for s in SPARSITIES:
        for a in ATTACKER_COUNTS:
            dir_name = f"train_n8_s{s:02d}_a{a}"
            dir_path = os.path.join(BASE_DIR, dir_name)
            X, labels = load_directory(dir_path, num_attackers=a)
            X_flat = X[:, 0, :, :].reshape(-1, X.shape[-1])
            l_flat = labels[:, 0, :].flatten()
            centroid = X_flat[l_flat == 0].mean(axis=0)
            d_normal = np.linalg.norm(X_flat[l_flat == 0] - centroid, axis=1).mean()
            d_attack = np.linalg.norm(X_flat[l_flat == 1] - centroid, axis=1).mean()
            results[(s, a)] = {'d_normal': d_normal, 'd_attack': d_attack, 'sep': d_attack - d_normal}
            print(f"  s={s/10:.1f} a={a}: d_normal={d_normal:.2f}, d_attack={d_attack:.2f}, sep={d_attack-d_normal:.2f}")

    # Heatmap of separation
    sep_matrix = np.array([[results[(s, a)]['sep'] for a in ATTACKER_COUNTS] for s in SPARSITIES])
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(sep_matrix, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(ATTACKER_COUNTS)))
    ax.set_xticklabels([f'a={a}' for a in ATTACKER_COUNTS])
    ax.set_yticks(range(len(SPARSITIES)))
    ax.set_yticklabels([f's={s/10:.1f}' for s in SPARSITIES])
    plt.colorbar(im, ax=ax, label='Mean Distance Separation (Attacker - Normal)')
    ax.set_title('Centroid Separation Heatmap')
    for i in range(len(SPARSITIES)):
        for j in range(len(ATTACKER_COUNTS)):
            ax.text(j, i, f'{sep_matrix[i,j]:.1f}', ha='center', va='center', fontsize=8)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "separation_heatmap.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")

    print("\nDone. Figures saved to:", OUTPUT_DIR)
