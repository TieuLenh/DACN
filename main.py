import numpy as np
import matplotlib.pyplot as plt
import time

from dca import ip_dca
from kmeans import kmeans

if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    k = 10
    random_state = 42
    # 1. Tạo dữ liệu dùng chung cho cả hai thuật toán
    X, _ = make_blobs(
        n_samples=10000,
        centers=k,
        n_features=2,
        cluster_std=1.0,
        random_state=random_state
    )
    n, d = X.shape
    rng = np.random.default_rng(random_state)
    c_init = X[rng.choice(n, size=k, replace=False)].copy()

    # 2. Chạy IP-DCA và đo thời gian
    start = time.perf_counter()

    result_ip_dca = ip_dca(
        X,
        n_clusters=10,
        c_init=c_init,
        t=10.0,
        max_iter=1000,
        tol=1e-6,
        random_state=random_state
    )

    time_ip_dca = time.perf_counter() - start

    # 3. Chạy K-means và đo thời gian
    start = time.perf_counter()

    result_kmeans = kmeans(
        X,
        k=10,
        c_init=c_init,
        random_state=random_state
    )

    time_kmeans = time.perf_counter() - start

    # 4. Lấy nhãn và tâm cụm
    labels_ip = result_ip_dca["labels"]
    centers_ip = result_ip_dca["centers"]

    labels_km = result_kmeans["labels"]
    centers_km = result_kmeans["centers"]

    # 5. In kết quả
    print("=" * 45)
    print("IP-DCA")
    print("=" * 45)
    print(f"Thời gian: {time_ip_dca:.6f} giây")
    print("Iterations:", result_ip_dca["iterations"])
    print("MSSC:", result_ip_dca["mssc"])
    print("Penalty:", result_ip_dca["penalty"])
    print("Penalized objective:",
          result_ip_dca["penalized_objective"])
    print('centers: ', result_ip_dca["centers"])

    print("\n" + "=" * 45)
    print("K-MEANS")
    print("=" * 45)
    print(f"Thời gian: {time_kmeans:.6f} giây")
    print("Iterations:", result_kmeans["iterations"])
    print("MSSC:", result_kmeans["inertia"])
    print('centers: ', result_kmeans["centers"])

    # 6. Vẽ hai sơ đồ cạnh nhau trong một figure
    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5),
        sharex=True,
        sharey=True
    )

    # Biểu đồ IP-DCA
    axes[0].scatter(
        X[:, 0],
        X[:, 1],
        c=labels_ip,
        cmap="viridis",
        s=35,
        alpha=0.8
    )

    axes[0].scatter(
        centers_ip[:, 0],
        centers_ip[:, 1],
        marker="X",
        s=220,
        c="red",
        edgecolors="black",
        label="Cluster centers"
    )

    axes[0].set_title(
        f"IP-DCA\n"
        f"MSSC = {result_ip_dca['mssc']:.2f} | "
        f"Time = {time_ip_dca:.5f}s"
    )
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    axes[0].legend()

    # Biểu đồ K-means
    axes[1].scatter(
        X[:, 0],
        X[:, 1],
        c=labels_km,
        cmap="viridis",
        s=35,
        alpha=0.8
    )

    axes[1].scatter(
        centers_km[:, 0],
        centers_km[:, 1],
        marker="X",
        s=220,
        c="red",
        edgecolors="black",
        label="Cluster centers"
    )

    axes[1].set_title(
        f"K-means\n"
        f"MSSC = {result_kmeans['inertia']:.2f} | "
        f"Time = {time_kmeans:.5f}s"
    )
    axes[1].set_xlabel("Feature 1")
    axes[1].set_ylabel("Feature 2")
    axes[1].legend()

    fig.suptitle(
        "Comparison of IP-DCA and K-means",
        fontsize=15
    )

    plt.tight_layout()
    plt.show()

    # 7. Vẽ lịch sử hàm mục tiêu IP-DCA riêng nếu cần
    plt.figure(figsize=(8, 4))
    plt.plot(result_ip_dca["objective_history"])
    plt.title("IP-DCA Objective History")
    plt.xlabel("Iteration")
    plt.ylabel("Penalized Objective F1(U, V)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
