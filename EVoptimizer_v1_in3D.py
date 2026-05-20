import numpy as np
import matplotlib.pyplot as plt

# 손실 함수
def loss_function(w):
    return w[0]**2 - w[1]**2

# 손실 함수의 도함수 정의
def gradient_function(w):
    return np.array([2 * w[0], -2 * w[1]])

# 최대-최소 정리 기반 최적화 알고리즘
def extreme_value_optimizer(w_init, lr=0.05, epochs=200, tol=0.0001):
    w = np.array(w_init, dtype=float) # 가중치 벡터 [x, y]
    w_history = [w.copy()]  # 과거 가중치 기록
    
    print(f"[학습 시작] 초기 가중치 w = [{w[0]:.4f}, {w[1]:.4f}], 초기 오차 = {loss_function(w):.4f}\n")
    
    for epoch in range(epochs):
        grad = gradient_function(w)
        loss_c = loss_function(w)
        
        # 기울기가 0일 때
        if np.linalg.norm(grad) < tol:
            print(f"★ [Epoch {epoch}] 기울기 0 근접 -> 안장점 c = [{w[0]:.4f}, {w[1]:.4f}] 감지!")
            
            delta = 0.1  # 구간 탐색을 위한 작은 값

            while delta < 10:
                c = w.copy()
                
                # 각 축 방향(상하좌우)으로 delta만큼 탐색
                points = {
                    'c': c,
                    'x+': c + np.array([delta, 0.0]),
                    'x-': c + np.array([-delta, 0.0]),
                    'y+': c + np.array([0.0, delta]),
                    'y-': c + np.array([0.0, -delta])
                }
                
                losses = {key: loss_function(pt) for key, pt in points.items()}
                min_key = min(losses, key=losses.get)
                
                if min_key == 'c':
                    delta *= 2
                    continue
                else:
                    w = points[min_key]
                    w_history.append(w.copy())
                    print(f"   ▶ f({min_key})가 더 낮으므로 w = [{w[0]:.4f}, {w[1]:.4f}] 지점으로 순간이동하여 하강 재개\n")
                    break
        
        # 일반 경사하강법 진행
        w = w - lr * grad
        w_history.append(w.copy())
        
    return w, w_history

w_final, w_hist = extreme_value_optimizer(w_init=[2.0, 0.0], lr=0.05, epochs=60)
w_hist = np.array(w_hist)


x_range = np.linspace(-2.5, 2.5, 100)
y_range = np.linspace(-2.5, 2.5, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = X**2 - Y**2  # 손실 함수 Z축 값 생성


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none', alpha=0.5, zorder=1)

z_hist = w_hist[:, 0]**2 - w_hist[:, 1]**2

ax.plot(w_hist[:, 0], w_hist[:, 1], z_hist, 'ro--', linewidth=2, markersize=5, zorder=3, label='Optimization Path')

ax.scatter(w_hist[0, 0], w_hist[0, 1], z_hist[0], color='green', s=100, marker='o', zorder=5, label='Start Point')
ax.scatter(w_final[0], w_final[1], loss_function(w_final), color='gold', edgecolor='black', s=200, marker='*', zorder=5, label='Final Position')

ax.view_init(elev=25, azim=-55)

# 그래프 스타일링
ax.set_xlabel('Weight X (w0)', fontsize=11)
ax.set_ylabel('Weight Y (w1)', fontsize=11)
ax.set_zlabel('Loss (Z)', fontsize=11)
ax.set_title('3D Visualized EV Optimizer on Saddle Surface ($x^2 - y^2$)', fontsize=13)
fig.colorbar(surf, shrink=0.5, aspect=10, label='Loss Value')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()