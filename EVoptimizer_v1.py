import numpy as np
import matplotlib.pyplot as plt

# 손실 함수 
def loss_function(w):
    return w**4 - 4*w**2 + w

# 손실 함수 도함수
def gradient_function(w):
    return 4*w**3 - 8*w + 1

# 최대-최소 정리 기반 최적화 알고리즘
def extreme_value_optimizer(w_init, lr=0.01, epochs=500, tol=1e-4):
    w = w_init # 초기 가중치
    w_history = [w]  # 과거 가중치 기록(시각화)
    
    print(f"[학습 시작] 초기 가중치 w = {w:.4f}, 초기 오차 = {loss_function(w):.4f}\n")
    
    for epoch in range(epochs):
        grad = gradient_function(w)
        loss_c = loss_function(w)
        
        # 기울기 = 0일 때
        if abs(grad) < tol:
            # print(f"★ [Epoch {epoch}] 기울기 0 달성 -> 로컬 미니마 c = {w:.4f} 감지!")
            
            delta = 0.1  # 구간 탐색을 위한 작은 값

            while delta < 10:
                c = w
                a = c - delta
                b = c + delta
                
                loss_a = loss_function(a)
                loss_b = loss_function(b)

                # 최대-최소 정리에 의거한 세 지점의 함숫값 비교
                losses = {'a': loss_a, 'b': loss_b, 'c': loss_c}
                min_key = min(losses, key=losses.get)
                
                if min_key == 'c':
                    # c가 가장 낮다면 delta를 늘려 더 넓은 구간 탐색
                    delta *= 2
                    continue
                else:
                    # f(a)나 f(b)가 더 작다면 그 가중치로 순간이동
                    w = a if min_key == 'a' else b
                    w_history.append(w)  # 순간이동한 위치 기록
                    print(f"[Epoch {epoch}] f({min_key})가 더 낮으므로 w = {w:.4f} 지점으로 이동하여 하강 재개\n")
                    break
        
        # 일반 경사하강법 진행
        w = w - lr * grad
        w_history.append(w)  # 이동 경로 기록
        
    return w, w_history

w_final, w_hist = extreme_value_optimizer(w_init=2.0, lr=0.02, epochs=500)

w_range = np.linspace(-2.5, 2.5, 500)
loss_range = loss_function(w_range)

plt.figure(figsize=(10, 6))

# 손실 함수
plt.plot(w_range, loss_range, label='Loss Function ($w^4 - 4w^2 + w$)', color='blue', linewidth=2)

# 가중치 이동 기록
loss_hist = [loss_function(x) for x in w_hist]
plt.plot(w_hist, loss_hist, 'ro--', alpha=0.5, markersize=4, label='Optimization Path')

# 최종 가중치
plt.scatter(w_final, loss_function(w_final), color='gold', edgecolor='black', s=250, marker='*', zorder=5, label='Final Global Minima')

# 그래프 스타일링
plt.xlabel('Weight (w)', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('EV Optimizer', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--')
plt.show()
