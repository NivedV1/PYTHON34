import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Parameters
# =========================================================
N = 16
n_iters = 1000
alpha = 0.559         # WGS weight (0 = GS, 0.3–0.7 typical)
eps = 1e-2           # numerical safety

# =========================================================
# Spatial grid
# =========================================================
x = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, x)

# =========================================================
# Source amplitude (Gaussian)
# =========================================================
A_source = np.exp(-(X**2 + Y**2) / 0.6)
A_source /= A_source.max()

# =========================================================
# Target INTENSITY (16×16)
# =========================================================
I_target = np.array([
 [50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50],
 [50,100,200,200,200,200,200,200,200,200,200,200,200,200,100,50],
 [50,200,100,100,100,100,100,100,100,100,100,100,100,100,200,50],
 [50,100,100,100,100,100,100,100,100,100,100,100,100,100,100,50],
 [50,100,100,100,100,255,100,100,100,100,255,100,100,100,100,50],
 [50,100,100,100,100,255,100,100,100,100,255,100,100,100,100,50],
 [50,100,100,100,100,100,100,100,100,100,100,100,100,100,100,50],
 [50,100,100,100,100,230,230,230,230,230,230,100,100,100,100,50],
 [50,100,100,100,100,230,230,230,230,230,230,100,100,100,100,50],
 [50,100,100,100,100,100,100,100,100,100,100,100,100,100,100,50],
 [50,100,100,100,100,100,100,100,100,100,100,100,100,100,100,50],
 [50,100,100,100,100,100,100,100,100,100,100,100,100,100,100,50],
 [50,150,100,100,100,100,100,100,100,100,100,100,100,100,150,50],
 [50,150,150,150,150,150,150,150,150,150,150,150,150,150,150,50],
 [50,100,150,150,150,150,150,150,150,150,150,150,150,150,100,50],
 [50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50]
], dtype=float)

A_target = np.sqrt(I_target)
A_target /= A_target.max()

# =========================================================
# Weighted Gerchberg–Saxton (WGS)
# =========================================================
phase = np.random.uniform(0, 2*np.pi, (N, N))
U = A_source * np.exp(1j * phase)

corr_history = []

for _ in range(n_iters):

    # Forward propagation
    U_f = np.fft.fftshift(np.fft.fft2(U))
    A_f = np.abs(U_f)

    # ---------- WGS amplitude update ----------
    weight = (A_target / (A_f + eps)) ** alpha
    A_new = A_f * weight
    U_f = A_new * np.exp(1j * np.angle(U_f))

    # Backward propagation
    U = np.fft.ifft2(np.fft.ifftshift(U_f))

    # Enforce source amplitude
    U = A_source * np.exp(1j * np.angle(U))

    # ----- convergence metric (proper!) -----
    U_test_f = np.fft.fftshift(np.fft.fft2(U))
    I_test = np.abs(U_test_f)**2
    I_test /= I_test.max()
    corr = np.corrcoef(I_test.ravel(), (I_target/I_target.max()).ravel())[0,1]
    corr_history.append(corr)

# Retrieved phase
phase_retrieved = np.angle(U)

# =========================================================
# Final evaluation (no constraints)
# =========================================================
U_test = A_source * np.exp(1j * phase_retrieved)
U_test_f = np.fft.fftshift(np.fft.fft2(U_test))
I_result = np.abs(U_test_f)**2
I_result /= I_result.max()

print(f"Final WGS correlation: {corr_history[-1]:.4f}")

# =========================================================
# Visualization
# =========================================================
plt.figure(figsize=(16,4))

plt.subplot(1,5,1)
plt.imshow(A_source**2, cmap="gray")
plt.title("Source intensity")
plt.axis("off")

plt.subplot(1,5,2)
plt.imshow(I_result, cmap="gray")
plt.title("WGS output intensity")
plt.axis("off")

plt.subplot(1,5,3)
plt.imshow(I_target/I_target.max(), cmap="gray")
plt.title("Target intensity")
plt.axis("off")

plt.subplot(1,5,4)
plt.imshow((phase_retrieved % (2*np.pi))/(2*np.pi), cmap="gray")
plt.title("Retrieved phase")
plt.axis("off")

plt.subplot(1,5,5)
plt.plot(corr_history)
plt.xlabel("Iteration")
plt.ylabel("Correlation")
plt.title("WGS convergence")
plt.grid(True)

plt.tight_layout()
plt.show()
