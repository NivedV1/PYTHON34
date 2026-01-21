import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Parameters
# =========================================================
N = 200
n_iters = 2000

# =========================================================
# Spatial grid
# =========================================================
x = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, x)

# =========================================================
# Source amplitude (Gaussian beam)
# =========================================================
A_source = np.exp(-(X**2 + Y**2) / 0.3)
A_source /= A_source.max()

# =========================================================
# TARGET: INTERFERENCE PATTERN
# Example: interference of 3 Gaussian beams
# =========================================================
sigma = 0.08   # beam waist
d = 0.35       # beam separation

U1 = np.exp(-((X - d)**2 + Y**2) / (2 * sigma**2))
U2 = np.exp(-((X     )**2 + Y**2) / (2 * sigma**2))
U3 = np.exp(-((X + d)**2 + Y**2) / (2 * sigma**2))

# Interference intensity
I_target = np.abs(U1 + U2 + U3)**2

# Convert intensity → amplitude
A_target = np.sqrt(I_target)
A_target /= A_target.max()

# =========================================================
# Gerchberg–Saxton Algorithm
# =========================================================
phase = np.random.uniform(0, 2*np.pi, (N, N))
U = A_source * np.exp(1j * phase)

for _ in range(n_iters):
    # Forward propagation (Fraunhofer)
    U_f = np.fft.fftshift(np.fft.fft2(U))

    # Enforce target amplitude
    U_f = A_target * np.exp(1j * np.angle(U_f))

    # Backward propagation
    U = np.fft.ifft2(np.fft.ifftshift(U_f))

    # Enforce source amplitude
    U = A_source * np.exp(1j * np.angle(U))

# Retrieved phase mask
phase_retrieved = np.angle(U)

# =========================================================
# PROPER EVALUATION (no constraints applied)
# =========================================================
U_test = A_source * np.exp(1j * phase_retrieved)
U_test_f = np.fft.fftshift(np.fft.fft2(U_test))
I_result = np.abs(U_test_f)**2
I_result /= I_result.max()

# Normalize target for comparison
I_target_norm = I_target / I_target.max()

# =========================================================
# Correlation metric
# =========================================================
corr = np.corrcoef(I_result.ravel(), I_target_norm.ravel())[0, 1]
print(f"Final correlation: {corr:.4f}")

# =========================================================
# Visualization
# =========================================================
plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.imshow(A_source**2, cmap="gray")
plt.title("Source intensity")
plt.colorbar()
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(I_result, cmap="gray")
plt.title("GS output intensity")
plt.colorbar()
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(I_target_norm, cmap="gray")
plt.title("Target interference intensity")
plt.colorbar()
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow((phase_retrieved % (2*np.pi)) / (2*np.pi), cmap="gray")
plt.title("Retrieved phase (0–2π)")
plt.colorbar()
plt.axis("off")

plt.tight_layout()
plt.show()
