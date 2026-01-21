import numpy as np
import matplotlib.pyplot as plt

# =================================================
# Parameters
# =================================================
N = 95
L = 2.0
w0 = 0.12
iterations = 80

# =================================================
# Spatial grid (SLM plane)
# =================================================
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

# =================================================
# 1. Input Gaussian amplitude & intensity
# =================================================
amplitude_in = np.exp(-(X**2 + Y**2) / (w0**2))
intensity_in = amplitude_in**2

# =================================================
# 2. Initial phase: left = 0, right = π
# =================================================
phase_init = np.zeros((N, N))
phase_init[:, N//2:] = np.pi

# =================================================
# Initial complex field
# =================================================
U_init = amplitude_in * np.exp(1j * phase_init)

# =================================================
# 3. Initial output (before GS)
# =================================================
U_f_init = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U_init)))
intensity_out_init = np.abs(U_f_init)**2
intensity_out_init /= intensity_out_init.max()

# =================================================
# 4. Target intensity: 3 Gaussian spots
# =================================================
def gaussian_spot(X, Y, x0, y0, sigma=0.04):
    return np.exp(-((X - x0)**2 + (Y - y0)**2) / (2*sigma**2))

fx = np.linspace(-1, 1, N)
fy = np.linspace(-1, 1, N)
FX, FY = np.meshgrid(fx, fy)

intensity_target = (
    gaussian_spot(FX, FY, -0.4, 0.0) +
    gaussian_spot(FX, FY,  0.0, 0.0) +
    gaussian_spot(FX, FY,  0.4, 0.0)
)
intensity_target /= intensity_target.max()
amplitude_target = np.sqrt(intensity_target)

# =================================================
# 5. Gerchberg–Saxton Algorithm
# =================================================
U = U_init.copy()

for _ in range(iterations):

    # Forward propagation
    U_f = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U)))

    # Enforce target amplitude
    U_f = amplitude_target * np.exp(1j * np.angle(U_f))

    # Back propagation
    U = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(U_f)))

    # Enforce input amplitude
    U = amplitude_in * np.exp(1j * np.angle(U))

# =================================================
# 6. Final phase and output intensity
# =================================================
final_phase = np.mod(np.angle(U), 2*np.pi)

U_f_final = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U)))
intensity_out_final = np.abs(U_f_final)**2
intensity_out_final /= intensity_out_final.max()

# =================================================
# 7. Correlation (Normalized Cross-Correlation)
# =================================================
I_t = intensity_target.flatten() - np.mean(intensity_target)
I_o = intensity_out_final.flatten() - np.mean(intensity_out_final)

correlation = np.sum(I_t * I_o) / np.sqrt(
    np.sum(I_t**2) * np.sum(I_o**2)
)

print(f"Correlation (target vs obtained): {correlation:.4f}")

# =================================================
# 8. Plot: 6 Subplots
# =================================================
fig, ax = plt.subplots(2, 3, figsize=(15, 8))

# 1. Input intensity
im0 = ax[0, 0].imshow(intensity_in, cmap="inferno")
ax[0, 0].set_title("1. Input Intensity (SLM plane)")
plt.colorbar(im0, ax=ax[0, 0])

# 2. Initial phase
im1 = ax[0, 1].imshow(phase_init, cmap="gray", vmin=0, vmax=np.pi)
ax[0, 1].set_title("2. Initial Phase (0 / π)")
plt.colorbar(im1, ax=ax[0, 1])

# 3. Initial output intensity
im2 = ax[0, 2].imshow(intensity_out_init, cmap="inferno")
ax[0, 2].set_title("3. Initial Output (2 spots)")
plt.colorbar(im2, ax=ax[0, 2])

# 4. Target intensity
im3 = ax[1, 0].imshow(intensity_target, cmap="inferno")
ax[1, 0].set_title("4. Target Intensity (3 spots)")
plt.colorbar(im3, ax=ax[1, 0])

# 5. Recovered phase (greyscale)
im4 = ax[1, 1].imshow(final_phase, cmap="gray", vmin=0, vmax=2*np.pi)
ax[1, 1].set_title("5. Recovered Phase (SLM)")
plt.colorbar(im4, ax=ax[1, 1])

# 6. Final output intensity + correlation
im5 = ax[1, 2].imshow(intensity_out_final, cmap="inferno")
ax[1, 2].set_title(
    f"6. Obtained Output (3 spots)\nCorrelation = {correlation:.3f}"
)
plt.colorbar(im5, ax=ax[1, 2])

plt.tight_layout()
plt.show()
