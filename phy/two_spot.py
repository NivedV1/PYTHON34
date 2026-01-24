import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Parameters
# -------------------------------------------------
N = 1000
L = 2.0
w0 = 0.08
A = 1.0

# -------------------------------------------------
# Spatial grid
# -------------------------------------------------
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

# -------------------------------------------------
# Initial Gaussian amplitude and intensity
# -------------------------------------------------
amplitude = A * np.exp(-((X**2 + Y**2) / (w0**2))**4)
intensity_in = amplitude**2
#print("Intenisty=",intensity_in)
print(np.shape(intensity_in))
# -------------------------------------------------
# Initial phase: left = 0, right = π
# -------------------------------------------------
binary_mask = np.zeros((N, N))
binary_mask[:, N//2:] = 1   # or any pattern you want

phase = np.pi * binary_mask
# -------------------------------------------------
# Complex field at SLM plane
# -------------------------------------------------
U = amplitude * np.exp(1j * phase)

# -------------------------------------------------
# Fourier transform (focal plane)
# -------------------------------------------------
U_f = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U)))
intensity_out = np.abs(U_f)**2
intensity_out /= intensity_out.max()   # normalize

# -------------------------------------------------
# Plot: 3 subplots
# -------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4))

# (1) Initial intensity
im0 = ax[0].imshow(
    abs(U),
    extent=[-L/2, L/2, -L/2, L/2],
    cmap="inferno"
)
ax[0].set_title("Initial Intensity (SLM plane)")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
plt.colorbar(im0, ax=ax[0])

# (2) Initial phase
im1 = ax[1].imshow(
    phase,
    extent=[-L/2, L/2, -L/2, L/2],
    cmap="gray"
)
ax[1].set_title("Initial Phase (0 / π)")
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
plt.colorbar(im1, ax=ax[1])

# (3) Focal plane intensity
im2 = ax[2].imshow(
    intensity_out,
    cmap="inferno"
)
ax[2].set_title("Focal Plane Intensity |FFT(U)|²")
ax[2].set_xlabel("fx")
ax[2].set_ylabel("fy")
plt.colorbar(im2, ax=ax[2])

plt.tight_layout()
plt.show()

print(np.shape(intensity_out))



