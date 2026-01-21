import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Parameters
# =========================================================
N = 100
n_iters = 2500
beta = 0.3   # strong background damping (important)

# =========================================================
# MRAF Algorithm
# =========================================================
def mraf(A_source_amp, A_target_amp, support_mask, n_iters, beta):

    phase = np.random.uniform(0, 2*np.pi, A_source_amp.shape)
    field_source = A_source_amp * np.exp(1j * phase)

    errors = []

    for _ in range(n_iters):

        # Forward propagation (Fraunhofer)
        field_target = np.fft.fftshift(
            np.fft.fft2(
                np.fft.ifftshift(field_source)
            )
        )

        # ---- ERROR BEFORE CONSTRAINT ----
        err = np.mean(
            (np.abs(field_target)[support_mask] -
             A_target_amp[support_mask])**2
        )
        errors.append(err)

        # ---- TARGET CONSTRAINT (MRAF) ----
        amp = np.abs(field_target)
        amp[support_mask] = A_target_amp[support_mask]
        amp[~support_mask] *= beta

        field_target = amp * np.exp(1j * np.angle(field_target))

        # Backward propagation
        field_source = np.fft.fftshift(
            np.fft.ifft2(
                np.fft.ifftshift(field_target)
            )
        )

        # ---- SOURCE CONSTRAINT ----
        field_source = A_source_amp * np.exp(1j * np.angle(field_source))

    return np.angle(field_source), field_target, errors


# =========================================================
# Visualization helpers
# =========================================================
def show_intensity_gray(I, title):
    I = I / I.max()
    plt.imshow((255 * I).astype(np.uint8), cmap="gray")
    plt.colorbar(label="0–255")
    plt.title(title)
    plt.axis("off")


def show_phase_gray(phase, title):
    phase = phase % (2*np.pi)
    plt.imshow((255 * phase / (2*np.pi)).astype(np.uint8), cmap="gray")
    plt.colorbar(label="0–255 ↔ 0–2π")
    plt.title(title)
    plt.axis("off")


# =========================================================
# SOURCE INTENSITY (Gaussian beam)
# =========================================================
x = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, x)

source_intensity = np.exp(-(X**2 + Y**2) / 0.3)
A_source_amp = np.sqrt(source_intensity)

# =========================================================
# TARGET INTENSITY (face pattern)
# =========================================================
target_intensity = np.zeros((N, N)) + 5  # background floor

# Eyes
target_intensity[40:50, 35:45] = 255
target_intensity[40:50, 55:65] = 255

# Mouth
target_intensity[60:70, 40:65] = 200

A_target_amp = np.sqrt(target_intensity)

# =========================================================
# SUPPORT WINDOW (CRITICAL)
# =========================================================
support_mask = np.zeros((N, N), dtype=bool)
support_mask[30:80, 30:80] = True

# =========================================================
# ENERGY NORMALIZATION (CRITICAL)
# =========================================================
A_source_amp /= np.sqrt(np.sum(A_source_amp**2))
A_target_amp /= np.sqrt(np.sum(A_target_amp[support_mask]**2))

# ========================================================= attach target only inside support
A_target_amp[~support_mask] = 0

# =========================================================
# RUN MRAF
# =========================================================
phase, field_target, errors = mraf(
    A_source_amp,
    A_target_amp,
    support_mask,
    n_iters,
    beta
)

output_intensity = np.abs(field_target)**2

# =========================================================
# DISPLAY RESULTS
# =========================================================
plt.figure(figsize=(18, 4))

plt.subplot(1, 5, 1)
show_intensity_gray(source_intensity, "Source intensity")

plt.subplot(1, 5, 2)
show_intensity_gray(output_intensity, "MRAF output intensity")

plt.subplot(1, 5, 3)
show_intensity_gray(target_intensity, "Target intensity")

plt.subplot(1, 5, 4)
show_phase_gray(phase, "Retrieved phase mask")

plt.subplot(1, 5, 5)
plt.plot(errors)
plt.title("Convergence (MSE)")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.grid(True)

plt.tight_layout()
plt.show()
