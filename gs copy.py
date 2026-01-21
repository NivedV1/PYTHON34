import numpy as np
import matplotlib.pyplot as plt

def matrix_correlation(A, B):
    A = np.asarray(A).ravel()
    B = np.asarray(B).ravel()
    return np.corrcoef(A, B)[0, 1]

def pretty_print(M, width=7, precision=2):
    for row in M:
        print(" ".join(f"{v:{width}.{precision}f}" for v in row))


# =========================================================
# Gerchberg–Saxton Algorithm
# =========================================================
def gerchberg_saxton(A_source_amp, A_target_amp, n_iters):

    phase = np.random.uniform(0, 2*np.pi, A_source_amp.shape)
    field_source = A_source_amp * np.exp(1j * phase)

    for i in range(n_iters):

        # Forward FFT
        field_target = np.fft.fft(field_source)

        # Enforce target amplitude
        field_target = A_target_amp * np.exp(1j * np.angle(field_target))

        # Backward IFFT
        field_source =np.fft.ifft(field_target)
        # Enforce source amplitude
        field_source = A_source_amp * np.exp(1j * np.angle(field_source))

    phase = np.angle(field_source)
    return phase, field_source, field_target


# =========================================================
# Visualization helpers
# =========================================================
def show_intensity_gray(I, title=""):
    I_norm = I / I.max()
    I_gray = (255 * I_norm).astype(np.uint8)
    plt.imshow(I_gray, cmap="gray", vmin=0, vmax=255)
    plt.colorbar(label="Intensity (0–255)")
    plt.title(title)
    plt.axis("off")


def show_phase_gray(phase, title="Phase mask"):
    phase_wrapped = phase % (2 * np.pi)
    phase_gray = (255 * phase_wrapped / (2 * np.pi)).astype(np.uint8)
    plt.imshow(phase_gray, cmap="gray", vmin=0, vmax=255)
    plt.colorbar(label="Phase (0–255 ↔ 0–2π)")
    plt.title(title)
    plt.axis("off")


# =========================================================
# YOUR INPUT MATRICES (Intensity)
# =========================================================
A_intensity = np.array([
 [256,2,3,253,252,6,7,249,248,10,11,245,244,14,15,241],
 [17,239,238,20,21,235,234,24,25,231,230,28,29,227,226,32],
 [33,223,222,36,37,219,218,40,41,215,214,44,45,211,210,48],
 [208,50,51,205,204,54,55,201,200,58,59,197,196,62,63,193],
 [192,66,67,189,188,70,71,185,184,74,75,181,180,78,79,177],
 [81,175,174,84,85,171,170,88,89,167,166,92,93,163,162,96],
 [97,159,158,100,101,155,154,104,105,151,150,108,109,147,146,112],
 [144,114,115,141,140,118,119,137,136,122,123,133,132,126,127,129],
 [128,130,131,125,124,134,135,121,120,138,139,117,116,142,143,113],
 [145,111,110,148,149,107,106,152,153,103,102,156,157,99,98,160],
 [161,95,94,164,165,91,90,168,169,87,86,172,173,83,82,176],
 [80,178,179,77,76,182,183,73,72,186,187,69,68,190,191,65],
 [64,194,195,61,60,198,199,57,56,202,203,53,52,206,207,49],
 [209,47,46,212,213,43,42,216,217,39,38,220,221,35,34,224],
 [225,31,30,228,229,27,26,232,233,23,22,236,237,19,18,240],
 [16,242,243,13,12,246,247,9,8,250,251,5,4,254,255,1]
], dtype=float)

B_intensity = np.array([
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
 [50,50,50,50,50,50,50,50,50,50,50,240,230,220,50,50]
], dtype=float)


# =========================================================
# Convert intensity → amplitude
# =========================================================
A_source_amp = np.sqrt(A_intensity)
A_target_amp = np.sqrt(B_intensity)

# =========================================================
# Run GS
# =========================================================
phase, U_source, U_target = gerchberg_saxton(
    A_source_amp,
    A_target_amp,
    n_iters=10000
)

phase = np.angle(U_source)

U_test = A_source_amp * np.exp(1j * phase)

U_test_f = np.fft.fft(U_test)

I_test = np.abs(U_test_f)**2


# Intensities
I_source = np.abs(U_source)**2
I_target = np.abs(U_target)**2


#pretty_print(B_intensity)
#print("nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn")
#pretty_print(I_target)

#corr = matrix_correlation(B_intensity, I_target)
#print("Correlation:", corr)

#O=B_intensity-I_target
#pretty_print(O)
corr = matrix_correlation(I_test, B_intensity)
print("Correct correlation:", corr)

pretty_print(B_intensity - I_test)
# =========================================================
# Display
# =========================================================
plt.figure(figsize=(16,4))

plt.subplot(1,4,1)
show_intensity_gray(A_intensity, "Source intensity (input)")

plt.subplot(1,4,2)
show_intensity_gray(I_test, "GS output intensity")

plt.subplot(1,4,3)
show_intensity_gray(B_intensity, "Target intensity (desired)")

plt.subplot(1,4,4)
show_phase_gray(phase, "Retrieved phase mask")

plt.tight_layout()
plt.show()
