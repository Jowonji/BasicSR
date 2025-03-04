import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns

# ---- Min-Max 정규화 ---- #
def min_max_normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)  # 0으로 나눔 방지

# ---- GLCM 계산 ---- #
def fast_glcm(img, levels=8, kernel_size=5, distance=1.0, angle=0.0):
    h, w = img.shape

    # 정규화된 이미지 (0~1)를 levels 단계로 변환 (0~levels-1)
    bins = np.linspace(0, 1 + 1e-8, levels + 1)
    gl1 = np.digitize(img, bins) - 1

    # 특정 거리와 각도로 이동한 픽셀과 비교
    dx = distance * np.cos(np.deg2rad(angle))
    dy = distance * np.sin(np.deg2rad(-angle))
    mat = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
    gl2 = cv2.warpAffine(gl1, mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

    # GLCM 행렬 초기화
    glcm = np.zeros((levels, levels, h, w), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            mask = (gl1 == i) & (gl2 == j)
            glcm[i, j, mask] = 1

    # 커널 적용하여 주변 영역 고려
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            glcm[i, j] = cv2.filter2D(glcm[i, j], -1, kernel)

    return glcm.astype(np.float32)

# ---- GLCM 특징 계산 ---- #
def calc_glcm_features(img, levels=8, ks=5, distances=[1], angles=[0, 45, 90, 135]):
    h, w = img.shape
    features = {}

    mean = np.zeros((h, w), dtype=np.float32)
    contrast = np.zeros((h, w), dtype=np.float32)
    dissimilarity = np.zeros((h, w), dtype=np.float32)
    homogeneity = np.zeros((h, w), dtype=np.float32)
    asm = np.zeros((h, w), dtype=np.float32)
    entropy = np.zeros((h, w), dtype=np.float32)

    for d in distances:
        for angle in angles:
            glcm = fast_glcm(img, levels=levels, kernel_size=ks, distance=d, angle=angle)

            # Mean
            for i in range(levels):
                for j in range(levels):
                    mean += glcm[i, j] * i / levels**2

            # Contrast
            for i in range(levels):
                for j in range(levels):
                    contrast += glcm[i, j] * (i - j) ** 2

            # Dissimilarity
            for i in range(levels):
                for j in range(levels):
                    dissimilarity += glcm[i, j] * abs(i - j)

            # Homogeneity
            for i in range(levels):
                for j in range(levels):
                    homogeneity += glcm[i, j] / (1.0 + (i - j) ** 2)

            # ASM (Angular Second Moment) & Energy
            for i in range(levels):
                for j in range(levels):
                    asm += glcm[i, j] ** 2

            # Entropy
            pnorm = glcm / (np.sum(glcm, axis=(0, 1)) + 1.0 / ks**2)
            entropy += np.sum(-pnorm * np.log(pnorm + 1e-8), axis=(0, 1))

    features["mean"] = np.mean(mean)
    features["contrast"] = np.mean(contrast)
    features["dissimilarity"] = np.mean(dissimilarity)
    features["homogeneity"] = np.mean(homogeneity)
    features["asm"] = np.mean(asm)
    features["entropy"] = np.mean(entropy)

    return features

# ---- 데이터 로드 ---- #
def load_data(lr_path, hr_path):
    """ NumPy 데이터 로드 (LR: 28000개, HR: 28000개) """
    lr_data = np.load(lr_path)  # (28000, 20, 20)
    hr_data = np.load(hr_path)  # (28000, 100, 100)

    # 정규화
    lr_data = np.array([min_max_normalize(lr) for lr in lr_data])
    hr_data = np.array([min_max_normalize(hr) for hr in hr_data])

    return lr_data, hr_data

# ---- 텍스처 분석 ---- #
def analyze_texture(lr_data, hr_data, sample_size=5000):
    """ LR과 HR의 텍스처 차이 분석 """
    lr_features = []
    hr_features = []

    for i in range(sample_size):
        lr_features.append(calc_glcm_features(lr_data[i]))
        hr_features.append(calc_glcm_features(hr_data[i]))

        if i % 1000 == 0:
            print(f"Processed {i}/{sample_size} samples...")

    # 평균값 계산
    lr_avg = {key: np.mean([f[key] for f in lr_features]) for key in lr_features[0]}
    hr_avg = {key: np.mean([f[key] for f in hr_features]) for key in hr_features[0]}

    return lr_avg, hr_avg

# ---- 시각화 ---- #
def plot_texture_comparison(lr_avg, hr_avg):
    """ GLCM 기반 텍스처 비교 시각화 """
    plt.figure(figsize=(12, 6))
    for i, key in enumerate(lr_avg.keys()):
        plt.subplot(2, 3, i+1)
        plt.bar(["LR", "HR"], [lr_avg[key], hr_avg[key]])
        plt.title(key)
    plt.tight_layout()
    plt.show()

# ---- 실행 ---- #
if __name__ == "__main__":
    train_lr_path = "/home/wj/works/Wind_Speed_Data/new_data/LR/train_lr.npy"
    train_hr_path = "/home/wj/works/Wind_Speed_Data/new_data/HR/train_hr.npy"

    # 데이터 로드
    train_lr, train_hr = load_data(train_lr_path, train_hr_path)

    # LR vs HR 텍스처 분석 (5000개 샘플)
    print("Analyzing texture differences...")
    lr_avg, hr_avg = analyze_texture(train_lr, train_hr, sample_size=5000)

    # 결과 시각화
    plot_texture_comparison(lr_avg, hr_avg)
