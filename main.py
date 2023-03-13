import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
import csv
from scipy.fftpack import dct, idct
from scipy.signal import wiener
from scipy.ndimage import median_filter
from skimage.metrics import structural_similarity as ssim
import torch
# PSNR
from math import log10, sqrt
# Proximal alternating linearilization minimization
# https://pyproximal.readthedocs.io/en/stable/api/generated/pyproximal.optimization.palm.PALM.html#pyproximal.optimization.palm.PALM
from pyproximal.optimization.palm import PALM
from pyproximal.utils.bilinear import BilinearOperator
from pyproximal import L1


def PSNR(original, denoised):
    mse = np.mean((original - denoised) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


# SSIM
def calculate_ssim(original, denoised):
    if not original.shape == denoised.shape:
        raise ValueError('Input images must have the same dimensions.')
    if original.ndim == 2:
        return ssim(original, denoised)
    elif original.ndim == 3:
        if original.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(original, denoised))
            return np.array(ssims).mean()
        elif original.shape[2] == 1:
            return ssim(np.squeeze(original), np.squeeze(denoised))
    else:
        raise ValueError('Wrong input image dimensions.')


def dct2(block):
    return dct(dct(block.T).T)


def idct2(block):
    return idct(idct(block.T).T)


def get_blocks(image, block_size=8):
    patches = [image[i:i + 8, j:j + 8] for i in range(0, image.shape[0], 8) for j in range(0, image.shape[1], 8)]
    # patches = []
    # for i in range(0, image.shape[0], 8):
    #     for j in range(0, image.shape[1], 8):
    #         patch = image[i:i + 8, j:j + 8]
    #         patches.append(patch)

    return np.array(patches)


def get_image(patches, h, w):
    image = np.zeros((h, w))
    c = 0
    for i in range(0, image.shape[0], 8):
        for j in range(0, image.shape[1], 8):
            image[i:i + 8, j:j + 8] = patches[c]
            c += 1
    return image


def get_patch_dct(patches):
    return [dct2(patch) for patch in patches]


def get_patch_idct(patches):
    return [idct2(patch) for patch in patches]


def generate_low_frequency_image(image):
    patches = get_blocks(image)
    patches = get_patch_dct(patches)

    low_freq = []
    for p in patches:
        p[1:] = 0
        p[:, 1:] = 0
        low_freq.append(p)

    low_patches = get_patch_idct(low_freq)
    low_image = get_image(low_patches, image.shape[0], image.shape[1])
    low_image = (low_image - low_image.min()) / (low_image.max() - low_image.min())

    return low_image


def minimize_error_function_9(YHi, DHi, MHi, NHi0, lam, p, mu, lr, iter):
    def h_mu(x, p, mu):
        return torch.sum((x ** 2 + mu) ** (p / 2))

    # YHi.shape = (8, 8) -> (64)
    # DHi.shape(k, 8, 8) -> (k * 64)
    # NHi.shape = (n * k)
    NHi = torch.autograd.Variable(NHi0, requires_grad=True)
    optimizer = torch.optim.Adam([NHi], lr=lr)
    history = []
    for i in range(iter):
        f = YHi - torch.matmul(NHi, DHi)
        f = h_mu(f, p, mu)
        f += lam * torch.sum(torch.abs(MHi - NHi))

        optimizer.zero_grad()
        f.backward()
        optimizer.step()
        history.append(f.item())

    plt.plot(history)
    plt.show()
    return NHi.detach().numpy(), f.item()


def construct_initial_dictionary(high_images, high_images_median, k=10, d=100):
    patches = []
    for image in high_images:
        patches += list(get_blocks(image))
    patches = np.array(patches)
    patches = patches.reshape(-1, 64)

    patches_median = []
    for image in high_images_median:
        patches_median += list(get_blocks(image))
    patches_median = np.array(patches_median)
    patches_median = patches_median.reshape(-1, 64)

    dictionaries = [patches[np.random.permutation(patches.shape[0])[:d]] for _ in range(k)]

    # dictionaries = []
    # for _ in range(k):
    #     di = patches[np.random.permutation(patches.shape[0])[:d]]
    #     dictionaries.append(di)

    return dictionaries, patches, patches_median


def calculate_sparse_representation(dictionaries, MH, NH0, patches, lam, p, mu, lr, iter):
    sparse_representations = []
    best_val = np.inf
    best_idx = 0
    for i in range(len(dictionaries)):
        ret = minimize_error_function_9(torch.from_numpy(patches).float(),
                                        torch.from_numpy(dictionaries[i]).float(),
                                        torch.from_numpy(MH[i]).float(),
                                        torch.from_numpy(NH0[i]).float(),
                                        lam, p, mu, lr, iter)
        sparse_representation, value = ret
        if best_val > value:
            best_idx = i
            best_val = value
        sparse_representations.append(sparse_representation)
    return sparse_representations, best_idx


class approx_dictionary_h(BilinearOperator):
    # x = NHi / MHi
    # y = DHi
    def __init__(self, YH, MHi, MDi, p, mu):
        super().__init__()
        self.YH = YH
        self.x = MHi
        self.y = MDi
        self.p = p
        self.mu = mu

    def h_mu(self, t, p, mu):
        return np.sum((t ** 2 + mu) ** (p / 2))

    def __call__(self, x, y):
        self.updatex(x)
        self.updatey(y)
        res = self.h_mu(self.YH - np.dot(self.x, self.y), self.p, self.mu)
        return res

    def gradx(self, x):
        # d/dMHi (h(Yi - MDi.MHi)) = d/dx h(x) * MDi
        # d/dx h(x) = p * x * (x^2 + mu) ^ (p / 2 - 1)
        # d/dMHi (h(Yi - MDi.MHi)) = p * x * (x^2 + mu) ^ (p / 2 - 1) * MDi
        f = self.YH - np.dot(x.reshape(self.x.shape), self.y)
        dMHi = -np.dot(self.p * f * (f ** 2 + self.mu) ** (self.p / 2 - 1), self.y.T)
        return dMHi

    def grady(self, y):
        # d/dMHi (h(Yi - MDi.MHi)) = d/dx h(x) * MDi
        # d/dx h(x) = p * x * (x^2 + mu) ^ (p / 2 - 1)
        # d/dMHi (h(Yi - MDi.MHi)) = p * x * (x^2 + mu) ^ (p / 2 - 1) * MDi
        f = self.YH - np.dot(self.x, y.reshape(self.y.shape))
        dMDi = -np.dot(self.x.T, self.p * f * (f ** 2 + self.mu) ** (self.p / 2 - 1))
        return dMDi

    def lx(self, x):
        # lx = d^2 / dx^2 H(w) , YH = Y.W
        # f = YH - X.Y, YH = Y.W => f = 0
        # d^2 / dx^2 H(w) = 2 * (p/2 - 1) * p * Y * f^2 * (f^2 + mu) ^ (P/2 - 2) + p * y^2 * (f^2 + mu)^(p/2 - 1)
        # lx = p * y^2 * mu^(p/2 - 1)
        return self.p * np.max(self.y ** 2) * mu ** (p / 2 - 1)

    def ly(self, y):
        # ly = d^2 / dy^2 H(w) , YH = W.X
        # f = YH - X.Y, YH = W.X => f = 0
        # d^2 / dy^2 H(w) = 2 * (p/2 - 1) * p * X * f^2 * (f^2 + mu) ^ (P/2 - 2) + p * X^2 * (f^2 + mu)^(p/2 - 1)
        # ly = p * X^2 * mu^(p/2 - 1)
        return self.p * np.max(self.x ** 2) * mu ** (p / 2 - 1)

    def updatex(self, x):
        self.x = x

    def updatey(self, y):
        self.y = y


def calculate_prox_update(Yi, MDi, MHi, p, mu, gamma1, gamma2, lam):
    # H(x, y) -> function of MHi and MDi
    # proxf(x) -> function of only MHi
    # proxg(y) -> function of only MDi

    H = approx_dictionary_h(Yi, MHi, MDi, p, mu)
    # print(H(MHi, MDi), H.gradx(MHi), H.grady(MDi))
    proxf = L1(lam)
    proxg = None

    MHi, MDi = PALM(H, proxf, proxg, MHi, MDi, gamma1, gamma2, niter=50, show=True)
    MHi, MDi = PALM(H, proxf, proxg, MHi, MDi, gamma1 / 2, gamma2 / 2, niter=50, show=True)
    MHi, MDi = PALM(H, proxf, proxg, MHi, MDi, gamma1 / 4, gamma2 / 4, niter=50, show=True)
    # MHi, MDi = PALM(H, proxf, proxg, MHi, MDi, gamma1/8, gamma2/8, niter=20, show=True)

    return MHi, MDi


def calculate_stop_condition(Yi, MHi, MHi1, MDi, MDi1, p, mu):
    H1 = approx_dictionary_h(Yi, MHi, MDi, p, mu)
    ci = H1.lx(MHi)
    di = H1.ly(MDi)

    H2 = approx_dictionary_h(Yi, MHi1, MDi1, p, mu)

    AMH = ci * (MHi - MHi1) + H2.gradx(MHi1) - H1.gradx(MHi)
    AMD = di * (MDi - MDi1) + H2.grady(MDi1) - H1.grady(MDi)

    return AMH, AMD

def add_sAndp_noise(image):
    imp_noise = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    image = np.asarray(image, np.uint8)

    cv2.randu(imp_noise, 0, 255)
    imp_noise = cv2.threshold(imp_noise, 245, 255, cv2.THRESH_BINARY)[1]

    # in_img = image + imp_noise
    in_img = cv2.add(image, imp_noise)

    # fig = plt.figure(dpi=300)
    #
    # fig.add_subplot(1, 3, 1)
    # plt.imshow(image, cmap='gray')
    # plt.axis("off")
    # plt.title("Original")
    #
    # fig.add_subplot(1, 3, 2)
    # plt.imshow(imp_noise, cmap='gray')
    # plt.axis("off")
    # plt.title("Impulse Noise")
    #
    # fig.add_subplot(1, 3, 3)
    # plt.imshow(in_img, cmap='gray')
    # plt.axis("off")
    # plt.title("Combined")
    # plt.show()

    return in_img


def add_gaussian_noise(image):
    gauss_noise = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    image = np.asarray(image, np.uint8)

    cv2.randn(gauss_noise, 128, 20)
    gauss_noise = (gauss_noise * 0.5).astype(np.uint8)
    print(img.shape)
    print(gauss_noise.shape)


    gn_img = cv2.add(image, gauss_noise)

    # fig = plt.figure(dpi=300)
    #
    # fig.add_subplot(1, 3, 1)
    # plt.imshow(image, cmap='gray')
    # plt.axis("off")
    # plt.title("Original")
    #
    # fig.add_subplot(1, 3, 2)
    # plt.imshow(gauss_noise, cmap='gray')
    # plt.axis("off")
    # plt.title("Gaussian Noise")
    #
    # fig.add_subplot(1, 3, 3)
    # plt.imshow(gn_img, cmap='gray')
    # plt.axis("off")
    # plt.title("Combined")
    # plt.show()

    return gn_img


results = []
for image in glob('data1/*'):

    k = 1  # number of dictionaries
    d = 400  # number of patterns in each dictionary
    epsilon = 1e-3  # optimization stop criteria
    lam = 1e-8  # L1 regularization
    p = 0.9  # power
    mu = 0.5
    gamma1 = 10000  # 1/lr -> 0.10000
    gamma2 = 10  # 1/lr -> 0.10

    img = cv2.imread(image, 0)
    new_size = (int(img.shape[1] / 8) * 8, int(img.shape[0] / 8) * 8)
    img = cv2.resize(img, new_size)
    img = (img - img.min()) / (img.max() - img.min()) * 255

    noisy_img1 = add_sAndp_noise(img)
    noisy_image = add_gaussian_noise(noisy_img1)


    fig = plt.figure(dpi=300)

    fig.add_subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.axis("off")
    plt.title("Original")

    fig.add_subplot(1, 2, 2)
    plt.imshow(noisy_image, cmap='gray')
    plt.axis("off")
    plt.title("Noisy Image")

    plt.show()

    # rows, cols = img.shape
    # noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
    # rng = np.random.default_rng()
    # noise[rng.random(size=noise.shape) > 0.5] *= -1
    #
    # img_noise = img + noise

    low_image = generate_low_frequency_image(noisy_image)

    high_image = noisy_image - low_image
    high_image_median = median_filter(high_image, 3)

    DH, patches, patches_median = construct_initial_dictionary([high_image], [high_image_median], k, d)

    init_zeros = np.zeros((len(DH), len(patches), DH[0].shape[0]))

    MH, _ = calculate_sparse_representation(DH, init_zeros, init_zeros, patches, lam, p, mu, 0.001, 100)
    NH, _ = calculate_sparse_representation(DH, MH, MH, patches_median, lam, p, mu, 0.0005, 100)

    Amh = np.array([0])
    Amd = np.array([0])
    for j in range(len(DH)):
        while max(np.abs(Amh).max(), np.abs(Amd).max()) < epsilon:
            # MH[j] = minimize_error_function_12(patches, DH[j], MH[j], p, ci, mu)
            # DH[j] = minimize_error_function_13(patches, DH[j], MH[j], p, ci, mu)
            MH_new, DH_new = calculate_prox_update(patches, DH[j], MH[j], p, mu, gamma1, gamma2, lam)
            Amh, Amd = calculate_stop_condition(patches, MH[j], MH_new, DH[j], DH_new, p, mu)
            MH[j], DH[j] = MH_new, DH_new
            print(np.abs(Amh).max(), np.abs(Amd).max())

    NH, best_idx = calculate_sparse_representation(DH, MH, NH, patches_median, lam, p, mu, 0.0001, 100)

    high_image = np.dot(NH[best_idx], DH[best_idx])
    high_image = get_image(high_image.reshape(-1, 8, 8), img.shape[0], img.shape[1])
    high_image = (high_image - high_image.min()) / (high_image.max() - high_image.min())

    # img_result = low_image + high_image
    # img_result = (img_result - img_result.min()) / (img_result.max() - img_result.min())

    low_image = wiener(low_image, (16, 16))

    img_result = cv2.addWeighted(low_image, 1, high_image, 1, 0.0)
    # img_result = median_filter(img_result, 8)
    img_result = (img_result - img_result.min()) / (img_result.max() - img_result.min()) * 255

    SSIM_noisy_value = calculate_ssim(img, noisy_image)
    PSNR_noisy_value = PSNR(img, noisy_image)

    SSIM_result_value = calculate_ssim(img, img_result)
    PSNR_result_value = PSNR(img, img_result)

    # result = (PSNR_result_value, SSIM_result_value)

    result = {'Image': image, 'PSNR': PSNR_result_value, 'SSIM': SSIM_result_value}
    results.append(result)

    # field names
    fields = ['Image', 'PSNR', 'SSIM']

    with open('Results.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)

        writer.writeheader()

        writer.writerows(results)

    plt.figure(figsize=(10., 10.))
    plt.subplot(1, 2, 1)
    plt.title(f'Noisy image: PSNR = {PSNR_noisy_value:.2f}, SSIM = {SSIM_noisy_value:.2f}')
    plt.imshow(noisy_image, cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title(f'Result: PSNR = {PSNR_result_value:.2f}, SSIM = {SSIM_result_value:.2f}')
    plt.imshow(img_result, cmap="gray")
    plt.show()

    file.close()
