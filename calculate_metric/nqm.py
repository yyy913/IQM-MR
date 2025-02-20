# https://github.com/lucia15/Image-Quality-Assessment/blob/master/IQA_indices.py

import numpy as np
from scipy.fftpack import fftshift as __fftshift

def mse(reference, query):
    """
    Computes the Mean Square Error (MSE) of two images
    :param reference: original image data
    :param query: modified image data to be compared
    :return: MSE value
    """
    (ref, que) = (reference.astype('double'), query.astype('double'))
    diff = ref - que
    square = (diff ** 2)
    mean = square.mean()
    return mean

def snr(reference, query):
    """
    Computes the Signal-to-Noise-Ratio (SNR)
    :param reference: original image data
    :param query: modified image data to be compared
    :return: SNR value
    """
    signal_value = (reference.astype('double') ** 2).mean()
    msev = mse(reference, query)
    if msev != 0:
        value = 10.0 * np.log10(signal_value / msev)
    else:
        value = float("inf")
    return value

def __convert_to_luminance(x):
    return np.dot(x[..., :3], [0.299, 0.587, 0.144]).astype('double')

def nqm(reference, query):
    """
    Computes the NQM metric
    :param reference: original image data
    :param query: modified image data to be compared
    :return: NQM value
    """
    def __ctf(f_r):
        """ Bandpass Contrast Threshold Function for RGB"""
        (gamma, alpha) = (0.0192 + 0.114 * f_r, (0.114 * f_r) ** 1.1)
        beta = np.exp(-alpha)
        num = 520.0 * gamma * beta
        return 1.0 / num

    def _get_masked(c, ci, a, ai, i):
        (H, W) = c.shape
        (c, ci, ct) = (c.flatten('F'), ci.flatten('F'), __ctf(i))
        ci[abs(ci) > 1.0] = 1.0
        T = ct * (0.86 * ((c / ct) - 1.0) + 0.3)
        (ai, a, a1) = (ai.flatten('F'), a.flatten('F'), (abs(ci - c) - T) < 0.0)
        ai[a1] = a[a1]
        return ai.reshape(H, W)

    def __get_thresh(x, T, z, trans=True):
        (H, W) = x.shape
        if trans:
            (x, z) = (x.flatten('F').T, z.flatten())
        else:
            (x, z) = (x.flatten('F'), z.flatten('F'))
        z[abs(x) < T] = 0.0
        return z.reshape(H, W)

    def __decompose_cos_log_filter(w1, w2, phase=np.pi):
        return 0.5 * (1 + np.cos(np.pi * np.log2(w1 + w2) - phase))

    def __get_w(r):
        w = [(r + 2) * ((r + 2 <= 4) * (r + 2 >= 1))]
        w += [r * ((r <= 4) * (r >= 1))]
        w += [r * ((r >= 2) * (r <= 8))]
        w += [r * ((r >= 4) * (r <= 16))]
        w += [r * ((r >= 8) * (r <= 32))]
        w += [r * ((r >= 16) * (r <= 64))]
        return w

    def __get_u(r):
        u = [4 * (np.logical_not((r + 2 <= 4) * (r + 2 >= 1)))]
        u += [4 * (np.logical_not((r <= 4) * (r >= 1)))]
        u += [0.5 * (np.logical_not((r >= 2) * (r <= 8)))]
        u += [4 * (np.logical_not((r >= 4) * (r <= 16)))]
        u += [0.5 * (np.logical_not((r >= 8) * (r <= 32)))]
        u += [4 * (np.logical_not((r >= 16) * (r <= 64)))]
        return u

    def __get_G(r):
        (w, u) = (__get_w(r), __get_u(r))
        phase = [np.pi, np.pi, 0.0, np.pi, 0.0, np.pi]
        dclf = __decompose_cos_log_filter
        return [dclf(w[i], u[i], phase[i]) for i in range(len(phase))]

    def __compute_fft_plane_shifted(ref, query):
        (x, y) = ref.shape
        (xplane, yplane) = np.mgrid[-y / 2:y / 2, -x / 2:x / 2]
        plane = (xplane + 1.0j * yplane)
        r = abs(plane)
        G = __get_G(r)
        Gshifted = list(map(__fftshift, G))
        return [Gs.T for Gs in Gshifted]

    def __get_c(a, l_0):
        c = [a[0] / l_0]
        c += [a[1] / (l_0 + a[0])]
        c += [a[2] / (l_0 + a[0] + a[1])]
        c += [a[3] / (l_0 + a[0] + a[1] + a[2])]
        c += [a[4] / (l_0 + a[0] + a[1] + a[2] + a[3])]
        return c

    def __get_ci(ai, li_0):
        ci = [ai[0] / (li_0)]
        ci += [ai[1] / (li_0 + ai[0])]
        ci += [ai[2] / (li_0 + ai[0] + ai[1])]
        ci += [ai[3] / (li_0 + ai[0] + ai[1] + ai[2])]
        ci += [ai[4] / (li_0 + ai[0] + ai[1] + ai[2] + ai[3])]
        return ci

    def __compute_contrast_images(a, ai, l, li):
        ci = __get_ci(ai, li)
        c = __get_c(a, l)
        return (c, ci)

    def __get_detection_thresholds():
        viewing_angle = (1.0 / 3.5) * (180.0 / np.pi)
        rotations = [2.0, 4.0, 8.0, 16.0, 32.0]
        return list(map(lambda x: __ctf(x / viewing_angle), rotations))

    def __get_account_for_supra_threshold_effects(c, ci, a, ai):
        r = range(len(a))
        return [_get_masked(c[i], ci[i], a[i], ai[i], i + 1) for i in r]

    def __apply_detection_thresholds(c, ci, d, a, ai):
        A = [__get_thresh(c[i], d[i], a[i], False) for i in range(len(a))]
        AI = [__get_thresh(ci[i], d[i], ai[i], True) for i in range(len(a))]
        return (A, AI)

    def __reconstruct_images(A, AI):
        return list(map(lambda x: np.add.reduce(x), (A, AI)))

    def __compute_quality(imref, imquery):
        return snr(imref, imquery)

    def __get_ref_basis(ref_fft, query_fft, GS):
        (L_0, LI_0) = list(map(lambda x: GS[0] * x, (ref_fft, query_fft)))
        (l_0, li_0) = list(map(lambda x: np.real(np.fft.ifft2(x)), (L_0, LI_0)))
        return (l_0, li_0)

    def __compute_inverse_convolution(convolved_fft, GS):
        convolved = [GS[i] * convolved_fft for i in range(1, len(GS))]
        return list(map(lambda x: np.real(np.fft.ifft2(x)), convolved))

    def __correlate_in_fourier_domain(ref, query):
        (ref_fft, query_fft) = list(map(lambda x: np.fft.fft2(x), (ref, query)))
        GS = __compute_fft_plane_shifted(ref, query)
        (l_0, li_0) = __get_ref_basis(ref_fft, query_fft, GS)
        a = __compute_inverse_convolution(ref_fft, GS)
        ai = __compute_inverse_convolution(query_fft, GS)
        return (a, ai, l_0, li_0)

    def __get_correlated_images(ref, query):
        (a, ai, l_0, li_0) = __correlate_in_fourier_domain(ref, query)
        (c, ci) = __compute_contrast_images(a, ai, l_0, li_0)
        d = __get_detection_thresholds()
        ai = __get_account_for_supra_threshold_effects(c, ci, a, ai)
        return __apply_detection_thresholds(c, ci, d, a, ai)

    if not len(reference.shape) < 3:
        reference = __convert_to_luminance(reference)
        query = __convert_to_luminance(query)
    (A, AI) = __get_correlated_images(reference, query)
    (y1, y2) = __reconstruct_images(A, AI)
    y = __compute_quality(y1, y2)
    return y