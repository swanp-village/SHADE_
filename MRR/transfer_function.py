"""
import numpy as np
import numpy.typing as npt


def simulate_transfer_function(
    wavelength: npt.NDArray[np.float_],
    L: npt.ArrayLike,
    K: npt.ArrayLike,
    alpha: float,
    eta: float,
    n_eff: float,
    n_g: float,
    center_wavelength: float,
) -> npt.NDArray[np.float_]:
    y: npt.NDArray[np.float_] = 20 * np.log10(np.abs(_D(L, K, alpha, wavelength, eta, n_eff, n_g, center_wavelength)))
    return y.reshape(y.size)


def _C(K_k: float, eta: float) -> npt.NDArray[np.float_]:
    C: npt.NDArray[np.float_] = (
        1
        / (-1j * eta * np.sqrt(K_k))
        * np.array([[1, -eta * np.sqrt(eta - K_k)], [np.sqrt(eta - K_k) * eta, -(eta ** 2)]])
    )
    return C


def _R(
    a_k: float,
    L_k: float,
    wavelength: npt.NDArray[np.float_],
    n_eff: float,
    n_g: float,
    center_wavelength: float,
) -> npt.NDArray[np.float_]:
    N_k = np.round(L_k * n_eff / center_wavelength)
    shifted_center_wavelength = L_k * n_eff / N_k
    x = (
        1j
        * np.pi
        * L_k
        * n_g
        * (wavelength - shifted_center_wavelength)
        / shifted_center_wavelength
        / shifted_center_wavelength
    )
    return np.array([[np.exp(x) / np.sqrt(a_k), 0], [0, np.exp(-x) * np.sqrt(a_k)]], dtype="object")


def _M(
    L: npt.ArrayLike,
    K: npt.ArrayLike,
    alpha: float,
    wavelength: npt.NDArray[np.float_],
    eta: float,
    n_eff: float,
    n_g: float,
    center_wavelength: float,
) -> npt.NDArray[np.float_]:
    L_: npt.NDArray[np.float_] = np.array(L)[::-1]
    K_: npt.NDArray[np.float_] = np.array(K)[::-1]
    a: npt.NDArray[np.float_] = np.exp(-alpha * L_)
    product = np.identity(2)
    for K_k, a_k, L_k in zip(K_[:-1], a, L_):
        product = np.dot(product, _C(K_k, eta))
        product = np.dot(product, _R(a_k, L_k, wavelength, n_eff, n_g, center_wavelength))
    product = np.dot(product, _C(K_[-1], eta))
    return product


def _D(
    L: npt.ArrayLike,
    K: npt.ArrayLike,
    alpha: float,
    wavelength: npt.NDArray[np.float_],
    eta: float,
    n_eff: float,
    n_g: float,
    center_wavelength: float,
) -> npt.NDArray[np.float_]:
    D: npt.NDArray[np.float_] = 1 / _M(L, K, alpha, wavelength, eta, n_eff, n_g, center_wavelength)[0, 0]
    return D
"""

import numpy as np
import numpy.typing as npt


def simulate_transfer_function(
    wavelength: npt.NDArray[np.float_],
    L: npt.ArrayLike,
    K: npt.ArrayLike,
    alpha: float,
    eta: float,
    n_eff: float,
    n_g: float,
    center_wavelength: float,
) -> npt.NDArray[np.float_]:
    y: npt.NDArray[np.float_] = 20 * np.log10(np.abs(_D(L, K, alpha, wavelength, eta, n_eff, n_g, center_wavelength)))
    return y.reshape(y.size)


def _C(K_k: float, eta: float) -> npt.NDArray[np.complex128]:
    """結合部の2x2行列(波長に依存しない、単一の(2,2)行列)"""
    C: npt.NDArray[np.complex128] = (
        1
        / (-1j * eta * np.sqrt(K_k))
        * np.array(
            [[1, -eta * np.sqrt(eta - K_k)], [np.sqrt(eta - K_k) * eta, -(eta**2)]],
            dtype=np.complex128,
        )
    )
    return C


def _R(
    a_k: float,
    L_k: float,
    wavelength: npt.NDArray[np.float_],
    n_eff: float,
    n_g: float,
    center_wavelength: float,
) -> npt.NDArray[np.complex128]:
    """
    リング内伝搬の行列。波長ごとに値が変わるため、
    shape=(len(wavelength), 2, 2) の3次元配列として返す。
    """
    N_k = np.round(L_k * n_eff / center_wavelength)
    shifted_center_wavelength = L_k * n_eff / N_k
    x = (
        1j
        * np.pi
        * L_k
        * n_g
        * (wavelength - shifted_center_wavelength)
        / shifted_center_wavelength
        / shifted_center_wavelength
    )  # shape=(len(wavelength),)

    exp_pos = np.exp(x) / np.sqrt(a_k)   # shape=(len(wavelength),)
    exp_neg = np.exp(-x) * np.sqrt(a_k)  # shape=(len(wavelength),)

    W = wavelength.size
    R = np.zeros((W, 2, 2), dtype=np.complex128)
    R[:, 0, 0] = exp_pos
    R[:, 1, 1] = exp_neg
    return R


def _M(
    L: npt.ArrayLike,
    K: npt.ArrayLike,
    alpha: float,
    wavelength: npt.NDArray[np.float_],
    eta: float,
    n_eff: float,
    n_g: float,
    center_wavelength: float,
) -> npt.NDArray[np.complex128]:
    L_: npt.NDArray[np.float_] = np.array(L)[::-1]
    K_: npt.NDArray[np.float_] = np.array(K)[::-1]
    a: npt.NDArray[np.float_] = np.exp(-alpha * L_)

    W = wavelength.size
    # productをshape=(W, 2, 2)の単位行列で初期化(波長ごとに独立して積算していく)
    product = np.broadcast_to(np.eye(2, dtype=np.complex128), (W, 2, 2)).copy()

    for K_k, a_k, L_k in zip(K_[:-1], a, L_):
        C = _C(K_k, eta)  # shape=(2,2)
        # np.matmulは、(W,2,2) @ (2,2) を自動的にブロードキャストして (W,2,2) にしてくれる
        product = np.matmul(product, C)
        R = _R(a_k, L_k, wavelength, n_eff, n_g, center_wavelength)  # shape=(W,2,2)
        product = np.matmul(product, R)

    C_last = _C(K_[-1], eta)
    product = np.matmul(product, C_last)  # shape=(W,2,2)

    return product


def _D(
    L: npt.ArrayLike,
    K: npt.ArrayLike,
    alpha: float,
    wavelength: npt.NDArray[np.float_],
    eta: float,
    n_eff: float,
    n_g: float,
    center_wavelength: float,
) -> npt.NDArray[np.complex128]:
    M = _M(L, K, alpha, wavelength, eta, n_eff, n_g, center_wavelength)  # shape=(W,2,2)
    D: npt.NDArray[np.complex128] = 1 / M[:, 0, 0]  # 各波長のM[0,0]要素を取り出す
    return D
