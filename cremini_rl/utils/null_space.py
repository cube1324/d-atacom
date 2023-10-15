import numpy as np
import torch
from scipy.linalg import qr, svd


def batch_qr_null(A, tol=None):
    Q, R = np.linalg.qr(np.transpose(A, axes=[0, 2, 1]), mode='complete')
    # tol = np.max(A, axis=(1, 2)) * np.finfo(R.dtype).eps if tol is None else tol
    #
    # R_diags = np.flip(np.abs(np.diagonal(R, axis1=1, axis2=2)), axis=1)
    # TODO assumes values are numerical stable, find way to use searchsorted in batch
    # rnk = min(A.shape) - np.abs(np.diag(R))[::-1].searchsorted(tol)
    # return Q[:, rnk:].conj()

    rnk = min(A.shape[1:])
    temp = Q[:, :, rnk:].conj()
    return temp


def batch_smooth_basis(A, T0=None):
    """
    Compute the null space matrix suggested by:
    On the computation of multidimensional solution manifolds of parametrized equations
    """
    Ux = batch_qr_null(A)
    if T0 is None:
        T0 = np.zeros(Ux.shape[1:], dtype=A.dtype)
        np.fill_diagonal(T0, 1.0)
    else:
        assert T0.shape == (Ux.shape[0], Ux.shape[0] - Ux.shape[1])

    U0 = np.transpose(Ux, axes=(0, 2, 1)) @ T0
    U, s, Vh = np.linalg.svd(U0)
    Q = U @ Vh
    return Ux @ Q


def batch_qr_null_tensor(A, tol=None):
    Q, R = torch.linalg.qr(torch.transpose(A, 2, 1), mode='complete')
    # tol = np.max(A, axis=(1, 2)) * np.finfo(R.dtype).eps if tol is None else tol
    #
    # R_diags = np.flip(np.abs(np.diagonal(R, axis1=1, axis2=2)), axis=1)
    # TODO assumes values are numerical stable, find way to use searchsorted in batch
    # rnk = min(A.shape) - np.abs(np.diag(R))[::-1].searchsorted(tol)
    # return Q[:, rnk:].conj()

    rnk = min(A.shape[1:])
    temp = Q[:, :, rnk:].conj()
    return temp


def batch_smooth_basis_tensor(A, T0=None):
    """
    Compute the null space matrix suggested by:
    On the computation of multidimensional solution manifolds of parametrized equations
    """
    Ux = batch_qr_null_tensor(A)
    if T0 is None:
        T0 = torch.zeros(Ux.shape[1:], dtype=A.dtype, device=A.device)
        T0.fill_diagonal_(1.)
    else:
        assert T0.shape == (Ux.shape[0], Ux.shape[0] - Ux.shape[1])

    U0 = torch.transpose(Ux, 2, 1) @ T0
    U, s, Vh = torch.linalg.svd(U0)
    Q = U @ Vh
    return Ux @ Q


def qr_null(A, tol=None):
    Q, R = qr(A.T, mode='full')
    tol = np.max(A) * np.finfo(R.dtype).eps if tol is None else tol
    rnk = min(A.shape) - np.abs(np.diag(R))[::-1].searchsorted(tol)
    return Q[:, rnk:].conj()


def smooth_basis(A, T0=None):
    """
    Compute the null space matrix suggested by:
    On the computation of multidimensional solution manifolds of parametrized equations
    """
    Ux = qr_null(A)  # [[ 0.40824829], [-0.81649658], [ 0.40824829]]
    if T0 is None:
        T0 = np.zeros(Ux.shape)
        np.fill_diagonal(T0, 1.0)
    else:
        assert T0.shape == (Ux.shape[0], Ux.shape[0] - Ux.shape[1])

    U0 = Ux.T @ T0  # [[0.40824829]]
    U, s, Vh = svd(U0)  # [[1.]] [0.40824829] [[1.]]
    Q = U @ Vh  # [[1.]]
    return Ux @ Q


if __name__ == "__main__":
    test_arr = np.arange(3).reshape(1, 3)
    print(test_arr)
    res1 = smooth_basis(test_arr)

    test_arr = np.array([test_arr, test_arr, 2 * test_arr, 3 * test_arr])

    print(test_arr.shape)
    print(test_arr)

    res2 = batch_smooth_basis(test_arr)

    print(res1, res2)
