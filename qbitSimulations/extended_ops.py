import numpy as np
import scipy as sp
from scipy.sparse import diags
from qutip.fastsparse import csr2fast
from qutip import Qobj
from enum import Enum


def charge_shift(Nc, shift_m=1):
    '''
    Operator increases charge number by `shift_m` in charge basis with maximum
    charge number equal to `Nc`.
    charge_p * |n> = |n+m>

    Parameters
    ----------
    Nc : int
        maximum charge number of Hilbert space (deg H = 2*Nc + 1)
    m : int
        requested shift in charge number basis (both positive and negative are accepted)

    Returns
    -------
    oper : QObj
        QObj for `charge shift` operator
    '''
    if not isinstance(Nc, (int, np.integer)):  # raise error if N not integer
        raise ValueError("Hilbert space dimension must be integer value")
    if not isinstance(shift_m, (int, np.integer)):
        raise ValueError("shift charge must be integer value")

    mat = diags(
        diagonals=[1.0], offsets=-shift_m,
        shape=(2*Nc+1, 2*Nc+1), format="csr", dtype=np.complex
    )

    # Qobj constructor will convert `scipy.sparse_matrix` to `qutip.fast_csr` instance internally
    return Qobj(mat, copy=False)


def charge_p(Nc):
    return charge_shift(Nc, 1)

def charge_m(Nc):
    return charge_shift(Nc, -1)


