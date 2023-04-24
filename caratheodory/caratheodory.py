import numpy as np


class Caratheodory:

    def __init__(self):
        self._dim = None  # dimension of the matrix (Green's function, self-energy, or cumulant) is _dim x _dim
        self._z = None  # to store the complex numbers (iwn - 1.j) / (iwn + 1.j)
        self._S = None
        self._L_inv = None
        self._R_inv = None

    def sqrtm(self, M):  # compute the square root of a matrix M
        e, v = np.linalg.eig(M)
        return v @ np.diag(np.sqrt(e)) @ np.linalg.inv(v)

    def build(self, w, data):
        if w.shape[0] != data.shape[0]:
            raise ValueError("The inputs have different number of Matsubara points.")
        self._dim = data.shape[1]
        self._z = np.zeros(w[w.imag > 0].shape, dtype=np.complex128)  # only data with positive Matsubara frequencies are used in calculations
        self._S = np.zeros(self._z.shape + (self._dim, self._dim), dtype=np.complex128)
        self._L_inv = np.zeros(self._z.shape + (self._dim, self._dim), dtype=np.complex128)
        self._R_inv = np.zeros(self._z.shape + (self._dim, self._dim), dtype=np.complex128)
        eye = np.eye(self._dim, dtype=np.complex128)

        for n, iwn in enumerate(w[w.imag > 0][::-1]):
            self._z[n] = (iwn - 1.j) / (iwn + 1.j)  # Eq. (1); _z is stored in reverse order compared to the paper for easier access later on
            Cn = data[w.shape[0] - n - 1, :, :]  # Eq. (2); only data with positive Matsubara frequencies are used in calculations
            self._S[n] = (eye - 1.j * Cn) @ np.linalg.inv(eye + 1.j * Cn)  # Eq. (3) with additional 1.j in front of data because 1.j * data is assumed to be the Caratheodory function

        for m in range(self._z.shape[0]-1, 0, -1):
            zm = self._z[m]
            Sm = self._S[m]  # refers to the S^{m}_{m} in the paper; also stored in a reversed order for latter convenience
            Lm_inv = self.sqrtm(eye - Sm @ np.conj(Sm).T)
            Lm = np.linalg.inv(Lm_inv)
            Rm = self.sqrtm(eye - np.conj(Sm).T @ Sm)
            Rm_inv = np.linalg.inv(Rm)
            for n in range(m-1, -1, -1):
                # Eq. (11)
                # after finishing the loop, _S[m-1] will be S^{m-1}_{m-1}, _S[n<m-1] wil be S^{m-1}_{n}
                # again, the calculation is performed in a reversed order compared to the paper for latter convenience
                zn = self._z[n]
                Sn = self._S[n]
                y_mn = np.abs(zm) * (zm - zn) / zm / (1.0 - np.conj(zm) * zn)
                self._S[n] = Lm @ (Sn - Sm) @ np.linalg.inv(eye - np.conj(Sm).T @ Sn) @ Rm / y_mn

            self._L_inv[m] = Lm_inv
            self._R_inv[m] = Rm_inv

        self._L_inv[0, :, :] = self.sqrtm(eye - self._S[0] @ np.conj(self._S[0]).T)
        self._R_inv[0, :, :] = np.linalg.inv(self.sqrtm(eye - np.conj(self._S[0]).T @ self._S[0]))

    def evaluate(self, wp):
        if self._dim is None:
            raise "Empty continuation data. Please run build(...) first."
        wp = wp[wp.imag > 0].copy()

        V = np.zeros(self._z.shape + (self._dim, self._dim), dtype=np.complex128)
        S = np.zeros(self._z.shape + (self._dim, self._dim), dtype=np.complex128)
        results = np.zeros(wp.shape + (self._dim, self._dim), dtype=np.complex128)
        eye = np.eye(self._dim, dtype=np.complex128)
        for i, iwnp in enumerate(wp):
            znp = (iwnp - 1.j) / (iwnp + 1.j)
            z0 = self._z[0]
            S0 = self._S[0]
            V[0] = np.abs(z0) * (z0 - znp) / z0 / (1.0 - np.conj(z0) * znp) * eye  # Eq. (9); since we instore _z and _S in a reversed order, we can do calculation in a forward order here
            S[0] = np.linalg.inv(eye + V[0] @ np.conj(S0).T) @ (V[0] + S0)  # Eq. (8)
            for mm, zm in enumerate(self._z[1:]):
                m = mm + 1
                Sm = self._S[m]
                # Eq. (9); S[m-1] instead of S[m+1] because we do the calculation in a forward order
                V[m] = np.abs(zm) * (zm - znp) / zm / (1.0 - np.conj(zm) * znp) * (self._L_inv[m] @ S[m - 1] @ self._R_inv[m])
                S[m] = np.linalg.inv(eye + V[m] @ np.conj(Sm).T) @ (V[m] + Sm)  # Eq. (8)

            val = S[-1, :, :]  # reverse the order of the frequency to match the paper
            results[i, :, :] = -1.j * np.linalg.inv(eye + val) @ (eye - val)  # Eq. (4)

        return results
