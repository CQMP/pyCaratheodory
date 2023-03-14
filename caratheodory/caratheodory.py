import numpy as np
import scipy

class Caratheodory:

    def __init__(self):
        self._dim = None
        self._mesh = None
        self._Ws = None
        self._sqrt_one = None
        self._sqrt_two = None

    def build(self, mesh, data):
        self._dim = data.shape[1]
        self._mesh = np.zeros(mesh[mesh.imag>0].shape, dtype=np.complex128)
        self._Ws = np.zeros(self._mesh.shape + (self._dim, self._dim), dtype=np.complex128)
        self._sqrt_one = np.zeros(self._mesh.shape + (self._dim, self._dim), dtype=np.complex128)
        self._sqrt_two = np.zeros(self._mesh.shape + (self._dim, self._dim), dtype=np.complex128)
        eye = np.eye(self._dim, dtype=np.complex128)

        for iw, w in enumerate(mesh[mesh.imag>0][::-1]):
            self._mesh[iw] = (w - 1.j) / (w + 1.j)
            val = data[mesh.data.shape[0] - iw - 1, :, :]
            val = (eye - 1.j * val) @ np.linalg.inv(eye + 1.j * val)
            self._Ws[iw] = val

        for iw in range(self._mesh.shape[0]-1, 0, -1):
            zi = self._mesh[iw]
            Wi = self._Ws[iw]
            sqrt_one_i = scipy.linalg.sqrtm(eye - Wi @ np.conj(Wi).T)
            sqrt_one_i_inv = np.linalg.inv(sqrt_one_i)
            sqrt_two_i = scipy.linalg.sqrtm(eye - np.conj(Wi).T @ Wi)
            for jw in range(iw-1, -1, -1):
                zj = self._mesh[jw]
                Wj = self._Ws[jw]
                y_ij = np.abs(zi) * (zi - zj) / zi / (1.0 - np.conj(zi) * zj)
                self._Ws[jw] = sqrt_one_i_inv @ (Wj - Wi) @ np.linalg.inv(eye - np.conj(Wi).T @ Wj) @ sqrt_two_i / y_ij

            self._sqrt_one[iw] = sqrt_one_i
            self._sqrt_two[iw] = np.linalg.inv(sqrt_two_i)

        self._sqrt_one[0, :, :]  = scipy.linalg.sqrtm(eye - self._Ws[0] @ np.conj(self._Ws[0]).T);
        self._sqrt_two[0, :, :]  = np.linalg.inv(scipy.linalg.sqrtm(eye - np.conj(self._Ws[0]).T @ self._Ws[0]));

    def evaluate(self, grid):
        if self._dim is None :
            raise "Empty continuation data. Please run solve(...) first."
        work_grid = grid[grid.imag>0].copy()

        Vs = np.zeros(self._mesh.shape + (self._dim, self._dim), dtype=np.complex128)
        Fs = np.zeros(self._mesh.shape + (self._dim, self._dim), dtype=np.complex128)
        results = np.zeros(work_grid.shape + (self._dim, self._dim), dtype=np.complex128)
        eye = np.eye(self._dim, dtype=np.complex128)
        for i, w in enumerate(work_grid):
            z = (w - 1.j) / (w + 1.j)
            z0 = self._mesh[0]
            W0 = self._Ws[0]
            Vs[0] = np.abs(z0) * (z0 - z) / z0 / (1.0 - np.conj(z0) * z) * eye
            Fs[0] = np.linalg.inv(eye + Vs[0] @ np.conj(W0).T) @ (Vs[0] + W0)
            for jj, zj in enumerate(self._mesh[1:]):
                j = jj + 1
                Wj = self._Ws[j]
                # See Eq.9 PhysRevB.104.165111
                Vs[j] = np.abs(zj) * (zj - z) / zj / (1.0 - np.conj(zj) * z) * (self._sqrt_one[j] @ Fs[j - 1] @ self._sqrt_two[j])
                # See Eq. 10 PhysRevB.104.165111
                Fs[j] = np.linalg.inv(eye + Vs[j] @ np.conj(Wj).T) @ (Vs[j] + Wj)

            val = Fs[-1, :, :]
            # See Eq.11 PhysRevB.104.165111
            results[i, :, :] = -1.j * np.linalg.inv(eye + val) @ (eye - val)

        return results
