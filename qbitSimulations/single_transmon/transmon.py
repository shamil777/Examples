import numpy as np
from numpy import pi, sqrt, abs

import qutip as qp

class Transmon:
    def __init__(self, Ec, Ej, d, gamma_rel, gamma_phi,
                 Nc, N_trunc, index, nonlinear_osc=False):
        """
        Parameters
        ----------
        Ec : float
            Transmon's charge energy Ec = e^2/2C
        Ej : float
            Transmon's larger JJ tunelling energy
        d : float
            assymetry of transmon's JJs. Equals to
            (EJ1 - EJ2) / (EJ1 + EJ2), where
            EJ1 > EJ2 - JJs tunneling energies
        gamma_rel : float
            effective rate for radial relaxation of the state due to
            coupling to the external bath (used for Lindblad equations)
        gamma_phi : float
            effective dephasing rate of the state due to
            coupling to the external bath (used for Lindblad equations)
        Nc : int
            maximum Cooper pairs number in charge basis used as a dimension
            of an approximating finitie-dimension Hilbert space.
             `deg(H) = 2*Nc + 1` - including |-Nc>, |0> and |Nc> cooper pair states.
        N_trunc : int
            ??
        index : int
            ??
        nonlinear_osc : bool
            Regulates how many terms to keep after substiting system operators
            by Bose's creation and destruction operators and exanding the equation.
                if `True` - the model corresponds to Duffing oscillator
            (first nonlinear term that keeps the number of excitations)
                if `False` - aprroximation is pure harmonic oscillator
            (all but first dropped)
        """
        self._Ec = Ec
        self._Ej = Ej
        self._d = d
        self._Nc = Nc
        self._Ns = Nc * 2 + 1
        self._gamma_rel = gamma_rel
        self._gamma_phi = gamma_phi
        self.index = index
        self._nonlinear_osc = nonlinear_osc

        self._linear_osc_H_stub = (sqrt(8 * Ej * Ec) - Ec) * qp.create(N_trunc) * qp.destroy(N_trunc)
        self._nonlinear_osc_H_stub = - Ec / 2 * qp.create(N_trunc) * qp.destroy(N_trunc) * \
                                     (qp.create(N_trunc) * qp.destroy(N_trunc) - 1)
        self._nonlinear_osc_raising = qp.create(N_trunc)
        self._nonlinear_osc_lowering = qp.destroy(N_trunc)
        self._nonlinear_osc_n = qp.destroy(N_trunc) + qp.create(N_trunc)

        self._N_trunc = N_trunc

        self._n_cache = {}
        self._c_ops_cache = {}
        self._Hdr_RF_RWA_cache = {}
        self._Hdr_cont_cache = {}
        self._H_diag_trunc_cache = {}

    def clear_caches(self):
        self._n_cache = {}
        self._c_ops_cache = {}
        self._Hdr_RF_RWA_cache = {}
        self._Hdr_cont_cache = {}
        self._H_diag_trunc_cache = {}

    def _truncate(self, operator):
        return qp.Qobj(operator[:self._N_trunc, :self._N_trunc])

    def Hc(self):
        return 4 * (self._Ec) * qp.charge(self._Nc) ** 2

    def Hj(self, phi, account_cosine_phase=False):
        if account_cosine_phase:
            return (- self._Ej * self._phi_coef_Hj(phi) * 1/2) * ()  # TODO: account for phase
        else:
            return (- self._Ej * self._phi_coef_Hj(phi) * 1/2) * qp.tunneling(self._Ns, 1)

    def Hj_td(self, phi_waveform):
        return [- self._Ej / 2 * qp.tunneling(self._Ns, 1), self._phi_coeff(phi_waveform)]

    def H_diag_trunc(self, phi):
        if self._nonlinear_osc:
            return self.H_nonlinear_osc(phi)

        try:
            return self._H_diag_trunc_cache[phi]
        except KeyError:
            H_charge_basis = self.Hc() + self.Hj(phi)
            evals, evecs = H_charge_basis.eigenstates()
            H = self._truncate(H_charge_basis.transform(evecs))
            self._H_diag_trunc_cache[phi] = H - H[0, 0]
            return self._H_diag_trunc_cache[phi]

    def H_nonlinear_osc(self, phi):
        return self._linear_osc_H_stub * self._phi_coeff(phi) + self._nonlinear_osc_H_stub

    def H_diag_trunc_approx(self, phi):
        H_charge_basis = self.Hc() + self.Hj(0)
        evals, evecs = H_charge_basis.eigenstates()
        H_dta = self._truncate(H_charge_basis.transform(evecs)) * sqrt(self._phi_coeff(phi))
        return H_dta - H_dta[0, 0]

    def H_td_diag_trunc_approx(self, waveform):
        # approximating f_q = f_q^max * sqrt(cos sqrt(1+ d^2tan^2))
        return [self.H_diag_trunc(0), sqrt(self._phi_coeff(waveform))]

    def g_state(self):
        #         evals, evecs = self.H(phi).eigenstates()
        #         return evecs[0]
        return qp.basis(self._N_trunc, 0)

    def e_state(self):
        return qp.basis(self._N_trunc, 1)

    def f_state(self):
        return qp.basis(self._N_trunc, 2)

    def eigenlevels_approx(self, phi):
        evals = self.H_diag_trunc_approx(phi).eigenenergies()
        return evals

    def ge_freq_approx(self, phi):
        """
        Returns calculated 0-1 state transition
        frequency.

        Parameters
        ----------
        phi : float
            external phase in units of `pi`

        Returns
        -------
        0-1 state transition frequency in same
        units as E_C and E_J are specified divided by `2 pi`.
        """
        evals = self.H_diag_trunc_approx(phi).eigenenergies()
        return (evals[1] - evals[0]) / 2 / pi

    def n(self, phi):
        if self._nonlinear_osc:
            return self._nonlinear_osc_n

        try:
            return self._n_cache[phi]
        except:
            H_charge_basis = self.Hc() + self.Hj(phi)
            evals, evecs = H_charge_basis.eigenstates()
            self._n_cache[phi] = self._truncate(qp.Qobj(abs(qp.charge(self._Nc).transform(evecs))))
            return self._n_cache[phi]

    def raising(self, phi):
        if self._nonlinear_osc:
            return self._nonlinear_osc_raising

        evecs = [qp.basis(self._N_trunc, i) for i in range(self._N_trunc)]
        return sum([abs(self.n(phi).matrix_element(evecs[j + 1], evecs[j])) /
                    abs(self.n(phi).matrix_element(evecs[0], evecs[1])) *
                    evecs[j + 1] * evecs[j].dag() for j in range(0, self._N_trunc - 1)])

    def lowering(self, phi):
        if self._nonlinear_osc:
            return self._nonlinear_osc_lowering

        evecs = [qp.basis(self._N_trunc, i) for i in range(self._N_trunc)]
        return sum([abs(self.n(phi).matrix_element(evecs[j], evecs[j + 1])) /
                    abs(self.n(phi).matrix_element(evecs[0], evecs[1])) *
                    evecs[j] * evecs[j + 1].dag() for j in range(0, self._N_trunc - 1)])

    def rotating_dephasing(self, phi):
        return self.raising(phi) * self.lowering(phi)

    def c_ops(self, phi):
        try:
            return self._c_ops_cache[phi]
        except KeyError:
            self._c_ops_cache[phi] = [sqrt(self._gamma_rel) * self.lowering(phi),
                                      sqrt(self._gamma_phi) * self.rotating_dephasing(phi)]
            return self._c_ops_cache[phi]

    def _phi_coeff(self, phi):
        """
        Returns coefficient for analytical approximation of
        eigenenergies.

        Parameters
        ----------
        phi : float
            external phase in units of `pi`

        Returns
        -------
        float
            coefficient for josephson junctions potential energy
        """
        return (np.cos(phi * pi)**2 + (self._d * np.sin(phi * pi)) ** 2) ** 0.25

    def _phi_coef_Hj(self, phi):
        return 2/(self._d + 1) * (np.cos(phi * pi)**2 + (self._d * np.sin(phi * pi)) ** 2)**0.5

    def get_Ns(self):
        return self._N_trunc

    def get_index(self):
        return self.index

    def Hdr(self, amplitude, duration, start, phase=0, freq=None):

        if freq is None:
            freq = self.ge_freq_approx(1 / 2)

        if self._nonlinear_osc:
            return [self._nonlinear_osc_n,
                    "%f*cos(2*pi*%.16f*t+%f)*(1+np.sign(t-%f))*(1+np.sign(-t+%f))/4" % \
                    (amplitude, freq, phase, start, start + duration)]

        return [self.n(0) / self.n(0).matrix_element(self.g_state(), self.e_state()),
                "%f*cos(2*pi*%.16f*t+%f)*(1+np.sign(t-%f))*(1+np.sign(-t+%f))/4" % \
                (amplitude, freq, phase, start, start + duration)]

    # driving!! utilized in double-tone spectroscopy
    def Hdr_cont(self, amplitude):

        if self._nonlinear_osc:
            op = self._nonlinear_osc_n
        else:
            op = self.n(0) / self.n(0).matrix_element(self.g_state(),
                                                      self.e_state())

        try:
            return self._Hdr_cont_cache[amplitude]
        except KeyError:
            self._Hdr_cont_cache[amplitude] = [
                amplitude * op,
                "cos(wd%d*t)" % self.index]
            return self._Hdr_cont_cache[amplitude]

    def Hdr_cont_RF_RWA(self, amplitude):
        if self._nonlinear_osc:
            op = self._nonlinear_osc_n
        else:
            op = self.n(0) / self.n(0).matrix_element(self.g_state(),
                                                      self.e_state())

        try:
            return self._Hdr_RF_RWA_cache[amplitude]
        except KeyError:
            self._Hdr_RF_RWA_cache[amplitude] = amplitude / 2 * op
            return self._Hdr_RF_RWA_cache[amplitude]

    def sz(self):
        return qp.ket2dm(qp.basis(3, 0)) - qp.ket2dm(qp.basis(3, 1))

    def sx(self):
        return qp.basis(3, 0) * qp.basis(3, 1).dag() + qp.basis(3, 1) * qp.basis(3, 0).dag()

    def sy(self):
        return -1j * qp.basis(3, 0) * qp.basis(3, 1).dag() + 1j * qp.basis(3, 1) * qp.basis(3, 0).dag()
