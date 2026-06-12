# -*- coding: utf-8 -*-
"""模块 1: GMAW 电弧-熔化自调节动力学 (集中参数 ODE)"""
import numpy as np
from scipy.integrate import solve_ivp


class GMAWDynamics:
    """状态 x=[s, I]: 干伸长 s [m], 电流 I [A]"""

    def __init__(self):
        self.Voc, self.Rs, self.Rl, self.Ls = 32.0, 0.004, 0.004, 3.0e-4
        self.V0, self.Ea, self.Ra = 15.5, 800.0, 0.022
        self.rw = 0.6e-3
        self.k1, self.k2, self.rho_r = 3.0e-4, 5.0e-5, 0.25

    def melting_rate(self, s, I):
        return self.k1 * I + self.k2 * s * I ** 2

    def arc_voltage(self, la, I):
        return self.V0 + self.Ea * la + self.Ra * I

    def rhs(self, t, x, WFS_fun, CTWD_fun):
        s, I = x
        la = max(CTWD_fun(t) - s, 1e-4)
        ds = WFS_fun(t) - self.melting_rate(s, I)
        dI = (self.Voc - (self.Rs + self.Rl + self.rho_r * s) * I
              - self.arc_voltage(la, I)) / self.Ls
        return [ds, dI]

    def simulate(self, t_end=1.0, x0=(6e-3, 150.0),
                 WFS_fun=None, CTWD_fun=None):
        WFS_fun = WFS_fun or (lambda t: 0.12)
        CTWD_fun = CTWD_fun or (lambda t: 0.018 + (0.003 if t >= 0.5 else 0))
        sol = solve_ivp(self.rhs, (0, t_end), x0, args=(WFS_fun, CTWD_fun),
                        method="LSODA", max_step=1e-3, dense_output=True)
        t = np.linspace(0, t_end, 2000)
        s, I = sol.sol(t)
        CTWD = np.array([CTWD_fun(ti) for ti in t])
        la = CTWD - s
        Va = self.arc_voltage(la, I)
        return dict(t=t, s=s, I=I, la=la, Va=Va, P=Va * I)
