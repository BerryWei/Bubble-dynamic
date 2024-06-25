import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys

class BubbleSimulator:
    def __init__(self, p0, pvap, rho, s, nu, gamma, amp, f, R0, h):
        self.p0 = p0 # ambient pressure
        self.pvap = pvap # vapour pressure in Pa (water) 31.7 hPa at 25 C
        self.rho = rho # density in kg per cubic metre
        self.s = s # surface tension in N/m 
        self.nu = nu # viscosity in Pa s (water)
        self.amp = amp * p0 # amplitude of acoustic forcing
        self.f = f # acoustic driving frequency in Hz
        self.R0 = R0 # equilibrium radius
        self.gamma = gamma # polytropic index 
        
        self.c = 1500.  # speed of sound in m/s
        self.h = h

    def p_acoustic(self, t):
        return -self.amp * np.sin(2 * np.pi * self.f * t)

    def rayleigh_plesset_with_radiation_loss(self, t, r):
        R = r[0]
        dR = r[1]

        # Form in https://iopscience.iop.org/article/10.1088/0143-0807/34/3/679
        p_gas = (self.p0 + 2 * self.s / self.R0) * (  (self.R0**3 - self.h **3) / (R**3 - self.h**3)) ** (self.gamma)
     
        p_ext = self.p_acoustic(t)

        if (self.P_gas_prev != -1) :
            #dPgdt = (p_gas - self.P_gas_prev) / self.dt

            # Form in https://iopscience.iop.org/article/10.1088/0143-0807/34/3/679
            dPgdt = -3 * self.gamma * p_gas*(R*R*dR)/(R**3 - self.h ** 3)
            radiation_loss = dPgdt * R / self.c
            self.P_gas_prev = p_gas
        else:
            radiation_loss = 0
            self.P_gas_prev = p_gas
            dPgdt = -3 * self.gamma*p_gas*(R*R*dR)/(R**3 - self.h ** 3)
            radiation_loss = dPgdt * R / self.c

        ddR = -3 * dR ** 2 / (2 * R) + 1 / (self.rho * R) * (p_gas - self.p0 - p_ext - 4 * self.nu * dR / R - 2 * self.s / R + radiation_loss)

        return [dR, ddR]
    
    @staticmethod
    def compute_chi(R):
        f'''
        Thermal conductivity (k_g): Approximately 0.025 W/(m·K)
        Gas density (ρ_g): Approximately 0.6 kg/m³ (assuming ideal gas behavior and pressure close to atmospheric)
        Specific heat at constant pressure (c_p,g): Approximately 1.996 kJ/(kg·K)
        '''
        k_g = 0.025
        rho_g = 0.6 # assume that rho_g is independant to R
        cp_g = 1.996
        return k_g / (rho_g * cp_g)

    def rayleigh_plesset_with_radiation_loss_2015(self, t, r):
        f''' https://www.sciencedirect.com/science/article/pii/S1350417714003472
        Influence of ultrasound power on acoustic streaming and micro-bubbles formations in a low frequency sono-reactor: Mathematical and 3D computational simulation'''
        R = r[0]
        dR = r[1]

        # van der Waals hard core radius
        h = self.R0 / 8.86

        # kappa need to redefined
        f''' https://www.nature.com/articles/18842
        '''
        A = 0.00367810602433701
        B = 1

        # chi: thermal diffusivity of gas

        chi = self.compute_chi(R)
        

        Pe = R * abs(dR) / chi
        if dR == 0:
            dR = 1.0e-7


        self.kappa = 1 + (self.gamma - 1) * np.exp( -1 * A / (Pe**B) )

        p_gas = (self.p0 + 2 * self.s / self.R0 - self.pvap) * ((self.R0**3 - h**3) / (R**3 - h**3)) ** (3 * self.kappa)
        self.P_gas_list.append(p_gas)
        p_surf = 2 * self.s / R
        p_liq = p_gas + self.pvap - p_surf
        p_ext = self.p0 + self.p_acoustic(t)

        if len(self.P_gas_list) > 2:
            dPgdt = (self.P_gas_list[-1] - self.P_gas_list[-2]) / self.dt
            term_dPgdt = R * dPgdt / self.c
        else:
            term_dPgdt = 0

        

        ddR = -3 * dR ** 2 / (2 * R) + 1 / (self.rho * R) * (p_liq - 4 * self.nu * dR / R - p_ext  + term_dPgdt)

        return [dR, ddR]
    
    def handle_event(self, t, y):
        progress = t / self.t_span[1] * 100  # Calculate progress percentage
        sys.stdout.write(f'\r{progress:.5f}%')  # Display progress
        sys.stdout.flush()
        return 0


    def simulate(self, t_span, t_eval):
        self.t_span = t_span
        y0 = [self.R0, 0]
        self.P_gas_prev= -1
        self.dt = t_eval[1]-  t_eval[0]
        solution = solve_ivp(self.rayleigh_plesset_with_radiation_loss, t_span, y0, t_eval=t_eval, rtol=1e-6, atol=1e-9, method='Radau', events=self.handle_event, dense_output=True)
        
        return solution.t, solution.y[0], solution.y[1]

    def plot(self, t, R, dRdt):
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        ax[0].plot(t * 1e6, R * 1e6) # Convert time to microseconds and radius to micrometers
        ax[0].set_xlabel('Time ($\mu$s)')
        ax[0].set_ylabel('Bubble radius ($\mu$m)')
        ax[0].set_title('Bubble radius vs. time')

        ax[1].plot(t * 1e6, dRdt)
        ax[1].set_xlabel('Time ($\mu$s)')
        ax[1].set_ylabel('Bubble growth rate [m/s]')
        ax[1].set_title('Bubble growth rate vs. time')

        plt.tight_layout()
        plt.show()



