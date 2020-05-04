import os
import datetime
import pandas as pd
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

CSV_CASE_PATH = os.path.join('Data', 'Data_2020-04-10Ny.txt')
DATE_COLUMN = 'Datum'
INCIDENSE_COLUMN = 'Incidens'
MEAN_LENGTH_SICK = 5
MEAN_LENGTH_INCUBATION=5.1

N=2374550#report page 9#int(648557/0.27+616655/0.26)/2 #Mean from Tabell 1 page 13 #approx 2.386M
i0=1
START_DATE = datetime.datetime(2020, 2, 17)
END_DATE_OPTIMIZE = datetime.datetime(2020, 4, 10)
END_DATE_SIMULATION = datetime.datetime(2020, 8, 10)
MIDDLE_THETA = datetime.datetime(2020, 3, 16)

p0 = 0.987 #Calibrated somehow to fit 2.5% in end march 
q0=0.11
delta=0.16
epsilon=-0.19
theta=10.9
rho = 1/MEAN_LENGTH_INCUBATION
gamma = 1/MEAN_LENGTH_SICK
t_b=(MIDDLE_THETA-START_DATE).days

def basic_reproduction(p0, b_t, gamma, q0):
    return (1 - p0) * b_t / gamma + p0 * q0 * b_t / gamma

def b_t_func(t, t_b, delta, epsilon, theta):
    exp_var = -epsilon * (t - t_b)
    return theta * (delta +(1 - delta)/(1 + np.exp(exp_var)))



def SEIR_derivative(y, t, t_b, delta, epsilon, theta, gamma, p0, q0):
    #Similar to R code
    global t_ode, R0, R_e

    R, S, E, I_o, I_r = y
    b_t = b_t_func(t, t_b, delta, epsilon, theta)
    nir = S *b_t/N #new_infected_ratio
    rhoE = rho * E

    dRdt = gamma * (I_o + I_r) # Recovered
    dSdt = -nir * I_r - nir * q0 * I_o # Susceptible
    dEdt = nir * I_r + nir * q0 * I_o - rhoE # Exposed
    dI_odt =  rhoE * p0 - gamma * I_o # Infected unobserved
    dI_rdt =  rhoE * (1- p0) - gamma * I_r # Infected reported
    # Calculate R0, R_e by appending for now. 
    R0.append(basic_reproduction(p0, b_t, gamma, q0))
    R_e.append(R0[-1] * S / N)
    t_ode.append(t)

    return dRdt, dSdt, dEdt, dI_odt, dI_rdt

df = pd.read_csv(CSV_CASE_PATH, sep=' ', parse_dates=[DATE_COLUMN]).set_index(DATE_COLUMN)
y0 = 0, N-i0, 0, 0, i0
daterange = [START_DATE + datetime.timedelta(days=x) for x in range(0, int((END_DATE_SIMULATION-START_DATE).days))]
t=np.arange(len(daterange))

global t_ode, R0, R_e
t_ode, R0, R_e = ([] for i in range(3))

return_vals = odeint(SEIR_derivative, y0, t, args=(t_b, delta, epsilon, theta, gamma, p0, q0))
R, S, E, I_o, I_r = return_vals.T
daterange_ode = [START_DATE + datetime.timedelta(days=x) for x in t_ode]

plt.plot(daterange_ode, R0)
plt.show()