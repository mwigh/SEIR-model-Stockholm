import os
import datetime
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
import scipy.stats
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy.special import logit, expit
import numdifftools as nd


CSV_CASE_PATH = os.path.join('Data', 'Data_2020-04-10Ny.txt')
CSV_IN_IVA_PATH = os.path.join('Data', 'Data_2020-05-04_iva.txt')
DATE_COLUMN = 'Datum'
INCIDENSE_COLUMN = 'Incidens'
IVA_COLUMN = 'IN_IVA'
MEAN_LENGTH_SICK = 5
MEAN_LENGTH_INCUBATION = 5.1

MEAN_LENGTH_IVA = 10  # Guess from their analysis #https://www.folkhalsomyndigheten.se/contentassets/4b4dd8c7e15d48d2be744248794d1438/skattning-av-behov-av-slutenvardsplatser-covid-lombardiet.pdf
IVA_DELAY = 11  # From iva register
RATE_IN_IVA = 0.121  # Test to fit curve

# report page 9#int(648557/0.27+616655/0.26)/2 #Mean from Tabell 1 page 13
# #approx 2.386M
N = 2374550
i0 = 1
START_DATE = datetime.datetime(2020, 2, 17)
START_DATE_IVA = datetime.datetime(2020, 3, 30)
END_DATE_OPTIMIZE = datetime.datetime(2020, 4, 10)
END_DATE_SIMULATION = datetime.datetime(2020, 8, 10)
MIDDLE_THETA = datetime.datetime(2020, 3, 16)
NB_OPTIMIZATIONS = 4  # Number of optimization runs
NB_BOOTSRAPING = 1000

p0 = 0.987  # Calibrated somehow to fit 2.5% in end march
q0 = 0.11  # Rate of how infective a non reported case is vs reported
delta = 0.16  # Init for test
epsilon = -0.19  # Init for test
theta = 10.9  # Init for test
rho = 1 / MEAN_LENGTH_INCUBATION
gamma = 1 / MEAN_LENGTH_SICK
t_b = (MIDDLE_THETA - START_DATE).days


def basic_reproduction(p0, b_t, gamma, q0):
    return (1 - p0) * b_t / gamma + p0 * q0 * b_t / gamma


def b_t_func(t, t_b, delta, epsilon, theta, gamma, p0, q0):
    exp_var = -epsilon * (t - t_b)
    if isinstance(exp_var, float) and exp_var > 200:
        exp_var = np.inf
    elif not isinstance(exp_var, float):
        exp_var[exp_var > 200] = np.inf
    return theta * (delta + (1 - delta) / (1 + np.exp(exp_var)))


def SEIR_derivative(y, t, t_b, delta, epsilon, theta, gamma, p0, q0):
    # Similar to R code
    global t_ode, R0, R_e

    R, S, E, I_o, I_r = y
    b_t = b_t_func(t, t_b, delta, epsilon, theta, gamma, p0, q0)
    nir = S * b_t / N  # new_infected_ratio
    rhoE = rho * E

    dRdt = gamma * (I_o + I_r)  # Recovered
    dSdt = -nir * I_r - nir * q0 * I_o  # Susceptible
    dEdt = nir * I_r + nir * q0 * I_o - rhoE  # Exposed
    dI_odt = rhoE * p0 - gamma * I_o  # Infected unobserved
    dI_rdt = rhoE * (1 - p0) - gamma * I_r  # Infected reported
    # Calculate R0, R_e by appending for now.
    R0.append(basic_reproduction(p0, b_t, gamma, q0))
    R_e.append(R0[-1] * S / N)
    t_ode.append(t)

    return dRdt, dSdt, dEdt, dI_odt, dI_rdt


def opt_guesses():
    # Similar as in R
    u_d = logit(np.random.uniform(0.05, 0.6))  # guess for delta
    u_e = np.random.uniform(-0.6, 0)    # guess for epsilon
    u_t = np.log(np.random.uniform(0, 15))    # guess for theta
    return u_d, u_e, u_t


def RSS(params, t_optimize, t_b, y0, p0, rho, incidence, gamma, q0):
    # Code for minimizing, similar as in R code

    delta = expit(params[0])
    epsilon = params[1]
    theta = np.exp(params[2])

    dummy_infectivity = b_t_func(
        np.arange(200), t_b, delta, epsilon, theta, gamma, p0, q0)
    if min(dummy_infectivity) < 0:
        return 10**12
    return_vals = odeint(SEIR_derivative, y0, t=t_optimize, args=(
        t_b, delta, epsilon, theta, gamma, p0, q0))
    R, S, E, I_o, I_r = return_vals.T
    fitted_incidence = (1 - p0) * E * rho
    # print(((incidence - fitted_incidence)**2).sum())

    return ((incidence - fitted_incidence)**2).sum()


def run_optimization(x0, args):
    t_optimize, t_b, y0, p0, rho, incidence, gamma, q0, tmpdict = args
    return minimize(RSS, x0, method=tmpdict['method'], options=tmpdict['options'],
                    args=(t_optimize, t_b, y0, p0, rho, incidence, gamma, q0))


def numerical_Hessian(params, args):
    # Calculate Hessian since they are using that for covariance matrix but it
    # is not returned from Nelder-Mead in python
    t_optimize, t_b, y0, p0, rho, incidence,  gamma, q0 = args
    return RSS(params, t_optimize, t_b, y0, p0, rho, incidence,  gamma, q0)


def run_odeint(paras, args):
    delta, epsilon, theta = paras
    t_b, gamma, p0, q0 = args
    return odeint(SEIR_derivative, y0, t, args=(t_b, delta, epsilon, theta, gamma, p0, q0))


def CRI(x, level=0.95, up=False):
    # Similar to in report. Credibility interval
    n = len(x)
    L = (1 - level) / 2
    U = 1 - (1 - level) / 2
    x = np.sort(x)
    if up:
        return x[np.int(n * U)]
    return x[np.int(n * L)]


def calculate_in_iva(I_r, rate_in_iva=RATE_IN_IVA, iva_delay=IVA_DELAY, mean_length_iva=MEAN_LENGTH_IVA):
    IVA_arr = np.zeros(IVA_DELAY + MEAN_LENGTH_IVA)
    IVA_arr[IVA_DELAY:] = 1
    return np.convolve(np.array(I_r), IVA_arr)[0:len(I_r)] * rate_in_iva


# Basic tests
print('Running. Should maybe take 10-20 seconds')
df = pd.read_csv(CSV_CASE_PATH, sep=' ', parse_dates=[
                 DATE_COLUMN]).set_index(DATE_COLUMN)
df_in_iva = pd.read_csv(CSV_IN_IVA_PATH, sep=' ', parse_dates=[
    DATE_COLUMN]).set_index(DATE_COLUMN)

y0 = 0, N - i0, 0, 0, i0
daterange = [START_DATE + datetime.timedelta(days=x) for x in range(
    0, int((END_DATE_SIMULATION - START_DATE).days))]
t = np.arange(len(daterange))

global t_ode, R0, R_e
t_ode, R0, R_e = ([] for i in range(3))

return_vals = odeint(SEIR_derivative, y0, t, args=(
    t_b, delta, epsilon, theta, gamma, p0, q0))
R, S, E, I_o, I_r = return_vals.T


# Optimize in parallel
t_optimize = np.arange((END_DATE_OPTIMIZE - START_DATE).days + 1)

incidence = df.to_numpy().flatten()
daterange_opt = [START_DATE + datetime.timedelta(days=x) for x in range(
    0, int((END_DATE_OPTIMIZE - START_DATE).days + 1))]

pool = mp.Pool(mp.cpu_count())
args = [t_optimize, t_b, y0, p0, rho, incidence[
    0:t_optimize.shape[0]],  gamma, q0]
opt_args = [{'method': 'Nelder-Mead', 'options': {'maxiter': 1000}}]
results = pool.starmap(run_optimization, [(
    opt_guesses(), args + opt_args) for i in range(NB_OPTIMIZATIONS)])
for i, res in enumerate(results):
    delta, epsilon, theta = res.x
    #print(delta, epsilon, theta)
    # print(res.fun)
    if i == 0 or res.fun < best_res.fun:
        best_res = res

delta, epsilon, theta = best_res.x

# Calculate Hessian since they are using that for covariance matrix but it
# is not returned from Nelder-Mead in python
# Use covariance instead of std
sigest = np.sqrt(best_res.fun / (t_optimize.shape[0] - 3))  # m-n
H = nd.Hessian(numerical_Hessian)([delta, epsilon, theta], args)

# This covariance matrix is transformed
# Calculated different NeginvH2 compared to report. Check this more some time.
# I think this is correct but they might be using some other opt algorithm
# that returns something else
NeginvH2 = np.linalg.inv(H) * sigest**2
sdParams = np.sqrt(np.diag(NeginvH2))

# Confidence intervals #Those this make sense with multivariate? Check
# this some time.
CI_level_05 = 0.025
delta_high = expit(scipy.stats.norm.ppf(
    loc=best_res.x[0], scale=sdParams[0], q=1 - CI_level_05))
epsilon_high = scipy.stats.norm.ppf(
    loc=best_res.x[1], scale=sdParams[1], q=1 - CI_level_05)
theta_high = np.exp(scipy.stats.norm.ppf(
    loc=best_res.x[2], scale=sdParams[2], q=1 - CI_level_05))

delta_low = expit(scipy.stats.norm.ppf(
    loc=best_res.x[0], scale=sdParams[0], q=CI_level_05))
epsilon_low = scipy.stats.norm.ppf(
    loc=best_res.x[1], scale=sdParams[1], q=CI_level_05)
theta_low = np.exp(scipy.stats.norm.ppf(
    loc=best_res.x[2], scale=sdParams[2], q=CI_level_05))

# Bootstraping
# Use cov matrix instead of sdparam d/t multivariate. Prob wrong in R code
paras_bootstrap = np.random.multivariate_normal(
    mean=best_res.x, cov=NeginvH2, size=NB_BOOTSRAPING)
paras_bootstrap[:, 0] = expit(paras_bootstrap[:, 0])
paras_bootstrap[:, 2] = np.exp(paras_bootstrap[:, 2])


pool = mp.Pool(mp.cpu_count())
res = pool.starmap(
    run_odeint, [(paras, [t_b, gamma, p0, q0]) for paras in paras_bootstrap])

# Calculate boostraping CI
t_ode, R0, R_e = ([] for i in range(3))
return_vals = odeint(SEIR_derivative, y0, t, args=(
    t_b, expit(delta), epsilon, np.exp(theta), gamma, p0, q0))  # Optimal
R, S, E, I_o, I_r = return_vals.T
R0_plot = R0
Re_plot = R_e
daterange_ode = [START_DATE + datetime.timedelta(days=x) for x in t_ode]

I_r_daily_new = (1 - p0) * E * rho  # New reported incidences estimates


I_r_bootsraping = [(1 - p0) * res[ind][:, 2:3] *
                   rho for ind in range(len(res))]
I_r_bootsraping = np.concatenate(I_r_bootsraping, axis=1)
S_bootstraping = [res[ind][:, 1:2] for ind in range(len(res))]
S_bootstraping = np.concatenate(S_bootstraping, axis=1)

I_r_Up = np.apply_along_axis(
    CRI, axis=1, arr=I_r_bootsraping, level=0.95, up=True)
I_r_Down = np.apply_along_axis(
    CRI, axis=1, arr=I_r_bootsraping, level=0.95, up=False)
S_Up = np.apply_along_axis(
    CRI, axis=1, arr=S_bootstraping, level=0.95, up=True)
S_Down = np.apply_along_axis(
    CRI, axis=1, arr=S_bootstraping, level=0.95, up=False)

# How many is in iva
in_iva = calculate_in_iva(I_r_daily_new)

# Plotting
fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(daterange, I_r_daily_new)
axs[0, 0].plot(df[INCIDENSE_COLUMN][[da.date()
                                     for da in daterange_opt]], 'o', mfc='none')
axs[0, 0].plot(daterange, in_iva)
axs[0, 0].plot(df_in_iva[IVA_COLUMN][[da.date()
                                     for da in daterange[(START_DATE_IVA - START_DATE).days:len(df_in_iva)]]], 'o', mfc='none')
axs[0, 0].plot(daterange, I_r_Up, ',')
axs[0, 0].plot(daterange, I_r_Down, ',')

axs[0, 0].set_title('# new cases every day with CI')
axs[0, 0].legend(
    ['Estimated new cases', 'Observed new cases', 'Estimated in IVA'])


axs[1, 0].plot(daterange, (N - np.array(S)) / N * 100)
axs[1, 0].plot(daterange, (N - np.array(S_Up)) / N * 100, ',')
axs[1, 0].plot(daterange, (N - np.array(S_Down)) / N * 100, ',')
axs[1, 0].set_title('Percent Non-Susceptible this date with CI')
axs[1, 0].set_ylabel('Percent')

axs[0, 1].plot(daterange, (np.array(I_o) + np.array(I_r)) / N * 100)
axs[0, 1].set_title('Percent infected this date')
axs[0, 1].set_ylabel('Percent')

axs[1, 1].plot(daterange_ode, R0_plot)
axs[1, 1].plot(daterange_ode, Re_plot, 'o', mfc='none')
axs[1, 1].set_title('R0 and R_e')
axs[1, 1].legend(['R0', 'R_e'])

for ax in axs.reshape(-1):
    ax.grid()
    ax.tick_params(labelrotation=20)
    mnthday = mdates.DateFormatter('%m-%d')
    ax.xaxis.set_major_formatter(mnthday)

plt.show()
