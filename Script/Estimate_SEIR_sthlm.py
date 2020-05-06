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
CSV_NEW_IN_IVA_PATH = os.path.join('Data', 'Antal_nya_iva_sthml_05-06.csv')
DATE_COLUMN = 'Datum'
INCIDENSE_COLUMN = 'Incidens'
IVA_COLUMN = 'IN_IVA'
MEAN_LENGTH_SICK = 5
MEAN_LENGTH_INCUBATION = 5.1

NEW_IN_IVA_COLUMN = 'NEW_IN_IVA'
MEAN_LENGTH_IVA = 10  # Guess from their analysis #https://www.folkhalsomyndigheten.se/contentassets/4b4dd8c7e15d48d2be744248794d1438/skattning-av-behov-av-slutenvardsplatser-covid-lombardiet.pdf
IVA_DELAY = 11  # From iva register
DEFAULT_RATE_IVA = 0.121  # Test to fit curve
# scipy.stats.norm.ppf(0.05, loc=10, scale=5) is approx = 1
STD_LENGTH_IVA = 5.4
STD_IVA_DELAY = 6  # scipy.stats.norm.ppf(0.05, loc=11, scale=6) is approx = 1
# OPTIMIZE_FOR_IN_IVA_OR_NEW_IN_IVA: IN_IVA=1, NEW_IN_IVA=2, ONLY_NEW_CASES=0
OPTIMIZE_FOR_IN_IVA_OR_NEW_IN_IVA = 0
IVA_NEW_IVA_SCALE_FOR_PLOT = 12
# Does it make sense to fit to IVA since they can be iva cases from
# patience infected abroad? #Now this is fi
IVA_OPT_WEGIHT = 1
# report page 9#int(648557/0.27+616655/0.26)/2 #Mean from Tabell 1 page 13
# #approx 2.386M
N = 2374550
i0 = 1
START_DATE = datetime.datetime(2020, 2, 17)
END_DATE_OPTIMIZE = datetime.datetime(2020, 5, 4)
END_DATE_INCIDENCE_OPTIMIZE = datetime.datetime(2020, 4, 10)
END_DATE_SIMULATION = datetime.datetime(2020, 6, 15)
MIDDLE_THETA = datetime.datetime(2020, 3, 16)
NB_OPTIMIZATIONS = 20  # Number of optimization runs
NB_BOOTSRAPING = 1000

p0 = 0.987  # Calibrated somehow to fit 2.5% in end march
q0 = 0.11  # Rate of how infective a non reported case is vs reported
delta = 0.16  # Init for test
epsilon = -0.19  # Init for test
theta = 10.9  # Init for test
rho = 1 / MEAN_LENGTH_INCUBATION
gamma = 1 / MEAN_LENGTH_SICK
t_b = (MIDDLE_THETA - START_DATE).days
if OPTIMIZE_FOR_IN_IVA_OR_NEW_IN_IVA:
    nb_paras = 4  # If fitting iva this is needed to be 4
else:
    nb_paras = 3  # If fitting not iva this is 3


def basic_reproduction(p0, b_t, gamma, q0):
    return (1 - p0) * b_t / gamma + p0 * q0 * b_t / gamma


def b_t_func(t, t_b, delta, epsilon, theta, gamma, p0, q0):
    exp_var = -epsilon * (t - t_b)
    if isinstance(exp_var, float) and exp_var > 200:
        exp_var = np.inf
    elif not isinstance(exp_var, float):
        exp_var[exp_var > 200] = np.inf
    return theta * (delta + (1 - delta) / (1 + np.exp(exp_var)))


def calculate_new_reported_infected(E, p0, rho):
    return (1 - p0) * E * rho


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


def opt_guesses(nb_paras):
    # Similar as in R
    u_d = logit(np.random.uniform(0.05, 0.6))  # guess for delta
    u_e = np.random.uniform(-0.6, 0)    # guess for epsilon
    u_t = np.log(np.random.uniform(0, 15))    # guess for theta
    if nb_paras == 3:
        return u_d, u_e, u_t
    else:
        rate_iva = np.log(np.random.uniform(0.05, 2))  # guess for rate_iva
        return u_d, u_e, u_t, rate_iva


def RSS(params, t_optimize, t_b, y0, p0, rho, incidence, gamma, q0, optimize_iva_arr, iva_opt_weight):
    # Code for minimizing, similar as in R code

    delta = expit(params[0])
    epsilon = params[1]
    theta = np.exp(params[2])
    if optimize_iva_arr[0] != 0:
        rate_iva = np.exp(params[3])
    dummy_infectivity = b_t_func(
        np.arange(200), t_b, delta, epsilon, theta, gamma, p0, q0)
    if min(dummy_infectivity) < 0:
        return 10**12
    return_vals = odeint(SEIR_derivative, y0, t=t_optimize, args=(
        t_b, delta, epsilon, theta, gamma, p0, q0))
    R, S, E, I_o, I_r = return_vals.T
    fitted_incidence = calculate_new_reported_infected(E, p0, rho)
    incidence_sq_sum = (
        (incidence - fitted_incidence[:len(incidence)])**2).sum()
    if optimize_iva_arr[0] == 0:
        return incidence_sq_sum
    elif optimize_iva_arr[0] == 1:
        fitted_iva = calculate_in_iva(
            fitted_incidence, rate_iva=rate_iva)
    else:
        fitted_iva = calculate_new_in_iva(
            fitted_incidence, rate_iva=rate_iva)

    fitted_iva = fitted_iva[len(t_optimize) - len(optimize_iva_arr[1]):]
    return incidence_sq_sum + iva_opt_weight * ((optimize_iva_arr[1] - fitted_iva)**2).sum()


def run_optimization(x0, args):
    t_optimize, t_b, y0, p0, rho, incidence, gamma, q0, optimize_iva_arr, iva_opt_weight, tmpdict = args
    return minimize(RSS, x0, method=tmpdict['method'], options=tmpdict['options'],
                    args=(t_optimize, t_b, y0, p0, rho, incidence, gamma, q0, optimize_iva_arr, iva_opt_weight))


def numerical_Hessian(params, args):
    # Calculate Hessian since they are using that for covariance matrix but it
    # is not returned from Nelder-Mead in python
    t_optimize, t_b, y0, p0, rho, incidence,  gamma, q0, optimize_iva_arr, iva_opt_weight = args
    return RSS(params, t_optimize, t_b, y0, p0, rho, incidence,  gamma, q0, optimize_iva_arr, iva_opt_weight)


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


def calculate_in_iva(I_r, rate_iva=DEFAULT_RATE_IVA, iva_delay=IVA_DELAY, mean_length_iva=MEAN_LENGTH_IVA):
    # Can move this outside if it takes time
    combined_std = np.sqrt(STD_LENGTH_IVA**2 + STD_IVA_DELAY**2)
    combined_mean_normal = IVA_DELAY + MEAN_LENGTH_IVA
    max_val = scipy.stats.norm.ppf(
        0.99, loc=combined_mean_normal, scale=combined_std)
    # max(scipy.stats.norm.ppf(0.01, loc=combined_mean_normal, scale=combined_std), 0)
    min_val = 0
    IVA_arr = scipy.stats.norm.pdf(np.arange(round(min_val), round(
        max_val)), loc=combined_mean_normal, scale=combined_std)
    # multiply just to get graphs approx right
    return np.convolve(np.array(I_r), IVA_arr)[0:len(I_r)] * rate_iva * IVA_NEW_IVA_SCALE_FOR_PLOT


def calculate_new_in_iva(I_r, rate_iva=DEFAULT_RATE_IVA, iva_delay=IVA_DELAY):
    IVA_arr = np.zeros(IVA_DELAY)
    IVA_arr[-1] = 1
    return np.convolve(np.array(I_r), IVA_arr)[0:len(I_r)] * rate_iva

# Basic tests
print('Running. Should maybe take 10-20 seconds')

df = pd.read_csv(CSV_CASE_PATH, sep=' ', parse_dates=[
                 DATE_COLUMN]).set_index(DATE_COLUMN)
df_in_iva = pd.read_csv(CSV_IN_IVA_PATH, sep=' ', parse_dates=[
    DATE_COLUMN]).set_index(DATE_COLUMN)
df_new_in_iva = pd.read_csv(CSV_NEW_IN_IVA_PATH, sep=',', parse_dates=[
                            DATE_COLUMN]).set_index(DATE_COLUMN)

y0 = 0, N - i0, 0, 0, i0
daterange = [START_DATE + datetime.timedelta(days=x) for x in range(
    0, int((END_DATE_SIMULATION - START_DATE).days))]
daterange_iva = [datetime.datetime(
    my_date.year, my_date.month, my_date.day) for my_date in df_in_iva.index.date]
daterange_new_in_iva = [datetime.datetime(
    my_date.year, my_date.month, my_date.day) for my_date in df_new_in_iva.index.date]
daterange_inc = [START_DATE + datetime.timedelta(days=x) for x in range(
    0, int((END_DATE_INCIDENCE_OPTIMIZE - START_DATE).days + 1))]

t = np.arange(len(daterange))
# Just for test
global t_ode, R0, R_e
t_ode, R0, R_e = ([] for i in range(3))
return_vals = odeint(SEIR_derivative, y0, t, args=(
    t_b, delta, epsilon, theta, gamma, p0, q0))
R, S, E, I_o, I_r = return_vals.T


# Optimize in parallel
t_optimize = np.arange((END_DATE_OPTIMIZE - START_DATE).days + 1)
incidence = df.to_numpy().flatten()
in_iva = df_in_iva.to_numpy().flatten()
new_in_iva = df_new_in_iva.to_numpy().flatten()

optimize_iva_arr = [0]
if OPTIMIZE_FOR_IN_IVA_OR_NEW_IN_IVA == 1:
    optimize_iva_arr = [OPTIMIZE_FOR_IN_IVA_OR_NEW_IN_IVA, in_iva]
elif OPTIMIZE_FOR_IN_IVA_OR_NEW_IN_IVA == 2:
    optimize_iva_arr = [OPTIMIZE_FOR_IN_IVA_OR_NEW_IN_IVA, new_in_iva]

pool = mp.Pool(mp.cpu_count())
args = [t_optimize, t_b, y0, p0, rho, incidence[
    0:t_optimize.shape[0]],  gamma, q0, optimize_iva_arr, IVA_OPT_WEGIHT]
opt_args = [{'method': 'Nelder-Mead', 'options': {'maxiter': 1000}}]
results = pool.starmap(run_optimization, [(
    opt_guesses(nb_paras=nb_paras), args + opt_args) for i in range(NB_OPTIMIZATIONS)])
for i, res in enumerate(results):
    if nb_paras == 3:
        delta, epsilon, theta = res.x
        print(round(expit(delta), 4), round(epsilon, 4), round(
            np.exp(theta), 4))
    else:
        delta, epsilon, theta, rate_iva = res.x
        print(round(expit(delta), 4), round(epsilon, 4), round(
            np.exp(theta), 4), round(np.exp(rate_iva), 4))
    print(round(res.fun, 4))
    if i == 0 or res.fun < best_res.fun:
        best_res = res

# Calculate Hessian since they are using that for covariance matrix but it
# is not returned from Nelder-Mead in python
# Use covariance instead of std

sigest = np.sqrt(best_res.fun / (t_optimize.shape[0] - nb_paras))  # m-n
H = nd.Hessian(numerical_Hessian)(best_res.x.tolist(), args)

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
paras_bootstrap = paras_bootstrap[:, 0:3]

pool = mp.Pool(mp.cpu_count())
res = pool.starmap(
    run_odeint, [(paras, [t_b, gamma, p0, q0]) for paras in paras_bootstrap])

# Calculate boostraping CI
t_ode, R0, R_e = ([] for i in range(3))
return_vals = odeint(SEIR_derivative, y0, t, args=(
    t_b, expit(best_res.x[0]), best_res.x[1], np.exp(best_res.x[2]), gamma, p0, q0))  # Optimal
R, S, E, I_o, I_r = return_vals.T
R0_plot = R0
Re_plot = R_e
daterange_ode = [START_DATE + datetime.timedelta(days=x) for x in t_ode]

I_r_daily_new = calculate_new_reported_infected(
    E, p0, rho)  # New reported incidences estimates


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
if len(best_res.x) == 3:
    rate_iva = np.log(DEFAULT_RATE_IVA)
else:
    rate_iva = best_res.x[3]
in_iva_est = calculate_in_iva(I_r_daily_new, rate_iva=np.exp(rate_iva))
new_in_iva_est = calculate_new_in_iva(I_r_daily_new, rate_iva=np.exp(rate_iva))
# Plotting
fig, axs = plt.subplots(2, 2)
titlestring = 'Fitted using Infected'
if OPTIMIZE_FOR_IN_IVA_OR_NEW_IN_IVA == 1:
    titlestring += ' and IN_IVA'
elif OPTIMIZE_FOR_IN_IVA_OR_NEW_IN_IVA == 2:
    titlestring += ' and NEW_IVA'

fig.suptitle(titlestring)
axs[0, 0].plot(daterange, I_r_daily_new)
axs[0, 0].plot(df[INCIDENSE_COLUMN][[da.date()
                                     for da in daterange_inc]], 'o', mfc='none')
axs[0, 0].plot(daterange, in_iva_est)
axs[0, 0].plot(df_in_iva[IVA_COLUMN][[da.date()
                                      for da in daterange_iva]], 'o', mfc='none')
axs[0, 0].plot(daterange, new_in_iva_est)
axs[0, 0].plot(df_new_in_iva[NEW_IN_IVA_COLUMN][[da.date()
                                                 for da in daterange_new_in_iva]], 'o', mfc='none')
axs[0, 0].plot(daterange, I_r_Up, ',')
axs[0, 0].plot(daterange, I_r_Down, ',')

axs[0, 0].set_title(
    'New cases/IVA every day with CI. \n Note there is a scale parameter between NEW_IVA and IN_IVA. \n Currently this scale is {}'.format(IVA_NEW_IVA_SCALE_FOR_PLOT))
axs[0, 0].legend(
    ['Estimated new cases', 'Observed new cases', 'Estimated IN_IVA', 'True IN_IVA', 'Estimated NEW_IVA', 'True NEW_IVA'])


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
