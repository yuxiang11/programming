
from datetime import datetime
import gen3
from gen3.auth import Gen3Auth
from gen3.submission import Gen3Submission
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import json
import requests
from matplotlib.dates import date2num, num2date
from scipy import integrate, optimize
from scipy.integrate import odeint
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore')





R0 = 2.4 # {type:"slider", min:0.9, max:5, step:0.1}
t_incubation = 5.1 # {type:"slider", min:1, max:14, step:0.1}
t_infective = 3.3 # {type:"slider", min:1, max:14, step:0.1}
# Population distribution
N = 14000 # {type:"slider", min:1000, max:350000, step: 1000}
# Initial exposure
n = 10 # {type:"slider", min:0, max:100, step:1}
#Begin social distancing after contact (weeks)
t_social_distancing = 2 # {type:"slider", min:0, max:30, step:0.1}
# Effectiveness of social distancing (0 to 100%)
u_social_distancing = 40 #@param {type:"slider", min:0, max:100, step:1}

# Initial infections and recoveries
e_initial = n/N
i_initial = 0.00
r_initial = 0.00
s_initial = 1 - e_initial - i_initial - r_initial

alpha = 1/t_incubation
gamma = 1/t_infective
beta = R0*gamma

def step(t):
    return 1 if t >= 7*t_social_distancing else 0

# SEIR model's derivative function
def deriv(x, t, u, alpha, beta, gamma):
    s, e, i, r = x
    dsdt = -(1-u*step(t)/100)*beta * s * i
    dedt =  (1-u*step(t)/100)*beta * s * i - alpha * e
    didt = alpha * e - gamma * i
    drdt =  gamma * i
    return [dsdt, dedt, didt, drdt]

t = np.linspace(0, 210, 210)
x_initial = s_initial, e_initial, i_initial, r_initial
s, e, i, r = odeint(deriv, x_initial, t, args=(u_social_distancing, alpha, beta, gamma)).T
s0, e0, i0, r0 = odeint(deriv, x_initial, t, args=(0, alpha, beta, gamma)).T

# Plot
fig = plt.figure(figsize=(12, 10))
ax = [fig.add_subplot(311, axisbelow=True), 
      fig.add_subplot(312)]

pal = sns.color_palette()

ax[0].stackplot(t/7, N*s, N*e, N*i, N*r, colors=pal, alpha=0.6)
ax[0].set_title('{0:3.0f}% susceptible and recovered populations with effective social distance'.format(u_social_distancing))
ax[0].set_xlabel('Number of weeks after initial exposure')
ax[0].set_xlim(0, t[-1]/7)
ax[0].set_ylim(0, N)
ax[0].legend([
    'susceptible', 
    'Exposure、asymptomatic', 
    'Infected、symptomatic',
    'rehabilitation'], 
    loc='best')
ax[0].plot(np.array([t_social_distancing, t_social_distancing]), ax[0].get_ylim(), 'r', lw=3)
ax[0].plot(np.array([0, t[-1]])/7, [N/R0, N/R0], lw=3, label='herd immunity')
ax[0].annotate("Start social distancing",
    (t_social_distancing, 0), (t_social_distancing + 1.5, N/10),
    arrowprops=dict(arrowstyle='->'))
ax[0].annotate("Herd immunity without social distancing",
    (t[-1]/7, N/R0), (t[-1]/7 - 8, N/R0 - N/5),
    arrowprops=dict(arrowstyle='->'))

ax[1].stackplot(t/7, N*i0,N*e0, colors=pal[2:0:-1], alpha=0.5)
ax[1].stackplot(t/7, N*i, N*e, colors=pal[2:0:-1], alpha=0.5)
ax[1].set_title('Infected population without social distance and the effective social distance is{0:3.0f}%'.format(u_social_distancing))
ax[1].set_xlim(0, t[-1]/7)
ax[1].set_ylim(0, max(0.3*N, 1.05*max(N*(e + i))))
ax[1].set_xlabel('Number of weeks after initial exposure')
ax[1].legend([
    'Infectious/symptomatic', 
    'exposed/asymptomatic'],
    loc='upper right')
ax[1].plot(np.array([t_social_distancing, t_social_distancing]), ax[0].get_ylim(), 'r', lw=3)

y0 = N*(e0 + i0)
k0 = np.argmax(y0)
ax[1].annotate("No social distancing", (t[k0]/7, y0[k0] + 100))

y = N*(e + i)
k = np.argmax(y)
ax[1].annotate("Effective social distance：{0:3.0f}%".format(u_social_distancing), (t[k]/7, y[k] + 100))

for a in ax:
    a.xaxis.set_major_locator(plt.MultipleLocator(5))
    a.xaxis.set_minor_locator(plt.MultipleLocator(1))
    a.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
    a.grid(True)

plt.tight_layout()





def base_seir_model(init_vals, params, t):
    '''
    
    beta: S --> E, epsilon: E --> I, gamma: I --> R
        
         
    
    '''
    S_0, E_0, I_0, R_0 = init_vals
    S, E, I, R = [S_0], [E_0], [I_0], [R_0]
    epsilon, beta, gamma = params
    dt = t[1] - t[0]
    for _ in t[1:]:
        next_S = S[-1] - (beta * S[-1] * I[-1]) * dt
        next_E = E[-1] + (beta * S[-1] * I[-1] - epsilon * E[-1]) * dt
        next_I = I[-1] + (epsilon * E[-1] - gamma * I[-1]) * dt
        next_R = R[-1] + (gamma * I[-1]) * dt
        S.append(next_S)
        E.append(next_E)
        I.append(next_I)
        R.append(next_R)
    return np.stack([S, E, I, R]).T





# Initial values
N = 5180493
S_0 = (N - 11) / N
E_0 = 10 / N
I_0 = 1 / N
R_0 = 0
init_vals = [S_0, E_0, I_0, R_0]

# Params
epsilon, beta, gamma = [0.2, 1.75, 0.5]
params = epsilon, beta, gamma

# define time interval 
t_max = 1000
dt = 1
t = np.linspace(0, t_max, int(t_max / dt) + 1)

# Run simulation
results = base_seir_model(init_vals, params, t)





def plot_model(
    simulated_susceptible, simulated_exposure, simulated_infectious, simulated_remove
):
    
    global times, numTimes
    startInd = 0
    numTimes = len(simulated_infectious)

    fig = plt.figure(figsize=[22, 12], dpi=120)
    fig.subplots_adjust(top=0.85, right=0.92)
    ind = np.arange(numTimes)
    indObs = np.arange(len(simulated_infectious))

    ax = fig.add_subplot(111)
    ax.yaxis.grid(True, color='black', linestyle='dashed')
    ax.xaxis.grid(True, color='black', linestyle='dashed')
    ax.set_axisbelow(True)
    fig.autofmt_xdate()

    (infectedp,) = ax.plot(indObs, simulated_infectious, linewidth=3, color='black')
    (sp,) = ax.plot(ind, simulated_susceptible, linewidth=3, color='red')
    (ep,) = ax.plot(ind, simulated_exposure, linewidth=3, color='purple')
    (ip,) = ax.plot(ind, simulated_infectious, linewidth=3, color='blue')
    (rp,) = ax.plot(ind, simulated_remove, linewidth=3, color='orange')
    ax.set_xlim(0, numTimes)
    ax.set_xlabel('Days')
    ax.set_ylabel('Population ratio')

    plt.legend(
        [sp, ep, ip, rp],
        [
            'susceptible',
            'Exposure, asymptomagtic',
            'Infected, symptomatic',
            'rehabilitation',
        ],
        loc='upper right',
        bbox_to_anchor=(1, 1.22),
        fancybox=True,
    )
    
plot_model(results[:200, 0], results[:200, 1], results[:200, 2], results[:200, 3])





df = pd.read_csv('./data/data.csv', parse_dates=['dateRep'])
df = df[(df.countriesAndTerritories == 'United_Kingdom')]
df = df.sort_values(by = 'dateRep')
df = df[['dateRep', 'cases', 'deaths', 'countriesAndTerritories']]
df.rename(columns = {'dateRep': 'date', 'countriesAndTerritories': 'county'}, inplace = True)
df.head(5)





def format_date(x, pos=None):
    thisind = np.clip(int(startInd + x + 0.5), startInd, startInd + numTimes - 1)
    return num2date(times[thisind]).strftime('%m/%d/%Y')

def validate_model(simulated_cases, cases):
    
    global times, numTimes
    startInd = 0
    times = [date2num(s) for (s) in df.date]
    numTimes = len(simulated_cases)

    fig = plt.figure(figsize=[22, 12], dpi=120)
    fig.subplots_adjust(top=0.85, right=0.92)
    ind = np.arange(numTimes)
    indObs = np.arange(len(simulated_cases))

    ax = fig.add_subplot(111)
    ax.yaxis.grid(True, color='black', linestyle='dashed')
    ax.xaxis.grid(True, color='black', linestyle='dashed')
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    fig.autofmt_xdate()

    (infectedp,) = ax.plot(indObs, simulated_cases, linewidth=3, color='black')
    (si,) = ax.plot(ind, simulated_cases, linewidth=3, color='orange')
    (i,) = ax.plot(ind, cases, linewidth=3, color='blue')
    ax.set_xlim(0, numTimes)
    ax.set_xlabel('Date')
    ax.set_ylabel('Population ratio')

    plt.legend(
        [si, i],
        ['Simulated', 'Actual'],
        loc='upper right',
        bbox_to_anchor=(1, 1.22),
        fancybox=True,
    )





days = len(df.cases)
startInd = 0
cases = results[:days, 1] + results[:days, 2]
validate_model((results[:days, 1] + results[:days, 2]) , (df.cases / N - df.deaths/N))





class OptimizeModelParameters(object):
    '''SEIR'''
    def __init__(self, init_vals, confirmed):
        
        self.init_vals = init_vals
        self.confirmed = confirmed

    def evaluate(self, params):
        
        S_0, E_0, I_0, R_0 = self.init_vals
        S, E, I, R = [S_0], [E_0], [I_0], [R_0]
        epsilon, beta, gamma = params
        dt = 1
        for _ in range(len(self.confirmed) - 1):
            next_S = S[-1] - (beta * S[-1] * I[-1]) * dt
            next_E = E[-1] + (beta * S[-1] * I[-1] - epsilon * E[-1]) * dt
            next_I = I[-1] + (epsilon * E[-1] - gamma * I[-1]) * dt
            next_R = R[-1] + (gamma * I[-1]) * dt
            S.append(next_S)
            E.append(next_E)
            I.append(next_I)
            R.append(next_R)
        return E, I

    def error(self, params):
        '''
    
        params: Epsilon, beta, gamma
        
        
        '''
        yEim, yIim = self.evaluate(params)
        yCim = [sum(i) for i in zip(yEim, yIim)]  
        res = sum(
              np.subtract(yCim, self.confirmed) ** 2
        )
        return res


    def optimize(self, params):
        '''
        
        params: Epsilon, beta, gamma

        
        '''
        res = optimize.minimize(
            self.error,
            params,
            method = 'L-BFGS-B',
            bounds = [(0.01, 20.0), (0.01, 20.0), (0.01, 20.0)],
            options = {'xtol': 1e-8, 'disp': True, 'ftol': 1e-7, 'maxiter': 1e8},
        )
        return res





# Set up population distribution in initial state
min_ratio = 221000/13816
max_ratio = 442000/13816
infected_cases = df.cases / N - df.deaths / N

# Instantiate the class
min_seir_eval = OptimizeModelParameters(init_vals, infected_cases * min_ratio)
max_seir_eval = OptimizeModelParameters(init_vals, infected_cases * max_ratio)

# Run optimiza function
min_opt_p = min_seir_eval.optimize(params)
max_opt_p = max_seir_eval.optimize(params)





min_results = base_seir_model(init_vals, min_opt_p.x, t)
max_results = base_seir_model(init_vals, max_opt_p.x, t)

min_simulated_cases = (min_results[:days,1] + min_results[:days,2]) * N/min_ratio
min_simulated_cases = [int(x) for x in min_simulated_cases]

max_simulated_cases = (max_results[:days,1] + max_results[:days,2]) * N/max_ratio
max_simulated_cases = [int(x) for x in max_simulated_cases]

avg_simulated_cases = [sum(i)/(2*N) for i in zip(min_simulated_cases, max_simulated_cases)]

validate_model(avg_simulated_cases, df.cases / N - df.deaths / N)





# Run simulation
results = base_seir_model(init_vals, params, t)
print('Forecasted maximum confrimed numbers: %s' % str(int(max(results[:, 2]) * N)))
plot_model(results[:200, 0], results[:200, 1], results[:200, 2], results[:200, 3])






