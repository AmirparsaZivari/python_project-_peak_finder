import numpy as np
import scipy as sci
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# +
## Single lorentzian function

def lorentzian(x,dc,a,gamma,centre):
    lorentzian = dc + a*gamma**2 / ((x-centre)**2+gamma**2)
    return lorentzian


# +
## Generating multiple Lorentzian function: this is the artificial data spectrum

def mult_lorentz(x,params):
#     n = 2
    n = round(np.size(params)/4)
    my_fun = np.zeros([np.size(x)])
#     param = np.array([dc0,dc1,a0,a1,g0,g1,f0,f1])
    for kk in range(n):
#         my_fun = my_fun + lorentzian(x,dc[kk],a[kk],gamma[kk],centre[kk])
        my_fun = my_fun + lorentzian(x,params[4*kk],params[4*kk+1],params[4*kk+2],params[4*kk+3])
        
    return my_fun

# Concatenating different step size values
x1 = np.linspace(0,5,100)
x2 = np.linspace(5.1,8,1000)
x3 = np.linspace(8.2,10,100)
x = np.concatenate((x1,x2,x3))

#Lorentzian parameter values: dc, a, gamma, centre for the 3 lorentzian functions in the fake spectrum
params = np.array([0,5,0.1,5,0,10,1,7,0,7,0.5,9])
# params = np.array([0,5,100e6,5e9])
fx = mult_lorentz(x,params)
plt.plot(x,fx)

## Adding normal distributed noise
fx_noisy = fx + np.random.normal(0,0.3,np.size(x))
plt.plot(x,fx_noisy)
# -

np.size(params)


# +
def res_fun(params,xdata,ydata):
    diff = mult_lorentz(xdata,params) - ydata
    return diff

# res_fun(params,x,fx_noisy)
print(type(res_fun(params,x,fx_noisy)))
# -

peaks2 , prop2 = find_peaks(fx_noisy, prominence = 3)
# print(type(peaks_2))
plt.scatter(x[peaks2],fx_noisy[peaks2],color = 'red')
plt.plot(x,fx_noisy)
len(peaks2)

# +
general_gamma = 0.1
# start = [0,5,1,general_gamma]
start = []

for ii in range(len(peaks2)):
    start = start + [0,10,general_gamma,x[peaks2[ii]]]
    print(x[peaks2[ii]])
#     f0 = f0+2
    
print(start)
# -

popt, ier = leastsq( res_fun, start, args=( x, fx_noisy ) )
print(popt)
popt[1]

fitted = mult_lorentz(x,popt)
plt.plot(x,fitted)
plt.plot(x,fx_noisy)



