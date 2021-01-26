# +
import numpy as np
import scipy as sci
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from scipy.signal import find_peaks




# +
## lorentzian

def lorentzian(x,dc,a,gamma,centre):
    lorentzian = dc + a*gamma**2 / ((x-centre)**2+gamma**2)
    return lorentzian

# +
## import data



# +
## generating data

x = np.linspace(1e9,10e9,1000)
y = lorentzian(x,0,10,100e6,5e9)
plt.plot(x,y)
plt.title('clean peak')
plt.xlabel('frequency (Hz)')
plt.ylabel('amplitude')
plt.grid()
plt.show()
y_noisy = y + np.random.normal(0,1,1000)
plt.plot(x,y_noisy)
plt.title('noisy peak')
plt.xlabel('frequency (Hz)')
plt.ylabel('amplitude')
plt.grid()
plt.show()
# -

peaks, properties = find_peaks(y_noisy,prominence = 9)
print(peaks)

plt.scatter(x[peaks],y[peaks])

# +
y2 = lorentzian(x,0,50,500e6,8e9)
y_double = y + y2

y_double_noisy = y_double + np.random.normal(0,1,1000)
plt.plot(x,y_double_noisy)
# -

peaks, properties = find_peaks(y_double_noisy,prominence = 9)
print(peaks)

plt.scatter(x[peaks],y_double[peaks],color = 'red')
plt.plot(x,y_double_noisy)

popt, pcov = curve_fit(lorentzian,x,y_double_noisy,bounds = ([0,0,1e6,7e9],[10,100,1e9,10e9]))
plt.plot(x,y_double_noisy)
plt.plot(x,lorentzian(x,*popt), color = 'red')

# +
n = 2

def my_fun(x,dc0,dc1,a0,a1,g0,g1,f0,f1):
#     dc = np.zeros([1,n])
#     a = np.zeros([1,n])
#     gamma = np.zeros([1,n])
#     centre = np.zeros([1,n])
#     n = np.size(dc)
#     print(n)
#     dc = params[0,:]
#     a = params[1,:]
#     gamma = params[2,:]
#     centre = params[3,:]
#     n = round(n)
    n = 2
    my_fun = np.zeros([len(x)])
    param = np.array([dc0,dc1,a0,a1,g0,g1,f0,f1])
    for kk in range(n):
#         my_fun = my_fun + lorentzian(x,dc[kk],a[kk],gamma[kk],centre[kk])
        my_fun = my_fun + lorentzian(x,param[kk],param[kk+2],param[kk+4],param[kk+6])
        
    return my_fun


# -

# dc = np.array([0,0])
# a = np.array([10,5])
# gamma = np.array([100e6,1000e6])
# centre = np.array([5e9,6e9])
param = np.array([0,0,10,5,100e6,1000e6,5e9,6e9])
fx = my_fun(x,0,0,10,5,100e6,1000e6,5e9,6e9)
plt.plot(x,fx)
fx_noisy = fx + np.random.normal(3,1,1000)
plt.plot(x,fx_noisy)

popt, pcov = curve_fit(my_fun,x,fx_noisy, bounds = ([0,0,0,0,100e6,1e9,1e9,1e9],[10,10,1000,1000,1e9,5e9,9e9,9e9]))
plt.plot(x,fx_noisy)
plt.plot(x,my_fun(x,*popt), color = 'red')

n = 2
popt, pcov = curve_fit(my_fun,x,fx_noisy, bounds = (n=1,n=3))


# +
def mult_lorentz(x,params):
#     n = 2
    n = round(np.size(params)/4)
    my_fun = np.zeros([np.size(x)])
#     param = np.array([dc0,dc1,a0,a1,g0,g1,f0,f1])
    for kk in range(n):
#         my_fun = my_fun + lorentzian(x,dc[kk],a[kk],gamma[kk],centre[kk])
        my_fun = my_fun + lorentzian(x,params[4*kk],params[4*kk+1],params[4*kk+2],params[4*kk+3])
        
    return my_fun

x1 = np.linspace(0,5,100)
x2 = np.linspace(5.1,8,1000)
x3 = np.linspace(8.2,10,100)
x = np.concatenate((x1,x2,x3))

params = np.array([0,5,0.1,5,0,10,1,7,0,7,0.5,9])
# params = np.array([0,5,100e6,5e9])
fx = mult_lorentz(x,params)
plt.plot(x,fx)
fx_noisy = fx + np.random.normal(2,0.3,np.size(x))
plt.plot(x,fx_noisy)
# -

np.size(params)


# +
def res_fun(params,xdata,ydata):
    diff = mult_lorentz(xdata,params) - ydata
    return diff

# res_fun(params,x,fx_noisy)
print(type(res_fun(params,x,fx_noisy)))

# +
# def res_fun(params,xdata,ydata):
#     diff = [mult_lorentz(x,params) - y for x,y in zip(xdata,ydata) ]
#     return diff

# res_fun(params,x,fx_noisy)
# print(type(res_fun(params,x,fx_noisy)))
# -

[1,2] +[3,4]

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


