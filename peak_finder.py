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


