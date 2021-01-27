import numpy as np
import scipy as sci
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# +
## Single lorentzian function

def lorentzian(x,dc,a,gamma,centre):
    """Single Lorentzian function. 
    
    dc: float
    gamma: float
    centre: float
    x: float   --- input parameter
    
    returns a Lorentzian function
    
    """
    lorentzian = dc + a*gamma**2 / ((x-centre)**2+gamma**2)
    return lorentzian


# +
## Generating multiple Lorentzian function: this is the artificial data spectrum

def mult_lorentz(x,params):
    
    """
    
    A multiple Lorentzian function. It determines the number of Lorentzians based on the size of the parameter array. Each Lorentzian takes 4 input parameters. 
    params: [dc,amplitude,gamma,centre] --- This is the format of our input parameters
    
    n: Determines the number of single Lorentzians generated
    
    
    
    
    """

    n = round(np.size(params)/4)
    my_fun = np.zeros([np.size(x)])
    for kk in range(n):
        my_fun = my_fun + lorentzian(x,params[4*kk],params[4*kk+1],params[4*kk+2],params[4*kk+3])
        
    return my_fun



#In the project description it was asked to check whether this program works for points that are not homogeneously spaced. Thus we created an x range that is not homoegeously spaced.


x1 = np.linspace(0,5,100)
x2 = np.linspace(5.1,8,1000)
x3 = np.linspace(8.2,10,100)

# Concatenating different step size values
x = np.concatenate((x1,x2,x3))

#Lorentzian parameter values: dc, a, gamma, centre for the 3 lorentzian functions in the fake spectrum
params = np.array([0,5,0.1,5,0,10,1,7,0,7,0.5,9])
fx = mult_lorentz(x,params)

## Adding normal distributed noise
fx_noisy = fx + np.random.normal(3,0.3,np.size(x))

# -

def res_fun(params,xdata,ydata):
    """ 
    We need this to later use the least squared fits function. It takes the difference between the fit and the raw data.
    
    """
    diff = mult_lorentz(xdata,params) - ydata
    return diff



# +
#This is the scipy peak finder function. Prominance is the minimum height between the summit and any higher terrain. 

peaks2 , prop2 = find_peaks(fx_noisy, prominence = 3)


# Verify that peaks are found
plt.scatter(x[peaks2],fx_noisy[peaks2],color = 'red')
plt.plot(x,fx_noisy)
plt.title("Finding the peaks of the spectrum")


print("Counting the number of peaks found =",len(peaks2))

# +
# Generating the initial parameters 
general_gamma = 0.1
dc = 0
a = 10
peak_loc = x[peaks2]



def start_val(general_gamma,peak_loc,dc,a):

    start = []

    for ii in range(len(peaks2)):
        start = start + [dc,a,general_gamma,peak_loc[ii]]
        
    return start        


    
print("Initial parameters",start_val(general_gamma,peak_loc,dc,a))

start = start_val(general_gamma,peak_loc,dc,a)

print(start)

# +
# The least squared fit function. 

popt, ier = leastsq( res_fun, start, args=( x, fx_noisy ) )
print("The optimised parameters of the least squared function are...",popt)


# +
fitted = mult_lorentz(x,popt)

plt.plot(x,fx_noisy,label='raw data')
plt.plot(x,fitted,label = 'fitted')

plt.title("Fit vs Data")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.legend()

# -







