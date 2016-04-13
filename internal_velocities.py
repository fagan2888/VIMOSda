from pyraf import iraf
import glob
import pandas as pd
import tqdm
import numpy as np
from matplotlib.pylab import plt

from astropy import units as u
from astropy import coordinates as coords
from astropy.coordinates import SkyCoord

GCs = pd.read_csv('/Volumes/VINCE/OAC/GCs_903.csv', dtype = {'ID': object}, comment = '#')

# ----------------------------------
rep1 = GCs[GCs.Alt1.isin(GCs.ID)]
df1 = pd.DataFrame()
df2 = pd.DataFrame()
for j in range(0,len(rep1)):
	df1.iloc[j] = rep1.iloc[j]



x = VIMOS['VREL_helio'] 
xerr = VIMOS['VERR'] 
y = SchuberthMatch['HRV'] 
yerr = SchuberthMatch['e.1'] 

print 'rms (VIMOS - Schuberth) GCs = ', np.std(x-y)

plt.close('all')
plt.figure(figsize=(6,6))
plt.errorbar(x, y, yerr= yerr, xerr = xerr, fmt = 'o', c ='black', label = 'Schuberth et al.')
plt.plot([-200, 2200], [-200, 2200], '--k')
plt.xlim(-200,2200)
plt.ylim(-200,2200)

x = VIMOS['r_auto'] 
y = SchuberthMatch['Rmag'] 

plt.scatter(x, y, c ='black')

