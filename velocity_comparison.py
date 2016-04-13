from pyraf import iraf
import glob
import pandas as pd
import tqdm
import numpy as np
from matplotlib.pylab import plt

from astropy import units as u
from astropy import coordinates as coords
from astropy.coordinates import SkyCoord

fmatch = pd.read_table('/Volumes/VINCE/OAC/master_adp_ugri_clean.txt', delim_whitespace = True)
fmatch = fmatch.replace((-9999.0, -8888.0), np.nan)
fmatch.index = fmatch.ID

GCs = pd.read_csv('/Volumes/VINCE/OAC/GCs_903.csv', dtype = {'ID': object}, comment = '#')
stars = pd.read_csv('/Volumes/VINCE/OAC/stars_903.csv', dtype = {'ID': object}, comment = '#')
stars.drop_duplicates(subset ='ID', inplace=True)

GCs.index = GCs.ID
stars.index = stars.ID


total = pd.concat([GCs,stars])
final = pd.concat([fmatch,total['flag']], axis = 1)

final[['ID_f11','RA_g', 'DEC_g', 'flag']].to_csv('spectro_with_flags.csv', index = False)
cols = ['RA_g','DEC_g','VREL_helio','VERR', 'u_auto', 'g_auto',
		'r_auto', 'i_auto','class']
GCsToAdd = pd.DataFrame(GCs[(GCs.flag == 'g') | (GCs.flag == 'm')][cols].values, 
				   columns = [cols])

# ----------------------------------

S_GCs =  pd.read_table('/Volumes/VINCE/OAC/fxcor/SchuberthGCs.csv', sep=r'\s+').replace(99.99, np.nan)
S_GCs = S_GCs.drop_duplicates(['Rmag', 'C-R','HRV']).reset_index(drop = True)
S_Stars =  pd.read_table('/Volumes/VINCE/OAC/fxcor/SchuberthStars.csv', sep=r'\s+').replace(99.99, np.nan)
S_Stars = S_Stars.drop_duplicates(['Rmag', 'C-R', 'HRV']).reset_index(drop = True)

cat1 = coords.SkyCoord(GCs['RA_g'], GCs['DEC_g'], unit=(u.degree, u.degree))
cat2 = coords.SkyCoord(S_GCs['RA'], S_GCs['DEC'], unit=(u.degree, u.degree))

index,dist2d, _ = cat1.match_to_catalog_sky(cat2)
mask = dist2d.arcsec < 0.4
new_idx = index[mask]

VIMOS = GCs.ix[mask].reset_index(drop = True)
SchuberthMatch = S_GCs.ix[new_idx].reset_index(drop = True)
print len(SchuberthMatch)

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

# ----------------------------------
# ----------------------------------

cat1 = coords.SkyCoord(stars['RA_g'], stars['DEC_g'], unit=(u.degree, u.degree))
cat2 = coords.SkyCoord(S_Stars['RA'], S_Stars['DEC'], unit=(u.degree, u.degree))

index,dist2d, _ = cat1.match_to_catalog_sky(cat2)
mask = dist2d.arcsec < 0.4
new_idx = index[mask]

VIMOS = stars.ix[mask].reset_index(drop = True)
SchuberthMatch = S_Stars.ix[new_idx].reset_index(drop = True)
print len(SchuberthMatch)

x = VIMOS['VREL_helio'] 
xerr = VIMOS['VERR'] 
y = SchuberthMatch['HRV'] 
yerr = SchuberthMatch['e.1'] 

plt.errorbar(x, y, yerr= yerr, xerr = xerr, fmt = 'o', c ='black')
plt.plot([400, 2200], [400, 2200], '--k')
print 'rms (VIMOS - Schuberth) stars = ', np.std(x-y)

# ----------------------------------
# ----------------------------------

Bergond = pd.read_table('/Volumes/VINCE/OAC/Bergond.txt', sep=';',comment='#')
BergondGCs = Bergond[Bergond['Type'] =='gc']
BergondGCs[['RAJ2000', 'DEJ2000']].to_csv('/Volumes/VINCE/OAC/imaging/BergondGCs_RADEC.reg', index = False, sep =' ', header = None)

cat1 = coords.SkyCoord(GCs['RA_g'], GCs['DEC_g'], unit=(u.degree, u.degree))
cat2 = coords.SkyCoord(BergondGCs['RAJ2000'], BergondGCs['DEJ2000'], unit=(u.degree, u.degree))

index,dist2d, _ = cat1.match_to_catalog_sky(cat2)
mask = dist2d.arcsec < 0.3
new_idx = index[mask]

VIMOS = GCs.ix[mask].reset_index(drop = True)
BergondMatch = BergondGCs.ix[new_idx].reset_index(drop = True)
print len(BergondMatch)

x = VIMOS['VREL_helio'] 
xerr = VIMOS['VERR'] 
y = BergondMatch['HRV'] 
yerr = BergondMatch['e_HRV'] 

plt.errorbar(x, y, yerr= yerr, xerr = xerr, fmt = 'o', c ='red',label = 'Bergond et al.')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')    
plt.xlabel(r'Velocity from this work [km s$^-1$ ]')
plt.ylabel(r'Velocity from the literature [km s$^-1$ ]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()

print 'rms (VIMOS - Bergond) GCs = ', np.std(x-y)
