from astropy.io import fits as pyfits
from glob import glob
from astropy.table import Table

import numpy as np
import pandas as pd
import os
from matplotlib.patches import Polygon
from matplotlib import rc
import matplotlib.pyplot as plt


def getXY(ra,dec, par):

    to_deg = 180 / np.pi
    to_rad = np.pi / 180.
    
    ra0,dec0,PA,axial = par
    
    ra = ra * np.pi / 180.
    dec = dec * np.pi / 180.
    ra0 = ra0 * np.pi / 180.
    dec0 = dec0 * np.pi / 180.
    PA = PA * np.pi / 180.
    
    xi = np.sin(ra - ra0) * np.cos(dec) * to_deg * 3600
    eta = ((np.sin(dec) * np.cos(dec0)) - np.cos(ra - ra0) * np.cos(dec) * np.sin(dec0) ) * to_deg * 3600
        
    return xi, eta

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

def mask_centres():
        
    files = []
    
    start_dir = '/Volumes/VINCE/vimos_new/'
    
    pattern         = '*MOS_SCIENCE_REDUCED.fits'

    for dir,_,_ in os.walk(start_dir):
        files.extend(glob(os.path.join(dir,pattern)))           
                
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    cols = ['RA', 'DEC', 'mask_name', 'date', 'seeing']
    
    df = pd.DataFrame(columns = cols )
        
    for f_data in files:
        
        hdu = pyfits.open(f_data)
        header = hdu[0].header 
        hdu.close()
    
        RA = header['RA']
        DEC = header['DEC']
        mask_name = header['OBJECT']
        date = pd.Timestamp(header['DATE-OBS'][0:10])
        seeing = float(header['HIERARCH ESO TEL IA FWHMLINOBS'])

        df_tmp = pd.DataFrame([[RA,DEC, mask_name,date, seeing ]], columns = cols)   
        df = df.append(df_tmp, ignore_index=True) 
                                           
    return df

if __name__ == '__main__':
    
    df = mask_centres()   
    df = df.drop_duplicates()   
        
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

fmatch = pd.read_table('/Volumes/VINCE/OAC/code/master_adp_ugri_clean.txt', delim_whitespace = True)
fmatch.drop(['Alt1', 'Alt2'], axis=1, inplace=True)  
fmatch = fmatch.replace((-9999.0, -8888.0), np.nan)

tmp = pd.read_csv('/Volumes/VINCE/OAC/imaging/Schuberth_poligon.txt', header = None)
xiS, etaS =  getXY(tmp[0], tmp[1], [54.620941, -35.450657, 64, 0.9])
vertsS = np.column_stack((-xiS/60., etaS/60.))

tmp = pd.read_csv('/Volumes/VINCE/OAC/imaging/Bergond_poligon.txt', header = None)
xiS, etaS =  getXY(tmp[0], tmp[1], [54.620941, -35.450657, 64, 0.9])
vertsB = np.column_stack((-xiS/60., etaS/60.))

xi, eta = getXY(fmatch.RA_adp.values, fmatch.DEC_adp.values, [54.620941, -35.450657, 64, 0.9])
xi = -xi / 60.
eta = eta / 60.

fig, ax = plt.subplots(figsize=(6,6))
hist, xbins, ybins = np.histogram2d(xi, eta, bins=(30,30), normed=False)
extent = [xbins.min(),xbins.max(),ybins.min(),ybins.max()]

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

ax.scatter(xi, eta, s = 4, c = 'black', marker = 'x', alpha = 0.5)
ax.set_axis_bgcolor('gray')
cax = ax.imshow(np.ma.masked_where(hist == 0, hist).T, interpolation='none', cmap = plt.cm.get_cmap('jet',10),
            origin='lower', vmin= 0, vmax = 20, extent = extent)
ax.add_patch(Polygon(vertsS, closed=True, fill=False, color='black', lw=2.4))
ax.add_patch(Polygon(vertsB, closed=True, fill=False, color='magenta', lw=2.4))

ax.set_xlabel(r'RA [arcmin]')
ax.set_ylabel(r'DEC [arcmin]')

ax.set_xlim(-27,33)
ax.set_ylim(-32,35)

cbar = fig.colorbar(cax, ticks = np.arange(0,22,2), fraction=0.052, pad=0.04)
labs = (2 * np.arange(0,22,2)).astype(str)
labs[-1] = r'$>$40'
cbar.ax.set_yticklabels(labs) 
cbar.ax.set_ylabel(r'Slits per arcmin$^2$')

fig.savefig('/Volumes/VINCE/OAC/paper/figures/slitdist.png', dpi = 300)



    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #


