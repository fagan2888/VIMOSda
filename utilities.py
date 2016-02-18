import numpy as np
import pandas as pd
import ppxf_util as util
from scipy import ndimage
from glob import glob
import random
from astropy.coordinates import SkyCoord
from pyraf import iraf
import re

def getdata_NEW(spec, wavelength, l1, l2):

     tmp = np.column_stack((wavelength, spec))
     mn = tmp[tmp[:,0] >  l1]
     mx = tmp[tmp[:,0] <  l2]

     lamRange = [mn[0,0] , mx[-1,0]]

     galaxy, logLam1, velscale = util.log_rebin(lamRange, spec)

     return galaxy, logLam1, velscale

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

def make_goodpixels(mask, logLam1, logLam2):
    regionmask = np.ones(logLam1.size, dtype=np.bool_)
    
    for lines in mask:    
        regionmask = regionmask & ~((logLam1 > lines[0]) & (logLam1 < lines[1]))
        goodpixels = np.nonzero(regionmask)[0]
        
    return goodpixels
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

def get_templates(velscale, l1, l2):
            
    library = glob('/Volumes/VINCE/OAC/libraries/Valdes/clean_short/*')
    ssp = pd.read_table(library[0], header=None)[0].values
    
    FWHM_gal = 13.1
    FWHM_tem = 1.35 
    FWHM_dif = np.sqrt(FWHM_gal**2 - FWHM_tem**2)
    scale = 0.4
    sigma = FWHM_dif/2.355/scale 
    lamRange2 = [3465., 9469.]
        
    sspNew, logLam2, _ = util.log_rebin(lamRange2, ssp, velscale = velscale)
    library_size = sspNew.size
    templates = np.empty((library_size,len(library)))
       
    for j in range(len(library)):
        tmp = pd.read_table(library[j], sep=r'\s+', header = None)
        ssp = tmp[0].values
        ssp = ndimage.gaussian_filter1d(ssp,sigma)
        sspNew, logLam2, _ = util.log_rebin(lamRange2, ssp, velscale = velscale)
        templates[:,j] = sspNew/np.median(sspNew) 
    
    return  logLam2, templates

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #


def get_vhelio(header):
    
        year = header['DATE-OBS'][0:4]
        month = header['DATE-OBS'][5:7]
        day = header['DATE-OBS'][8:10]
        ut = header['DATE-OBS'][11:16]
        
        coord = SkyCoord(ra = header['RA'], dec = header['DEC'] , unit = 'deg' )
        
        iraf.astutil.rvcorrect.ra = coord.ra.hour
        iraf.astutil.rvcorrect.dec = coord.dec.deg
        
        iraf.astutil.rvcorrect.observatory = 'esovlt'
        iraf.astutil.rvcorrect.year = year
        iraf.astutil.rvcorrect.month = month
        iraf.astutil.rvcorrect.day = day
        iraf.astutil.rvcorrect.ut = ut
        
        tmp = iraf.astutil.rvcorrect(Stdout = 1)
        
        vhelio = float(re.findall(r'(\S+)', tmp[5])[2])
        
        return vhelio
 