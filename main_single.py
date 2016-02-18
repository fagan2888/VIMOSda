from glob import glob
import ppxf
from astropy.table import Table

import numpy as np
from tqdm import *
import time
import multiprocessing
import matplotlib.pyplot as plt
import pandas as pd
import os
import utilities 
reload(utilities)
import ppxfplot as ppplot
from astropy.io.fits import getheader, getdata
from pyraf import iraf
import ppxf_util as util
from astropy.io import fits
import ppxfplot as ppplot

c = 299792.458 

pixelmask_normal = np.log(np.array([[6850, 6950], [7560, 7700], [7150, 7350],
            [5565,5590], [5874, 5910], [6290,6312], [6355,6374]]))

pixelmask_Q1Q3 = np.log(np.array([[6850, 6950], [7560, 7700], [7150, 7350], [5600,6200],
            [5565,5590], [6290,6312], [6355,6374]]))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

def run_ppxf(wavelength_log, templates, galaxy, noise, velscale, start, goodpixels, dv):

     pp = ppxf.ppxf(templates, galaxy, noise, velscale, start, bias = 0, moments=2, goodpixels = goodpixels, vsyst=dv, quiet=True, degree = -1, mdegree = 6, oversample = 20)
                                                                  
     rv    =  round(pp.sol[0],1)
     sigma =  round(pp.sol[1],1)
     chi2  =  pp.chi2
     error_rv =  pp.error[0]*np.sqrt(chi2)
     error_sigma =  pp.error[1]*np.sqrt(chi2)
     bestfit = pp.bestfit
     residuals =  bestfit - galaxy

     zp1 = 1 + rv/c
     output_wavelengths = wavelength_log / zp1  

     return rv, sigma, error_rv, error_sigma, chi2, bestfit, residuals, zp1, output_wavelengths

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

def _boot_VIMOS_ppxf(sample):
     '''
     Convenience function for multiprocessing
     '''
     return (run_ppxf(*sample)) 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #


def VIMOS_ppxf(l1, l2, start, plot = True, nsimulations = False):
    
     spec = getdata(result['ORIGINALFILE'].loc[0], 1)    
     header = getheader(result['ORIGINALFILE'].loc[0], 1)  
     
     quadrant = header['ESO OCS CON QUAD']

     lamRange = header['CRVAL1']  + np.array([0., header['CD1_1'] * (header['NAXIS1'] - 1)]) 
     wavelength = np.linspace(lamRange[0],lamRange[1], header['NAXIS1'])

     ix_start = np.where(wavelength > l1)[0][0]
     ix_end = np.where(wavelength < l2)[0][-1]

     w_start = wavelength[ix_start]
     w_end = wavelength[ix_end]

     lamRange = [w_start, w_end]

     galaxy, logLam1, velscale = util.log_rebin(lamRange, spec)

     wavelength_log = np.exp(logLam1)   

     logLam2, templates = utilities.get_templates(velscale, l1, l2)
     dv = (logLam2[0] - logLam1[0]) * c 

     df_columns = ['slit_id','rv_ppxf','rv_ppxf_err','sigma_ppxf','sigma_ppxf_err', 'seeing', 'vhelio']
     final_frame = pd.DataFrame(columns = df_columns)               
       
     # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        
     for i in tqdm(range(0,len(result))):

         single = result.iloc[i]

         hdu = fits.open(single['ORIGINALFILE'])

         header = hdu[0].header  
         galaxy,_,_ = util.log_rebin(lamRange, hdu[1].data[ix_start: ix_end], velscale = velscale)
         noise,_,_ = util.log_rebin(lamRange, hdu[2].data[ix_start: ix_end], velscale = velscale)
         noise = noise + 0.001 # Some spectra have noise = 0
         twoD = hdu[3].data
         thumbnail = hdu[4].data

         hdu.close()
                
         seeing = header['ESO TEL IA FWHMLINOBS']   # Delivered seeing on IA detector  
         slit_id = single['ID']
         quadrant = slit_id[2]
         rv_fxcor = single['VREL']
         sg = header['SG_NORM']
         sn1 = header['SN1']
         sn2 = header['SN2']
         RA_g = single['RA_g']
         DEC_g = single['DEC_g']

         vhelio = utilities.get_vhelio(header)               # Calculate heliocentric velocity 
         start = [rv_fxcor, 10]

         if (quadrant == '1') | (quadrant == '3'):
             pixelmask = pixelmask_Q1Q3
         else:
             pixelmask = pixelmask_normal

         goodpixels = utilities.make_goodpixels(pixelmask, logLam1, logLam2)

         rv, sigma, error_rv, error_sigma, chi2, bestfit, residuals, zp1, output_wavelengths = run_ppxf(wavelength_log, templates, galaxy, noise, velscale, start, goodpixels, dv)

         if plot:

             kwargs = {'output_wavelengths': output_wavelengths,
                       'ix_start': ix_start,
                       'ix_end': ix_end,
                       'slit_id': slit_id,
                       'rv': rv,
                       'sigma': sigma,
                       'chi2': chi2,
                       'pixelmask': pixelmask,
                       'zp1': zp1,
                       'rv_fxcor': rv_fxcor,
                       'sg' : sg,
                       'sn1': sn1,
                       'sn2': sn2,
                       'RA_g': RA_g,
                       'DEC_g': DEC_g}

             figure = ppplot.ppxfplot(twoD, galaxy, noise, bestfit, residuals, thumbnail, **kwargs)     
             figure.savefig('/Volumes/VINCE/OAC/ppxf/figures/{}.png'.format(slit_id))  
             plt.close(figure)

         df_tmp = pd.DataFrame([[slit_id, rv, error_rv, sigma, error_sigma, seeing, vhelio]], columns = df_columns)    
         final_frame = final_frame.append(df_tmp, ignore_index=True)

         final_frame.to_csv('ppxf_results.csv', index = False)
                    
         if nsimulations:

             samples = []
             rv_mc = []

             for j in xrange(nsimulations):
                 noise_mc = noise * np.random.normal(size = noise.size)
                 galaxy_mc = bestfit + noise_mc

                 samples.append((wavelength_log, templates, galaxy_mc, noise, velscale, start, goodpixels, dv))
                 rv, sigma, chi2, error_rv, error_sigma, bestfit, residuals, zp1, output_wavelengths = run_ppxf(wavelength_log, templates, galaxy_mc, noise, velscale, start, goodpixels, dv)
                 rv_mc.append(rv)


                         #workers = min(multiprocessing.cpu_count(), 2)
                         #pool = multiprocessing.Pool(processes=workers) 

                         #print 'Using', workers, 'workers'

                         #sample_results = pool.map(_boot_VIMOS_ppxf, samples)

                         #print 'past sample_results'
     return final_frame

if __name__ == '__main__':
    
     t = time.clock()
         
     l1 = 4950         # Start wavelength
     l2 = 8800         # End wavelength
     factor = 40
     nsimulations = 20
        
     results = VIMOS_ppxf(l1, l2, start, plot=True, nsimulations = False)     
     results.to_csv('results2.csv', index = False)

     print 'Done in {} minutes'.format((time.clock() - t)/60.)









