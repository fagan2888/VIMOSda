from pyraf import iraf
import glob
import pandas as pd
from tqdm import *
import matplotlib.gridspec as gridspec
from matplotlib import rc
import utilities 
from scipy.ndimage.filters import gaussian_filter 
from astropy.io import fits
from pyraf import iraf
import os
iraf.noao()
import functools
import operator
import numpy as np
from matplotlib.pylab import plt

fmatch = pd.read_table('/Volumes/VINCE/OAC/master_adp_ugri_clean.txt', delim_whitespace = True)
fmatch = fmatch.replace((-9999.0, -8888.0), np.nan)
par = [54.620941, -35.450657, 64, 0.9]

# - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - 

def mad(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score > thresh

# - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - 

residuals = pd.read_csv('/Volumes/VINCE/OAC/VIMOSda/residuals.txt', 
                          delim_whitespace = True, comment = '#')
residuals.index = residuals.Mask

def correct_vel(maskid, multiplex):
    if multiplex == 0:
        rms = residuals.loc[maskid]['M1']
    else:
        rms = residuals.loc[maskid]['M3']
    return (rms * 90.4)

def fxcor_subplot(result, GCs, stars):
        '''
        Makes subplot of TDR vs VERR and VREL.
        Returns a figure object
        '''
        plt.close('all')
        fig = plt.figure(figsize=(6,6))
        gs = gridspec.GridSpec(70,40,bottom=0.10,left=0.15,right=0.98, top = 0.95)

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        R_vel = plt.subplot(gs[10:40,0:40])
        R_err = plt.subplot(gs[40:70,0:40])
        R_hist = plt.subplot(gs[0:10,0:40])
        
        R_hist.axis('off')
        plt.setp(R_vel.get_xticklabels(), visible=False)
        
        x = result.TDR
        y = result.VREL_helio
        R_vel.scatter(x, y, s=10, c='gray', edgecolor='none', alpha = 0.6, label = 'All')

        x = GCs.TDR
        y = GCs.VREL_helio
        R_vel.scatter(x, y, s=11, c='orange', edgecolor='none', alpha = 0.8, label = 'GCs')

        x = stars.TDR
        y = stars.VREL_helio
        R_vel.scatter(x, y, s=11, c='green', edgecolor='none', alpha = 0.8, label = 'Stars')

        R_vel.set_xlim(1,20)
        R_vel.set_ylim(-2000,5000)
        R_vel.set_ylabel(r'$v$ $[km \, s^{-1}]$')
        plt.setp(R_vel.get_yticklabels()[0], visible=False)  
        
        x = result.TDR
        y = result.VERR
        R_err.scatter(x, y,s=10, c='gray', edgecolor='none', alpha = 0.6)

        x = GCs.TDR
        y = GCs.VERR
        R_err.scatter(x, y,s=11, c='orange', edgecolor='none', alpha = 0.8)

        x = stars.TDR
        y = stars.VERR
        R_err.scatter(x, y,s=11, c='green', edgecolor='none', alpha = 0.8)

        R_err.set_ylim(2,80)
        R_err.set_xlim(1,20)
        R_err.set_ylabel(r'$\delta v$ $[km \, s^{-1}]$')
        R_err.set_xlabel(r'TDR')
        plt.setp(R_err.get_yticklabels()[-1], visible=False)
        R_vel.legend()
        
        R_hist.hist([GCs.TDR,stars.TDR], range = (1,20), bins = 50, normed=True,
                                 color=['orange','green'])
        return fig

def fetchfrom_fits(fitsfile):
        '''
        Open the fits header. Gets the S/N, the SG parameter
        and calculate the heliocentric velocity. 
        Returns a tuple. 
        '''
        header = fits.getheader(fitsfile, hdu = 0)
        SN1 = header['SN1']
        SG = header['sg_norm']
        SN2 = header['SN2']
        MULTIPLEX = header['MULTIPLEX']

        vhelio = utilities.get_vhelio(header)

        output = [SN1, SN2, SG, vhelio, MULTIPLEX]

        return output

def getXYR(ra,dec, par):

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

    X = -xi * np.sin(PA) - eta*np.cos(PA)
    Y = -xi * np.cos(PA) + eta*np.sin(PA)
    
    R = np.sqrt((Y**2 / axial) + (X**2 * axial))
        
    return X, Y, R

def get_fresult(filename, fmatch):

          f = pd.read_table('/Volumes/VINCE/OAC/fxcor/{}'.format(filename), comment = '#', delim_whitespace =True, header = None)
          
          f.columns = ['OBJECT', 'IMAGE', 'REF', 'HJD', 'AP', 'CODES', 'SHIFT', 
                           'HGHT', 'FWHM', 'TDR','VOBS','VREL','VHELIO','VERR']
          
          f['ID'] = f['IMAGE'].str.split('[/._]').str[-4]              # get the slit ID
          f['DETECT'] = f['IMAGE'].str.split('[/._]').str[-3]          # get the object detection per slit
          f.drop(['REF', 'OBJECT' , 'HJD', 'AP', 'CODES' , 'VOBS', 'VHELIO'], axis=1, inplace=True) # Drop unimportant columns 
          
          cat = f.replace('INDEF', np.nan) # Fill INDEF with a missing value
          cat[['SHIFT', 'HGHT', 'VERR', 'TDR','FWHM']] = cat[['SHIFT', 'HGHT', 'VERR', 'TDR','FWHM']].astype(float) # Transform to float (currently strings)
          cat['DETECT'] = cat['DETECT'].astype(int)
          
          # Fill missing values with the median of the nonnull values
          groupcat = cat.groupby(['ID', 'DETECT'])

          cat['VERR'] = groupcat['VERR'].transform(lambda x: x.fillna(x.median())) 
          cat['FWHM'] = groupcat['FWHM'].transform(lambda x: x.fillna(x.median())) 
          cat['TDR']  = groupcat['TDR'].transform(lambda x: x.fillna(x.median()))
          
          # Fill rows with null TDR and/or VERR with 0.1
          cat['VERR'] = cat['VERR'].fillna(0.1) 
          cat['TDR'] = cat['TDR'].fillna(0.1) 
          cat['FWHM'] = cat['FWHM'].fillna(0.1) 

          cat['ORIGINALFILE'] = '/Volumes/VINCE/OAC/extracted_new/' + cat.ID + '_' + cat.DETECT.astype(str) + '.fits'
          
          grouped = pd.DataFrame(columns = np.append(cat.columns, ['OUTLIERS', 'SCATTER']))
          tempnum = 40
          
          for i in tqdm(range(0,(len(cat)/tempnum))):
           
            j = i * tempnum  # Indexes of the groups of len tepnum
            
            vrad = cat['VREL'].loc[0+j:tempnum - 1 +j]
            verr = cat['VERR'].loc[0+j:tempnum - 1 +j]
          
            toremove = mad(vrad)
            
            vradcorr = vrad[~toremove]
            verrcorr = verr[~toremove]
          
            outliers = tempnum - len(vradcorr)
          
            grouped.loc[i] = pd.Series({'ID':cat['ID'].loc[j], 
                                        'DETECT':cat['DETECT'].loc[j],
                                        'SHIFT': np.median(cat['SHIFT'].loc[0+j:tempnum - 1 +j]),
                                        'IMAGE': cat['IMAGE'].loc[j],
                                        'FWHM': np.median(cat['FWHM'].loc[j]),
                                        'TDR': np.median(cat['TDR'].loc[j]),
                                        'HGHT': np.median(cat['HGHT'].loc[j]),
                                        'VREL': np.median(vradcorr),
                                        'VERR': np.sqrt(np.median(verrcorr)**2 + np.std(vradcorr)**2),
                                        'OUTLIERS': outliers,
                                        'ORIGINALFILE': cat['ORIGINALFILE'].loc[j],
                                        'SCATTER': np.std(vradcorr)
                                        })
  
          result = pd.merge(grouped, fmatch, on = ['ID'])

          print 'Computing heliocentric velocity . . .'
          toadd = np.array(map(lambda x: fetchfrom_fits(x), result['ORIGINALFILE']))
          
          result['SN1'] = toadd[:,0]
          result['SN2'] = toadd[:,1]

          print 'Appending SG classifier . . .'
          result['SG'] = toadd[:,2] 

          print 'Appending Heliocentric velocity correction . . .'
          result['vhelio'] = toadd[:,3]
          result['MULTIPLEX'] = toadd[:,4]
          
          print 'Appending misalignment correction . . .'
          result['maskid'] = pd.Series('fnx' + result.ID.str[0:2] + 'q' + result.ID.str[2])
          result['correction'] = result.apply(lambda x: correct_vel(x['maskid'], x['MULTIPLEX']), axis=1)
          
          result['VREL_helio'] = result['VREL'] + result['vhelio'] - result['correction']
          XYR = result.apply(lambda x: getXYR(x['RA_g'], x['DEC_g'], par), axis=1).values
          
          X = []
          Y = []
          R = []

          for i in range(0,len(XYR)):
            X.append(XYR[i][0])
            Y.append(XYR[i][1])
            R.append(XYR[i][2])

          result['X'] = X
          result['Y'] = Y
          result['R'] = R

          resultClean = result[np.isfinite(result['RA_g'])]

          resultClean['obj_ID'] = resultClean.ID + '_' + resultClean.DETECT.astype(int).astype(str)


          return resultClean

def Michele_spectra(GCssorted):
  for i in range(0,100):
    hdu = fits.open(GCssorted['ORIGINALFILE'].iloc[i])
    header1 = hdu[1].header
    data = hdu[1].data
    hdu.close()
    lamRange = header1['CRVAL1']  + np.array([0., header1['CD1_1'] * (header1['NAXIS1'] - 1)])
    zp = 1. + (GCssorted['VREL'].iloc[i] / 299792.458)
    wavelength = np.linspace(lamRange[0],lamRange[1], header1['NAXIS1']) / zp

    df = pd.DataFrame({'wavelength':wavelength, 'counts':data})
    df.to_csv('/Volumes/VINCE/OAC/highSN/{}.csv'.format(GCssorted['ID'].iloc[i]))
    plt.plot(wavelength,data)
  return df  

def plot_spectrum(result, correct = True, interactive = False):
    
    plt.close('all')
    plt.ioff()

    if interactive:
      plt.ion()

    hdu = fits.open(result['ORIGINALFILE'])
    galaxy = gaussian_filter(hdu[1].data, 1)
    thumbnail = hdu['THUMBNAIL'].data
    twoD = hdu['2D'].data
    header = hdu[0].header
    header1 = hdu[1].header
    hdu.close()

    lamRange = header1['CRVAL1']  + np.array([0., header1['CD1_1'] * (header1['NAXIS1'] - 1)]) 
    
    if correct:
      zp = 1. + (result['VREL'] / 299792.458)
    else:
      zp = 1.

    wavelength = np.linspace(lamRange[0],lamRange[1], header1['NAXIS1']) / zp
    ymin, ymax = np.min(galaxy), np.max(galaxy)
    ylim = [ymin, ymax] + np.array([-0.02, 0.1])*(ymax-ymin)
    ylim[0] = 0.

    xmin, xmax = np.min(wavelength), np.max(wavelength)

    ### Define multipanel size and properties
    fig = plt.figure(figsize=(8,6))
    gs = gridspec.GridSpec(200,130,bottom=0.10,left=0.10,right=0.95)

    ### Plot the object in the sky
    ax_obj = fig.add_subplot(gs[0:70,105:130])
    ax_obj.imshow(thumbnail, cmap = 'gray', interpolation = 'nearest')
    ax_obj.set_xticks([]) 
    ax_obj.set_yticks([]) 

    ### Plot the 2D spectrum
    ax_2d = fig.add_subplot(gs[0:11,0:100])
    ix_start = header['START_{}'.format(int(result['DETECT']))]
    ix_end = header['END_{}'.format(int(result['DETECT']))]
    ax_2d.imshow(twoD, cmap='spectral',
                aspect = "auto", origin = 'lower', extent=[xmin, xmax, 0, 1], 
                vmin = -0.2, vmax=0.2) 
    ax_2d.set_xticks([]) 
    ax_2d.set_yticks([]) 
    
    ### Add spectra subpanels
    ax_spectrum = fig.add_subplot(gs[11:85,0:100])
    ax_blue = fig.add_subplot(gs[110:200,0:50])
    ax_red = fig.add_subplot(gs[110:200,51:100])
    
    ### Plot some atomic lines  
    line_wave = [4861., 5175., 5892., 6562.8, 8498., 8542., 8662.] 
    #           ['Hbeta', 'Mgb', 'NaD', 'Halpha', 'CaT', 'CaT', 'CaT']
    for i in range(len(line_wave)):
        x = [line_wave[i], line_wave[i]]
        y = [ylim[0], ylim[1]]
        ax_spectrum.plot(x, y, c= 'gray', linewidth=1.0)
        ax_blue.plot(x, y, c= 'gray', linewidth=1.0)
        ax_red.plot(x, y, c= 'gray', linewidth=1.0)

    ### Plot the spectrum 
    ax_spectrum.plot(wavelength, galaxy, 'k', linewidth=1.3)
    ax_spectrum.set_ylim(ylim)
    ax_spectrum.set_xlim([xmin,xmax])
    ax_spectrum.set_ylabel(r'Arbitrary Flux')
    ax_spectrum.set_xlabel(r'Restframe Wavelength [ $\AA$ ]')
    
    ### Plot blue part of the spectrum
    x1, x2 = 300, 750 
    ax_blue.plot(wavelength[x1:x2], galaxy[x1:x2], 'k', linewidth=1.3)
    ax_blue.set_xlim(wavelength[x1],wavelength[x2])
    ax_blue.set_ylim(galaxy[x1:x2].min(), galaxy[x1:x2].max())
    ax_blue.set_yticks([]) 
    
    ### Plot red part of the spectrum
    x1, x2 = 1400, 1500
    ax_red.plot(wavelength[x1:x2], galaxy[x1:x2], 'k', linewidth=1.3)
    ax_red.set_xlim(wavelength[x1],wavelength[x2])
    ax_red.set_ylim(galaxy[x1:x2].min(), galaxy[x1:x2].max())
    ax_red.set_yticks([]) 

    ### Plot text
    #if interactive:
    textplot = fig.add_subplot(gs[80:200,105:130])
    kwarg = {'va' : 'center', 'ha' : 'left', 'size' : 'medium'}
    textplot.text(0.1, 1.0,r'ID = {} \, {}'.format(result.ID, int(result.DETECT)),**kwarg)
    textplot.text(0.1, 0.9,r'$v =$ {}'.format(int(result.VREL)), **kwarg)
    textplot.text(0.1, 0.8,r'$\delta \, v = $ {}'.format(int(result.VERR)), **kwarg)
    textplot.text(0.1, 0.7,r'SN1 = {0:.2f}'.format(result.SN1), **kwarg)
    textplot.text(0.1, 0.6,r'SN2 = {0:.2f}'.format(result.SN2), **kwarg)
    textplot.text(0.1, 0.5,r'TDR = {0:.2f}'.format(result.TDR), **kwarg)
    textplot.text(0.1, 0.4,r'SG = {}'.format(result.SG), **kwarg)
    textplot.axis('off')

    return fig

def plot_piledspectra():
    fig = plt.figure(figsize = (6,8)) 
    plt.xlim(5000,9000)
    
    specindex = range(0,100,10)
    offset = np.arange(0,len(specindex)) * 0.5
    ylim = [0.5, offset[-1] + 1.3]
    plt.ylim(ylim[0], ylim[1])
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    plt.xlabel(r'Restrame Wavelength [ \AA\ ]')
    plt.ylabel(r'Flux')
    
    line_wave = [5175., 5892., 6562.8, 8498., 8542., 8662.] 
        #       ['Mgb', 'NaD', 'Halpha', 'CaT', 'CaT', 'CaT']
    for line in line_wave:
            x = [line, line]
            y = [ylim[0], ylim[1]]
            plt.plot(x, y, c= 'gray', linewidth=1.0)
    
    plt.annotate(r'CaT', xy=(8540.0, ylim[1] + 0.05), xycoords='data', annotation_clip=False)
    plt.annotate(r'H$\alpha$', xy=(6562.8, ylim[1] + 0.05), xycoords='data', annotation_clip=False)
    plt.annotate(r'NaD', xy=(5892., ylim[1] + 0.05), xycoords='data', annotation_clip=False)
    plt.annotate(r'Mg$\beta$', xy=(5175., ylim[1] + 0.05), xycoords='data', annotation_clip=False)
    
    for i,j in zip(specindex,offset):
        iraf.noao.onedspec.continuum(input = GCssorted.ORIGINALFILE.iloc[i] + '[1]', output = '/Volumes/VINCE/OAC/continuum.fits',
            type = 'ratio', naverage = '3', function = 'spline3',
            order = '5', low_reject = '2.0', high_reject = '2.0', niterate = '10')
    
        data = fits.getdata('/Volumes/VINCE/OAC/continuum.fits', 0)
        
        hdu = fits.open(GCssorted.ORIGINALFILE.iloc[i])
        header1 = hdu[1].header
        lamRange = header1['CRVAL1']  + np.array([0., header1['CD1_1'] * (header1['NAXIS1'] - 1)]) 
        wavelength = np.linspace(lamRange[0],lamRange[1], header1['NAXIS1'])
        hdu.close()
    
        zp = 1. + (GCssorted.VREL.iloc[i] / 299792.458)
      
        plt.plot(wavelength/zp, gaussian_filter(data,2) + j, c = 'black', lw=1)
        os.remove('/Volumes/VINCE/OAC/continuum.fits')

# - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - 

#result = get_fresult('bobo.txt', fmatch)
#result.to_csv('/Volumes/VINCE/OAC/result.csv', index=False)

result = pd.read_csv('/Volumes/VINCE/OAC/result.csv')

tmp = result.copy()
candidates = tmp[(tmp['VREL'] > 450) & 
             (tmp['VREL'] < 2500) &
             (tmp['VERR'] > 10) & 
             (tmp['VERR'] < 150) & 
             (tmp['TDR'] > 2) & 
             ((tmp['g_auto'] - tmp['i_auto']) < 2.) &
             ((tmp['g_auto'] - tmp['i_auto']) > 0.) & 
             (tmp['i_auto'] > 19.)].reset_index(drop = True)

#classify = []
#for i in range(0,len(candidates)):
#  fig = plot_spectrum(candidates.iloc[i], correct = True, interactive = True)
#  plt.show()
#  classify.append(raw_input())
#  fig.savefig('/Volumes/VINCE/OAC/spectra/{}_{}.png'.format(candidates.ID.iloc[i],candidates.DETECT.iloc[i]), dpi = 200)

classify = pd.read_csv('/Volumes/VINCE/OAC/classify_GCs_total.csv')
classify.columns = ['obj_ID', 'flag'] 
candidates = pd.merge(candidates,classify,on='obj_ID')
marginals = candidates[candidates['flag'] == 'm']

# - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - 

stars_candidates = result[(result['VREL'] < 450) & (result['VREL'] > -450) &
               (result['VERR'] < 100) &
               (result['VERR'] > 5) &
               (result['TDR'] > 2)].reset_index(drop = True)

#classify_stars_candidates = []
#for i in range(0,len(stars_candidates)):
#  fig = plot_spectrum(stars_candidates.iloc[i], correct = True, interactive = True)
#  plt.show()
#  classify_stars_candidates.append(raw_input())

classify_stars_candidates = pd.read_csv('/Volumes/VINCE/OAC/classify_stars_all.csv')
classify_stars_candidates.columns = ['obj_ID', 'flag'] 
stars_candidates = pd.merge(stars_candidates,classify_stars_candidates,on='obj_ID')

# - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - 

GCs = candidates[candidates.flag == 'g']
GCssorted = GCs.sort_values('SN1', ascending = False)
stars = stars_candidates[stars_candidates.flag == 's']

# - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - 

f, ax = plt.subplots(2, 3, sharex='col', sharey='row')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

ax[0][0].scatter(result['g_auto'] - result['r_auto'], result['g_auto'] - result['i_auto'], 
                s=5, c='gray', edgecolor='none', alpha = 0.5)
ax[0][0].scatter(GCs['g_auto'] - GCs['r_auto'], GCs['g_auto'] - GCs['i_auto'], 
                s=13, c='red', edgecolor='none')
ax[0][0].scatter(stars['g_auto'] - stars['r_auto'], stars['g_auto'] - stars['i_auto'], 
                s=10,  c = 'green', edgecolor='none')
ax[0][0].set_xlabel('(g - r)')
ax[0][0].set_ylabel('(g - i)')
ax[0][0].set_xlim(-0.2,2)
ax[0][0].set_ylim(0,3)


ax[1][0].scatter(result['g_auto'] - result['r_auto'], result['u_auto'] - result['g_auto'], 
                s=5, c='gray', edgecolor='none', alpha = 0.5)
ax[1][0].scatter(GCs['g_auto'] - GCs['r_auto'], GCs['u_auto'] - GCs['g_auto'], 
                s=13, c='red', edgecolor='none')
ax[1][0].scatter(stars['g_auto'] - stars['r_auto'], stars['u_auto'] - stars['g_auto'], 
                s=10, c='green', edgecolor='none')
ax[1][0].set_xlabel('(g - r)')
ax[1][0].set_ylabel('(u - g)')
ax[1][0].set_xlim(-0.2,2)
ax[1][0].set_ylim(0.,3)


ax[0][1].scatter(result['i_auto'], result['g_auto'] - result['i_auto'], 
                s=5, c='gray', edgecolor='none', alpha = 0.5)
ax[0][1].scatter(GCs['i_auto'], GCs['g_auto'] - GCs['i_auto'], 
                s=13, c='red', edgecolor='none')
ax[0][1].scatter(stars['i_auto'], stars['g_auto'] - stars['i_auto'], 
                s=10, c='green', edgecolor='none')
ax[0][1].set_xlabel('i')
ax[0][1].set_ylabel('(g - i)')
ax[0][1].set_xlim(16,23.)
ax[0][1].set_ylim(0.,3)

ax[1][1].scatter(result['i_auto'], result['u_auto'] - result['g_auto'], 
                s=5, c='gray', edgecolor='none', alpha = 0.5)
ax[1][1].scatter(GCs['i_auto'], GCs['u_auto'] - GCs['g_auto'], 
                s=13, c='red', edgecolor='none')
ax[1][1].scatter(stars['i_auto'], stars['u_auto'] - stars['g_auto'], 
                s=10, c='green',  edgecolor='none')
ax[1][1].set_xlabel('i')
ax[1][1].set_ylabel('(u - g)')
ax[1][1].set_xlim(16,23)
ax[1][1].set_ylim(0.5,3)

ax[1][2].scatter(GCs['VREL_helio'], GCs['u_auto'] - GCs['g_auto'], s=13, c='red', edgecolor='none')
ax[1][2].scatter(stars['VREL_helio'], stars['u_auto'] - stars['g_auto'], s=13, c='green', marker='x', edgecolor='none')
ax[1][2].set_xlabel(r'$v [km s^{-1}]$')
ax[1][2].set_ylabel('(u - g)')
ax[1][2].set_xticks(np.arange(-500, 2600,500))

ax[0][2].scatter(GCs['VREL_helio'], GCs['g_auto'] - GCs['i_auto'], s=13, c='red', edgecolor='none')
ax[0][2].scatter(stars['VREL_helio'], stars['g_auto'] - stars['i_auto'], s=13, c='green', marker='x', edgecolor='none')
ax[0][2].set_xlabel(r'$v [km s^{-1}]$')
ax[0][2].set_ylabel('(g - i)')
ax[0][2].set_xticks(np.arange(-500, 2600,500))

# - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - 

x = GCs['R'].copy().sort_values()
y = np.arange(1,len(GCs)+1)
plt.plot(x,y,label='GCs')

x = stars['R'].copy().sort_values()
y = np.arange(1,len(stars)+1)
plt.plot(x,y,label='Stars')

plt.legend()

# - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - 


mask = (((result['g_auto'] - result['r_auto']) < (0.2 + 0.6 * (result['g_auto'] - result['i_auto']))) &
        ((result['g_auto'] - result['r_auto']) > (-0.2 + 0.6 * (result['g_auto'] - result['i_auto']))) &
        ((result['g_auto'] - result['i_auto']) > 0.5) & 
        ((result['g_auto'] - result['i_auto']) < 1.3) &
        ((result['i_auto']) < 24))

subset = result[mask]
subset = subset.sample(n=1000)

plt.figure()
plt.scatter(result['g_auto'] - result['i_auto'], result['g_auto'] - result['r_auto'], s=10, c='gray', edgecolor='none', alpha = 0.5)
plt.scatter(subset['g_auto'] - subset['i_auto'], subset['g_auto'] - subset['r_auto'], s=20, c='blue', edgecolor='none')
plt.scatter(GCs['g_auto'] - GCs['i_auto'], GCs['g_auto'] - GCs['r_auto'], s=10, c='red', edgecolor='none')
plt.xlabel('(g - i)')
plt.ylabel('(g - r)')
plt.xlim(-1,4)
plt.ylim(-1,4)

plt.figure()
plt.scatter(subset['g_auto'] - subset['r_auto'], subset['r_auto'], s=30, c='blue', edgecolor='none')
plt.scatter(GCs['g_auto'] - GCs['r_auto'], GCs['i_auto'], s=8, c='red', edgecolor='none')
plt.ylim(13,24)
plt.gca().invert_yaxis()
plt.xlabel('(g - i)')
plt.ylabel('i')

plt.figure()
plt.scatter(result['g_auto'] - result['u_auto'], result['g_auto'] - result['r_auto'], s=10, c='gray', edgecolor='none', alpha = 0.5)
plt.scatter(subset['g_auto'] - subset['u_auto'], subset['g_auto'] - subset['r_auto'], s=20, c='blue', edgecolor='none')
plt.scatter(GCs['g_auto'] - GCs['u_auto'], GCs['g_auto'] - GCs['r_auto'], s=10, c='red', edgecolor='none')
plt.scatter(galaxies['g_auto'] - galaxies['u_auto'], galaxies['g_auto'] - galaxies['r_auto'], s=10, c='green', edgecolor='none')
plt.xlabel('(g - u)')
plt.ylabel('(g - r)')
plt.xlim(-4,3)
plt.ylim(-1,2)


for i in tqdm(range(0,len(subset))):
  fig = plot_spectrum(subset.iloc[i], correct = True, interactive = False)
  fig.savefig('/Volumes/VINCE/OAC/class_test/{}_{}.png'.format(subset.ID.iloc[i],subset.DETECT.iloc[i]), dpi = 300)

dir = '/Volumes/VINCE/OAC/class_test/' 
groups = [dir + 'group_1',dir + 'group_2',dir + 'group_3',dir + 'group_4',dir + 'group_5',dir + 'group_6']
images = glob.glob('/Volumes/VINCE/OAC/class_test/*png*')

for i,g in enumerate(groups):
  for f in images[200*i : 200*(i+1)]:
    shutil.copy(f,g)


