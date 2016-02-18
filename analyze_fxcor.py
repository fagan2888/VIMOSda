from pyraf import iraf
import glob
import pandas as pd
from tqdm import *
import matplotlib.gridspec as gridspec
from astropy.io.fits import getval, getheader, getdata
from matplotlib import rc

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)

fmatch = pd.read_table('/Volumes/VINCE/OAC/master_adp_ugri_clean.txt', delim_whitespace = True)
fmatch = fmatch.replace((-9999.0, -8888.0), np.nan)

def change_name(list, suff):
  names = [list[i].split('/')[-1] for i in range(len(list))]
  return [names[i].split('.')[0] + suff for i in range(len(list))]

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

def fxcor_subplot(result):
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
        y = result.VREL
        R_vel.scatter(x, y, s=10, c='gray', edgecolor='none', alpha = 0.6)
        R_vel.set_xlim(1,20)
        R_vel.set_ylim(-2000,5000)
        R_vel.set_ylabel(r'$v$ $[km \, s^{-1}]$')
        plt.setp(R_vel.get_yticklabels()[0], visible=False)  
        
        
        y = result.VERR
        R_err.set_ylim(2,80)
        R_err.set_xlim(1,20)
        R_err.set_ylabel(r'$\delta v$ $[km \, s^{-1}]$')
        R_err.set_xlabel(r'TDR')
        R_err.scatter(x, y,s=10, c='gray', edgecolor='none', alpha = 0.6)
        plt.setp(R_err.get_yticklabels()[-1], visible=False)
        
        R_hist.hist(result.TDR, range = (1,20), bins = 50, normed=True,
                                 color='black', histtype='step')
        return fig

# - - - - - - - - - - - - - - - - - - - - - - - 

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
          
          grouped = pd.DataFrame(columns = append(cat.columns, ['OUTLIERS', 'SCATTER']))
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
          result['SN1'] = result['ORIGINALFILE'].map(lambda x: getval(x, 'SN1'))
          result['SG'] = result['ORIGINALFILE'].map(lambda x: getval(x, 'sg_norm'))
          
          resultClean = result[np.isfinite(result['RA_g'])]

          return resultClean

# - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - 

result = get_fresult('teretere.txt', fmatch)

tmp = result.copy()
GCs = tmp[(tmp['VREL'] > 550) & 
             (tmp['VREL'] < 2500) &
             (tmp['VERR'] > 10) & 
             (tmp['VERR'] < 150) & 
             (tmp['TDR'] > 2) & 
             ((tmp['g_auto'] - tmp['i_auto']) < 2.) &
             ((tmp['g_auto'] - tmp['i_auto']) > 0.) & 
             (tmp['i_auto'] > 19.)].reset_index(drop = True)

iraf.onedspec()
for i in tqdm.tqdm(range(0,len(GCs))):
    f = '/Volumes/VINCE/OAC/extracted_new/' + str(GCs.ID[i]) + '_' +str(int(GCs.DETECT[i])) + '.fits[1]'
    output = '/Volumes/VINCE/OAC/fxcor/datad/' + str(GCs.ID[i]) + '_' +str(int(GCs.DETECT[i])) + '_z.fits'
    iraf.onedspec.dopcor(input = f, output = output, redshift = str(GCs.VREL[i]), 
                         isvelocity = 'yes', dispersion = 'yes')

# - - - - - - - - - - - - - - - - - - - - - - - 

stars = result[(result['VREL'] < 300) & (result['VREL'] > -300) &
               (result['VERR'] < 100) &
               (result['VERR'] > 0) &
               (result['TDR'] > 2)]
stars.to_csv('cat_stars.csv', index = False)
stars[['RA_g', 'DEC_g']].to_csv('stars_RADEC.reg', index = False, header = None)

# - - - - - - - - - - - - - - - - - - - - - - - 

plt.figure()
plt.scatter(GCs['g_auto'] - GCs['i_auto'], GCs['VREL'], s=13, c='red', edgecolor='none')
plt.scatter(stars['g_auto'] - stars['i_auto'], stars['VREL'], s=13, c='green', edgecolor='none')
plt.xlabel('(g - i)')
plt.ylabel('VREL')

plt.figure()
plt.scatter(fmatch['g_auto'] - fmatch['i_auto'], fmatch['g_auto'] - fmatch['r_auto'], s=10, c='gray', edgecolor='none', alpha = 0.5)
plt.scatter(GCs['g_auto'] - GCs['i_auto'], GCs['g_auto'] - GCs['r_auto'], s=13, c='red', edgecolor='none')
plt.scatter(stars['g_auto'] - stars['i_auto'], stars['g_auto'] - stars['r_auto'], s=13, c='green', edgecolor='none')
plt.xlabel('(g - i)')
plt.ylabel('(g - r)')

plt.figure()
plt.scatter(fmatch['g_auto'] - fmatch['r_auto'], fmatch['i_auto'], s=20, c='gray', edgecolor='none', alpha = 0.5)
plt.scatter(GCs['g_auto'] - GCs['r_auto'], GCs['i_auto'], s=15, c='red', edgecolor='none')
plt.scatter(stars['g_auto'] - stars['r_auto'], stars['i_auto'], s=15, c='green', edgecolor='none')
plt.ylim(13,24)
plt.gca().invert_yaxis()
plt.xlabel('(g - i)')
plt.ylabel('i')

# - - - - - - - - - - - - - - - - - - - - - - - 

from astropy import units as u
from astropy import coordinates as coords
from astropy.coordinates import SkyCoord

S_GCs =  pd.read_table('/Volumes/VINCE/OAC/fxcor/SchuberthGCs.csv', sep=r'\s+').replace(99.99, np.nan)
S_GCs = S_GCs.drop_duplicates(['Rmag', 'C-R','HRV']).reset_index(drop = True)

S_Stars =  pd.read_table('/Volumes/VINCE/OAC/fxcor/SchuberthStars.csv', sep=r'\s+').replace(99.99, np.nan)
S_Stars = S_Stars.drop_duplicates(['Rmag', 'C-R', 'HRV']).reset_index(drop = True)

cat1 = coords.SkyCoord(GCs['RA_g'], GCs['DEC_g'], unit=(u.degree, u.degree))
cat2 = coords.SkyCoord(S_GCs['RA'], S_GCs['DEC'], unit=(u.degree, u.degree))

index,dist2d, _ = cat1.match_to_catalog_sky(cat2)
mask = dist2d.arcsec < 0.4
new_idx = index[mask]

GCsMatch = GCs.ix[mask].reset_index(drop = True)
SchuberthMatch = S_GCs.ix[new_idx].reset_index(drop = True)

x = GCsMatch['VREL'] 
xerr = GCsMatch['VERR'] 
y = SchuberthMatch['HRV'] 
yerr = SchuberthMatch['e.1'] 

plt.errorbar(x, y, yerr= yerr, xerr = xerr, fmt = 'o')
plt.plot([800, 2200], [800, 2200])

# ---------------------
nicola = pd.read_csv('N1387_check_vel_stream.csv')
resultClean.rename(columns = {'RA_g': 'RA', 'DEC_g': 'DEC'}, inplace = True)
nicolaM = pd.merge(nicola, GCs, on = ['RA','DEC'])

