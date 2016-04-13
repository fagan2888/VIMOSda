from astropy.io import fits
from glob import glob
from astropy.table import Table
from astropy.io.fits import getheader, getdata

import numpy as np
import time
import pandas as pd
import os
from pyraf import iraf
from tqdm import *
from astropy import wcs
from scipy.interpolate import interp1d
from scipy.fftpack import fft, ifft
iraf.noao()


def mad_based_outlier(points, thresh=3.):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def getfiles(start_dir):
    '''
    Reads files containing the data
    '''        
    files, files_error, files_table, files_2d = [], [], [], []
        
    pattern         = ['*MOS_SCIENCE_REDUCED.fits','*MOS_SCI_ERROR_REDUCED.fits',
                       '*OBJECT_SCI_TABLE.fits',  '*MOS_SCIENCE_EXTRACTED.fits' ]

    for dir,_,_ in os.walk(start_dir):
         files.extend(glob(os.path.join(dir,pattern[0])))   
    for dir,_,_ in os.walk(start_dir):
         files_error.extend(glob(os.path.join(dir,pattern[1])))     
    for dir,_,_ in os.walk(start_dir):
         files_table.extend(glob(os.path.join(dir,pattern[2])))    
    for dir,_,_ in os.walk(start_dir):
         files_2d.extend(glob(os.path.join(dir,pattern[3])))
        
    return files, files_error, files_table, files_2d

def fillgaps(wp, data):
    mask = np.array([[6850, 6950], [7560, 7700], [7150, 7350],
            [5565,5590], [5874, 5910], [6290,6312], [6355,6374]])
    for j in mask:
        gap = wp[(wp.wavelength >= j[0]) & (wp.wavelength <= j[1])]
    
        x = [gap.index[0], gap.index[-1]]
        y = [data[x[0]], data[x[1]]]
        f = interp1d(x, y, kind='linear')
    
        data[gap.pixels] = f(gap.pixels)
    return data

def galaxy_star(d):
    smed = pd.rolling_median(d, 5)
    median = np.median(smed)
    outliers = smed[mad_based_outlier(smed,thresh=3.0)]
    return len(outliers[outliers > median])


def fill_continuum(datafile, mask, wavelength):

    iraf.noao.onedspec.continuum(input = datafile + '[1]', output = '/Volumes/VINCE/OAC/extracted_new/continuum.fits',
        type = 'fit', naverage = '3', function = 'spline3',
        order = '5', low_reject = '2.0', high_reject = '2.0', niterate = '10')

    continuum = getdata('/Volumes/VINCE/OAC/extracted_new/continuum.fits')
    os.remove('/Volumes/VINCE/OAC/extracted_new/continuum.fits')
    data = getdata(datafile, 0)
    
    for j in mask:
        gap = np.where((wavelength >= j[0]) & (wavelength <= j[1])) # Select pixels in the gap
        data[gap] = continuum[gap]                                  # Fill the gap

    return data

def modify_header(head, tab, index, keys, row): 
     '''
     Append the entries of the *OBJECT_SCI_TABLE.fits table to the fits header.
     '''     
     for j in range(0, len(keys)):

         keyword = keys[j]
         value = int(tab.values[index][j])

         head.update(keyword, value)
         head.update('OBJECT_NUM', row, 'The objet identification number in this header')

     return head

def cutlow(array, cut = 80):

    x = np.arange(0,len(array))

    w = rfft(array.astype(float))
    spectrum = w**2

    cutoff_idx = spectrum < np.percentile(spectrum, cut)
    w2 = w.copy()
    w2[cutoff_idx] = 0

    return irfft(w2)


def get_thumbnail(hduimg, w, ra, dec, size): 
     '''
     Return a thumbnail image of the object.
     '''     
     center = w.wcs_world2pix([[ra,dec]], 1).flatten()

     x1 = center[0] - size
     x2 = center[0] + size
     y1 = center[1] - size
     y2 = center[1] + size

     data = hduimg[0].data[y1:y2,x1:x2]

     return data


def repeated(files):
    df = pd.DataFrame(columns = ['mask', 'quadrant'])

    for i in range(0,len(files)):
        h = getheader(files[i]) 
        df.loc[i] = pd.Series({'mask' : h['OBJECT'], 
                          'quadrant': h['ESO OCS CON QUAD']})
    return df

def extract(start_dir):
   
   files, files_error, files_table, files_2d = getfiles(start_dir)
   
   # Load the photometric master catalogue
   fmatch = pd.read_table('/Volumes/VINCE/OAC/code/master_adp_ugri_clean.txt', delim_whitespace = True)
   fmatch = fmatch.replace((-9999.0, -8888.0), np.nan)
   fmatch.index = fmatch.ID.astype(str)

   # Load the VST fits image
   hduimg = fits.open('/Volumes/VINCE/OAC/imaging/mosaic_F11_F16_ex.fit')
   w = wcs.WCS(hduimg[0].header)

   # Calculate wavelengths corresponding to each pixel
   h = getheader(files[0], 1)
   lamRange = h['CRVAL1']  + np.array([0., h['CD1_1'] * (h['NAXIS1'] - 1)]) 
   wavelength = np.linspace(lamRange[0],lamRange[1], h['NAXIS1'])

   mask = np.array([[6850, 6950], [7560, 7700], [7150, 7350], [5565,5590], [5874, 5910], [6290,6312], [6355,6374]])

   # Loop through the REFLEX reduced files 
   for f_data, f_error, f_table, f_twod in tqdm(zip(files, files_error, files_table, files_2d)):

     hdu = fits.open(f_data)
     headerP = hdu[0].header 
     quadrant = headerP['ESO OCS CON QUAD']
     mask_name = headerP['OBJECT'][3:5]
     hdu.close()

     qnum = range(1,len(hdu))
     j = 0

     for index in qnum:
         
         header = getheader(f_data, ext = index)

         t = Table.read(f_table, hdu = index)                            # read fits table 
         t = t.filled(-99)                                               # fill masked values with -99
         fitstable = pd.DataFrame.from_records(t, columns=t.colnames)    # transform to dataframe    
         fitstable = fitstable.replace((np.nan, -99), (int(0), int(0)))  # Fills nan and -99 with zeros
                                                                         
         a = fitstable.columns.str.split('_').str[1]      # Split the fitstable entries before and after '_'
         row_num = a.unique()[2:].astype(int)             # Select unique elements and drop the 
                                                          # first two entries ('id', nan)
         keys = fitstable.columns                                    # Get table column names
         keys = [keys[i].upper() for i in range(0,len(keys))]        # Transform to uppercase (the header wants uppercase)
         
         data_all = getdata(f_data, ext = index)                   # Load the flux data
         noise_all = getdata(f_error, ext = index)                 # Load the error data
         j += index                
         twod_all = getdata(f_twod, ext = j)

         for i in range(0, len(fitstable)) :
             
             for row in row_num:
                if fitstable.ix[i]['row_{}'.format(row)] != 0:
                    
                    id = int(fitstable.ix[i]['row_{}'.format(row)])

                    data_original = data_all[id]
                    hduP = fits.PrimaryHDU()
                    hdu1 = fits.ImageHDU(data_original)
                    hdulist = fits.HDUList([hduP, hdu1])  

                    datafile = '/Volumes/VINCE/OAC/extracted_new/spec_tmp.fits'
                    hdulist.writeto(datafile, clobber = True)

                    data_filled = fill_continuum(datafile, mask, wavelength)
                    os.remove(datafile)

                    noise = noise_all[id]

                    sn1 = np.median(data_filled[440:700]/ noise[440:700])
                    if np.isfinite(sn1) == False:
                        sn1 = 0.

                    sn2 = np.median(data_filled[1360:1500]/ noise[1360:1500])
                    if np.isfinite(sn2) == False:
                        sn2 = 0.
                    
                    data_tmp = pd.Series(data_filled)
                    data_tmp = data_tmp[(data_tmp.index < 750) | (data_tmp.index > 1105)]
                    sub = np.split(data_tmp,8)
                    sg_norm = np.array(map(lambda x: galaxy_star(x), sub)).max()

                    # Get the 2D image
                    start = int(fitstable.ix[i]['start_{}'.format(row)])
                    end = int(fitstable.ix[i]['end_{}'.format(row)])
                    twod = twod_all[start - 5 : end + 5, :]

                    # Append data to each HDU
                    hdu0 = fits.PrimaryHDU()    # Empty 
                    hdu1 = fits.ImageHDU(data = data_filled, name = 'filled')  # Data
                    hdu2 = fits.ImageHDU(data = noise, name = 'noise') # Noise
                    hdu3 = fits.ImageHDU(data = twod, name = '2d')  # 2D image

                    hdulist = fits.HDUList([hdu0, hdu1, hdu2, hdu3])  # Create HDU list
             
                    hdulist[0].header = modify_header(headerP, fitstable, i, keys, row)
                    hdulist[1].header = modify_header(header, fitstable, i, keys, row)

                    hdulist[0].header.update('SN1', sn1, 'Signal to Noise')
                    hdulist[1].header.update('SN1', sn1, 'Signal to Noise')
                    hdulist[0].header.update('SN2', sn2, 'Signal to Noise')
                    hdulist[1].header.update('SN2', sn2, 'Signal to Noise')
                    hdulist[0].header.update('sg_norm', sg_norm, 'Number of outliers rebinned')

                    slit_id = str(hdulist[1].header['SLIT_ID'])

                    if len(list(slit_id)) == 4:
                        slit_id = slit_id
                    elif len(list(slit_id)) == 3:
                        slit_id = '0' + slit_id
                    elif len(list(slit_id)) == 2:
                        slit_id = '00' + slit_id
                    elif len(list(slit_id)) == 1:
                        slit_id = '000' + slit_id 

                    hdulist[0].header.set('SLIT_ID', slit_id, 'Slit ID')
                    hdulist[1].header.set('SLIT_ID', slit_id, 'Slit ID')

                    outname = '{}{}{}'.format(mask_name, quadrant, slit_id)

                    if np.isfinite(fmatch.loc[outname].RA_g) == False:
                        RA_g = 54.0000
                        DEC_g = -35.0000
                    else:
                        RA_g = fmatch.loc[outname].RA_g
                        DEC_g = fmatch.loc[outname].DEC_g

                    hdulist[0].header.set('RA_g', RA_g, 'RA in the g-band')
                    hdulist[0].header.set('DEC_g', DEC_g, 'DEC in the g-band')

                    hdulist[1].header.set('RA_g', RA_g, 'RA in the g-band')
                    hdulist[1].header.set('DEC_g', DEC_g, 'DEC in the g-band')

                    hduthumb = fits.ImageHDU(data = get_thumbnail(hduimg, w, RA_g, DEC_g, 25), name = 'thumbnail')
                    hdulist.append(hduthumb)
                    hdulist.append(fits.ImageHDU(data = data_original, name = 'original'))

                    hdulist.writeto('/Volumes/VINCE/OAC/extracted_new/{}_{}.fits'.format(outname, row), clobber = True)
                    hdulist.close()

start_dir = '/Volumes/VINCE/OAC/vimos_new'
extract(start_dir)


