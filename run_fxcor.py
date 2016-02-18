from pyraf import iraf
import time
import os
import glob
from tqdm import *
import pandas as pd
from astropy.io.fits import getval, getheader, getdata
from astropy.io import fits

def change_name(l, suff):
	names = [l[i].split('/')[-1] for i in range(len(l))]
	return [names[i].split('.')[0] + suff for i in range(len(l))]

def create_list(path, output):
	ser = pd.Series(glob.glob(path))
	ser.to_csv('/Volumes/VINCE/OAC/fxcor/{}'.format(output), index = False, header = False)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# PREPARE THE TEMPLATE SPECTRA

# Load the original .fits files. Keep them in a separate directory as backup
files = glob.glob('/Volumes/VINCE/OAC/fxcor/temp_original/*.fits*')

# Select only stars with spectral type > F
l = []
for f in files:
	stype = getval(f, 'SPTYPE', 0)
	if stype > 'F':
		l.append(f)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

iraf.imfilter()
FWHM = 7000 / 580.
sig = FWHM / 2.355 # The sigma of the Gaussian to be convolved with

# Define file names for the output
namesd = ['/Volumes/VINCE/OAC/fxcor/tempd/redispersed/' + change_name(l, 'd.fits')[i] for i in range(len(l))]

print 'Changing the spectral resolution . . .'

for file, output in tqdm(zip(l, namesd)):
	input = file + '[0]'
	iraf.imfilter.gauss(input = input, output = output, sigma = sig)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# w1 = 5100.0, w2 = 8800.0
# w1 = 8400.0, w2 = 8850.0,
# w1 = 6510.0, w2 = 8850.0, 
filesd = glob.glob('/Volumes/VINCE/OAC/fxcor/tempd/redispersed/*d.fits*')
nameR = ['/Volumes/VINCE/OAC/fxcor/tempd/Ha/' + change_name(filesd, '_Ha.fits')[i] for i in range(len(filesd))]

iraf.onedspec()
for file,output in tqdm(zip(filesd,nameR)):
	iraf.onedspec.dispcor(input = file + '[0]', output = output, w1 = 6510.0, w2 = 8850.0, 
					      dw = 2.6, nw = 15011, verbose = 'no')

create_list('/Volumes/VINCE/OAC/fxcor/tempd/Ha/*Ha.fits', 'templates_Ha.txt')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# PREPARE THE SCIENCE SPECTRA
#w1 = 8400., w2 = 8850.
files = glob.glob('/Volumes/VINCE/OAC/extracted_new/*.fits*')
nameR = ['/Volumes/VINCE/OAC/fxcor/datad/' + change_name(files, '_cat.fits')[i] for i in range(len(files))]

iraf.onedspec()
for f, outputR in tqdm(zip(files, nameR)):
	sn = float(getval(f, 'SN1', 0))
	sg = float(getval(f, 'sg_norm', 0))

	if sn > 1: #and sg <= 10:
		f = f + '[1]'
		iraf.onedspec.dispcor(input = f, output = outputR, w1 = 8400., w2 = 8850., verbose = 'no')

create_list('/Volumes/VINCE/OAC/fxcor/datad/*cat.fits*', 'datacat.txt')

# Full range 3 330 220 500 5150-8700 5140-8690
# CaT range   

from pyraf import iraf

cuton = [1,2,3,4,5,6]
fullon = [180,190,200,210,220]
cutoff = [250,280,290,300,310,320,330]
fulloff = [480,490,500,520,530,600]

myfile = open('xyz.cl', 'w')
for a in cuton:
	for b in fullon:
		for c in cutoff:
			for d in fulloff:
   				 myfile.write("filtpars.cuton = {}\n".format(a))
   				 myfile.write("filtpars.fullon = {}\n".format(b))
   				 myfile.write("filtpars.cutoff = {}\n".format(c))
   				 myfile.write("filtpars.fulloff = {}\n".format(d))
   				 myfile.write("fxcor obj.fits[0] temp.fits[1]\n")
myfile.close()

myfile = open('reference.txt', 'w')
for a in cuton:
	for b in fullon:
		for c in cutoff:
			for d in fulloff:
   				 myfile.write("{} {} {} {}\n".format(a,b,c,d))
myfile.close()

aa = pd.read_csv('aa.txt', comment = '#', delim_whitespace = True, header = None)
