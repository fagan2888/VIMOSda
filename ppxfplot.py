import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)

def ppxfplot(twoD, galaxy, noise, bestfit, residuals, thumbnail, output_wavelengths,
             ix_start, ix_end, slit_id, rv, sigma, chi2, pixelmask, zp1, rv_fxcor, sg,
             sn1, sn2, RA_g, DEC_g):
    
    ymin = np.min(galaxy)
    ymax = np.max(galaxy)
    ylim = [ymin, ymax] + np.array([-0.02, 0.1])*(ymax-ymin)
    ylim[0] = 0.

    xmin = np.min(output_wavelengths)
    xmax = np.max(output_wavelengths)
    
    ### Define multipanel size and properties
    fig = plt.figure(figsize=[9,4])
    gs = gridspec.GridSpec(100,130,bottom=0.15,left=0.15,right=0.95)

    ### Plot the object in the sky
    ax_obj = fig.add_subplot(gs[0:30,105:125 ])
    
    ax_obj.imshow(thumbnail, cmap = 'gray', interpolation = 'nearest')
    ax_obj.set_xticks([]) 
    ax_obj.set_yticks([]) 

    ### Plot the 2D spectrum
    ax_2d = fig.add_subplot(gs[0:11,0:99])
    ax_2d.imshow(twoD[:, ix_start : ix_end], cmap='spectral',
                aspect = "auto", origin = 'lower', extent=[xmin,xmax,0,1], 
                vmin = -0.2, vmax=0.2)
    ax_2d.set_xticks([]) 
    ax_2d.set_yticks([]) 

    #### Plot the masked regions
    ax_spectrum = fig.add_subplot(gs[11:85,0:99])
    for pair in pixelmask:
        pair = np.exp(pair)/ zp1
        ax_spectrum.add_patch(patches.Rectangle((pair[0], ylim[0]), pair[1] - pair[0], ylim[1], 
                                            edgecolor="gray", facecolor="gray", 
                                            zorder=10, alpha=.5))
    ### Plot some atomic lines                                    
    line_wave = [6562.8, 5895.9, 5183.6, 8498, 8542, 8662 ]
    #line_label1 = ['Halpha', 'NaD', 'Mgb', 'CaT', 'CaT', 'CaT']

    for i in range(len(line_wave)):
        x = [line_wave[i], line_wave[i]]
        y = [ylim[0], ylim[1]]
        ax_spectrum.plot(x, y, 'k--', linewidth=1.2)

    ### Plot the spectrum and the bestfit
    ax_spectrum.plot(output_wavelengths, galaxy, 'k', linewidth=1.4)
    ax_spectrum.plot(output_wavelengths, bestfit,'r', linewidth=1.6)
    
    ### Define plot boundaries
    ax_spectrum.set_ylim(ylim)
    ax_spectrum.set_xlim([xmin,xmax])
    ax_spectrum.set_ylabel(r'Arbitrary Flux')

    #### Plot residuals
    ax_resid = fig.add_subplot(gs[86:99,0:99])
    ax_resid.set_xlim([xmin,xmax])
    ax_resid.set_ylim([-0.2,0.2])
    ax_resid.set_yticks([-0.1,0.1])
    ax_resid.plot(output_wavelengths, bestfit - galaxy, 'd', 
                  color='LimeGreen', mec='LimeGreen', ms=3)
    x = [xmin,xmax]
    y = [0, 0]
    ax_resid.plot(x, y, 'k-')
    ax_resid.set_ylabel(r'O - C')

    ax_resid.set_xlabel(r'Restframe Wavelength [ $\AA$ ]')
    plt.setp(ax_spectrum.get_xticklabels(), visible=False)

    textplot = fig.add_subplot(gs[40:100,105:130 ])
    textplot.text(0.1, 1.0,r'slit ID = {}'.format(slit_id), va="center", ha="left", size = 'smaller')
    textplot.text(0.1, 0.9,r'RV = {}'.format(rv), va="center", ha="left", size = 'smaller')
    textplot.text(0.1, 0.8,r'sigma = {}'.format(sigma), va="center", ha="left", size = 'smaller')
    textplot.text(0.1, 0.7,r'$\chi^2$ = {0:.2f}'.format(chi2), va="center", ha="left", size = 'smaller')

    textplot.text(0.1, 0.6,r'rv_fxcor = {}'.format(rv_fxcor), va="center", ha="left", size = 'smaller')
    textplot.text(0.1, 0.5,r'delta v = {}'.format(abs(rv_fxcor - rv)), va="center", ha="left", size = 'smaller')
    textplot.text(0.1, 0.4,r'sg = {}'.format(sg), va="center", ha="left", size = 'smaller')
    textplot.text(0.1, 0.3,r'sn1 = {0:.2f}'.format(sn1), va="center", ha="left", size = 'smaller')
    textplot.axis('off')
  
    #plt.show()
    
    return fig
