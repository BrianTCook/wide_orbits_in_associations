import gzip
import numpy as np 
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

import glob

def maps(bg_str, time_arrow):
    
    datadir = '/Users/BrianTCook/Desktop/wide_orbits_in_associations/data/'
    snapshots = glob.glob(datadir+'%s_%s/PhaseSpace_*.ascii'%(time_arrow, bg_str))

    sim_times = np.linspace(0., 64., 9)  
    
    for i, t in enumerate(sim_times):
        
        print('t = %.0f Myr, %s'%(t, bg_str))
        
        phase_space_data = np.loadtxt(snapshots[i])
        center_of_mass = np.average(phase_space_data[:,1:], axis=0, weights=phase_space_data[:,0])
        
        x_com, y_com, z_com, vx_com, vy_com, vz_com = center_of_mass
        
        x = [ x - x_com for x in phase_space_data[:, 1] ]
        y = [ y - y_com for y in phase_space_data[:, 2] ]
        z = [ z - z_com for z in phase_space_data[:, 3] ]
        vx = [ vx - vx_com for vx in phase_space_data[:, 4] ]
        vy = [ vy - vy_com for vy in phase_space_data[:, 5] ]
        vz = [ vz - vz_com for vz in phase_space_data[:, 6] ]
        
        dx = (np.percentile(x, 95) - np.percentile(x, 5))/2.
        dy = (np.percentile(y, 95) - np.percentile(y, 5))/2.
        dz = (np.percentile(z, 95) - np.percentile(z, 5))/2.
        dvx = (np.percentile(vx, 95) - np.percentile(vx, 5))/2.
        dvy = (np.percentile(vy, 95) - np.percentile(vy, 5))/2.
        dvz = (np.percentile(vz, 95) - np.percentile(vz, 5))/2.
        
        fig, axs = plt.subplots(5, 5)
        
        #first column      
        
        xy = np.vstack([x,y])
        z00 = gaussian_kde(xy)(xy)
        idx = z00.argsort()
        x, y, colors = [ x[i] for i in idx ], [ y[i] for i in idx ], [ z00[i] for i in idx ]
        axs[0, 0].scatter(x, y, s=0.5, c=colors, edgecolor='')
        axs[0, 0].set_ylabel(r'$y$ (pc)', fontsize=8)
        axs[0, 0].tick_params(labelsize='xx-small')
        axs[0, 0].set_title(r'$x \in %.01f \pm %.02f$ pc'%(x_com, dx), fontsize=3)
        
        xz = np.vstack([x,z])
        z10 = gaussian_kde(xz)(xz)
        idx = z10.argsort()
        x, z, colors = [ x[i] for i in idx ], [ z[i] for i in idx ], [ z10[i] for i in idx ]
        axs[1, 0].scatter(x, z, s=0.5, c=colors, edgecolor='')
        axs[1, 0].set_ylabel(r'$z$ (pc)', fontsize=8)
        axs[1, 0].tick_params(labelsize='xx-small')
        
        xvx = np.vstack([x,vx])
        z20 = gaussian_kde(xvx)(xvx)
        idx = z20.argsort()
        x, vx, colors = [ x[i] for i in idx ], [ vx[i] for i in idx ], [ z20[i] for i in idx ]                        
        axs[2, 0].scatter(x, vx, s=0.5, c=colors, edgecolor='')
        axs[2, 0].set_ylabel(r'$v_{x}$ (km/s)', fontsize=8)
        axs[2, 0].tick_params(labelsize='xx-small')
        
        xvy = np.vstack([x,vy])
        z30 = gaussian_kde(xvy)(xvy)
        idx = z30.argsort()
        x, vy, colors = [ x[i] for i in idx ], [ vy[i] for i in idx ], [ z30[i] for i in idx ]                       
        axs[3, 0].scatter(x, vy, s=0.5, c=colors, edgecolor='')
        axs[3, 0].set_ylabel(r'$v_{y}$ (km/s)', fontsize=8)
        axs[3, 0].tick_params(labelsize='xx-small')
        
        xvy = np.vstack([x,vz])
        z40 = gaussian_kde(xvy)(xvy)
        idx = z40.argsort()
        x, vz, colors = [ x[i] for i in idx ], [ vz[i] for i in idx ], [ z40[i] for i in idx ]  
        axs[4, 0].scatter(x, vz, s=0.5, c=colors, edgecolor='')
        axs[4, 0].set_xlabel(r'$x$ (pc)', fontsize=8)
        axs[4, 0].set_ylabel(r'$v_{z}$ (km/s)', fontsize=8)
        axs[4, 0].tick_params(labelsize='xx-small')
        
        #second column
        
        axs[0, 1].axis('off')
        
        yz = np.vstack([y,z])
        z11 = gaussian_kde(yz)(yz)
        idx = z11.argsort()
        y, z, colors = [ y[i] for i in idx ], [ z[i] for i in idx ], [ z11[i] for i in idx ]  
        axs[1, 1].scatter(y, z, s=0.5, c=colors, edgecolor='')
        axs[1, 1].tick_params(labelsize='xx-small')
        axs[1, 1].set_title(r'$y \in %.01f \pm %.02f$ pc'%(y_com, dy), fontsize=3)
        
        
        yvx = np.vstack([y,vx])
        z21 = gaussian_kde(yvx)(yvx)
        idx = z21.argsort()
        y, vx, colors = [ y[i] for i in idx ], [ vx[i] for i in idx ], [ z21[i] for i in idx ]  
        axs[2, 1].scatter(y, vx, s=0.5, c=colors, edgecolor='')
        axs[2, 1].tick_params(labelsize='xx-small')
        
        yvy = np.vstack([y,vy])
        z31 = gaussian_kde(yvy)(yvy)
        idx = z31.argsort()
        y, vy, colors = [ y[i] for i in idx ], [ vy[i] for i in idx ], [ z31[i] for i in idx ] 
        axs[3, 1].scatter(y, vy, s=0.5, c=colors, edgecolor='')
        axs[3, 1].tick_params(labelsize='xx-small')
        
        yvz = np.vstack([y,vx])
        z41 = gaussian_kde(yvz)(yvz)
        idx = z41.argsort()
        y, vz, colors = [ y[i] for i in idx ], [ vz[i] for i in idx ], [ z41[i] for i in idx ] 
        axs[4, 1].scatter(y, vz, s=0.5, c=colors, edgecolor='')
        axs[4, 1].set_xlabel(r'$y$ (pc)', fontsize=8)
        axs[4, 1].tick_params(labelsize='xx-small')
        
        #third column
        
        axs[0, 2].axis('off')
        axs[1, 2].axis('off')
        
        zvx = np.vstack([z,vx])
        z22 = gaussian_kde(zvx)(zvx)
        idx = z22.argsort()
        z, vx, colors = [ z[i] for i in idx ], [ vx[i] for i in idx ], [ z22[i] for i in idx ] 
        axs[2, 2].scatter(z, vx, s=0.5, c=colors, edgecolor='')
        axs[2, 2].tick_params(labelsize='xx-small')
        axs[2, 2].set_title(r'$z \in %.01f \pm %.02f$ pc'%(z_com, dz), fontsize=3)
        
        zvy = np.vstack([z,vy])
        z32 = gaussian_kde(zvy)(zvy)
        idx = z32.argsort()
        z, vy, colors = [ z[i] for i in idx ], [ vy[i] for i in idx ], [ z32[i] for i in idx ] 
        axs[3, 2].scatter(z, vy, s=0.5, c=colors, edgecolor='')
        axs[3, 2].tick_params(labelsize='xx-small')
        
        zvz = np.vstack([z,vz])
        z42 = gaussian_kde(zvz)(zvz)
        idx = z42.argsort()
        z, vz, colors = [ z[i] for i in idx ], [ vz[i] for i in idx ], [ z42[i] for i in idx ] 
        axs[4, 2].scatter(z, vz, s=0.5, c=colors, edgecolor='')
        axs[4, 2].set_xlabel(r'$z$ (pc)', fontsize=8)
        axs[4, 2].tick_params(labelsize='xx-small')
        
        #fourth column
        
        axs[0, 3].axis('off')
        axs[1, 3].axis('off')                        
        axs[2, 3].axis('off')
        
        vxvy = np.vstack([vx,vy])
        z33 = gaussian_kde(vxvy)(vxvy)
        idx = z33.argsort()
        vx, vy, colors = [ vx[i] for i in idx ], [ vy[i] for i in idx ], [ z33[i] for i in idx ] 
        axs[3, 3].scatter(vx, vy, s=0.5, c=colors, edgecolor='')
        axs[3, 3].tick_params(labelsize='xx-small')
        axs[3, 3].set_title(r'$v_{x} \in %.01f \pm %.02f$ km/s'%(vx_com, dvx), fontsize=3)
           
        
        vxvz = np.vstack([vx,vz])
        z43 = gaussian_kde(vxvz)(vxvz)
        idx = z43.argsort()
        vx, vz, colors = [ vx[i] for i in idx ], [ vz[i] for i in idx ], [ z43[i] for i in idx ]
        axs[4, 3].scatter(vx, vz, s=0.5, c=colors, edgecolor='')
        axs[4, 3].set_xlabel(r'$v_{x}$ (km/s)', fontsize=8)
        axs[4, 3].tick_params(labelsize='xx-small')
        
        #fifth column
        
        axs[0, 4].axis('off')
        axs[1, 4].axis('off')
        axs[2, 4].axis('off')
        axs[3, 4].axis('off')
        
        vyvz = np.vstack([vy,vz])
        z44 = gaussian_kde(vyvz)(vyvz)
        idx = z44.argsort()
        vx, vz, colors = [ vy[i] for i in idx ], [ vz[i] for i in idx ], [ z44[i] for i in idx ]
        axs[4, 4].scatter(vy, vz, s=0.5, c=colors, edgecolor='')
        axs[4, 4].set_xlabel(r'$v_{y}$ (km/s)', fontsize=8) 
        axs[4, 4].tick_params(labelsize='xx-small')
        axs[4, 4].set_title(r'$v_{y} \in %.01f \pm %.02f$ km/s'%(vy_com, dvy), fontsize=3)
            
        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()
        
        axs[4, 4].set_ylabel(r'$v_{z} \in %.01f \pm %.02f$ km/s'%(vz_com, dvz), fontsize=3, rotation=270, labelpad=12)
        axs[4, 4].yaxis.set_label_position("right")
        
        fig.align_ylabels(axs[:, 0])
        
        if bg_str == 'with_background':
        
            fig.suptitle(r'LCC in Tidal Field, $t=%.0f \, {\rm Myr}$'%(t), fontsize=10)
            
        if bg_str == 'without_background':
        
            fig.suptitle(r'LCC by itself, $t=%.0f \, {\rm Myr}$'%(t), fontsize=10)
            
        plt.savefig('PSMap_%s_t_%s_Myr.pdf'%(bg_str, str(int(t))))
        plt.close()
            
    return 0

if __name__ in '__main__':
    
    arrows = [ 'forward' ]
    backgrounds = [ 'with_background', 'without_background' ]
    
    for bg_str in backgrounds:
        for arrow in arrows:
            maps(bg_str, arrow)
