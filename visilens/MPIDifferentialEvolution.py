__all__ = ['MPI_DifferentialEvolution']

import numpy as np
import scipy.sparse
import os
import sys
import copy
from astropy.cosmology import WMAP9
from class_utils import *
from lensing import *
from utils import *
from calc_likelihood import calc_vis_lnlike
from mpi4py import MPI

arcsec2rad = np.pi/180/3600

class MPI_DifferentialEvolution(object):
    """
    A class that can perform MPI Self-Adaptive Differential Evolution
    Using the visilens modeling framework.  
    """
    
    def __init__(self,data,lens,source,xmax=30.,highresbox=[-3.,3.,3.,-3.],emitres=None,fieldres=None,sourcedatamap=None, scaleamp=False, shiftphase=False,modelcal=True,cosmo=WMAP9,nwalkers=1e3):
        """
        Initialize the Differential Evolution class, along with the initial positions of walkers.
        Setup the data, grids, and model.
        """
        
        # Making these lists just makes later stuff easier since we now know the dtype.
        # They're also the args that are passed to calc_vis_lnlike, which is important because
        # we want to make this particular code only have to permute the vectors at any point.
        lens = list(np.array([lens]).flatten())
        source = list(np.array([source]).flatten()) # Ensure source(s) are a list
        data = list(np.array([data]).flatten())     # Same for dataset(s)
        scaleamp = list(np.array([scaleamp]).flatten())
        shiftphase = list(np.array([shiftphase]).flatten())
        modelcal = list(np.array([modelcal]).flatten())
        
        if len(scaleamp)==1 and len(scaleamp)<len(data): scaleamp *= len(data)
        if len(shiftphase)==1 and len(shiftphase)<len(data): shiftphase *= len(data)
        if len(modelcal)==1 and len(modelcal)<len(data): modelcal *= len(data)
        if sourcedatamap is None: sourcedatamap = [None]*len(data)
        
        
        # Get all of the parameter guesses into a single array
        ndim, p0, colnames = 0, [], []
        # Lens(es) first
        for i,ilens in enumerate(lens):
            if ilens.__class__.__name__=='SIELens':
                for key in ['x','y','M','e','PA']:
                    if not vars(ilens)[key]['fixed']:
                        ndim += 1
                        p0.append(vars(ilens)[key]['value'])
                        colnames.append(key+'L'+str(i))
            if ilens.__class__.__name__=='PowerKappa':
                for key in ['x','y','M','ex','ey','gamma','rc']:
                    if not vars(ilens)[key]['fixed']:
                        ndim += 1
                        p0.append(vars(ilens)[key]['value'])
                        colnames.append(key+'L'+str(i))
            elif ilens.__class__.__name__=='ExternalShear':
                for key in ['shear','shearangle']:
                    if not vars(ilens)[key]['fixed']:
                        ndim += 1
                        p0.append(vars(ilens)[key]['value'])
                        colnames.append(key)
            elif ilens.__class__.__name__=='Multipoles':
                for key in ['A2','B2','A3','B3','A4','B4']:
                    if not vars(ilens)[key]['fixed']:
                        ndim += 1
                        p0.append(vars(ilens)[key]['value'])
                        colnames.append(key)
        # Then source(s)
        for i,src in enumerate(source):
            if src.__class__.__name__=='GaussSource':
                for key in ['xoff','yoff','flux','width']:
                    if not vars(src)[key]['fixed']:
                        ndim += 1
                        p0.append(vars(src)[key]['value'])
                        colnames.append(key+'S'+str(i))
            elif src.__class__.__name__=='SersicSource':
                for key in ['xoff','yoff','flux','reff','index','axisratio','PA']:
                    if not vars(src)[key]['fixed']:
                        ndim += 1
                        p0.append(vars(src)[key]['value'])
                        colnames.append(key+'S'+str(i))
            elif src.__class__.__name__=='PointSource':
                for key in ['xoff','yoff','flux']:
                    if not vars(src)[key]['fixed']:
                        ndim += 1
                        p0.append(vars(src)[key]['value'])
                        colnames.append(key+'S'+str(i))
        # Then flux rescaling; only matters if >1 dataset
        for i,t in enumerate(scaleamp[1:]):
            if t:
                ndim += 1
                p0.append(1.) # Assume 1.0 scale factor to start
                colnames.append('ampscale_dset'+str(i+1))
        # Then phase/astrometric shift; each has two vals for a shift in x&y
        for i,t in enumerate(shiftphase[1:]):
            if t:
                ndim += 2
                p0.append(0.); p0.append(0.) # Assume zero initial offset
                colnames.append('astromshift_x_dset'+str(i+1))
                colnames.append('astromshift_y_dset'+str(i+1))
                    

        
        # Get any model-cal parameters set up. The process involves some expensive
        # matrix inversions, but these only need to be done once, so we'll do them
        # now and pass the results as arguments to the likelihood function. See docs
        # in calc_likelihood.model_cal for more info.
        for i,dset in enumerate(data):
            if modelcal[i]:
                uniqant = np.unique(np.asarray([dset.ant1,dset.ant2]).flatten())
                dPhi_dphi = np.zeros((uniqant.size-1,dset.u.size))
                for j in range(1,uniqant.size):
                      dPhi_dphi[j-1,:]=(dset.ant1==uniqant[j])-1*(dset.ant2==uniqant[j])
                C = scipy.sparse.diags((dset.sigma/dset.amp)**-2.,0)
                F = np.dot(dPhi_dphi,C*dPhi_dphi.T)
                Finv = np.linalg.inv(F)
                FdPC = np.dot(-Finv,dPhi_dphi*C)
                modelcal[i] = [dPhi_dphi,FdPC]
        
        
        # Create our lensing grid coordinates now, since those shouldn't be
        # recalculated with every call to the likelihood function
        xmap,ymap,xemit,yemit,indices = GenerateLensingGrid(data,xmax,highresbox,
                                                  fieldres,emitres)
                                                  
        # Calculate the uv coordinates we'll interpolate onto; only need to calculate
        # this once, so do it here.
        kmax = 0.5/((xmap[0,1]-xmap[0,0])*arcsec2rad)
        ug = np.linspace(-kmax,kmax,xmap.shape[0])
        

        # Calculate some distances; we only need to calculate these once.
        # This assumes multiple sources are all at same z; should be this
        # way anyway or else we'd have to deal with multiple lensing planes
        if cosmo is None: cosmo = WMAP9
        Dd = cosmo.angular_diameter_distance(lens[0].z).value
        Ds = cosmo.angular_diameter_distance(source[0].z).value
        Dds= cosmo.angular_diameter_distance_z1z2(lens[0].z,source[0].z).value
        
        # Everything that is created in the preamble to the MCMC function we now want to set as a class
        # attribute so that it can be passed between functions.
        self.lens = lens
        self.source = source
        self.data = data
        self.scaleamp = scaleamp
        self.shiftphase = shiftphase
        self.sourcedatamap = sourcedatamap
        self.p0 = p0
        self.Ndim = ndim
        self.colnames = colnames
        self.modelcal = modelcal
        self.xmap = xmap
        self.ymap = ymap
        self.xemit = xemit
        self.yemit = yemit
        self.indices = indices
        self.ug = ug
        self.Dd = Dd
        self.Ds = Ds
        self.Dds = Dds
        self.Nwalkers = nwalkers
        
        # All the lens objects know if their parameters have been altered since the last time
        # we calculated the deflections. If all the lens pars are fixed, we only need to do the
        # deflections once. This step ensures that the lens object we create the sampler with
        # has these initial deflections.
        for i,ilens in enumerate(self.lens):
            if ilens.__class__.__name__ == 'SIELens': ilens.deflect(self.xemit,self.yemit,self.Dd,self.Ds,self.Dds)
            elif ilens.__class__.__name__ == 'PowerKappa': ilens.deflect(self.xemit,self.yemit,self.Dd,self.Ds,self.Dds)
            elif ilens.__class__.__name__ == 'ExternalShear': ilens.deflect(self.xemit,self.yemit,self.lens[0])
            elif ilens.__class__.__name__ == 'Multipoles': ilens.deflect(self.xemit,self.yemit,self.lens[0])
            
    def Initialization(self,priorup,priordn):
        self.VECTORS = np.zeros([self.Nwalkers,self.Ndim])
        for i in range(self.Nwalkers):
            # Initialize the vectors randomly
            self.VECTORS[i,:] = (np.random.random(self.Ndim)) * (priorup-priordn) + priordn
        
        self.DONORS = np.zeros(self.VECTORS.shape)
        self.TRIALS = np.zeros(self.VECTORS.shape)
        self.CURRENT_FX = np.ones(self.VECTORS.shape[0]) * np.inf
        self.priorup = np.inf * np.ones(self.VECTORS.shape[1])
        self.priordn = -np.inf * np.ones(self.VECTORS.shape[1])
    
    def SetPriors(self,priorup,priordn):
        """
        If we want to set hard priors, this is where to do so.  Otherwise priors are just
        -inf to inf
        """
        self.priorup = priorup
        self.priordn = priordn
    
    def Mutation(self):
        """
        Mutate the vectors.
        """
        for i in range(self.VECTORS.shape[0]):
            options = range(self.VECTORS.shape[0])
            options.remove(i)
            choices = np.random.choice(options,3,replace=False)
            F = np.random.normal(0.5,0.3)
            self.DONORS[i,:] = self.VECTORS[choices[0],:]+F*(self.VECTORS[choices[1],:]-self.VECTORS[choices[2],:])
        
    def Recombination(self):
        """
        Recombine the donor vectors with the current vectors
        to create the trial vectors, including crossover.
        """
        for i in range(self.VECTORS.shape[0]):
            CR = np.random.normal(0.5,0.1)
            for j in range(self.VECTORS.shape[1]):
                if ((np.random.random() <= CR) or (j == np.random.choice(range(0,self.DONORS.shape[1])))):
                    self.TRIALS[i,j] = self.DONORS[i,j]
                else:
                    self.TRIALS[i,j] = self.VECTORS[i,j]
                    
    def Selection(self,f2):
        """
        look through the chi2 values, and see which trial vectors should be retained
        """
        Naccepted = 0
        for i in range(self.VECTORS.shape[0]):
            if f2[i] < self.CURRENT_FX[i]:
                self.VECTORS[i,:] = self.TRIALS[i,:]
                self.CURRENT_FX[i] = f2[i]
                Naccepted += 1
            else:
                pass
            
        return Naccepted
        
    def ConvergenceCheck(self,):
        '''
        Check if current vectors have a small enough RMS scatter
        or if their chi2 are within a certain precision
        
        if either are met return True, else return False
        '''
        
    def Optimize(self,Niter,num_gangs=1,ConvThresh=1e-7,FileWriteNumIter=False,filename_prefix='temp',chi2_thresh=1e-7):
        
        # setup MPI communicator
        comm = MPI.COMM_WORLD
        # Get process rank
        rank = comm.Get_rank()
        # Get size of world
        size = comm.Get_size()
        
        # All gangs must have the same number of CPUs
        if (size%num_gangs) != 0:
            print "Error: bad number of gangs.  Terminating..."
            return
        
        # All gangs must have the same number of walkers.
        if (self.Nwalkers % num_gangs!=0):
            print "Error: bad number of gangs.  Terminating..."
            return
        
        # How many walkers are in a gang.
        num_walk_per_gang = self.Nwalkers / num_gangs
        
        # How many processes are in a gang
        num_gang_members = size/num_gangs
        
        # Which gang is my process in
        my_gang = np.floor(rank/num_gang_members).astype(int)
        
        # What is my rank within the gang
        my_rank = (rank % num_gang_members)
        
        # Set up communicators for within rank and between gangs
        GangComm = MPI.COMM_WORLD.Split(my_gang,my_rank)
        RankComm = MPI.COMM_WORLD.Split(my_rank,my_gang)
        
        # Create an array to transport chi2 back to process 0
        local_proposed_fx = np.zeros(num_walk_per_gang)
        
        # Create arrays to store samples
        self.SAMPLES = np.zeros([self.Nwalkers,Niter,self.Ndim])
        self.CHI2 = np.zeros([self.Nwalkers,Niter])
        
        # Begin the optimization
        for i in range(Niter):
            comm.Barrier()
            if rank ==0:
                self.Mutation()
                self.Recombination()
            comm.Bcast(self.VECTORS,root=0)
            comm.Bcast(self.DONORS,root=0)
            comm.Bcast(self.TRIALS,root=0)
            comm.Bcast(self.CURRENT_FX,root=0)
        
            comm.Barrier()
            for j in range(num_walk_per_gang):
                # Calculate the likelihood
                local_proposed_fx[j] = -calc_vis_lnlike(self.TRIALS[j+my_gang*num_walk_per_gang,:],self.data,self.lens,self.source, \
                        self.Dd , self.Ds , self.Dds , self.ug , self.xmap , self.ymap ,\
                        self.xemit , self.yemit , self.indices , self.sourcedatamap ,\
                        self.scaleamp , self.shiftphase , self.modelcal)[0]
                
                # Use Trial Function for debugging
                #local_proposed_fx[j] = Ackley(self.TRIALS[j+my_gang*num_walk_per_gang,:])
                
            # Root process must now catch all the data, and put it in the right place
            f2 = np.zeros(self.Nwalkers)
            comm.Barrier()
            RankComm.Gatherv(local_proposed_fx,[f2,tuple(num_walk_per_gang for j in range(num_gangs)),\
                                tuple(j*num_walk_per_gang for j in range(num_gangs)),MPI.DOUBLE],root=0)
            
            # Now the root process has the correct chi2 list, but others may not.  Might as well synch
            comm.Bcast(f2,root=0)
            
            comm.Barrier()
            if rank ==0:
                # Most of everything else can be run in single process
                Naccepted = self.Selection(f2)
                self.SAMPLES[:,i,:] = self.VECTORS
                self.CHI2[:,i] = self.CURRENT_FX
            
                print "---------- Interation:  " , i , "/", float(Niter) , "----------"
                print "Number of accepted proposals:  " ,  Naccepted
                print "Best chi2:  " , np.min(self.CURRENT_FX)
                print "Vector RMS:  " , np.std(self.VECTORS,axis=0)
            
                # Save the data if asked to do so
                try:
                    if i % FileWriteNumIter ==0:
                        self.SaveData(i,filename_prefix)
                except:
                    pass  
                
            # broadcast current set of vectors (just in case)
            comm.Bcast(self.VECTORS,root=0)
            comm.Bcast(self.CURRENT_FX,root=0)
            
            # Each process should check if sampling has converged
            if self.ConvergenceCheck():
                break
        if rank ==0:
            print "Optimization has Converged"
            print "Best Parameters:  " , self.VECTORS[np.argmin(self.CURRENT_FX),:] 
                
            
            
def Ackley(parameters):
    '''
    Good test function for debugging a DE optimizer'''
    n = len(parameters)
    fx = -20*np.exp(-0.2*np.sqrt(np.sum(parameters**2)/float(n)))-np.exp(np.sum(np.cos(2*np.pi*parameters))/float(n))+20+np.exp(1)
    return fx                
                
                
        
        
        
