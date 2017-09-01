__all__ = ['Visdata','SersicSource','GaussSource','PointSource','SIELens','ExternalShear',
            'read_visdata','concatvis','bin_visibilities','Multipoles','PowerKappa']

import numpy as np
import astropy.constants as co
from astropy import units
from utils import cart2pol,pol2cart
try:
    import _fastell
    Fastell_loaded = True
except ImportError:
    print "Could not load the PowerKappa lens model.\n"
    Fastell_loaded = False

c = co.c.value # speed of light, in m/s
G = co.G.value # gravitational constant in SI units
Msun = co.M_sun.value # solar mass, in kg
Mpc = 1e6*co.pc.value # 1 Mpc, in m
arcsec2rad = (np.pi/(180.*3600.))
rad2arcsec =3600.*180./np.pi
deg2rad = np.pi/180.

class Visdata(object):
      """
      Class to hold all necessary info relating to one set of visibilities.
      Auto-updates amp&phase or real&imag if those values are changed, but
      MUST SET WITH, eg, visobj.amp = (a numpy array of the new values);
      CANNOT USE, eg, visobj.amp[0] = newval, AS THIS DOES NOT CALL THE
      SETTER FUNCTIONS.
      
      Parameters:
      u     numpy ndarray
            The Fourier plane u coordinates of the visibilities to follow.
      v     numpy ndarray
            The Fourier plane v coordinates of the visibilities to follow.
      real  numpy ndarray
            The real parts of the visibilities
      imag  numpy ndarray
            The imaginary parts of the visibilities
      ant1  numpy ndarray
            The first antenna number or name of the visibility on each baseline
      ant2  numpy ndarray
            The second antenna number or name of the visibility on each baseline
      PBfwhm      float
            The FWHM of the antenna primary beam at this wavelength (at present
            assumes a homogeneous antenna array)
      filename    str
            A filename associated with these data.      
      """
      
      def __init__(self,u,v,real,imag,sigma,ant1=None,ant2=None,PBfwhm=None,filename=None):
            self.u = u
            self.v = v
            self.real = real
            self.imag = imag
            self.sigma = sigma
            self.ant1 = ant1
            self.ant2 = ant2
            self.PBfwhm = PBfwhm
            self.filename = filename
      
      @property
      def uvdist(self):
            return np.sqrt(self.u**2. + self.v**2.)

      @property
      def real(self):
            return self._real
      @real.setter
      def real(self,val):
            self._real = val
            # Setting amp & phase during __init__ will fail since imag is still unknown
            # Doing so during conjugate() will also fail, but gives a ValueError
            try:
                  self._amp = np.sqrt(self._real**2. + self.imag**2.)
                  self._phase = np.arctan2(self.imag,self._real)
            except (AttributeError,ValueError):
                  self._amp = None
                  self._phase = None
            
      @property
      def imag(self):
            return self._imag
      @imag.setter
      def imag(self,val):
            self._imag = val
            try:
                  self._amp = np.sqrt(self.real**2. + self._imag**2.)
                  self._phase = np.arctan2(self._imag,self.real)
            except (AttributeError,ValueError):
                  self._amp = None
                  self._phase = None

      @property 
      def amp(self):
            return  self._amp
      @amp.setter
      def amp(self,val):
            self._amp = val
            self._real = val * np.cos(self.phase)
            self._imag = val * np.sin(self.phase)

      @property
      def phase(self):
            return self._phase
      @phase.setter
      def phase(self,val):
            self._phase = val
            self._real = self.amp * np.cos(val)
            self._imag = self.amp * np.sin(val)

      def __add__(self,other):
            return Visdata(self.u,self.v,self.real+other.real,self.imag+other.imag,\
                  (self.sigma**-2. + other.sigma**-2.)**-0.5)

      def __sub__(self,other):
            return Visdata(self.u,self.v,self.real-other.real,self.imag-other.imag,\
                  (self.sigma**-2. + other.sigma**-2.)**-0.5)

      def conjugate(self):
            u = np.concatenate((self.u,-self.u))
            v = np.concatenate((self.v,-self.v))
            real = np.concatenate((self.real,self.real))
            imag = np.concatenate((self.imag,-self.imag))
            sigma = np.concatenate((self.sigma,self.sigma))
            ant1 = np.concatenate((self.ant1,self.ant2))
            ant2 = np.concatenate((self.ant2,self.ant1))
            self.u = u
            self.v = v
            self.real = real
            self.imag = imag
            self.sigma = sigma
            self.ant1 = ant1
            self.ant2 = ant2

class SIELens(object):
      """
      Class to hold parameters for an SIE lens, with each parameter (besides
      redshift) a dictionary.
      
      Example format of each parameter:
      x = {'value':x0,'fixed':False,'prior':[xmin,xmax]}, where x0 is the
      initial/current value of x, x should not be a fixed parameter during fitting,
      and the value of x must be between xmin and xmax.
      
      Note: in my infinite future free time, will probably replace e and PA with
      the x and y components of the ellipticity, which are better behaved as e->0.

      Parameters:
      z
            Lens redshift. If unknown, any value can be chosen as long as it is
            less than the source redshift you know/assume.
      x, y
            Position of the lens, in arcseconds relative to the phase center of
            the data (or any other reference point of your choosing). +x is west 
            (sorry not sorry), +y is north.
      M
            Lens mass, in Msun. With the lens and source redshifts, sets the 
            overall "strength" of the lens. Can be converted to an Einstein radius
            using theta_Ein = (4*G*M * D_LS / (c**2 * D_L * D_S))**0.5, in radians, 
            with G and c the gravitational constant and speed of light, and D_L, D_S
            and D_LS the distances to the lens, source, and between the lens and source,
            respectively.
      e
            Lens ellipticity, ranging from 0 (a circularly symmetric lens) to 1 (a very
            elongated lens).
      PA
            Lens major axis position angle, in degrees east of north.
      """      
      def __init__(self,z,x,y,M,e,PA):
            # Do some input handling.
            if not isinstance(x,dict):
                  x = {'value':x,'fixed':False,'prior':[-30.,30.]}
            if not isinstance(y,dict):
                  y = {'value':y,'fixed':False,'prior':[-30.,30.]}
            if not isinstance(M,dict):
                  M = {'value':M,'fixed':False,'prior':[1e7,1e15]}
            if not isinstance(e,dict):
                  e = {'value':e,'fixed':False,'prior':[0.,1.]}
            if not isinstance(PA,dict):
                  PA = {'value':PA,'fixed':False,'prior':[0.,180.]}

            if not all(['value' in d for d in [x,y,M,e,PA]]): 
                  raise KeyError("All parameter dicts must contain the key 'value'.")

            if not 'fixed' in x: x['fixed'] = False
            if not 'fixed' in y: y['fixed'] = False
            if not 'fixed' in M: M['fixed'] = False  
            if not 'fixed' in e: e['fixed'] = False
            if not 'fixed' in PA: PA['fixed'] = False
            
            if not 'prior' in x: x['prior'] = [-30.,30.]
            if not 'prior' in y: y['prior'] = [-30.,30.]
            if not 'prior' in M: M['prior'] = [1e7,1e15]
            if not 'prior' in e: e['prior'] = [0.,1.]
            if not 'prior' in PA: PA['prior'] = [0.,180.]

            self.z = z
            self.x = x
            self.y = y
            self.M = M
            self.e = e
            self.PA = PA
            self.gamma = {'value':1.0,'fixed':True}
            
            # Here we keep a Boolean flag which tells us whether one of the lens
            # properties has changed since the last time we did the lensing
            # deflections. If everything is the same, we don't need to lens twice.
            self._altered = True
      
      def deflect(self,xim,yim,Dd,Ds,Dds):
            """
            Follow Kormann+1994 for the lensing deflections.
            
            Parameters:
            xim, yim
                  2D Arrays of image coordinates we're going to lens,
                  probably generated by np.meshgrid.
            Dd, Ds, Dds
                  Distances to the lens, source and between the source
                  and lens (units don't matter as long as they're the 
                  same). Can't be calculated only from lens due to source
                  distances.
            """
            if self._altered: # Only redo if something is new.
                  ximage, yimage = xim.copy(), yim.copy() # for safety.
            
                  f = 1. - self.e['value']
                  fprime = np.sqrt(1. - f**2.)
            
                  # K+94 parameterizes in terms of LOS velocity dispersion and then
                  # basically the Einstein radius.
                  sigma = ((self.M['value']*Ds*G*Msun*c**2.)/(4*np.pi**2. * Dd*Dds*Mpc))**(1/4.)
                  Xi0 = 4*np.pi * (sigma/c)**2. * (Dd*Dds/Ds)
            
                  # Flip units, the recenter and rotate grid to lens center and major axis
                  ximage *= arcsec2rad; yimage *= arcsec2rad
                  ximage -= (self.x['value']*arcsec2rad)
                  yimage -= (self.y['value']*arcsec2rad)
                  if not np.isclose(self.PA['value'], 0.):
                        r,theta = cart2pol(ximage,yimage)
                        ximage,yimage = pol2cart(r,theta-(self.PA['value']*deg2rad))
                  phi = np.arctan2(yimage,ximage)
            
                  # Calculate the deflections, account for e=0 (the SIS), which has
                  # cancelling infinities. K+94 eq 27a.
                  if np.isclose(f, 1.):
                        dxs = -(Xi0/Dd)*np.cos(phi)
                        dys = -(Xi0/Dd)*np.sin(phi)
                  else:
                        dxs = -(Xi0/Dd)*(np.sqrt(f)/fprime)*np.arcsinh(np.cos(phi)*fprime/f)
                        dys = -(Xi0/Dd)*(np.sqrt(f)/fprime)*np.arcsin(np.sin(phi)*fprime)
            
                  # Rotate and shift back to sky frame
                  if not np.isclose(self.PA['value'], 0.):
                        r,theta = cart2pol(dxs,dys)
                        dxs,dys = pol2cart(r,theta+(self.PA['value']*deg2rad))
                  dxs *= rad2arcsec; dys *= rad2arcsec
            
                  self.deflected_x = dxs
                  self.deflected_y = dys
                  self._altered = False
                  
class PowerKappa(object):
    """
    Class to hold parameters for an Elliptical Power-law lens, with each 
    parameter (besides redshift) a dictionary.  This model simplifies to
    an SIE model in the case of fixed gamma = 1.
      
    Example format of each parameter:
    x = {'value':x0,'fixed':False,'prior':[xmin,xmax]}, where x0 is the
    initial/current value of x, x should not be a fixed parameter during fitting,
    and the value of x must be between xmin and xmax.
      
    Parameter Convention will be the same as that from the Ripples software,
    for easy compatibility.

    Parameters:
    z
            Lens redshift. If unknown, any value can be chosen as long as it is
            less than the source redshift you know/assume.
    x, y
            Position of the lens, in arcseconds relative to the phase center of
            the data (or any other reference point of your choosing). +x is west 
            (sorry not sorry), +y is north.
    M
            Lens-mass.  For better MCMC sampling, this will take the base-10 logarithm
            of the mass.
    ex,ey
            The x and y components of the ellipticity.  For sampling efficiency, we will
            write the components as 10 times the ellipticity components.
            A de-facto prior is included here.  Since the ellipticity must be less than 1,
            but the _fastell software still produces a real number for any e>1, we will 
            return the deflections as np.inf for all e>1.
    gamma
            The slope of the power-law.  Only works properly for 0 < gamma < 2.  Choose
            your priors accordingly.
    rc
            The EPMD model allows for a non-singular core at the center.  We give this as 
            an option.
    """
    
    def __init__(self,z,x,y,M,ex,ey,gamma,rc):
        # Do some input handling.
        if not isinstance(x,dict):
            x = {'value':x,'fixed':False,'prior':[-30.,30.]}
        if not isinstance(y,dict):
            y = {'value':y,'fixed':False,'prior':[-30.,30.]}
        if not isinstance(M,dict):
            M = {'value':M,'fixed':False,'prior':[0.5,1.5]}
        if not isinstance(ex,dict):
            ex = {'value':ex,'fixed':False,'prior':[-10,10]}
        if not isinstance(ey,dict):
            PA = {'value':ey,'fixed':False,'prior':[-10.,10]}
        if not isinstance(ey,dict):
            gamma = {'value':gamma,'fixed':False,'prior':[0,2]}
        if not isinstance(rc,dict):
            rc = {'value':rc,'fixed':False,'prior':[0,1]}

        if not all(['value' in d for d in [x,y,M,ex,ey,gamma]]): 
              raise KeyError("All parameter dicts must contain the key 'value'.")

        if not 'fixed' in x: x['fixed'] = False
        if not 'fixed' in y: y['fixed'] = False
        if not 'fixed' in M: M['fixed'] = False  
        if not 'fixed' in ex: ex['fixed'] = False
        if not 'fixed' in ey: ey['fixed'] = False
        if not 'fixed' in gamma: gamma['fixed'] = False
        if not 'fixed' in rc: rc['fixed'] = False
        
        if not 'prior' in x: x['prior'] = [-30.,30.]
        if not 'prior' in y: y['prior'] = [-30.,30.]
        if not 'prior' in M: M['prior'] = [0.5,1.5]
        if not 'prior' in ex: ex['prior'] = [-10,10]
        if not 'prior' in ey: ey['prior'] = [-10,10]
        if not 'prior' in gamma: gamma['prior'] = [0.,180.]
        if not 'prior' in rc: rc['prior'] = [0.,180.]

        self.z = z
        self.x = x
        self.y = y
        self.M = M
        self.ex = ex
        self.ey = ey
        self.gamma = gamma
        self.rc = rc
        
        self.PA = {'value':np.arctan2(ey['value'],ex['value']),'fixed':False,'prior':[0,180.]}
        
        # Here we keep a Boolean flag which tells us whether one of the lens
        # properties has changed since the last time we did the lensing
        # deflections. If everything is the same, we don't need to lens twice.
        self._altered = True
        
    def deflect(self,xim,yim,Dd,Ds,Dds):
        """
        Follow Barkana+1998 for the lensing deflections.  Uses a numerically
        integrated routine accurate to roughly 1e-7
        
        Parameters:
        xim, yim
              2D Arrays of image coordinates we're going to lens,
              probably generated by np.meshgrid.
        Dd, Ds, Dds
              Distances to the lens, source and between the source
              and lens (units don't matter as long as they're the 
              same). Can't be calculated only from lens due to source
              distances.
        """
        if self._altered: # Only redo if something is new.
            ximage, yimage = xim.copy(), yim.copy() # for safety.
            
            Q = self.compute_Q(Dd,Ds,Dds)
            # ellipticity and axis ratio
            elp = 0.1 * np.sqrt(self.ex['value']**2+self.ey['value']**2)
            q = 1-elp
            # rotation angle
            ang = np.arctan2(self.ey['value'],self.ex['value']) + np.pi / 2.
            
            
            # Flip units, the recenter and rotate grid to lens center and major axis
            ximage *= arcsec2rad; yimage *= arcsec2rad
            ximage -= (self.x['value']*arcsec2rad)
            yimage -= (self.y['value']*arcsec2rad)
            
            # Rotate image coordinates
            if not np.isclose(ang, 0.):
                r,theta = cart2pol(ximage,yimage)
                ximage,yimage = pol2cart(r,theta-ang)
            
            dxs = np.empty(len(ximage.ravel()))
            dys = np.empty(len(ximage.ravel()))
            
            if q > 0:
                _fastell.fastelldefl_array(ximage.ravel(),yimage.ravel(),Q,self.gamma['value']/2.,q,self.rc['value'],dxs,dys,len(dxs))
            else:
                dxs -= np.inf
                dys -= np.inf
            
            # reshape deflection angles back to 2d
            dxs = -dxs.reshape(ximage.shape)
            dys = -dys.reshape(yimage.shape)
            
            
            
            # Rotate and shift back to sky frame
            if not np.isclose(ang, 0.):
                r,theta = cart2pol(dxs,dys)
                dxs,dys = pol2cart(r,theta+(ang))
            dxs *= rad2arcsec; dys *= rad2arcsec
            
            self.deflected_x = dxs
            self.deflected_y = dys
            self._altered = False
            
            # Need to update position angle, so that shear works properly
            self.PA['value'] = np.arctan2(self.ey['value'],self.ex['value'])
            
    def compute_Q(self,Dd,Ds,Dds):
        """
        The Barkana 1998 model uses an annoying parameter Q for the deflection scale.
        It is a mixed combination of a number of physically relevant things, so we set 
        this function aside so that all the ugly math can be done here rather than in 
        the deflect function.
        """
        Mass = 10**(10*self.M['value']) * units.solMass
        q = 1- 0.1 * np.sqrt(self.ex['value']**2+self.ey['value']**2)
        rc = self.rc['value'] * arcsec2rad
        gam = self.gamma['value']/2.
        
        # Units should be Mpc from astropy.cosmology, this is risky.
        Dd = Dd * units.Mpc
        Ds = Ds * units.Mpc
        Dds = Dds * units.Mpc
        SigmaCrit = (co.c**2 / (4*np.pi * co.G) * Ds/(Dd*Dds)*Dd**2).decompose()
        
        
        
        sigma = np.sqrt(Mass / np.sqrt(q) * co.G / (10. * units.kpc) / np.pi).to(units.km/units.s)
        R_mass = 10e03 / Dd.to(units.pc).value 
        
        Q = (Mass / (2*np.pi * SigmaCrit * (rc**(2*(1-gam))-(R_mass**2+rc**2)**(1-gam))/(2.0*(gam-1.0)) ) / (q))
        
        
        
        return Q.decompose().value
        
            

class ExternalShear(object):
      """
      Class to hold the two parameters relating to an external tidal shear,
      where each parameter is a dictionary.
      
      Example format of each parameter:
      x = {'value':x0,'fixed':False,'prior':[xmin,xmax]}, where x0 is the
      initial/current value of x, x should not be a fixed parameter during fitting,
      and the value of x must be between xmin and xmax.

      Parameters:
      shear:
            The strength of the external shear. Should be 0 to 1 (although treating
            other objects in the lensing environment like this is really only valid
            for shear <~ 0.3).
      shearangle
            The position angle of the tidal shear, in degrees east of north.
      """
      def __init__(self,shear,shearangle):
            # Do some input handling.
            if not isinstance(shear,dict):
                  shear = {'value':shear,'fixed':False,'prior':[0.,1.]}
            if not isinstance(shearangle,dict):
                  shearangle = {'value':shearangle,'fixed':False,'prior':[0.,180.]}

            if not all(['value' in d for d in [shear,shearangle]]): 
                  raise KeyError("All parameter dicts must contain the key 'value'.")

            if not 'fixed' in shear: shear['fixed'] = False
            if not 'fixed' in shearangle: shearangle['fixed'] = False

            if not 'prior' in shear: shear['prior'] = [0.,1.]
            if not 'prior' in shearangle: shearangle['prior'] = [0.,180.]

            self.shear = shear
            self.shearangle = shearangle
            
      def deflect(self,xim,yim,lens):
            """
            Calculate deflection following Keeton,Mao,Witt 2000.
            
            Parameters:
            xim, yim
                  2D Arrays of image coordinates we're going to lens,
                  probably generated by np.meshgrid.
            lens
                  A lens object; we use this to shift the coordinate system
                  to be centered on the lens.
            """
            
            ximage,yimage = xim.copy(), yim.copy()
            
            ximage -= lens.x['value']; yimage -= lens.y['value']
            
            if not np.isclose(lens.PA['value'], 0.):
                  r,theta = cart2pol(ximage,yimage)
                  ximage,yimage = pol2cart(r,theta-(lens.PA['value']*deg2rad))
                  
            # KMW2000, altered for our coordinate convention.
            g,thg = self.shear['value'], (self.shearangle['value']-lens.PA['value'])*deg2rad
            dxs = -g*np.cos(2*thg)*ximage - g*np.sin(2*thg)*yimage
            dys = -g*np.sin(2*thg)*ximage + g*np.cos(2*thg)*yimage
            
            if not np.isclose(lens.PA['value'], 0.):
                  r,theta = cart2pol(dxs,dys)
                  dxs,dys = pol2cart(r,theta+(lens.PA['value']*deg2rad))
                  
            self.deflected_x = dxs; self.deflected_y = dys
            
            
class Multipoles(object):
    """
    Class to hold higher order angular multipoles in the isophote
    shape kappa(theta) = kappa0 + (Am cos(m theta)+ Bm sin(m theta)).  
    Each parameter should be a dictionary
    
    Example format of each parameter:
    x = {'value':x0,'fixed':False,'prior':[xmin,xmax]}, where x0 is the
    initial/current value of x, x should not be a fixed parameter during fitting,
    and the value of x must be between xmin and xmax.
    
    
    Parameters:
    
    A2 B2
            The 2nd order angular multipole components. (shear)
    
    A3 B3 
            The 3rd order angular multipole components.  Together they 
            define the amplitude and phase.
    A4 B4
            The 4th order angular multipole components.  Together they 
            define the amplitude and phase
    
    ***  WM: Later version of this could specify multipoles as a list, and
    Iterate over the elements of the list.
    """
    
    def __init__(self,A2,B2,A3,B3,A4,B4):
        # Handle inputs
        if not isinstance(A2,dict):
              A2 = {'value':A2,'fixed':False,'prior':[-100.,100.]}
        if not isinstance(B3,dict):
              B2 = {'value':B2,'fixed':False,'prior':[-100.,100.]}
        if not isinstance(A3,dict):
              A3 = {'value':A3,'fixed':False,'prior':[-100.,100.]}
        if not isinstance(B3,dict):
              B3 = {'value':B3,'fixed':False,'prior':[-100.,100.]}
        if not isinstance(A4,dict):
              A4 = {'value':A4,'fixed':False,'prior':[-100.,100.]}
        if not isinstance(B3,dict):
              B4 = {'value':B4,'fixed':False,'prior':[-100.,100.]}
        if not all(['value' in d for d in [A3,B3,A4,A4]]): 
              raise KeyError("All parameter dicts must contain the key 'value'.")
        
        if not 'fixed' in A2: A2['fixed'] = False
        if not 'fixed' in B2: B2['fixed'] = False
        if not 'fixed' in A3: A3['fixed'] = False
        if not 'fixed' in B3: B3['fixed'] = False
        if not 'fixed' in A3: A4['fixed'] = False
        if not 'fixed' in B3: B4['fixed'] = False

        if not 'prior' in A2: A2['prior'] = [-100.,100.]
        if not 'prior' in B2: B2['prior'] = [-100.,100.]
        if not 'prior' in A3: A3['prior'] = [-100.,100.]
        if not 'prior' in B3: B3['prior'] = [-100.,100.]
        if not 'prior' in A4: A4['prior'] = [-100.,100.]
        if not 'prior' in B4: B4['prior'] = [-100.,100.]
        
        self.A2 = A2
        self.B2 = B2
        self.A3 = A3
        self.B3 = B3
        self.A4 = A4
        self.B4 = B4
    def deflect(self,xim,yim,lens):
       
        """
        Calculate deflection following ....
         
        Parameters:
        xim, yim
               2D Arrays of image coordinates we're going to lens,
               probably generated by np.meshgrid.
        lens
               A lens object; we use this to shift the coordinate system
               to be centered on the lens and scale the multipoles
               by the slope of the mass distribution.
        """     
        ximage,yimage = xim.copy(), yim.copy()
       
        #Center coordinates on lens
        ximage -= lens.x['value']; yimage -= lens.y['value']
        
        # Convert positions to polar coordinates, and rotate
        r,theta = cart2pol(ximage,yimage)
        #theta -= (lens.PA['value']*deg2rad)
        r *= np.pi / 180.0 / 3600.0
        
        # Rs defines radial scaling of multipole deflection
        Rs = np.pi / 180.0 / 3600.0
        # Gamma defines slope of this radial scaling
        GAMMA = lens.gamma['value']
        a2 = self.A2['value']
        b2 = self.B2['value']
        a3 = self.A3['value']
        b3 = self.B3['value']
        a4 = self.A4['value']
        b4 = self.B4['value']
        
        # Compute non-rotated deflections in polar coordinates
        
        # Second order multipole (shear)
        alphaR = a2 * r * np.cos(2*theta) + b2* r * np.sin(2*theta)
        alphaTheta = -a2 * r * np.sin(2*theta) + b2 * r * np.cos(2*theta)
        
        # Third order multipole
        alphaR += (1./(GAMMA**2-4.*GAMMA-5.)) * (2.-GAMMA)*(r**(1.-GAMMA)/(Rs)**(-GAMMA)) * (a3*np.cos(3.0*theta)+b3*np.sin(3.0*theta))
        alphaTheta += (3./(GAMMA**2.-4.*GAMMA-5.) * (r**(1-GAMMA)/(Rs)**-GAMMA)) * (-a3*np.sin(3.0*theta)+b3*np.cos(3.0*theta))
        
        # Fourth order multipole
        alphaR += (1./(GAMMA**2-4.*GAMMA-12.)) * (2.-GAMMA)*(r**(1.-GAMMA)/(Rs)**(-GAMMA)) * (a4*np.cos(4.0*theta)+b4*np.sin(4.0*theta))
        alphaTheta += (4./(GAMMA**2-4.*GAMMA-12.) * (r**(1.-GAMMA)/(Rs)**-GAMMA)) * (-a4*np.sin(4.0*theta)+b4*np.cos(4.0*theta))
        
        # Convert back to cartesian coords.
        alphaX = alphaR * np.cos(theta) - alphaTheta * np.sin(theta)
        alphaY = alphaR * np.sin(theta) + alphaTheta * np.cos(theta)
        
        self.deflected_x = alphaX * 3600.0 * 180.0 / np.pi
        self.deflected_y = alphaY * 3600.0 * 180.0 / np.pi
        
        
        
    

class SersicSource(object):
      """
      Class to hold parameters of an elliptical Sersic light profile, ie
      I(x,y) = A * exp(-bn*((r/reff)^(1/n)-1)),
      where bn makes reff enclose half the light (varies with Sersic index), 
      and all the variable parameters are dictionaries.
      
      Example format of each parameter:
      x = {'value':x0,'fixed':False,'prior':[xmin,xmax]}, where x0 is the
      initial/current value of x, x should not be a fixed parameter during fitting,
      and the value of x must be between xmin and xmax.
      
      Parameters:
      z
            Source redshift. Can be made up, as long as it's higher than
            the lens redshift.
      lensed
            True/False flag determining whether this object is actually lensed
            (in which case it gets run through the lensing equations) or not (in
            which case it's simply added to the model of the field without lensing).
            This also determines the convention for the source position coordinates,
            see below.
      x, y
            Position of the source in arcseconds. If lensed is True, this position
            is relative to the position of the lens (or the first lens in a list of
            lenses). If lensed is False, this position is relative to the field
            center (or (0,0) coordinates). +x is west (sorry not sorry), +y is north.
      flux
            Total integrated flux density of the source (ie, NOT peak pixel value), in
            units of Jy.
      reff
            The source major axis in arcseconds.
      index
            The Sersic profile index n (0.5 is ~Gaussian, 1 is ~an exponential disk, 4
            is a de Vaucoleurs profile). Recommend that this not be much less than 0.3,
            or the approximation to b(n) which makes reff enclose half the light begins
            to break down.
      axisratio
            The source minor/major axis ratio, varying from 1 (circularly symmetric) to
            0 (highly elongated).
      PA
            Source position angle. If lensed is True, this is in degrees CCW from the
            lens major axis (or first lens in a list of them). If lensed is False, this
            is in degrees east of north.
      """

      def __init__(self,z,lensed=True,xoff=None,yoff=None,flux=None,reff=None,\
                  index=None,axisratio=None,PA=None):
            # Do some input handling.
            if not isinstance(xoff,dict):
                  xoff = {'value':xoff,'fixed':False,'prior':[-10.,10.]}
            if not isinstance(yoff,dict):
                  yoff = {'value':yoff,'fixed':False,'prior':[-10.,10.]}
            if not isinstance(flux,dict):
                  flux = {'value':flux,'fixed':False,'prior':[1e-5,1.]} # 0.01 to 1Jy source
            if not isinstance(reff,dict):
                  reff = {'value':reff,'fixed':False,'prior':[0.,2.]} # arcsec
            if not isinstance(index,dict):
                  index = {'value':index,'fixed':False,'prior':[0.3,4.]}
            if not isinstance(axisratio,dict):
                  axisratio = {'value':axisratio,'fixed':False,'prior':[0.01,1.]}
            if not isinstance(PA,dict):
                  PA = {'value':PA,'fixed':False,'prior':[0.,180.]}

            if not all(['value' in d for d in [xoff,yoff,flux,reff,index,axisratio,PA]]): 
                  raise KeyError("All parameter dicts must contain the key 'value'.")

            if not 'fixed' in xoff: xoff['fixed'] = False
            if not 'fixed' in yoff: yoff['fixed'] = False
            if not 'fixed' in flux: flux['fixed'] = False  
            if not 'fixed' in reff: reff['fixed'] = False
            if not 'fixed' in index: index['fixed'] = False
            if not 'fixed' in axisratio: axisratio['fixed'] = False
            if not 'fixed' in PA: PA['fixed'] = False
            
            if not 'prior' in xoff: xoff['prior'] = [-10.,10.]
            if not 'prior' in yoff: yoff['prior'] = [-10.,10.]
            if not 'prior' in flux: flux['prior'] = [1e-5,1.]
            if not 'prior' in reff: reff['prior'] = [0.,2.]
            if not 'prior' in index: index['prior'] = [1/3.,10]
            if not 'prior' in axisratio: axisratio['prior'] = [0.01,1.]
            if not 'prior' in PA: PA['prior'] = [0.,180.]

            self.z = z
            self.lensed = lensed
            self.xoff = xoff
            self.yoff = yoff
            self.flux = flux
            self.reff = reff
            self.index = index
            self.axisratio = axisratio
            self.PA = PA


class GaussSource(object):
      """
      Class to hold parameters of a circularly symmetric Gaussian light
      profile, where all the variable parameters are dictionaries.
      
      Example format of each parameter:
      x = {'value':x0,'fixed':False,'prior':[xmin,xmax]}, where x0 is the
      initial/current value of x, x should not be a fixed parameter during fitting,
      and the value of x must be between xmin and xmax.
      
      Parameters:
      z
            Source redshift. Can be made up, as long as it's higher than
            the lens redshift.
      lensed
            True/False flag determining whether this object is actually lensed
            (in which case it gets run through the lensing equations) or not (in
            which case it's simply added to the model of the field without lensing).
            This also determines the convention for the source position coordinates,
            see below.
      x, y
            Position of the source in arcseconds. If lensed is True, this position
            is relative to the position of the lens (or the first lens in a list of
            lenses). If lensed is False, this position is relative to the field
            center (or (0,0) coordinates). +x is west (sorry not sorry), +y is north.
      flux
            Total integrated flux density of the source (ie, NOT peak pixel value), in
            units of Jy.
      width
            The Gaussian width (sigma) of the light profile, in arcseconds.
      """

      def __init__(self,z,lensed=True,xoff=None,yoff=None,flux=None,width=None):
            # Do some input handling.
            if not isinstance(xoff,dict):
                  xoff = {'value':xoff,'fixed':False,'prior':[-10.,10.]}
            if not isinstance(yoff,dict):
                  yoff = {'value':yoff,'fixed':False,'prior':[-10.,10.]}
            if not isinstance(flux,dict):
                  flux = {'value':flux,'fixed':False,'prior':[1e-5,1.]} # 0.01 to 1Jy source
            if not isinstance(width,dict):
                  width = {'value':width,'fixed':False,'prior':[0.,2.]} # arcsec

            if not all(['value' in d for d in [xoff,yoff,flux,width]]): 
                  raise KeyError("All parameter dicts must contain the key 'value'.")

            if not 'fixed' in xoff: xoff['fixed'] = False
            if not 'fixed' in yoff: yoff['fixed'] = False
            if not 'fixed' in flux: flux['fixed'] = False  
            if not 'fixed' in width: width['fixed'] = False
            
            if not 'prior' in xoff: xoff['prior'] = [-10.,10.]
            if not 'prior' in yoff: yoff['prior'] = [-10.,10.]
            if not 'prior' in flux: flux['prior'] = [1e-5,1.]
            if not 'prior' in width: width['prior'] = [0.,2.]

            self.z = z
            self.lensed = lensed
            self.xoff = xoff
            self.yoff = yoff
            self.flux = flux
            self.width = width
            
class PointSource(object):
      """
      Class to hold parameters of an (unlensed) object unresolved by
      the data, where all the variable parameters are dictionaries.
      
      Example format of each parameter:
      x = {'value':x0,'fixed':False,'prior':[xmin,xmax]}, where x0 is the
      initial/current value of x, x should not be a fixed parameter during fitting,
      and the value of x must be between xmin and xmax.
      
      NOTE: Having a lensed point source is not currently implemented.      
      
      Parameters:
      z
            Source redshift. Can be made up, as long as it's higher than
            the lens redshift.
      lensed
            True/False flag determining whether this object is actually lensed
            (in which case it gets run through the lensing equations) or not (in
            which case it's simply added to the model of the field without lensing).
            This also determines the convention for the source position coordinates,
            see below.
      x, y
            Position of the source in arcseconds. If lensed is False (it must be), 
            this position is relative to the field center (or (0,0) coordinates). 
            +x is west (sorry not sorry), +y is north.
      flux
            Total flux density of the source, in units of Jy.
      """
      
      def __init__(self,z,lensed=True,xoff=None,yoff=None,flux=None):
            # Do some input handling.
            if not isinstance(xoff,dict):
                  xoff = {'value':xoff,'fixed':False,'prior':[-10.,10.]}
            if not isinstance(yoff,dict):
                  yoff = {'value':yoff,'fixed':False,'prior':[-10.,10.]}
            if not isinstance(flux,dict):
                  flux = {'value':flux,'fixed':False,'prior':[1e-5,1.]} # 0.01 to 1Jy source

            if not all(['value' in d for d in [xoff,yoff,flux]]): 
                  raise KeyError("All parameter dicts must contain the key 'value'.")

            if not 'fixed' in xoff: xoff['fixed'] = False
            if not 'fixed' in yoff: yoff['fixed'] = False
            if not 'fixed' in flux: flux['fixed'] = False  
            
            if not 'prior' in xoff: xoff['prior'] = [-10.,10.]
            if not 'prior' in yoff: yoff['prior'] = [-10.,10.]
            if not 'prior' in flux: flux['prior'] = [1e-5,1.]

            self.z = z
            self.lensed = lensed
            self.xoff = xoff
            self.yoff = yoff
            self.flux = flux


def read_visdata(filename):
      """
      Function to read in visibility data from file and create a visdata object
      to hold it afterwards. So far only .bin files from get_visibilities.py are
      supported; idea is eventually to be able to not mess with that and get straight
      from a CASA ms, but don't currently know how to do that without bundling the 
      casacore utilities directly...

      Params:
      filename
            Name of file to read from. Should contain all the visibility data needed,
            including u (Lambda), v (Lambda), real, imag, sigma, antenna1, and antenna 2.

      Returns:
      visdata
            A visdata object containing the data from filename.
      """
      
      if not filename.split('.')[-1].lower() in ['bin']:
            raise ValueError('Only .bin files are supported for now...')

      data = np.fromfile(filename)
      PBfwhm = data[-1]
      data = data[:-1]
      data = data.reshape(7,data.size/7) # bin files lose array shape, so reshape to match

      return Visdata(*data,PBfwhm=PBfwhm,filename=filename)

      
def concatvis(visdatas):
      """
      Concatenate multiple visibility sets into one larger set.
      Does no consistency checking of any kind, so beware.
      
      :param visdatas:
            List of visdata objects

      This method returns:

      * ``concatvis'' - The concatenated visibility set.
      """

      newu, newv, newr, newi = np.array([]),np.array([]),np.array([]),np.array([])
      news, newa1,newa2 = np.array([]),np.array([]),np.array([])

      for vis in visdatas:
            newu = np.concatenate((newu,vis.u))
            newv = np.concatenate((newv,vis.v))
            newr = np.concatenate((newr,vis.real))
            newi = np.concatenate((newi,vis.imag))
            news = np.concatenate((news,vis.sigma))
            newa1= np.concatenate((newa1,vis.ant1))
            newa2= np.concatenate((newa2,vis.ant2))

      return Visdata(newu,newv,newr,newi,news,newa1,newa2,visdatas[0].PBfwhm,'Combined Data')      

def bin_visibilities(visdata,maxnewsize=None):
      """
      WARNING: DOESN'T WORK CURRENTLY(?)
      Bins up (ie, averages down) visibilities to reduce the total
      number of them.  Note that since we fit directly to the visibilities,
      this is slightly different (and easier) than gridding in preparation for
      imaging, as we won't need to FFT and so don't need a convolution function.

      :param visdata
            A Visdata object.
      :param maxnewsize = None
            If desired, the maximum number of visibilities post-binning can
            be specified. As long as this number meets other criteria (ie,
            we don't have bin sizes smaller than an integration time or
            bandwidth in wavelengths), the total number in the returned
            Visdata will have fewer than maxnewsize visibilities.

      This method returns:
      * ``BinnedVisibilities'' - A Visdata object containing binned visibilities.
      """

      if maxnewsize is None: maxnewsize = visdata.u.size/2

      # Bins should be larger than an integration; strictly only valid for an EW array,
      # and assumes a 20s integration time. Thus, this is a conservative estimate.
      minbinsize = 20. * visdata.uvdist.max() / (24*3600.)

      # Bins should be smaller than the effective field size
      maxbinsize = (visdata.PBfwhm * arcsec2rad)**-1

      print minbinsize,maxbinsize

      # We're going to find a binning solution iteratively; this gets us set up
      Nbins, binsizeunmet, Nvis, it, maxiter = [3000,3000], True, visdata.u.size, 0, 250
      
      while (binsizeunmet or Nvis >= maxnewsize):
            print Nbins
            # Figure out how to bin up the data
            counts,uedges,vedges,bins = stats.binned_statistic_2d(
                  visdata.u,visdata.v,values=visdata.real,statistic='count',
                  bins=Nbins)
            
            du, dv = uedges[1]-uedges[0], vedges[1]-vedges[0]

            # Check that our bins in u and v meet our conditions
            if (du > minbinsize and du < maxbinsize and
                dv > minbinsize and dv < maxbinsize): binsizeunmet = False
            # Otherwise we have to adjust the number of bins to adjust their size...
            #elif (du <= minbinsize or dv <= minbinsize): Nbins = int(Nbins/1.2)
            #elif (du >= maxbinsize or dv >= maxbinsize): Nbins = int(Nbins*1.2)
            elif du <= minbinsize: Nbins[0] = int(Nbins[0]/1.1); binsizeunmet=True
            elif dv <= minbinsize: Nbins[1] = int(Nbins[1]/1.1); binsizeunmet=True
            elif du >= maxbinsize: Nbins[0] = int(Nbins[0]*1.1); binsizeunmet=True
            elif dv >= maxbinsize: Nbins[1] = int(Nbins[1]*1.1); binsizeunmet=True

            # If we still have more than the desired number of visibilities, make
            # fewer bins (we'll loop after this).
            if np.unique(bins).size > maxnewsize: Nbins[0],Nbins[1] = int(Nbins[0]/1.1),int(Nbins[1]/1.1)
            Nvis = np.unique(bins).size
            it += 1
            if it > maxiter: raise ValueError("It's impossible to split your data into that few bins!  "
                                    "Try setting maxnewsize to a larger value!")
            print Nvis,du,dv


      # Get us some placeholder arrays for the binned data
      u,v,real,imag,sigma,ant1,ant2 = np.zeros((7,Nvis))

      for i,filledbin in enumerate(np.unique(bins)):
            # This tells us which visibilities belong to the current bin
            points = np.where(bins==filledbin)[0]
            # This unravels the indices to uedges,vedges from the binned_statistic binnumber
            uloc = int(np.floor(filledbin/(vedges.size+1)) - 1)
            vloc = int(filledbin - (vedges.size+1)*(uloc+1) - 1)
            # Get our new data, place at center of uv bins
            u[i],v[i] = uedges[uloc]+0.5*du, vedges[vloc]+0.5*dv
            real[i],sumwt = np.average(visdata.real[points],weights=visdata.sigma[points]**-2.,returned=True)
            imag[i] = np.average(visdata.imag[points],weights=visdata.sigma[points]**-2.)
            sigma[i] = sumwt**-0.5
            # We can keep the antenna numbers if we've only selected points from the same baseline,
            # otherwise get rid of them (CHECK IF MODELCAL FAILS WITH None ANTENNAS)
            ant1[i] = visdata.ant1[points][0] if (visdata.ant1[points]==visdata.ant1[points][0]).all() else None
            ant2[i] = visdata.ant2[points][0] if (visdata.ant2[points]==visdata.ant2[points][0]).all() else None

      return Visdata(u,v,real,imag,sigma,ant1,ant2,visdata.PBfwhm,'BIN{0}'.format(Nvis)+visdata.filename)
