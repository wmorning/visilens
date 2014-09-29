import numpy as np
from utils import cart2pol,pol2cart
import astropy.constants as co
c = co.c.value # speed of light, in m/s
G = co.G.value # gravitational constant in SI units
Msun = co.M_sun.value # solar mass, in kg
Mpc = 1e6*co.pc.value # 1 Mpc, in m
arcsec2rad = (np.pi/(180.*3600.))
rad2arcsec =3600.*180./np.pi
deg2rad = np.pi/180.
rad2deg = 180./np.pi

__all__ = ['LensRayTrace','RayTraceSIE','TraceExternalShear','CausticsSIE']

def LensRayTrace(xim,yim,lens,Dd,Ds,Dds,shear=None):
      """
      Wrapper to pass off lensing calculations to any number of functions
      defined below, accumulating lensing offsets from multiple lenses
      and shear as we go.
      """
      # Ensure lens is a list, for convenience
      lens = list(np.array([lens]).flatten())
      
      ximage = xim.copy()
      yimage = yim.copy()
      
      for i,ilens in enumerate(lens):
            if ilens.__class__.__name__ == 'SIELens':
                  dxt,dyt = xim.copy(),yim.copy()
                  dxt,dyt = RayTraceSIE(dxt,dyt,ilens,Dd,Ds,Dds)
                  ximage += dxt; yimage += dyt
                  
            else: raise ValueError("Only SIE Lenses supported for now...")
      
      if shear is not None:
            dxt,dyt = xim.copy(),yim.copy()
            dxt,dyt = TraceExternalShear(dxt,dyt,lens[0],shear)
            ximage += dxt; yimage += dyt
      
      return ximage,yimage

def RayTraceSIE(xim,yim,SIELens,Dd,Ds,Dds):
      """
      Routine to transform image-plane coordinates to source-plane coordinates
      for an SIE lens.

      Inputs:
      ximage,yimage:
            2D arrays defining a grid of coordinates to map, in arcsec. These should
            be relative to the field center, i.e., we'll use the lens locations to
            recenter this grid on the lens and then shift back to the coordinate system
            given.

      SIELens:
            An SIELens object, containing mass, offset, etc. properties.
      
      Dd,Ds,Dds:
            Angular diameter distances (in Mpc) to the lens, source, and lens-source, 
            respectively. These could be calculated by passing a Lens and Source object,
            but since redshift is fixed, we can just calculate the distances once
            elsewhere and use those values repeatedly.

      Returns:
      xsource,ysource:
            Source-plane coordinates for the grid points given in ximage, yimage, in arcsec.
      """
      
      ximage,yimage = xim.copy(), yim.copy()

      # Following Kormann+ 1994 for the lensing. Easier to work with axis ratio than ellipticity
      f = 1. - SIELens.e['value']
      fprime = np.sqrt(1. - f**2.)

      # K+94 parameterize lens in terms of LOS velocity dispersion; calculate here in m/s
      sigma_lens = ((SIELens.M['value']*Ds*G*Msun*c**2.)/(4.*np.pi**2. * Dd*Dds*Mpc))**(1./4.)

      # K+94, eq 15., units of Mpc; a polar coordinate radius normalization
      Xi0 = 4 * np.pi * (sigma_lens/c)**2. * (Dd*Dds/Ds)

      ximage *= arcsec2rad
      yimage *= arcsec2rad

      # Recenter grid on the lens
      ximage -= SIELens.x['value']*arcsec2rad
      yimage -= SIELens.y['value']*arcsec2rad

      # Rotate the coordinate system to align with the lens major axis
      if not np.isclose(SIELens.PA['value'], 0.):
            r,theta = cart2pol(ximage,yimage)
            ximage,yimage = pol2cart(r,theta-(SIELens.PA['value']*deg2rad))

      # Polar coordinate angle from lens centroid
      phi = np.arctan2(yimage,ximage)

      # Calculate the transformation of the convergence term; K+94 eq 27a
      # Need to account for case of ellipticity=0 (the SIS), which has canceling infinities
      if np.isclose(f,1.):
            dxs =  - (Xi0/Dd)*np.cos(phi)
            dys =  - (Xi0/Dd)*np.sin(phi)
      else:
            dxs = - (Xi0/Dd)*(np.sqrt(f)/fprime)*np.arcsinh(np.cos(phi)*fprime/f)
            dys = - (Xi0/Dd)*(np.sqrt(f)/fprime)*np.arcsin(np.sin(phi)*fprime)

      # Rotate and shift the coordinate system back to the sky frame
      if not np.isclose(SIELens.PA['value'],0.):
            r,theta = cart2pol(dxs,dys)
            dxs,dys = pol2cart(r,theta+(SIELens.PA['value']*deg2rad))

      dxs *= rad2arcsec
      dys *= rad2arcsec
      
      #print dxs,dys

      return dxs,dys
      
def TraceExternalShear(xim,yim,lens,shear):
      """
      Calculate deflections due to an external shear component.
      
      Inputs:
      xim,yim:
            Map coordinates to perform the shear on.
            
      lens:
            A lens object; we use this to shift the coordinate system
            to be centered on the lens. This should be a single lens,
            not a list, since external shear isn't really meant to be
            applied if there are multiple lenses anyway.
            
      shear:
            An ExternalShear object, which gives the strength and angle of
            the deflection. The angle is defined in degrees N of E.
            
      Returns:
      xsource,ysource:
            Deflections calculated for the given inputs.
      """
      
      ximage,yimage = xim.copy(),yim.copy()
      
      ximage -= lens.x['value']
      yimage -= lens.y['value']
      
      if not np.isclose(lens.PA['value'], 0.):
            r,theta = cart2pol(ximage,yimage)
            ximage,yimage = pol2cart(r,theta-(lens.PA['value']*deg2rad))
      
      # Calculate contribution from shear term; see Chen,Kochanek&Hewitt1995, altered for this coord convention
      gamma,thg = shear.shear['value'],(shear.shearangle['value'] + lens.PA['value'])*deg2rad
      dxs = gamma*np.cos(2*thg)*ximage + gamma*np.sin(2*thg)*yimage
      dys = gamma*np.sin(2*thg)*ximage - gamma*np.cos(2*thg)*yimage  
      
      return dxs,dys

def CausticsSIE(SIELens,Dd,Ds,Dds,Shear=None):
      """
      Routine to calculate and return the analytical solutions for the caustics
      of an SIE Lens, following Kormann+94.

      Inputs:
      SIELens:
            An SIELens object for which to calculate the caustics.

      Dd,Ds,Dds:
            Angular diameter distances to the lens, source, and lens-source, respectively.

      Shear:
            An ExternalShear object describing the shear of the lens.

      Returns:
      2xN list:
            Arrays containing the x and y coordinates for the caustics that exist (i.e.,
            will have [xr,yr] for the radial caustic only if lens ellipticity==0, otherwise
            will have [[xr,yr],[xt,yt]] for radial+diamond caustics)
      """

      # Following Kormann+ 1994 for the lensing. Easier to work with axis ratio than ellipticity
      f = 1. - SIELens.e['value']
      fprime = np.sqrt(1. - f**2.)

      # K+94 parameterize lens in terms of LOS velocity dispersion; calculate here in m/s
      sigma_lens = ((SIELens.M['value']*Ds*G*Msun*c**2.)/(4.*np.pi**2. * Dd*Dds*Mpc))**(1./4.)

      # Einstein radius, for normalizing the size of the caustics, b in notation of Keeton+00
      b = 4 * np.pi * (sigma_lens/c)**2. * (Dds/Ds) * rad2arcsec      

      # Caustics calculated over a full 0,2pi angle range
      phi = np.linspace(0,2*np.pi,2000)

      # K+94, eq 21c; needed for diamond caustic
      Delta = np.sqrt(np.cos(phi)**2. + f**2. * np.sin(phi)**2.)
      
      if ((Shear is None) or (np.isclose(Shear.shear['value'],0.))):
            # Need to account for when ellipticity=0, as caustic equations have cancelling infinities
            #  In that case, Delta==1 and there's only one (radial and circular) caustic
            if np.isclose(f,1.):
                  xr,yr = -b*np.cos(phi)+SIELens.x['value'], -b*np.sin(phi)+SIELens.y['value']
                  caustic = np.atleast_3d([xr,yr])
                  return caustic.reshape(caustic.shape[2],caustic.shape[0],caustic.shape[1])

            else:
                  # Calculate the radial caustic coordinates
                  xr = (-b*np.sqrt(f)/fprime)*np.arcsinh(np.cos(phi)*fprime/f) + SIELens.y['value']
                  yr = (b*np.sqrt(f)/fprime)*np.arcsin(np.sin(phi)*fprime) - SIELens.x['value']
                  
                  # Now rotate the caustic to match the PA of the lens
                  r,th = cart2pol(xr,yr)
                  xr,yr = pol2cart(r,th+SIELens.PA['value']*deg2rad)

                  # Calculate the tangential caustic coordinates
                  xt = b*(((np.sqrt(f)/Delta) * np.cos(phi)) - ((np.sqrt(f)/fprime)*np.arcsinh(fprime/f * np.cos(phi)))) + SIELens.y['value']
                  yt = -b*(((np.sqrt(f)/Delta) * np.sin(phi)) - ((np.sqrt(f)/fprime)*np.arcsin(fprime * np.sin(phi)))) - SIELens.x['value']

                  # ... and rotate it to match the lens
                  r,th = cart2pol(xt,yt)
                  xt,yt = pol2cart(r,th+SIELens.PA['value']*deg2rad)

                  return np.atleast_3d([[xr,yr],[xt,yt]])

      else: # Blerg, complicated expressions... Keeton+00, but at least radial pseudo-caustic doesn't depend on shear       
            s, sa = Shear.shear['value'], (Shear.shearangle['value']-SIELens.PA['value'])*deg2rad            

            if np.isclose(f,1.):
                  rcrit = b * (1.+s*np.cos(2*(phi-sa)))/(1.-s**2.)

                  xr = -b*np.cos(phi) + SIELens.y['value']
                  yr = b*np.sin(phi) - SIELens.x['value']

                  xt = (np.cos(phi) + s*np.cos(phi-2*sa))*rcrit + xr
                  yt = (np.sin(-phi) - s*np.sin(-phi+2*sa))*rcrit + yr

                  r,th = cart2pol(xt,yt)
                  xt,yt = pol2cart(r,th+SIELens.PA['value']*deg2rad)
                  
                  r,th = cart2pol(xr,yr)
                  xr,yr = pol2cart(r,th+SIELens.PA['value']*deg2rad)
                  
                  return np.atleast_3d([[xr,yr],[xt,yt]])

            else:
                  rcrit = np.sqrt(2.*f)*b*(1.+s*np.cos(2.*(phi-sa))) / ((1.-s**2.)*np.sqrt((1+f**2.) - (1-f**2.)*np.cos(2*phi)))

                  xi = np.sqrt((2*(1-f**2.)) / ((1+f**2.)-(1-f**2.)*np.cos(2*phi)))
                  xr = -(b*np.sqrt(f)/fprime)*np.arctanh(xi*np.sin(phi)) + SIELens.y['value']
                  yr = (b*np.sqrt(f)/fprime)*np.arctan(xi*np.cos(phi)) - SIELens.x['value']

                  xt = (np.sin(phi)-s*np.sin(phi-2*sa))*rcrit + xr
                  yt = (np.cos(np.pi-phi)+s*np.cos(np.pi-phi+2*sa))*rcrit + yr                  

                  r,th = cart2pol(xt,yt)
                  xt,yt = pol2cart(r,th+SIELens.PA['value']*deg2rad)

                  r,th = cart2pol(xr,yr)
                  xr,yr = pol2cart(r,th+SIELens.PA['value']*deg2rad)

                  return np.atleast_2d([[xr,yr],[xt,yt]])
