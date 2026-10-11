import numpy as np

Z0_OHM = 376.730313668


def integrate(y, x, axis=-1):
    # numpy 1.21 environments and numpy 2.x use different public names.
    function = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return function(y, x, axis=axis)


def propagate(e, q, neff, wavelength_um, z_um, amplitude=1.0):
    """Return complex SI E[V/m], H[A/m], axes (z, x, xyz).

    q=e'/k0; exp(-i omega t) gives Hx=-neff*Ey/Z0 and Hz=-i*q/Z0.
    """
    phase = amplitude*np.exp(1j*2*np.pi/wavelength_um*neff*np.asarray(z_um))[:, None]
    shape = (len(z_um), len(e), 3)
    electric = np.zeros(shape, dtype=np.complex128)
    magnetic = np.zeros(shape, dtype=np.complex128)
    electric[..., 1] = phase*np.asarray(e)[None, :]
    magnetic[..., 0] = -neff*electric[..., 1]/Z0_OHM
    magnetic[..., 2] = -1j*phase*np.asarray(q)[None, :]/Z0_OHM
    return electric, magnetic


def poynting(electric, magnetic):
    return 0.5*np.real(np.cross(electric, np.conj(magnetic)))


def gaussian_projection(x, e, waist):
    """L2 modal projection in a uniform TE guide; radiation/higher modes omitted.

    Fraction is an electric-profile L2 fraction, not port coupling efficiency.
    """
    source = np.exp(-(np.asarray(x)/waist)**2)
    overlap = integrate(source*e, x)
    norm = integrate(e*e, x)
    coefficient = overlap/norm
    fraction = overlap**2/(norm*integrate(source*source, x))
    return coefficient, fraction, source
