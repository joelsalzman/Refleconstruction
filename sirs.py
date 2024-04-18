# SINGLE IMAGE REFLECTANCE SEPARATION
import numpy as np
import utils
from scipy.fft import fft2, ifft2
from scipy.signal import convolve2d

def separate_layers(I, lambda_, beta_, i_max, eta_, lb, ub):
    """
    Separates front and back layers based on gradient smoothness prior.
    Implemented from "Single Image Layer Separation using Relative Smoothness"
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6909748
    This is essentially Algorithm 1

    Parameters
    ----------
    I : np.array[m, n, 3]
        Image
    lambda_ : float
        Smoothness weight
    beta_ : float
        Initial weight that we will increase during training
    i_max : int
        Number of iterations to perform
    eta_ : float
        Factor by which to increase the beta_ weight
    lb : float
        Lower bound for image values
    ub : float
        Upper bound for image values

    Returns
    -------
    L1 : np.array[m, n, 3]
        The sharper layer (specular)
    L2 : np.array[m, n, 3]
        The smoother layer (diffused)
    """
    
    # Pre-computed values
    L1 = I.copy()
    tau = 10e-16

    f1 = np.array([[-1, 1]])
    f2 = f1.T
    f3 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    nor1 = np.tile(psf2otf(f3, I.shape[:2])**2, (1, 1, 3)) * fft2(I)
    den1 = np.tile(psf2otf(f3, I.shape[:2])**2, (1, 1, 3))
    den2 = np.tile(
        psf2otf(f1, I.shape[:2])**2 + psf2otf(f2, I.shape[:2])**2, 
        (1, 1, 3)
    )

    # Training loop
    for _ in range(i_max):

        # Equation 7
        filtered = [
            np.stack([convolve2d(L1[:, :, c], f) for c in range(3)], axis=-1) 
            for f in [f1, f2, f3]
        ]
        g1, g2, _ = [np.where(f**2 > 1/beta_, f, 0) for f in filtered]
        
        # Equation 8
        nor2 = -np.diff(g1, axis=1) - np.diff(g2, axis=0)
        # nor2 = np.concatenate((
        #         (g1[:, -1, :] - g1[:, 0, :])[:, np.newaxis, :], 
        #         -np.diff(g1, axis=1)
        #         ), axis=1
        #     ) + np.concatenate((
        #         (g2[-1, :, :] - g2[0, :, :])[np.newaxis, :, :], 
        #         -np.diff(g2, axis=0)
        #         ), axis=0
        #     )
        nor = lambda_ * nor1 + beta_ * fft2(nor2)
        den = lambda_ * den1 + beta_ * den2 + tau
        L1 = np.real(ifft2(nor / den))

        # Equation 9
        for i in range(I.shape[2]):
            
            for _ in range(100):

                LB_t = L1[:, :, i]
                threshold = 1 / np.prod(LB_t.shape)
                dt_nor = np.sum(LB_t[LB_t < lb[:, :, i]])
                dt = -2 * (dt_nor + np.sum(LB_t[LB_t > ub[:, :, i]])) / \
                    np.prod(LB_t.shape)
                LB_t = LB_t + dt
                
                if np.abs(dt) < threshold:
                    break
        
            L1[:, :, i] = LB_t

        # Weight update
        beta_ *= eta_
        
        # Thresholding
        L1[L1 < lb] = lb[L1 < lb]
        L1[L1 > ub] = ub[L1 > ub]
    
    # Return values
    L2 = I - L1
    return L1, L2

def psf2otf(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.

    Args:
        psf: PSF array
        shape: Output shape of the OTF array

    Returns:
        otf: OTF array
    """
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf_padded = np.zeros(shape, dtype=np.complex128)
    psf_padded[:inshape[0], :inshape[1]] = psf
    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf_padded = np.roll(psf_padded, -int(axis_size / 2), axis=axis)
    # Compute the OTF
    otf = fft2(psf_padded)
    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf_padded.size * np.log2(psf_padded.shape))
    otf = np.real_if_close(otf, tol=n_ops)
    return otf[:, :, np.newaxis]

def binarize(img):
    
    return np.where(img.sum(axis=2), 1, 0)

def separate(img):
    
    return separate_layers(
        I = img,
        lambda_ = 0,
        beta_ = 10,
        i_max = 5,
        eta_ = 2,
        lb = np.zeros(img.shape),
        ub = img
    )

if __name__ == '__main__':
    from load import load_rgb
    img = load_rgb(r'data\parrot_test_5_Color.png')
    L1, L2 = separate(img)
    utils.imwrite(img, L1, L2)