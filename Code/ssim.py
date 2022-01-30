import sys
import numpy
from scipy import signal
from scipy import ndimage

import gauss


def ssim(img1, img2, cs_map=False):
  
    img1 = img1.astype(numpy.float64)
    img2 = img2.astype(numpy.float64)
    size = 11
    sigma = 1.5
    window = gauss.fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve( img1, window, mode='valid')
    mu2 = signal.fftconvolve(  window, img2, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

def msssim(img1, img2):
   
    level = 5
    weight = numpy.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = numpy.ones((2, 2))/4.0
    im1 = img1.astype(numpy.float64)
    im2 = img2.astype(numpy.float64)
    mssim = numpy.array([])
    mcs = numpy.array([])
    for l in range(level):
        ssim_map, cs_map = ssim(im1, im2, cs_map=True)
        mssim = numpy.append(mssim, ssim_map.mean())
        mcs = numpy.append(mcs, cs_map.mean())
        filtered_im1 = ndimage.filters.convolve(im1, downsample_filter, 
                                                mode='reflect')
        filtered_im2 = ndimage.filters.convolve(im2, downsample_filter, 
                                                mode='reflect')
        im1 = filtered_im1[::2, ::2]
        im2 = filtered_im2[::2, ::2]
    return (numpy.prod(mcs[0:level-1]**weight[0:level-1])*
                    (mssim[level-1]**weight[level-1]))

def main():
    """Compute the SSIM index on two input images specified on the cmd line."""
    import pylab
    argv = sys.argv
    if len(argv) != 3:
        print (sys.stderr, 'usage: python -m sp.ssim image1.tif image2.tif')
        sys.exit(2)

    try:
        from PIL import Image
        img1 = numpy.asarray(Image.open(argv[1]))
        img2 = numpy.asarray(Image.open(argv[2]))
    except Exception as e:
        e = 'Cannot load images' + str(e)
        print  (sys.stderr, e)

    ssim_map = ssim(img1, img2)
    ms_ssim = msssim(img1, img2)

    pylab.figure()
    pylab.subplot(131)
    pylab.title('Image1')
    pylab.imshow(img1, interpolation='nearest', cmap=pylab.gray())
    pylab.subplot(132)
    pylab.title('Image2')
    pylab.imshow(img2, interpolation='nearest', cmap=pylab.gray())
    pylab.subplot(133)
    pylab.title('SSIM Map\n SSIM: %f\n MSSSIM: %f' % (ssim_map.mean(), ms_ssim))
    pylab.imshow(ssim_map, interpolation='nearest', cmap=pylab.gray())
    pylab.show()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())