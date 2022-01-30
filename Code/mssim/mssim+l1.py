import numpy as np
from scipy import signal
from PIL import Image 
from scipy.ndimage.filters import convolve
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


tf.flags.DEFINE_string('original_image', None, 'Path to PNG image.')
tf.flags.DEFINE_string('compared_image', None, 'Path to PNG image.')
FLAGS = tf.flags.FLAGS



def _tf_fspecial_gauss(size, sigma, channels=1):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
 
    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)
 
    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)
 
    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)
 
    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
 
    window = g / tf.reduce_sum(g)
    return tf.tile(window, (1,1,channels,channels))

	
def tf_gauss_conv(img, filter_size=11, filter_sigma=1.5):
    _, height, width, ch = img.get_shape().as_list()
    size = min(filter_size, height, width)
    sigma = size * filter_sigma / filter_size if filter_size else 0
    window = _tf_fspecial_gauss(size, sigma, ch) # window shape [size, size]
    padded_img = tf.pad(img, [[0, 0], [size//2, size//2], [size//2, size//2], [0, 0]], mode="CONSTANT")
    return tf.nn.conv2d(padded_img, window, strides=[1,1,1,1], padding='VALID')
 

def tf_gauss_weighted_l1(img1, img2, mean_metric=True, filter_size=11, filter_sigma=1.5):
    diff = tf.abs(img1 - img2)
    L1 = tf_gauss_conv(diff, filter_size=filter_size, filter_sigma=filter_sigma)
    if mean_metric:
        return tf.reduce_mean(L1)
    else:
        return L1
 

def tf_ssim(img1, img2, cs_map=False, mean_metric=True, filter_size=11, filter_sigma=1.5):
    _, height, width, ch = img1.get_shape().as_list()
    size = min(filter_size, height, width)
    sigma = size * filter_sigma / filter_size if filter_size else 0
 
    window = _tf_fspecial_gauss(size, sigma, ch)
    K1 = 0.01
    K2 = 0.03
    L = 1  
    C1 = (K1*L)**2
    C2 = (K2*L)**2
 
    padded_img1 = tf.pad(img1, [[0, 0], [size//2, size//2], [size//2, size//2], [0, 0]], mode="CONSTANT")      
    padded_img2 = tf.pad(img2, [[0, 0], [size//2, size//2], [size//2, size//2], [0, 0]], mode="CONSTANT")       
    mu1 = tf.nn.conv2d(padded_img1, window, strides=[1,1,1,1], padding='VALID') 
    mu2 = tf.nn.conv2d(padded_img2, window, strides=[1,1,1,1], padding='VALID')
    mu1_sq = mu1*mu1     
    mu2_sq = mu2*mu2    
    mu1_mu2 = mu1*mu2   
 

    paddedimg11 = padded_img1*padded_img1
    paddedimg22 = padded_img2*padded_img2
    paddedimg12 = padded_img1*padded_img2
 
    sigma1_sq = tf.nn.conv2d(paddedimg11, window, strides=[1,1,1,1],padding='VALID') - mu1_sq  
    sigma2_sq = tf.nn.conv2d(paddedimg22, window, strides=[1,1,1,1],padding='VALID') - mu2_sq   
    sigma12 = tf.nn.conv2d(paddedimg12, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2    
 
    ssim_value = tf.clip_by_value(((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)), 0, 1)
    if cs_map:         
        cs_map_value = tf.clip_by_value((2*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2), 0, 1)    
        value = (ssim_value, cs_map_value)
    else:
        value = ssim_value
    if mean_metric:            
        value = tf.reduce_mean(value)
    return value
 
def tf_ssim_l1_loss(img1, img2, mean_metric=True, filter_size=11, filter_sigma=1.5, alpha=0.84):
    L1 = tf_gauss_weighted_l1(img1, img2, mean_metric=False, filter_size=filter_size, filter_sigma=filter_sigma)
    if mean_metric:
        loss_ssim= 1 - tf_ssim(img1, img2, cs_map=False, mean_metric=True, filter_size=filter_size, filter_sigma=filter_sigma)
        loss_L1 = tf.reduce_mean(L1)
        value = loss_ssim * alpha + loss_L1 * (1-alpha)
    else:
        loss_ssim= 1 - tf_ssim(img1, img2, cs_map=False, mean_metric=False, filter_size=filter_size, filter_sigma=filter_sigma)
        value = loss_ssim * alpha + L1 * (1-alpha)
 
    return value, loss_ssim


img1 = Image.open('01_A.jpg','r')
img2 = Image.open('IFCNN-MAX-RGB-RGB.png','r')
img1 = np.array(img1,dtype=np.float32).reshape([48,4,1043,8])
img2 = np.array(img2,dtype=np.float32).reshape([48,4,1043,8])
 
l1_loss = tf_ssim_l1_loss(tf.constant(img1),tf.constant(img2))
 
with tf.Session() as sess:
    print(sess.run(l1_loss))
