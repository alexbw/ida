import numpy as np

def sh(i, figsize=(8,1)):
    from pylab import figure, imshow
    figure(figsize=figsize)
    if i.ndim == 3:
        imshow(np.swapaxes(i,0,1))
    else:
        imshow(i.T)

def get_norep(labels):
    labels_norep = []
    durations = []
    labels_norep.append(labels[0])
    durations.append(1)
    for i in range(1,len(labels)):
        if labels[i] == labels_norep[-1]:
            durations[-1] += 1
        else:
            labels_norep.append(labels[i])
            durations.append(1)
    return np.array(labels_norep), np.array(durations)

def find_highs(data, thresh=100, min_length=15):
    highs, durations = get_norep((data>thresh).astype('int32'))
    idx = np.argwhere((durations>min_length)&(highs==1)).ravel()
    cdur = np.r_[0,np.cumsum(durations)]
    regions = np.vstack([cdur[idx],cdur[idx+1]]).T
    return regions

def unique_indices(x,unique_x=None):
    if unique_x == None:
        unique_x = np.unique(x)
    new_x = np.zeros_like(x)
    for i,ux in enumerate(unique_x):
        new_x[x==ux] = i
    return new_x, unique_x

def do_shift(image, shift, dest_size = (150,20)):
    out = np.zeros(dest_size, dtype='float32')
    h,w = np.minimum(image.shape[0]-shift, image.shape[0]), np.minimum(image.shape[1], dest_size[1])
    if image.ndim == 2:
        out[:h,:w] = image[shift:,:w]
    elif image.ndim == 3:
        out[:h,:w] = image[shift:,:w,:]
    else: 
        raise ValueError("Incorrect image dimensions")
    return out

def smooth(signal, std=2):
    "Smooths a 1D signal"
    from scipy.signal import gaussian
    smoothingKernel = gaussian(std*5,std)
    smoothingKernel /= np.sum(smoothingKernel)
    signal = np.convolve(signal, smoothingKernel, 'same')    
    return signal

def local_minima(a):
    return np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True]

def local_maxima(a):
    return local_minima(-a)

def get_boundary_data(data, k=7, sigma=3):
    w = np.r_[[0]*(k/2),data[k:] - data[:-k],[0]*(k/2)]
    return smooth(w,sigma)

def extract_redness(img, saturation_threshold=0.5):
    import skimage.color
    img_lab = skimage.color.rgb2hsv(img).astype('float32')
    saturation_img = img_lab[:,:,1]
    value_img = img_lab[:,:,2]
    value_img[saturation_img < saturation_threshold] = 1.0
    SoverV = saturation_img/value_img
    return SoverV
    
def detect_plug(this_data, threshold = 0.1, valid_range=(3,25)):
    d = np.diff(this_data)
    d[d<threshold] = 0.0
    peaks = np.argwhere(local_maxima(d)).ravel()
    peaks = peaks[(peaks > valid_range[0]) & (peaks < valid_range[1])]
    if len(peaks) > 0:
        return peaks[np.argmax(d[peaks]).ravel()[0]].ravel()[0]-1
    else:
        return -1