import numpy as np
import scipy.ndimage as ndimage
import skimage.io

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

def process(img,downsample_fact=0.05):

    # Downsample the image
    r = []
    for i in range(3):
        r.append(ndimage.zoom(img[:,:,i], downsample_fact))
    r = np.dstack(r)
    rs = r.mean(2)

    # Extract light/dark boundaries
    h = ndimage.median_filter(np.max(rs,axis=1), (rs.shape[1]/10.0,))
    w = ndimage.median_filter(np.max(rs,axis=0), (rs.shape[0]/50.0,))

    # Find the index of the boundaries
    height_ticks = find_highs(h,min_length=100)
    width_ticks = find_highs(w,min_length=10)

    # Extract the sub-images
    test_imgs_bw = []
    test_imgs = []
    position = []
    for iheight,height_tick in enumerate(height_ticks):
        h1,h2 = height_tick
        for iwidth,width_tick in enumerate(width_ticks):
            w1,w2 = width_tick
            Ibw = rs[h1:h2,w1:w2]
            test_imgs_bw.append(Ibw)
            I = r[h1:h2,w1:w2,:]
            test_imgs.append(I)
            position.append((iheight,iwidth))
            
    # Save a copy of the extracted images
    out_imgs = test_imgs[:]
            
    # Shift the sub-images to remove the plug. 
    shifts = []
    for test_img, pos in zip(test_imgs, position):
        plug_data = np.mean(test_img[:,:,0] - test_img.mean(2), axis=1)
        plug_data = plug_data[2:] - plug_data[:-2] # lagged derivative, smooths things out a little
        locs = np.argwhere(local_maxima(plug_data[:20])).ravel() # only look at the beginning of the tube for the plug
        highs = [h+1 for h in locs[np.argsort(plug_data[locs])[::-1]]]
        if len(highs) == 0:
            print "NO PLUG FOUND FOR POSITION %s" % str(pos)
        shifts.append(highs[0])
    test_imgs = [I[shift:,:,:] for I,shift in zip(test_imgs, shifts)]
    
    # Extract fittable data
    data = [-np.median(test_img,axis=1).mean(1) for test_img in test_imgs]

    return data, position, out_imgs, shifts

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