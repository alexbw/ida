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




def go(img_path,downsample_fact=0.05):
    import ndimage
    import skimage.io


    
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

    data = []
    for i,test_img in enumerate(test_imgs):
        subplot(gs[i])
        t = test_img[:,:,0]-test_img.mean(2)
        t = t[:150]
        data.append(t.mean(1)/t.mean(1).max())

    return data, position