import numpy as np
from skimage import exposure

def znorm_rescale(img):
    """Applies Z-score normalization and rescaling to a MRI image."""

    # Z-score norm
    movingNan=np.copy(img)
    movingNan[movingNan==0]=np.nan
    movingMean=np.nanmean(movingNan)
    movingSTD=np.nanstd(movingNan)
    moving=(img-movingMean)/movingSTD
    b=255/(1-(moving.max()/moving.min()))
    a=-b/moving.min()
    movingNorm=np.copy(moving)
    movingNorm=np.round((movingNorm*a)+b,2)

    # Rescaling
    p2, p98 = np.percentile(movingNorm, (1, 99)) # These parameters may not be optimal, further testing could be done
    moving_rescale = exposure.rescale_intensity(movingNorm, in_range=(p2, p98))

    return moving_rescale

X_START, X_END, Y_START, Y_END, Z_START, Z_END = (56,184, 24,216, 14,142)

def center_crop(img):
    return img[X_START:X_END, Y_START:Y_END, Z_START:Z_END]

def undo_center_crop(input):
    out = np.zeros((240, 240, 155))
    out[X_START:X_END, Y_START:Y_END, Z_START:Z_END] = input 
    return out