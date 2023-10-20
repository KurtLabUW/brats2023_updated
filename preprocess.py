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
    p2, p98 = np.percentile(movingNorm, (1, 99))
    moving_rescale = exposure.rescale_intensity(movingNorm, in_range=(p2, p98))

    return moving_rescale