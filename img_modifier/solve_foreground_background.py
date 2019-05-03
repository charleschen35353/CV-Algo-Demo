

import numpy as np


CONST_ALPHA_MARGIN = 0.4



def get_original_background(image, alpha):
    falpha = alpha.flatten()
    weights = falpha < CONST_ALPHA_MARGIN
    background = weights[:, np.newaxis] * image.reshape((alpha.size, -1))
    background = background.reshape(*image.shape)
    return background



if __name__ == "__main__":
    main()
