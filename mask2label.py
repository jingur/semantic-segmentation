import numpy as np
import scipy.misc
import argparse
import os
import imageio


def read_masks(path):
        '''
        Read masks from directory and tranform to categorical
        '''
        masks = np.empty((512, 512), dtype = "int")
        mask = imageio.imread(path)
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[mask == 2] = 3  # (Green: 010) Forest land 
        masks[mask == 1] = 4  # (Blue: 001) Water 
        masks[mask == 7] = 5  # (White: 111) Barren land 
        masks[mask == 0] = 6  # (Black: 000) Unknown
        masks[mask == 4] = 6  # (Red: 100) Unknown
        
        return masks
