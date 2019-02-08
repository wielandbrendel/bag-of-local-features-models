import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, transform

def plot_heatmap(heatmap, original, ax, cmap='RdBu_r', 
                 percentile=99, dilation=0.5, alpha=0.25):
    """
    Plots the heatmap on top of the original image 
    (which is shown by most important edges).
    
    Parameters
    ----------
    heatmap : Numpy Array of shape [X, X]
        Heatmap to visualise.
    original : Numpy array of shape [X, X, 3]
        Original image for which the heatmap was computed.
    ax : Matplotlib axis
        Axis onto which the heatmap should be plotted.
    cmap : Matplotlib color map
        Color map for the visualisation of the heatmaps (default: RdBu_r)
    percentile : float between 0 and 100 (default: 99)
        Extreme values outside of the percentile range are clipped.
        This avoids that a single outlier dominates the whole heatmap.
    dilation : float
        Resizing of the original image. Influences the edge detector and
        thus the image overlay.
    alpha : float in [0, 1]
        Opacity of the overlay image.
    
    """
    if len(heatmap.shape) == 3:
        heatmap = np.mean(heatmap, 0)
    
    dx, dy = 0.05, 0.05
    xx = np.arange(0.0, heatmap.shape[1], dx)
    yy = np.arange(0.0, heatmap.shape[0], dy)
    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    cmap_original = plt.get_cmap('Greys_r')
    cmap_original.set_bad(alpha=0)
    overlay = None
    if original is not None:
        # Compute edges (to overlay to heatmaps later)
        original_greyscale = original if len(original.shape) == 2 else np.mean(original, axis=-1)
        in_image_upscaled = transform.rescale(original_greyscale, dilation, mode='constant', 
                                              multichannel=False, anti_aliasing=True)
        edges = feature.canny(in_image_upscaled).astype(float)
        edges[edges < 0.5] = np.nan
        edges[:5, :] = np.nan
        edges[-5:, :] = np.nan
        edges[:, :5] = np.nan
        edges[:, -5:] = np.nan
        overlay = edges
    
    abs_max = np.percentile(np.abs(heatmap), percentile)
    abs_min = abs_max
    
    ax.imshow(heatmap, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
    if overlay is not None:
        ax.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_original, alpha=alpha)
        

def generate_heatmap_pytorch(model, image, target, patchsize):
    """
    Generates high-resolution heatmap for a BagNet by decomposing the
    image into all possible patches and by computing the logits for
    each patch.
    
    Parameters
    ----------
    model : Pytorch Model
        This should be one of the BagNets.
    image : Numpy array of shape [1, 3, X, X]
        The image for which we want to compute the heatmap.
    target : int
        Class for which the heatmap is computed.
    patchsize : int
        The size of the receptive field of the given BagNet.
    
    """
    import torch
    
    with torch.no_grad():
        # pad with zeros
        _, c, x, y = image.shape
        padded_image = np.zeros((c, x + patchsize - 1, y + patchsize - 1))
        padded_image[:, (patchsize-1)//2:(patchsize-1)//2 + x, (patchsize-1)//2:(patchsize-1)//2 + y] = image[0]
        image = padded_image[None].astype(np.float32)
        
        # turn to torch tensor
        input = torch.from_numpy(image).cuda()
        
        # extract patches
        patches = input.permute(0, 2, 3, 1)
        patches = patches.unfold(1, patchsize, 1).unfold(2, patchsize, 1)
        num_rows = patches.shape[1]
        num_cols = patches.shape[2]
        patches = patches.contiguous().view((-1, 3, patchsize, patchsize))

        # compute logits for each patch
        logits_list = []

        for batch_patches in torch.split(patches, 1000):
            logits = model(batch_patches)
            logits = logits[:, target][:, 0]
            logits_list.append(logits.data.cpu().numpy().copy())

        logits = np.hstack(logits_list)
        return logits.reshape((224, 224))