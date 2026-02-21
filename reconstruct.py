import numpy as np
import matplotlib.pyplot as plt
from cam_utils import Camera,triangulate

def decode(imprefix, start, threshold, suffix="_u"):
    """
    Given a sequence of 20 images of a scene showing projected 10 bit gray code, 
    decode the binary sequence into a decimal value in (0,1023) for each pixel.
    Mark those pixels whose code is likely to be incorrect based on the user 
    provided threshold.  Images are assumed to be named "imageprefixN.png" where
    N is a 2 digit index (e.g., "img00.png,img01.png,img02.png...")
    Parameters
    ----
    imprefix : str
       Image name prefix
      
    start : int
       Starting index
       
    threshold : float
       Threshold to determine if a bit is decodeable
       
    Returns
    -------
    code : 2D numpy.array (dtype=float)
        Array the same size as input images with entries in (0..1023)
        
    mask : 2D numpy.array (dtype=logical)
        Array indicating which pixels were correctly decoded based on the threshold
    
    """
    
    # we will assume a 10 bit code
    nbits = 10

    # to store recovered gray bits
    gray_bits = []

    # to store undecodable masks for each bit
    undecodable_masks = []

    # loop from n = 0 to n = 9 (nbits - 1)
    for nbit in range(nbits):
        # images in pairs
        index = start + 2*nbit
      #   image1 = plt.imread(f"{imprefix}{index:02d}.png") # formatting is ##
      #   image2 = plt.imread(f"{imprefix}{index+1:02d}.png") # formatting is ##
        image1 = plt.imread(f"{imprefix}{index:02d}{suffix}.png")
        image2 = plt.imread(f"{imprefix}{index+1:02d}{suffix}.png")

        # convert images to grayscale
        if image1.ndim == 3:
           image1 = np.mean(image1,axis=2)
        if image2.ndim == 3:
           image2 = np.mean(image2,axis=2)
        
        # convert images to float date type and scale to [0,1]
        if image1.dtype == np.uint8:
            image1 = image1.astype(float) / 256
        if image2.dtype == np.uint8:
           image2 = image2.astype(float) / 256

        # use the absolute differece to determine undecodable pixels
        difference = np.abs(image1 - image2)
        undecodable_pixels = difference < threshold
        undecodable_masks.append(undecodable_pixels)

        # gray code bit (image1 > image2)
        bit = image1 > image2
        gray_bits.append(bit)
        
    # create a stack to binary bits and undecodable masks
    gray_bits_stack = np.stack(gray_bits, axis=0)
    undecodable_masks_stack = np.stack(undecodable_masks, axis=0)

    # identify the mask and "bad" pixels (any of the 10 bits were undecodeable)
    bad_pixels = np.any(undecodable_masks_stack, axis=0)
    mask = np.logical_not(bad_pixels)

    # convert 10 bit gray code to binary
    binary_bits = [gray_bits_stack[0]]
    for i in range(1, nbits):
        # XOR
        binary_bits.append(binary_bits[i-1] ^ gray_bits_stack[i])
    binary_stack = np.stack(binary_bits, axis=0)
   
    # convert binary to decimal
    code = (binary_stack * (2 ** np.arange(nbits)[::-1]).reshape((nbits, 1, 1))).sum(axis=0)
        
    return code,mask

def reconstruct(imprefixL, imprefixR, threshold, diff_thr, camL, camR, color_obj_path, color_bkg_path, use_denoiser=False, denoiser_device="cpu", use_depth_model=False, depth_device="cpu", midas_type="MiDaS_small"):
    """
    Performing matching and triangulation of points on the surface using structured
    illumination. This function decodes the binary graycode patterns, matches 
    pixels with corresponding codes, and triangulates the result.
    
    The returned arrays include 2D and 3D coordinates of only those pixels which
    were triangulated where pts3[:,i] is the 3D coordinte produced by triangulating
    pts2L[:,i] and pts2R[:,i]

    Parameters
    ----------
    imprefixL, imprefixR : str
        Image prefixes for the coded images from the left and right camera
        
    threshold : float
        Threshold to determine if a bit is decodeable
   
    camL,camR : Camera
        Calibration info for the left and right cameras

    color_obj_path : str
        File path to the color image of the scene with the object present

    color_bkg_path : str
        File path to the background image with no object present
        
    Returns
    -------
    pts2L,pts2R : 2D numpy.array (dtype=float)
        The 2D pixel coordinates of the matched pixels in the left and right
        image stored in arrays of shape 2xN
        
    pts3 : 2D numpy.array (dtype=float)
        Triangulated 3D coordinates stored in an array of shape 3xN

    colors : 2D numpy.array
        Color values of the image
    """

    # Decode the H and V coordinates for the two views
    HL, HL_mask = decode(imprefixL,  0, threshold)
    VL, VL_mask = decode(imprefixL, 20, threshold)
    HR, HR_mask = decode(imprefixR,  0, threshold)
    VR, VR_mask = decode(imprefixR, 20, threshold)

    # Mask for valid codes
    final_decodeL = HL_mask | VL_mask
    final_decodeR = HR_mask | VR_mask

    # Construct the combined 20 bit code C = H + 1024*V and mask for each view
    CL = 1024 * HL + VL
    CR = 1024 * HR + VR

    # Use the object and background images to create a mask for the object
    obj = plt.imread(color_obj_path)
    bkg = plt.imread(color_bkg_path)

    # Optional PyTorch denoiser: convert to float [0,1], apply, then continue
    if use_denoiser:
        try:
            from torch_denoiser import denoise_numpy_image
        except Exception:
            denoise_numpy_image = None

        if denoise_numpy_image is not None:
            # ensure float in [0,1]
            def to_float01(im):
                imf = im.astype(float)
                if imf.max() > 1.0:
                    imf = imf / 255.0
                return imf

            objf = to_float01(obj.copy())
            bkgf = to_float01(bkg.copy())

            try:
                obj = denoise_numpy_image(objf, device=denoiser_device)
                bkg = denoise_numpy_image(bkgf, device=denoiser_device)
            except Exception:
                # on any failure, fall back to original images
                obj = objf
                bkg = bkgf

    # Convert to grayscale if needed
    if obj.ndim == 3:
        obj = obj.mean(axis=2)
    if bkg.ndim == 3:
        bkg = bkg.mean(axis=2)

    # Object mask
    diff = np.abs(obj - bkg)
    obj_mask = diff > diff_thr

    # Final masks
    maskL = final_decodeL & obj_mask
    maskR = final_decodeR

    # Find the indices of pixels in the left and right code image that 
    # have matching codes. If there are multiple matches, just
    # choose one arbitrarily.
    _, mL, mR = np.intersect1d(CL[maskL], CL[maskR], return_indices=True)

    h, w = CL.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    xxL = np.reshape(xx[maskL], (-1, 1))
    yyL = np.reshape(yy[maskL], (-1, 1))
    xxR = np.reshape(xx[maskR], (-1, 1))
    yyR = np.reshape(yy[maskR], (-1, 1))

    # Use matching indices to select coordinates
    pts2L = np.concatenate((xxL[mL], yyL[mL]), axis=1).T 
    pts2R = np.concatenate((xxR[mR], yyR[mR]), axis=1).T

    # Triangulate the points
    pts3 = triangulate(pts2L, camL, pts2R, camR)

    # Get the colors
    colors = obj[pts2L[1].astype(int), pts2L[0].astype(int)]
    colors = colors.T

    # Optional learned depth comparison using MiDaS
    if use_depth_model:
        try:
            from torch_depth import load_midas, predict_depth_numpy, depth_to_point_cloud, compare_point_clouds
            # predict depth on left color image (obj may be float or uint8)
            img = obj.copy()
            if img.dtype != np.uint8:
                img_for = (np.clip(img, 0.0, 1.0) * 255).astype('uint8')
            else:
                img_for = img

            model, transform = load_midas(midas_type, device=depth_device)
            depth_map = predict_depth_numpy(img_for, model, transform, device=depth_device)

            # Convert predicted depth to point cloud in world coords
            pc_pred = depth_to_point_cloud(depth_map, camL, mask=obj_mask)

            # Compare geometric pts3 (3xN) with predicted point cloud
            try:
                compare_point_clouds(pts3, pc_pred)
            except Exception as e:
                print("Depth comparison failed:", e)
        except Exception as e:
            print("Depth model unavailable or failed:", e)

    return pts2L, pts2R, pts3, colors