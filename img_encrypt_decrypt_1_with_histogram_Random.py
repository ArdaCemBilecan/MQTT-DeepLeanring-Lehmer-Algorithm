"""

*** CLASSIC RANDOM GENERATOR BASED BITWISE (XOR-based) ENCRYTPTION/DECRYPTION ***

After loading, the original image is stored in a 3D array with r rows, c columns and t color values between 0 and 255.
Next, a key is generated that corresponds to an identically sized array whose elements are randomly generated.
For encryption, it is iterated over the original image and each value of the image array ix XORed with the corresponding value of the key array.
For decryption, an analogous iteration is performed over the encrypted image and each value of the encrypted image array is XORed with the corresponding value of the key array.

Please note that this logic is not to be considered as the right way, but as a possible one. It is based on a one-time-pad encryption, which is information-theoretically secure when used as specified. However, this is not fulfilled here, since the key would have to be chosen truly randomly, whereas here it is chosen pseudo randomly. For a one-time-pad, the key has to be as long as the message, i.e. the key array has to be as large as the image data. In principle, other algorithms can be used as well, e.g. AES, wich is however more complex to implement (with regard to padding, IV etc.).

Note also that when storing the encrypted image, a format must be used that does not change the image data (which is actually the ciphertext). Otherwise, after loading the encrypted image, decryption may be incorrect or even impossible, depending on the algorithm. A format that generally compresses the data and thus changes the data is e.g. jpg, a format that does not change the data is e.g. bmp, see e.g. here for more details. The format can be controlled by the file extension when saving with imwrite.

"""
from __future__ import print_function

print(__doc__)

import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import os


# Load original image
original_image = cv2.imread("test/2012_002808.png")
r = original_image.shape[0]
c = original_image.shape[1]
t = original_image.shape[2]

# Display original image
#cv2.imshow("Original image", original_image)
#cv2.waitKey()

# Create random encryption key
key = np.random.randint(256, size = (r, c, t))
print("Encryption key:\n")
print(key)
### Random anahtari hardiske kaydedelim
import base64
key_save = base64.urlsafe_b64encode(key)
key_file = open("classic_random_key_file.txt", "wb")
key_file.write(key_save)
key_file.close()
######################################################

# Encryption
# Iterate over the image
encrypted_image = np.zeros((r, c, t), np.uint8)
for row in range(r):
    for column in range(c):
        for depth in range(t):
            encrypted_image[row, column, depth] = original_image[row, column, depth] ^ key[row, column, depth] 
            
#cv2.imshow("Encrypted image", encrypted_image)
filename = "enc/1.png"
imgSize = (r, c)
cv2.imwrite(filename, encrypted_image)
#cv2.waitKey()

# Decryption
# Iterate over the encrypted image
decrypted_image = np.zeros((r, c, t), np.uint8)
for row in range(r):
    for column in range(c):
        for depth in range(t):
            decrypted_image[row, column, depth] = encrypted_image[row, column, depth] ^ key[row, column, depth] 
            
#cv2.imshow("Decrypted Image", decrypted_image)
filename2 = "decrpyted_img.png"
imgSize2 = (r, c)
cv2.imwrite(filename2, decrypted_image)
#cv2.waitKey()



def calc_npcr(ori_img, enc_img):
    diff = np.array(ori_img) - np.array(enc_img)
    diff = np.abs(diff)
    binary_diff = np.where(diff > 0.5, 1, 0)
    total_sum = 0
    for i in range(ori_img.shape[2]):
        total_sum += np.sum(np.sum(binary_diff[:, :, i]))
    total_sum = total_sum * 100.0 / (ori_img.shape[0] * ori_img.shape[1] * ori_img.shape[2])

    return total_sum

def calc_uaci(ori_img, enc_img):
    diff = np.array(ori_img) - np.array(enc_img)

    total_sum = 0
    for i in range(ori_img.shape[2]):
        total_sum += sum(sum(np.abs(diff)[:, :, i]))
    total_sum = total_sum * 100.0 / (ori_img.shape[0] * ori_img.shape[1] * ori_img.shape[2])

    return total_sum / 255.0


print("\n\nNumber of pixels change rate (NPCR) degeri:\n")
print(calc_npcr(original_image, encrypted_image))
print("\n")
print("The unified average changing intensity (UACI) degeri:\n")
print(calc_uaci(original_image, encrypted_image))
print("\n")

from PIL import Image

def entropy(signal):
        '''
        function returns entropy of a signal
        signal must be a 1-D numpy array
        '''
        lensig=signal.size
        symset=list(set(signal))
        numsym=len(symset)
        propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]
        ent=np.sum([p*np.log2(1.0/p) for p in propab])
        return ent

def imageEntropy(file_name, image_caption, image_height, image_width, DPI):
        colorIm=Image.open(file_name)
        base = os.path.basename(file_name)
        name = os.path.splitext(base)[0]
        ext = os.path.splitext(base)[1]
        greyIm=colorIm.convert('L')
        colorIm=np.array(colorIm)
        greyIm=np.array(greyIm)
        N=5
        S=greyIm.shape
        E=np.array(greyIm)
        for row in range(S[0]):
                for col in range(S[1]):
                        Lx=np.max([0,col-N])
                        Ux=np.min([S[1],col+N])
                        Ly=np.max([0,row-N])
                        Uy=np.min([S[0],row+N])
                        region=greyIm[Ly:Uy,Lx:Ux].flatten()
                        E[row,col]=entropy(region)

        plt.subplot(1,3,3)
        plt.imshow(E, cmap=plt.cm.jet)
        plt.xlabel(image_caption)
        figure = plt.gcf() # get current figure
        figure.set_size_inches(image_height, image_width)#in inches
        # when saving, specify the DPI
        new_file_name = "entropy_"+name+ext
        print("\nThe filename is: "+new_file_name)
        plt.savefig(new_file_name, dpi = DPI, bbox_inches='tight')
        #plt.colorbar()
        ## Ozellikle plot show kismini kapattim zaten harddiske kaydediyor elde ettigi entropy resmin        
        #plt.show()
        ## Entropy degeri dizisini yazdiralim. Bu dzinin ortalam degerini buldurulam
        print("Entropy array of given image:\n")
        print(E)
        print("Average Entropy of given image:\n")
        print(np.average(E))



imageEntropy('test/2012_002808.png', "Entropy of original image in 10 x 10 neighbourhood", 20, 10, 600)
imageEntropy('enc/1.png', "Entropy of encrypte image in 10 x 10 neighbourhood", 20, 10, 600)
imageEntropy('decrpyted_img.png', "Entropy of decrypte image in 10 x 10 neighbourhood", 20, 10, 600)

from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.morphology import disk
from skimage.morphology import ball
from skimage.filters import rank

plt.rcParams['font.size'] = 11

def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins)
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')

    xmin, xmax = dtype_range[image.dtype.type]
    ax_hist.set_xlim(xmin, xmax)

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')

    return ax_img, ax_hist, ax_cdf

# Load an original image
orig_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
orig_img = img_as_ubyte(orig_img)

# Encrytped image's histogram
img_enc_hist = cv2.cvtColor(encrypted_image, cv2.COLOR_BGR2RGB)
img_enc_hist = img_as_ubyte(img_enc_hist)

# Decrytped image's histogram
img_dec_hist = cv2.cvtColor(decrypted_image, cv2.COLOR_BGR2RGB)
img_dec_hist = img_as_ubyte(img_dec_hist)

# Display results
fig = plt.figure(figsize=(8, 5))
axes = np.zeros((2, 3), dtype=object)
axes[0, 0] = plt.subplot(2, 3, 1)
axes[0, 1] = plt.subplot(2, 3, 2, sharex=axes[0, 0], sharey=axes[0, 0])
axes[0, 2] = plt.subplot(2, 3, 3, sharex=axes[0, 0], sharey=axes[0, 0])
axes[1, 0] = plt.subplot(2, 3, 4)
axes[1, 1] = plt.subplot(2, 3, 5)
axes[1, 2] = plt.subplot(2, 3, 6)

ax_img, ax_hist, ax_cdf = plot_img_and_hist(orig_img, axes[:, 0])
ax_img.set_title('Original image')
ax_hist.set_ylabel('Number of pixels')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_enc_hist, axes[:, 1])
ax_img.set_title('Encrypted image')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_dec_hist, axes[:, 2])
ax_img.set_title('Decrypted image')
ax_cdf.set_ylabel('Fraction of total intensity')

# prevent overlap of y-axis labels
fig.tight_layout()
new_histogram_file_name = "histogram_org_enc_dec.png"
plt.savefig(new_histogram_file_name, dpi = 600, bbox_inches='tight')
plt.show()
####################################################
####################################################

######################################
#cv2.waitKey()
cv2.destroyAllWindows()