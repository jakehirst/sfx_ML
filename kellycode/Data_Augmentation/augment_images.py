from Posterize import *
from Rotation import *
from verticalflip import *
from horizontal_and_verticalflip import *
from horizontalflip import *
from Solarize import *
from AutoContrast import *
from Brightness import *
from Contrast import *
from Equalize import *
from Gaussian_Noise import *
from Identity import *
from Sharpness import *
from Shear import *
from Shift import *
from Zoom import *
from Color import *

pathname1 = 'C:\\Users\\u1056\\sfx\\images_sfx\\Original\\OG'
pathname2 = 'C:\\Users\\u1056\\sfx\\images_sfx\\Original\\' 

pathname1 = 'C:\\Users\\u1056\\sfx\\images_sfx\\Original_new_dataset\\OG'
pathname2 = 'C:\\Users\\u1056\\sfx\\images_sfx\\Original_new_dataset\\' 

pathname1 = 'C:\\Users\\u1056\\sfx\\images_sfx\\Visible_cracks_new_dataset_\\OG'
pathname2 = 'C:\\Users\\u1056\\sfx\\images_sfx\\Visible_cracks_new_dataset_\\'


# pathname1 = 'C:\\Users\\u1056\\sfx\\images_sfx\\Original_from_test_matrix\\OG'
# pathname2 = 'C:\\Users\\u1056\\sfx\\images_sfx\\Original_from_test_matrix\\' 

""" If you are running this, make sure to go ahead and do the color data augmentation as well. You have to do it individually because of the multiprocessing. """
#Color(pathname1, pathname2)


print("Zooming")
Zoom(pathname1, pathname2)
print("Shifting")
Shift(pathname1, pathname2)
print("Shearing")
Shear(pathname1, pathname2)
print("Sharpness")
Sharpness(pathname1, pathname2)
print("Identity")
Identity(pathname1, pathname2) 
print("GaussianNoise")
GaussianNoise(pathname1, pathname2)
print("Equalize")
Equalize(pathname1, pathname2)
print("Contrast")
Contrast(pathname1, pathname2)
print("Brightness")
Brightness(pathname1, pathname2)
print("autoContrast")
autoContrast(pathname1, pathname2)
print("vertical_flip")
vertical_flip(pathname1, pathname2)
print("horizontal_flip")
horizontal_flip(pathname1, pathname2)
print("horizontal_and_verticalflip")
horizontal_and_verticalflip(pathname1, pathname2)
print("rotate")
rotate(pathname1, pathname2)
print("posterize")
posterize(pathname1, pathname2)
print("solarize")
solarize(pathname1, pathname2)


