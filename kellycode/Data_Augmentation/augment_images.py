from Posterize import *
from Color import *
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

pathname1 = 'C:\\Users\\u1056\\sfx\\images_sfx\\Original\\OG'
pathname2 = 'C:\\Users\\u1056\\sfx\\images_sfx\\Original\\' 

Identity(pathname1, pathname2)
GaussianNoise(pathname1, pathname2)
Equalize(pathname1, pathname2)
Contrast(pathname1, pathname2)
Brightness(pathname1, pathname2)
autoContrast(pathname1, pathname2)
vertical_flip(pathname1, pathname2)
horizontal_flip(pathname1, pathname2)
horizontal_and_verticalflip(pathname1, pathname2)
rotate(pathname1, pathname2)
posterize(pathname1, pathname2)
solarize(pathname1, pathname2)
Color(pathname1, pathname2)


