from SSIM_PIL import compare_ssim
from PIL import Image

image1 = Image.open(open(r"D:\ssm\IFCNN-MAX-RGB-RGB.png", 'rb'))
image1.show()
image2 = Image.open(open(r"D:\ssm\01_A.jpg", 'rb'))
image2.show()
value = compare_ssim(image1, image2)

print(value)