import os
from pdf2image import convert_from_path
import time

print("\nConversion started..\n\n")
start_time=time.time()

for file in os.listdir('PDFs/'):
	images = convert_from_path(os.path.abspath(os.path.join('PDFs/',file)))

	for i, image in enumerate(images):
		fname = "image"+str(i)+'.jpg'
		img = str(os.path.abspath(os.path.join('images_all/',file)))[:-4]+"-"+str(i)+'.jpg'
		image.save(img,"JPEG")

end_time=time.time()

print('-'*40)
print("Conversion complete!!")
print('-'*40)

print("Time taken: ",round((end_time-start_time),3),"seconds")