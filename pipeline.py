from keras_retinanet.utils.image import preprocess_image
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.image import resize_image
from keras_retinanet import models

from pdf2image import convert_from_path
import matplotlib.pyplot as plt
from PyPDF2 import PdfFileWriter, PdfFileReader

import numpy as np
import cv2
import camelot
import copy

# load the class label mappings
#model for headers
#labels = '/content/gdrive/My Drive/Colab Notebooks/retinanet_classes_RN3.csv'

#model for table
labels = 'keras-retinanet/retinanet_classes.csv'

LABELS = open(labels).read().strip().split("\n")
print(LABELS)
LABELS = {int(L.split(",")[1]): L.split(",")[0] for L in LABELS}
print(LABELS)

#model for table
model_path = 'keras-retinanet/output.h5'
model = models.load_model(model_path, backbone_name='resnet50')

def norm_pdf_page(pdf_file, pg):
    pdf_doc = PdfFileReader(open(pdf_file, "rb"))
    pdf_page = pdf_doc.getPage(pg-1)
    pdf_page.cropBox.upperLeft = (0, list(pdf_page.mediaBox)[-1])
    pdf_page.cropBox.lowerRight = (list(pdf_page.mediaBox)[-2], 0)
    return pdf_page

def pdf_page2img(pdf_file, pg, save_image=True):
    img_page = convert_from_path(pdf_file, first_page=pg, last_page=pg)[0]
    if save_image:
        img=pdf_file[:-4]+"-"+str(pg)+".jpg"
        img_page.save(img)
    return np.array(img_page), img

def img_dim(img, bbox):
    H_img,W_img,_=img.shape
    x1_img, y1_img, x2_img, y2_img=bbox
    w_table, h_table=x2_img-x1_img, y2_img-y1_img
    return [[x1_img, y1_img, x2_img, y2_img], [w_table, h_table], [H_img,W_img]]

def norm_bbox(img, bbox, x_corr=0.05, y_corr=0.05):
    [[x1_img, y1_img, x2_img, y2_img], [w_table, h_table], [H_img,W_img]]=img_dim(img, bbox)
    x1_img_norm,y1_img_norm,x2_img_norm,y2_img_norm=x1_img/W_img, y1_img/H_img, x2_img/W_img, y2_img/H_img
    w_img_norm, h_img_norm=w_table/W_img, h_table/H_img
    w_corr=w_img_norm*x_corr
    h_corr=h_img_norm*x_corr

    return [x1_img_norm-w_corr,y1_img_norm-h_corr/2,x2_img_norm+w_corr,y2_img_norm+2*h_corr]

def bboxes_pdf(img, pdf_page, bbox, save_cropped=False):
    W_pdf=float(pdf_page.cropBox.getLowerRight()[0])
    H_pdf=float(pdf_page.cropBox.getUpperLeft()[1])
    print(bbox)

    [x1_img_norm,y1_img_norm,x2_img_norm,y2_img_norm]=norm_bbox(img, bbox)
    x1, y1 = x1_img_norm*W_pdf, (1-y1_img_norm)*H_pdf
    x2, y2 = x2_img_norm*W_pdf, (1-y2_img_norm)*H_pdf
    
    if save_cropped:
        page=copy.copy(pdf_page)
        page.cropBox.upperLeft = (x1, y1)
        page.cropBox.lowerRight = (x2, y2)
        output = PdfFileWriter()
        output.addPage(page)

        with open(pdf_file[:-4]+"-"+str(pg)+".pdf", "wb") as out_f:
            output.write(out_f)

    return [x1, y1, x2, y2]
    
def detect_table(imgfname):
    image_path = imgfname
    image = read_image_bgr(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = image.copy()
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    
    image = preprocess_image(image)
    (image, scale) = resize_image(image)
    image = np.expand_dims(image, axis=0)

    # detect objects in the input image and correct for the image scale

    (boxes, scores, labels) = model.predict_on_batch(image)
    boxes /= scale

    confidence = 0.2
    label_out = []

    # loop over the detections

    for (box, score, label) in zip(boxes[0], scores[0], labels[0]):

        # filter out weak detections

        if score < confidence:
            continue

        # convert the bounding box coordinates from floats to integers
        box = box.astype('int')

        # build the label and draw the label + bounding box on the output image
        labeli = label

        if LABELS[labeli] not in label_out:
          label_out.append(LABELS[labeli])
          print(box)

          return (box[0], box[1], box[2], box[3])

page = int(input("Enter the Page no: "))
PDF_PATH = 'PDFs/Yang_BERT_representations_for_Video_Question_Answering_WACV_2020_paper.pdf'

img_page = convert_from_path(PDF_PATH, first_page=page, last_page=page+1, output_folder="/content/", fmt="jpg")
pdf_file = PDF_PATH
pg = page
img, imgfname = pdf_page2img(pdf_file, pg, save_image=True)


import numpy as np
import cv2
import matplotlib.pyplot as plt

image_path = imgfname
#image = cv2.imread(image_path)
image = read_image_bgr(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

output = image.copy()
output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
print(output.shape) # row (height) x column (width) x color (3)

image = preprocess_image(image)
(image, scale) = resize_image(image)
image = np.expand_dims(image, axis=0)

# detect objects in the input image and correct for the image scale
(boxes, scores, labels) = model.predict_on_batch(image)
boxes /= scale

confidence =  0.2
label_out = []
result = ""

# loop over the detections
for (box, score, label) in zip(boxes[0], scores[0], labels[0]):
	# filter out weak detections
	if score < confidence:
		continue

	# convert the bounding box coordinates from floats to integers
	box = box.astype("int")

	# build the label and draw the label + bounding box on the output
	# image
	labeli = label
	label = "{}: {:.2f}".format(LABELS[label], score)
	print(label)
	
	if LABELS[labeli] not in label_out:
	    label_out.append(LABELS[labeli])
	    cv2.rectangle(output, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 12)
	    print(box[0])
	    print(box[1])
	    print(box[2])
	    print(box[3])
	    #result 		 
	    cv2.putText(output, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 4.5, (255, 1, 1), 12)
     	    
plt.figure(figsize=(20, 20))
plt.imshow(output)
plt.show()

## Get table

pdf_page=norm_pdf_page(PDF_PATH, pg)
x1, y1, x2, y2 = detect_table(imgfname)
print(x1, y1, x2, y2)

interesting_areas=[]

output = [[x1, y1, x2, y2]]
for x in output:
  [x1, y1, x2, y2] = bboxes_pdf(img, pdf_page, x)
  bbox_camelot = [
            ",".join([str(x1), str(y1), str(x2), str(y2)])
        ][0]  # x1,y1,x2,y2 where (x1, y1) -> left-top and (x2, y2) -> right-bottom in PDF coordinate space
        #print(bbox_camelot)
  interesting_areas.append(bbox_camelot)


print(interesting_areas)
output_camelot = camelot.read_pdf(
    filepath=pdf_file, pages=str(pg), flavor="stream", table_areas=interesting_areas
)

output_camelot[0].df

output_camelot=[x.df for x in output_camelot]
for i,db in enumerate(output_camelot):
  db.to_excel(pdf_file[:-4]+"-"+str(pg)+"-table-"+str(i)+".xlsx")