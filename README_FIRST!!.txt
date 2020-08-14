################################################

Advanced Table Parsing using DL

################################################

1. virtualenv for this proj - venv

Dependencies:
1.pdf2image (pip install pdf2image)
2.poppler-utils
	* For Windows, download latest stable release from - "http://blog.alivate.com.au/poppler-windows/"
	* Extract the contents, place "directory_location/bin" under path
	as the environment variable.
3. "labelimg" - tool for annotating images
	* Clone the repository - https://github.com/tzutalin/labelImg

	You need to build the tool from source:
	* PyQt5 (pip install pyqt5)
	* lxml (pip install lxml)
	* Go to the directory. Run the command: 
		"pyrcc5 -o resources.py resources.qrc"
	*  Run the command: "python labelImg.py" and the labelImg tool will launch.

4. keras==2.3.1 
You may run into some runtime errors. 
Refer: https://github.com/tensorflow/tensorflow/issues/38589

5. tensorflow==2.1.0
6. PyPDF2 (pip install PyPDF2)
7. camelot-py[cv] (pip install camelot-py[cv])

* pdf_to_image.py - Script to read PDF(s) from "PDFs" folder 
and convert all the pages of the PDFs to ".jpg" format in "images"
folder for creating training database.

* labelImg tool - Use the tool to select the "images" folder and label each
  table in each image with class "table". Save the contents in "annotations"
  folder as XML.
  NOTE: Save the annotation format as PASCALVOC.
* Navigate to the folder "dataset" and from the "images" folder in the root 	  directory, place images in "images_train"/"images_test" respectively to create
  the training/testing dataset. 
  	Run the command: "python jpg_to_txt.py" and it will add the images' names in train.txt/test.txt respectively.
* Have all the images ( combined train+test stored in Images folder) and delete the folders "images_train"/"images_test".

* Copy contents of dataset folder into logos folder and run Python build_logos.py. This will create the necessary csv files.

* Run: git clone https://github.com/fizyr/keras-retinanet.

* Copy paste train.txt/test.txt, Images folder and all the 3 csv files into this cloned directory from logos folder.

* Run: 
retinanet-train --batch-size 1 --steps 50  --epochs 10 --weights logos/resnet50_coco_best_v2.1.0.h5 --snapshot-path logos/snapshots csv retinanet_train.csv retinanet_classes.csv

(Adjust batch size and step size accordingly, I used this as figures above this gave GPU OOM error to me)

* Run:
retinanet-convert-model C:\Users\ralha\OneDrive\Desktop\Table_parsing\keras-retinanet\logos\snapshots\resnet50_csv_10.h5 output.h5 

* Run:
retinanet-evaluate csv retinanet_test.csv retinanet_classes.csv output.h5

* Run:
python pipeline.py


