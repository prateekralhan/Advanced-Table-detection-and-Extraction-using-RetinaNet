# Advanced-Table-detection-and-Extraction-using-RetinaNet

## Pre-trained Weights can be found [here.](https://drive.google.com/drive/u/2/folders/1Xv7voEYUPOdyTCXy46jBKplQe7i2VBHl)

### Some sample outputs:

![1](https://user-images.githubusercontent.com/29462447/90297676-ec64ed80-deac-11ea-99ba-4eb7719a1d02.png)

![1_table](https://user-images.githubusercontent.com/29462447/90297677-ecfd8400-deac-11ea-9227-6aba9c85bafb.png)


![2](https://user-images.githubusercontent.com/29462447/90297666-e5d67600-deac-11ea-813f-15e3087fd778.png)

![2_table](https://user-images.githubusercontent.com/29462447/90297667-e7a03980-deac-11ea-96a7-d3c4ee315b03.png)


![3](https://user-images.githubusercontent.com/29462447/90297668-e838d000-deac-11ea-8937-f756d3d15f8b.png)

![3_table](https://user-images.githubusercontent.com/29462447/90297673-ea9b2a00-deac-11ea-8de3-c2ae3daef11e.png)

#### The model was trained only on images from around 10 to 12 PDFs ( ~ 70 images ). Definitely, the model's performance ( mAP ) will improve more when you have more data :wink:

Next steps:

***Create an E2E pipeline where you can just dump multiple reports at a time and then the pipeline will automatically detect tables from the PDFs and export them to Excel spreadsheets accordingly in another folder.***
