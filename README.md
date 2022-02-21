
## Information-Extraction-from-Receipts

![project] ![research]



- <b>Project Mentor</b>
    1. Dr Uthayasanker Thayasivam
- <b>Contributors</b>
    1. Thumilan Muhunthan
    2. Chuhaashanan Nagenthiran
    
---

## Summary

Retail stores play a vital role in the supply chain of the business process and act as a bridge between the customers and manufacturers of the products. For a particular supermarket, receipts share some common elements such as shop name, shop address but they vary in product details. Along with that supermarkets maintain a unique template for their receipts so finding a common template for cross supermarket receipts is not a viable effort. Hence our project is to develop an Automated Information Extraction System to identify information from printed grocery receipts and deliver the information as the interoperable format.

## Description

Information extraction from receipts is a prominent area of research since it aids the business organization to perceive knowledge about their customers. 
This project provide an approach to extract information from printed grocery receipts. We demonstrated F1 score of nearly 1 for single supermarket and we performed error analysis on template independent information extraction from grocery receipts.
we have investigated whether it is feasible to extract information from unseen receipts using sequential features and sequential model CRF from printed grocery receipts. 



## Dataset

We have collected 300 receipts as 100 receipts from one supermarket. So totally we have collected three different receipt types for our research. Then to increase our data corpus another three different supermarket receipts are included. Finally, we have collected 100 receipt texts of six different supermarkets altogether 600 receipt texts


## Model

### Receipt Image Processing
<p align="center">
<img src="https://github.com/aaivu/aaivu-information-extraction-from-receipts/blob/newBranch/docs/receipt_image_processing.png" width="600">
</p>

|  Steps	|  Description	| 
|:---------|:-----------	|
| Background Removal    	| Canny Edge detection , hough line transform algorithms|
| Receipt Image chunking     	| Efficient and Accurate Scene Text Detector (EAST), the pre-trained model was used to detect text regions in receipts.   	| 
|Binarization  	| Otsu’s binarization was used in receipt chunks to binarize receipt images.     	|
|Tilting/Deskewing & Resizing| By chunking receipts we can tilt particular part of receipts.  	|
|OCR Client API  	| Tesseract , Google OCR     	|
| Rebuild Text Builder|We developed an algorithm which uses bounding box coordinates from google OCR result |

### Receipt Text to JSON

NLTK POS tagging was used to chunk the receipt text so that we can extract layout features.
Conditional Random Field (CRF) is used as a sequential model.

<p align="center">
<img src="https://github.com/aaivu/aaivu-information-extraction-from-receipts/blob/newBranch/docs/receipt_text_to_JSON.png" width="600">
</p>

## Results

The scope of the experiment is to demonstrate how our approach extract information from unseen receipt template. Here we used Bi-directional sequential features with content and layout features


|          	| Precision 	| Recall 	| F1_Score 	|
|:---------:|:-----------:	|:---------:|:----------:|
| Ammma   	| 0.8281     	| 0.8298  	| 0.7948    	|
| Annai     	| 0.8098     	| 0.8661  	| 0.76    	|
| Keells   	| 0.647     	| 0.641  	| 0.5547    	|


## More references

1. Reference
2. Link

---

### License

Apache License 2.0

### Code of Conduct

Please read our [code of conduct document here](https://github.com/aaivu/aaivu-introduction/blob/main/docs/code_of_conduct.md).

[project]: https://img.shields.io/badge/-Project-blue
[research]: https://img.shields.io/badge/-Research-yellowgreen



Please read our [code of conduct document here](https://github.com/aaivu/aaivu-introduction/blob/master/docs/code_of_conduct.md).

[project]: https://img.shields.io/badge/-Project-blue
[research]: https://img.shields.io/badge/-Research-yellowgreen
