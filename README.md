# Vision Transformer for Medical Images

### Abdulshaheed Alqunber

### <Abdulshaheed.qunber@kaust.edu.sa>

*This project aims to explore application of the Vision Transformer
(ViT) architecture for medical images classification. The model is
implemented with TensorFlow and is similar to the original ViT with some
changes to fit the new type of problem.*

## **Introduction**

CNN slow training time is big flaw. Since AlexNet in 2012, different
architectures of CNNs have brought a tremendous contribution to real
business operations and academic researches. A major flaw of CNN exists
in Pooling layers because it loses a lot of valuable information and it
ignores the relationship between part of the image and the whole. 

In replacement of CNN, ViT was introduced in October 2020. ViT gives
High Accuracy with Less Computation Time for Training. Vision
Transformer achieved State-of-the Art results in image recognition tasks
with standard Transformer encoder and fixed-size patches.

![](.//media/image1.gif)

"*An Image is Worth 16x16 Words: Transformers for Image Recognition at
Scale*" by Google Research

How does it work?

-   Split an image into patches

-   Flatten the patches

-   Produce lower-dimensional linear embeddings from the flattened
    patches

-   Add positional embeddings

-   Feed the sequence as an input to a standard transformer encoder

-   Pretrain the model with image labels (fully supervised on a huge
    dataset)

-   Finetune on the downstream dataset for image classification

We wanted to benchmark this new architecture with different medical
dataset for an image classification task as a start. 

## **Datasets**

### 1.  **Brain MRI Images for Brain Tumor classification**

<https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri>

<https://www.kaggle.com/jaykumar1607/brain-tumor-mri-classification-tensorflow-cnn#Evaluation>

#### **Background**

A Brain tumor is considered as one of the aggressive diseases, among
children and adults. Brain tumors account for 85 to 90 percent of all
primary Central Nervous System (CNS) tumors. Every year, around 11,700
people are diagnosed with a brain tumor. The 5-year survival rate for
people with a cancerous brain or CNS tumor is approximately 34 percent
for men and 36 percent for women. Brain Tumors are classified as: Benign
Tumor, Malignant Tumor, Pituitary Tumor, etc. Proper treatment,
planning, and accurate diagnostics should be implemented to improve the
life expectancy of the patients. The best technique to detect brain
tumors is Magnetic Resonance Imaging (MRI). A huge amount of image data
is generated through the scans. These images are examined by the
radiologist. A manual examination can be error-prone due to the level of
complexities involved in brain tumors and their properties.

#### **Description of Dataset**

The dataset has 3264 MRI images with 4 different classes. The task is to
classify those MRI images into four classes. Glioma tumor, meningioma
tumor, pituitary tumor and no tumor. About 900 images for each class
except for no tumor which has about 500 images.

#### **Results Table**

  |Model             | Accuracy   |Time to train|
  |:------------------: | :----------:| :-------------:|
  |CNN: ResNet152V2  | 96%       | 35 minutes|
  |ViT               | 98.9%     | 6 minutes|

### 2.  **Chest X-ray images**

<https://www.kaggle.com/tolgadincer/labeled-chest-xray-images>

### <https://www.kaggle.com/zivaharon/pneumonia-detection-dl-inceptionv3-tl>

#### **Background**

Pneumonia is an inflammatory condition of the lung affecting primarily
the small air sacs known as alveoli. It kills more children younger than
5 years old each year than any other infectious disease, such as HIV
infection, malaria, or tuberculosis. Symptoms typically include some
combination of productive or dry cough, chest pain, fever and difficulty
breathing. The severity of the condition is variable. Pneumonia is
usually caused by infection with viruses or bacteria and less commonly
by other microorganisms, certain medications or conditions such as
autoimmune diseases. Risk factors include cystic fibrosis, chronic
obstructive pulmonary disease (COPD), asthma, diabetes, heart failure, a
history of smoking, a poor ability to cough such as following a stroke
and a weak immune system. Diagnosis is often based on symptoms and
physical examination. Chest X-ray, blood tests, and culture of the
sputum may help confirm the diagnosis. The disease may be classified by
where it was acquired, such as community- or hospital-acquired or
healthcare-associated pneumonia.

#### **Description of Dataset**

This dataset contains 5,856 Chest X-Ray images. Images are labeled as
one of the three classes NORMAL/BACTERIA/VIRUS. For details of the data
collection and description, see the referenced paper below. According to
the paper, the images (anterior-posterior) were selected from
retrospective cohorts of pediatric patients of one to five years old
from Guangzhou Women and Children's Medical Center, Guangzhou.

#### **Results Table**

  |Model             | Accuracy   |Time to train|
  |:------------------: | :----------:|:-------------:|
  |CNN: inception v3 |  97%     |   51 minutes|
  |ViT               |  98.2%   |   31 minutes|

### 3.  **Lyme Disease**

<https://www.kaggle.com/sshikamaru/lyme-disease-rashes>

<https://www.kaggle.com/sshikamaru/lyme-disease-detection-with-cnn#Modeling>

#### **Background**

Lyme Disease is a bacterial infection, also known as the \"Silent
Epidemic\". It affects more than 300,000 people each year. Lyme disease
is caused by the bacterium Borrelia burgdorferi and rarely, Borrelia
mayonii. It is transmitted to humans through the bite of infected
blacklegged ticks. Typical symptoms include fever, headache, fatigue,
and a characteristic skin rash called erythema migraines. The rash can
appear up to 3 months after being bitten by a tick and usually lasts for
several weeks.

#### **Description of Dataset**

The data contains images of the EM (Erythema Migraines) also known as
the \"Bull\'s Eye Rash\" It is one of the most prominent symptoms of
Lyme disease. Also, in the data contains several other types of rashes
which may be often confused with EM rash by doctors and most of the
medical field. Given 882 images of various rashes, let\'s try to predict
if a given rash is a symptom of Lyme disease.

#### **Results Table**

  |Model             | Accuracy   |Time to train|
  |:------------------: | :----------:| :-------------:|
  |CNN: ResNet-50  | 91%    |    12 minutes|
  |ViT            |  81.6%   |   8:20 minutes|

### 4.  **Lung and Colon Cancer Histopathological Images**

<https://www.kaggle.com/usmantahirkiani/lungs>

<https://www.kaggle.com/andrewmvd/lung-and-colon-cancer-histopathological-images>

#### **Background**

Lung and colon cancers are two of the most common malignancies, which,
in some cases, may develop synchronously. Patients with lung cancer may
develop other malignancies as may those with colon cancer.
Epidemiologically, it has been suggested that cigarette smoking is
closely associated with an increased risk of cancer in various organs,
including the lung and the colon. During a 76-month study period, from
April 2009 up to July 2016, 17 (0.54%) of 3,102 patients with lung
cancer were diagnosed with colon cancer within 1 month.

#### **Description of Dataset**

This dataset contains 25,000 histopathological images with 5 classes.
All images are 768 x 768 pixels in size and are in jpeg file format. The
images were generated from an original sample of HIPAA compliant and
validated sources, consisting of 750 total images of lung tissue (250
benign lung tissue, 250 lung adenocarcinomas, and 250 lung squamous cell
carcinomas) and 500 total images of colon tissue (250 benign colon
tissue and 250 colon adenocarcinomas) and augmented to 25,000 using
the Augmenter package.

There are five classes in the dataset, each with 5,000 images, being:

-   Lung benign tissue

-   Lung adenocarcinoma

-   Lung squamous cell carcinoma

-   Colon adenocarcinoma

-   Colon benign tissue

#### **Results Table**

  |Model             | Accuracy   |Time to train|
  |:------------------: | :----------:| :-------------:|
  |CNN: VGG19 |  96%     |   86 minutes|
  |ViT        |  93.8%   |   83 minutes|

### 5.  **Leukemia (Blood Cancer)**

<https://www.kaggle.com/andrewmvd/leukemia-classification>

<https://www.kaggle.com/rishirajak/blood-cancer-detection-lenet-and-alexnet>

<https://www.kaggle.com/gauravrajpal/leukemia-classification-v1-3-inceptionv3-65-29>

#### **Background**

[Acute lymphoblastic
leukemia](https://en.wikipedia.org/wiki/Acute_lymphoblastic_leukemia) (ALL)
is the most common type of childhood cancer and accounts for
approximately 25% of the pediatric cancers. These cells have been
segmented from microscopic images and are representative of images in
the real-world because they contain some staining noise and illumination
errors, although these errors have largely been fixed in the course of
acquisition.

#### **Description of Dataset**

The task of identifying immature leukemic blasts from normal cells under
the microscope is challenging due to morphological similarity and thus
the ground truth labels were annotated by an expert oncologist. In total
there are 15,135 images from 118 patients with two labelled classes:

-   Normal cell

-   Leukemia blast

#### **Results Table**

  |Model             | Accuracy   |Time to train|
  |:------------------: | :----------:| :-------------:|
  |CNN: Inception V3 |  65.4%   |   14 minutes|
  |ViT               |  72.25%  |   28:20 minutes|

## **Discussion**

The results were better than CNN in the case we have a big dataset, as
seen in the brain tumor MRI, Chest X-ray and Leukemia datasets. Since
the paper mentions that the vision transformer does not perform well in
small dataset. We can see that in the Lyme disease dataset that has
about 400 images in total.

Regarding timing, the new architecture is indeed fast, in almost all
cases the time was improved by a lot. For example, the time improved
almost 6x in the brain tumor dataset. The chest x-ray has about 6000
images, and took 30 minutes in ViT compared to 50 minutes for CNN.
