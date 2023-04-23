# FREEDOM : Effective Early Depression Detection through Online Media Posts

### **Abstract**: 
With an explosion in the availability of instant web-based services and enhancement in the pace of propagation of digital media content being published on an individual’s social accounts, there exists an emergent need to monitor if the content being posted or searched by an individual is indicative of the existence of early symptoms of depression. Early and effective analysis and prediction through pre-learned representations of depression by visual or textual data form the basis of the problem of the proposed research-based project. The aim of the proposed research-based project is the development and deployment of Artificial Intelligence (AI) based mathematical models to analyze uni-modal and multi-modal aspects of the input data to predict depression. The initial evaluation of baseline models involves scraping the data as well as the images and text from web platforms where user-based content is generated. The proposed research project suggests combining the predictions of distinct input modalities to provide efficient results on the media posts of an individual.

> The proposed methods in the projects is specific to user posts.
> In general, the publicly available data set focuses only on
> particular kinds of images as facial features or postures whereas it
> is not necessary that an individual will always be able to infer a
> feeling of depression from such posts only. There is a need to train
> the Artificial Intelligence(AI) based model on a data set, which is
> directly relevant or related to the content being shared online.

### Model Architecture
![architecture](https://github.com/mohit15-iiitd/Course-Project_Group8/blob/master/images/Multi-Modal%20Architecture.jpeg?raw=true)

## Dataset Creation

For creating the dataset, we have scraped data from shuttlestock. We use python and beautifulSoup for scraping images, and their corresponding text from ShuttleStock.

*The tags which we have used for scraping depressive data are:*

 - **Depression Anxiety**
 - **Sadness**

*And for scraping non-depressive data are:*

 - **Happiness**
 - **Joy**

The Structure of our dataset is:

> <pre>
> /data
>     - /img
>     - data.csv
>     - test.jsonl
>     - train.jsonl
>     - val.jsonl
> </pre>

In the above file structure. The img folder contains all the images. The train, test and val.jsonl files contains the text and labels of the whole data split. The whole dataset is splitted into train, test and validation with ratio of 80:10:10 .

## Codebase

**We have experimented on this dataset in different setups**
1. Unimodal Image Classification
2. Unimodal Text Classification
3. Multimodal Image + Text Classification

#### Unimodal Image Classification
In this setup, we have trained 4 Deep Neural-Network based Image models like ResNet50, VGG19, InceptionV3, and Xception. Out of these four models, **InceptionV3** outperforms all the other models with a test accuracy of 82.0 %.

The hyper-parameter configuration used for training all four models are:
	- **learning rate**: 0.01
	- **Optimizer**: SGD.
	- **Loss Function**: Binary Cross Entropy.

#### Unimodal Text Classification
For this, we have trained four different configurations to understand the text and build our classifier.
1. LSTM.
2. FastText Embeddings + SVM classifier.
3. BERT Embeddings + SVM classifier.
4. Fine-tuned BERT classifier.

Out of these four configurations, **fine-tuned BERT classifier** outperforms all other configurations with 98.0 %.

The techniques which we have used for pre-processing the text is removal of stop-words, punctuations, emojis, and stemming and lemmatization.

#### Multimodal Image + Text Class

In multimodal setup, we have three multimodal setups:
1. ResNet50 + DistillBERT
2. VGG-16 + BERT
3. InceptionV2 + BERT

The configuration used for these setups are:
	- **Learning Rate**: 1e-5
	- **Number of Epochs**: 10
	- **Dropout**: 0.5

The setup-3 which is **InceptionV2 + BERT** outperforms the other setup with the test accuracy of 96.89 %.

For Explainability, we have used **GradCam**. This shows, which portion of the image our model is focusing while classifying.

![gradcam](https://raw.githubusercontent.com/mohit15-iiitd/Course-Project_Group8/master/images/gradcam.png?token=GHSAT0AAAAAAB6U6QRGSJHFBCGFZBPQEIKMZCFJ5IQ)
