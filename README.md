# Tweep-Fake

# Table of Contents
1. [Problem Statement](#problem-statement)
2. [Approach Used](#approach-used)
3. [Architecture Used](#architecture-used)
3. [Results Achieved](#results-achieved)
4. [Technology Used](#technology-used)
5. [How to Use This API](#how-to-use-this-api)

# Problem Statement
The problem statement is to classify whether the Tweet is written by a Human or Bot and also to classify the type of language model used to generate the Bot Tweet. 

# Approach Used
Used Bidirectional Encoder Representations from Transformers(BERT) to train the model.

# Architecture Used
![fig2-1](https://user-images.githubusercontent.com/37527532/95648499-525ca280-0af5-11eb-9178-74c37bad07d9.png)

# Results Achieved
Binary Model:-Achieved an overall accuracy of 82.46% on the Training Set, 82.06% on the validation set and 80.96 percent on the Testing Set.Also Achieved Macro F1 Score,precision and recall of 0.83, 0.82 and 0.82.

Multiclass Model:-Achieved an overall accuracy of 83.09% on the Training Set, 81.06% on the validation set and 80.68 percent on the Testing Set.Also Achieved Macro F1 Score,precision and recall of 0.79, 0.80 and 0.79.

# Technology Used
![112719_Python_Software_Foundation_Logo large](https://user-images.githubusercontent.com/37527532/91639130-21874900-ea32-11ea-8c44-b7c20a76452c.jpg)
![flask](https://user-images.githubusercontent.com/37527532/91639099-c2293900-ea31-11ea-9b8e-6a4309abc1df.png)

# How to Use this API
Step 1:Clone this Repository By typing ``` git clone https://github.com/rishabh706/Tweep-Fake.git```.

Step 2:Download the model weights from ```https://drive.google.com/drive/folders/106jCWwF3uVZEDeT4exR-e7M_-fzgaRX8?usp=sharing``` and put it inside the model weights folder.

Step 2:Open your Terminal or Bash shell and get into the project directory by typing command ```cd```.

Step 3:Then Type ```conda create -n Virtual Environment Name``` to create virtual environment if you donot have conda then you can install from ```https://docs.conda.io/en/latest/miniconda.html``` as per you PC specification.

Step 4:Activate the virtual environment by typing the following command ```conda activate Name of YOUR Enviroment```.

Step 4:Install the required Packages by typing ```pip install -r requirements.txt```.

Step 5:Type the following command to run the server ```python app.py```.

Step 6:Open Postman If you don't have you can download from ```https://www.postman.com/downloads/```.

Step 7:Type  ```http://127.0.0.1:5000/welcome``` to the Post Request and Hit Send you will see a response like this
![Capture](https://user-images.githubusercontent.com/37527532/95648877-2989dc80-0af8-11eb-82f9-d32a91d0800a.JPG)

If you see this welcome message then you are good to go

Step 8:Type ```http://127.0.0.1:5000/predict``` and in the Body head pass the Test Image as per the following picture and Hit Send
![Capture2](https://user-images.githubusercontent.com/37527532/91639686-f141a980-ea35-11ea-9f5f-73003c346bf3.JPG)
And there it is the Response of the predicted model.

