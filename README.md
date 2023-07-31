# Kavida_AI_Assignment


# Problem Statement

Problem Statement:
Kavida.ai is a supply-chain disruption decision platform. We build a digital twin (replica) of enterprise supply chains and use artificial intelligence to detect disruption threats before they occur. Our AI then simulates the threat's impact and the optimal resiliency decision to minimize the financial and consumer impact with a mission to protect supply chains with integrated data so companies can proactively manage disruption risks. With the evolving business scope, it is crucial to set up processes that can help in delivering intelligent data-driven solutions.
Given you have joined the team as a Data Science Intern to help supply quality “Insights!”
You have been introduced to a vital challenge, a few of our clients want data for specific niche like alerts which might be related to environment/strike etc. related issues. As we proceed to scale the data offerings, we need an intelligent model to classify the articles which outperforms regular keywords-based classification!
# TASK 1:
Pls find the attached live data from our systems which contain thousands of news articles and their headline with the corresponding categories.
Use transfer learning to fine-tune a classification model for the attached dataset.
Follow the guidelines below for this task,
• Focus on the proper EDA of the dataset, and comment on your views!
• For model training you can use Kaggle or collab or personal GPU.
• Accuracy is not the concern, hence please limit your training time, the process followed by the rationale is more critical.
• apart from the code and proper comment, provide the rationale behind the models you use.
• Evaluate your trained model and provide the rationale behind the metrics used.
# TASK 2:
The intelligent model we created for the above problem is not valuable unless we deploy it to provide insights from the live data!
Create a RESTful API to get the inference from the model!
- Docker deployment (Recommended):
o Containerize the application to get the inference from the model.
o The request should respond with the predictions in a Json format.
(It is “optional” to make an end-to-end deployment on any of the cloud service)
# TASK 3:
As a Data Scientist you will deal with a lot of problems related to data sourcing and distribution, what will be your approach to deal with the following:
1. For building a classification model for example for categorizing data into various genres like sports, science, etc., how would you get labeled data for this task? How would you collect such data from the internet?
2. How can you leverage the attached data and trained classification model (Task 1) to inform companies about supply chain disruptions? Please illustrate using a visualization.





## You can find all the colab notebook by opening this colab link:
https://drive.google.com/drive/folders/18_Iv5dllXmK6wO4ZnGre1uurgWBbqwF6?usp=sharing


# Rationale behind the BERT model

BERT is a transformer-based model, and its key innovation lies in the concept of bidirectionality. Unlike traditional language models that read text in one direction (e.g., from left to right or right to left), BERT employs a bidirectional context prediction technique. It predicts each word based on the words that come before and after it, thus capturing richer contextual information.
Since we have to use transfer learning to fine tune this classification , so BERT is a good choice Transfer Learning: BERT's pre-trained weights are valuable knowledge that can be transferred to downstream NLP tasks. By fine-tuning BERT on a specific task, it can adapt its contextual embeddings to the task-specific domain, leading to better performance with fewer labeled training samples.

In the context of the provided problem statement (classifying articles), using BERT allows the model to leverage its pre-trained knowledge of language, context, and semantics. Fine-tuning BERT on the task-specific dataset enables it to learn to classify articles based on the textual information present in the 'paragraph' column and predict the corresponding category ('news_list') effectively.




### Rationale behind the Evaluation metrics -  Classification Report,Accuracy_Score

To evaluate the predictions of the BERT model, we can use several standard evaluation metrics that are commonly used for multi-class classification tasks. Here are some of the main metrics along with their rationales:

#Classification Report: The classification report combines precision, recall, F1 score, and support for each class in a compact format.

Rationale: The classification report provides a comprehensive summary of the model's performance for all classes. It is a convenient way to compare the model's metrics across different classes.


Accuracy: Accuracy is the most straightforward metric and measures the proportion of correctly predicted labels out of all the samples in the dataset.
It is a simple and intuitive metric that provides an overall assessment of the model's performance.
Rationale: Accuracy is useful when the classes in the dataset are balanced, i.e., there are approximately equal samples for each class. However, accuracy may not
be suitable for imbalanced datasets, where one class dominates the others, as it can be misleading and mask the true performance of the model.





**1. The solutions for the Task1 is available in Task - 1 : -> Deep Learning.ipynb**

**2. The solutions for the Task3 is available in Task 3:.ipynb**


**3. The solutions for the Task 2 is available in app.py file**
