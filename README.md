### **README: Malicious URL Detection from QR Codes**

---

### **Project Overview**  
This project focuses on detecting malicious URLs embedded in QR codes using machine learning techniques. With the increasing adoption of QR codes in daily activities, such as payments and authentication, the risk of **QR code phishing (quishing)** has grown significantly. The goal of this research is to identify the most effective machine learning model to classify URLs as **benign** or **malicious** while analyzing performance trade-offs between precision, recall, and accuracy.  

The study evaluates **four classification models**:  
1. **Random Forest**  
2. **Gaussian Naive Bayes (GNB)**  
3. **XGBoost**  
4. **LightGBM**  

The dataset comprises **195,794 unique URLs** decoded from QR codes, and features such as URL length, entropy, and character counts are extracted to train and test the models.  

---

### **Project Structure**

The project is organized into the following files and directories:

1. **Data**  
   - `decoded_urls.csv` : Contains QR code-decoded URLs with labels (benign or malicious).  
   - `updated_features_extracted.csv` : Dataset with extracted features for each URL.  
   - Preprocessed datasets are stored as training and testing files in CSV format.

2. **Scripts**  
   - **`download_dataset.py`**: Downloads the dataset using Kaggle’s API.  
   - **`decode_qr_codes.py`**: Decodes QR codes from images and extracts URLs.  
   - **`feature_extraction.py`**: Extracts features (e.g., length, entropy, counts) from the URLs.  
   - **`data_preprocessing.py`**: Cleans the dataset, encodes categorical variables, and scales numerical features.  
   - **`fix_tld_extraction.py`**: Updates TLD extraction for accurate feature representation.  
   - **`validate_features.py`**: Validates feature quality and checks for missing or erroneous values.  
   - **`process_decoded_data.py`**: Processes and previews the decoded URLs dataset.  
   - **`read_file.py`**: Reads and displays the preprocessed feature dataset.  

3. **Models and Results**  
   - Machine learning models were implemented using **Scikit-learn** and **LightGBM**.  
   - Performance metrics, such as accuracy, precision, recall, and F1-score, are reported for each model.  

---

### **Requirements**

To set up the project, ensure the following dependencies are installed:

- **Python 3.8+**  
- Libraries:  
   - pandas  
   - scikit-learn  
   - xgboost  
   - lightgbm  
   - tldextract  
   - pyzbar  
   - pillow  
   - matplotlib  

Install all dependencies using the following command:

```bash
pip install -r requirements.txt
```

---

### **Steps to Run the Project**

1. **Dataset Preparation**  
   - Run `download_dataset.py` to download the dataset.  
   - Run `decode_qr_codes.py` to decode URLs from QR code images and clean the data.  

2. **Feature Extraction**  
   - Execute `feature_extraction.py` to extract relevant features, such as URL length, entropy, and special character counts.  

3. **Data Preprocessing**  
   - Use `data_preprocessing.py` to handle missing values, encode categorical features (like TLD), and standardize numerical columns.  

4. **Model Training and Evaluation**  
   - Train and test machine learning models using the preprocessed data.  
   - Evaluate model performance based on metrics like accuracy, precision, recall, and F1-score.  

5. **Model Results**  
   - Review the outputs and confusion matrices for all models.  
   - Analyze results to identify the most effective model for detecting malicious URLs.  

---

### **Results**

- **Gaussian Naive Bayes** achieved an accuracy of **56.08%**, performing poorly on complex feature interactions.  
- **Random Forest** emerged as the best-performing model with an accuracy of **93.06%**, balancing precision and recall for both benign and malicious URLs.  
- **XGBoost** achieved **92.37% accuracy**, offering competitive performance with faster runtime.  
- LightGBM demonstrated efficiency and scalability but showed slightly lower performance than Random Forest.  

Random Forest's robust handling of non-linear relationships and its resilience against overfitting make it the most suitable model for phishing URL detection.

---

### **Future Work**

- Incorporate additional features such as content-based URL attributes or geographic patterns.  
- Explore real-time deployment of the models for live phishing detection.  
- Investigate hybrid models that combine strengths of Random Forest, XGBoost, and LightGBM.  

---

### **Contact**  

For questions or collaboration, feel free to contact:  
**Name**: Sabah Hagos
**Email**: sabah.hagos17@gmail.com
**Institution**: Fordham University

---

