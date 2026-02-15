# Document Classification System
A Machine Learning powered web application built with Streamlit to classify documents into 5 categories: **Sports, Politics, Technology, Business, and Entertainment**.
## Features
- **Multi-Format Support**: Upload `.txt`, `.pdf`, or `.docx` files.
- **Batch Processing**: Classify multiple files at once and get a summary table.
- **Live Text Input**: Paste text directly for instant analysis.
- **Visualizations**:
    - **Prediction Probability**: Bar chart showing confidence scores for all categories.
    - **Keyword Highlighting**: Highlights top contributor words in the text.
    - **Word Cloud**: Visual representation of the most frequent words.
- **Downloadable Reports**: Generate and download a PDF report containing classification results, keywords, and a text snippet.
## Installation
1. **Clone the repository** (or download the files):
   ```bash
   git clone <repository-url>
   cd document-classification-system
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **(Optional) Retrain Model**:
   The project comes with a pre-trained model (`model_pipeline.pkl`). If you wish to retrain it:
   ```bash
   python train_model.py
   ```
## Usage
1. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
2. **Open in Browser**:
   The app will open automatically at `http://localhost:8501`.
3. **Classify**:
   - **File Upload Tab**: Drag and drop files to classify them.
   - **Live Text Input Tab**: Paste text and click "Classify Text".
## Technologies Used
- **Python**: Core programming language.
- **Streamlit**: Web application framework.
- **Scikit-learn**: Machine learning (Multinomial Naive Bayes, TF-IDF).
- **NLTK**: Text preprocessing (stopwords).
- **PyPDF2 & python-docx**: Document parsing.
- **FPDF**: PDF report generation.
- **WordCloud & Matplotlib**: Visualization.
