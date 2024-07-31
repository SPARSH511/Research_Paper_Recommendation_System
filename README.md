# Research Paper Recommendation System

## Introduction

The Research Paper Recommendation System aims to assist researchers and academics in discovering relevant literature efficiently. Leveraging a dataset containing ArXiv papers' information, the system uses TF-IDF vectorization for feature extraction and cosine similarity for identifying and recommending similar papers. This tool enhances research efficiency, improves information accessibility, fosters interdisciplinary collaboration, supports novice researchers, and addresses information overload.

## Objectives

- **Enhancing Research Efficiency**: Reduce the time and effort required for literature searches.
- **Improving Information Accessibility**: Simplify the process of discovering and accessing scholarly papers.
- **Fostering Interdisciplinary Collaboration**: Encourage exploration and engagement with literature across different domains.
- **Supporting Novice Researchers**: Provide personalized recommendations that align with their interests and research goals.
- **Addressing Information Overload**: Curate and recommend high-quality papers relevant to the researcher's interests.

## Methodology

1. **Data Acquisition and Preprocessing**:
    - Gathered a diverse dataset of ArXiv papers including metadata such as titles, authors, abstracts, and publication dates.
    - Handled missing values, removed duplicates, and cleaned the text for quality and consistency.

2. **Feature Extraction with TF-IDF Vectorization**:
    - Transformed textual data of abstracts into numerical feature vectors.
    - Calculated term importance relative to its frequency in the entire corpus.

3. **Similarity Computation using Cosine Similarity**:
    - Measured similarity between pairs of documents based on TF-IDF representations.
    - Identified papers with similar themes, topics, or concepts.

4. **User Interaction and Customization**:
    - Designed a user-friendly interface for interaction with the recommendation system.
    - Allowed users to input preferences, specify the number of similar papers, and refine search criteria.

5. **Evaluation and Performance Metrics**:
    - Defined metrics such as precision, recall, and user satisfaction to assess system effectiveness.
    - Conducted experiments to evaluate recommendation accuracy across diverse research domains.

6. **Scalability and Deployment**:
    - Optimized the system for scalability and explored local and cloud-based deployment options.
    - Ensured efficient processing and analysis of large datasets while maintaining high performance and reliability.

## System Architecture

The system architecture includes data acquisition, preprocessing, TF-IDF vectorization, cosine similarity computation, and a user-friendly interface built with Streamlit.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/research-paper-recommendation-system.git
    cd research-paper-recommendation-system
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Extract the preprocessed data embeddings .pkl file from the zip and keep this in the same directory as the python script. 
2. Launch the Streamlit application after running the script by opening the terminal in the directory of project.
    ```bash
    python web_ui.py
    ```bash
    python -m streamlit run web_ui.py
3. Input your topic of interest.
4. Specify the number of recommendations desired.
5. Click on "Get Recommendations" to receive a list of relevant papers.

## Conclusion

This Research Paper Recommendation System empowers researchers to navigate scholarly literature with ease and precision. 
By providing personalized recommendations and facilitating interdisciplinary connections, the system accelerates discovery and fosters collaboration. 
I am committed to continuous improvement and expanding the system to meet the evolving needs of the research community.


