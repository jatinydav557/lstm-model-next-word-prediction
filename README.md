🔗 👉 **[Watch the Demo on YouTube](https://www.youtube.com/watch?v=eJ1H5GUIVL4&list=PLe-YIIlt-fbMg0B4DsrA8Xa2kgRv_pqA1&index=2&ab_channel=Jatin)**
-----

# 📖 Next Word Prediction using LSTM

**Predicting the Next Word in a Sequence with Deep Learning and Streamlit**

This project develops a deep learning model that can predict the next word in a given sequence of text. Built using Long Short-Term Memory (LSTM) networks, which are excellent for understanding patterns in sequences, the model is trained on Shakespeare's "Hamlet." An interactive Streamlit web application allows users to experience real-time next word predictions.

-----

## 🎯 Project Overview

Next word prediction is a fundamental task in Natural Language Processing (NLP) with applications ranging from autocorrect to advanced language models. This project demonstrates how a deep learning model can learn the structure and flow of language from a classic text like "Hamlet" to intelligently guess the word that logically follows a given sequence.

**Key Objectives:**

  * **Build a Sequence Prediction Model:** Develop a powerful LSTM model capable of predicting the next word in a sentence.
  * **Process Text Data:** Implement robust techniques to prepare raw text for a deep learning model.
  * **Create an Interactive User Interface:** Provide a simple and intuitive Streamlit app for real-time predictions.
  * **Showcase LSTM Capabilities:** Highlight how LSTMs excel at understanding context and dependencies in sequential data like text.

-----

## ✨ Key Features

  * **Deep Learning Model (LSTM):** Uses a sophisticated Long Short-Term Memory network, ideal for understanding language sequences.
  * **Trained on Shakespeare's "Hamlet":** The model learns linguistic patterns, vocabulary, and sentence structures from this rich classic text.
  * **Real-time Next Word Prediction:** Enter a phrase into the Streamlit app, and the model will predict the most probable next word(s).
  * **Predict Multiple Words:** Option to predict a sequence of several words, building a sentence piece by piece.
  * **Automated Text Preprocessing:** Handles tokenization, sequence generation, and padding of input text behind the scenes.
  * **Model Persistence:** The trained model and tokenizer are saved and loaded for immediate use in the Streamlit app.

-----

## 🧠 Model Details: How the LSTM Learns to Predict Words

The core of this project is a **Long Short-Term Memory (LSTM) network**, which is a special type of Recurrent Neural Network (RNN). LSTMs are designed to remember information for long periods, making them highly effective for understanding context in sentences.

Here's a look at the main parts of the model:

### **1. Embedding Layer**

  * **What it does:** Converts each word in your input text into a dense numerical representation (a "word embedding"). Think of it as giving each word a unique, meaningful numerical fingerprint. Words that are used in similar contexts or have similar meanings will end up with similar numerical representations.
  * **Why it's used:** Computers understand numbers, not words directly. This layer translates words into a format that the neural network can process effectively, capturing semantic relationships between words.

### **2. LSTM Layers**

  * **What they do:** These are the "brain" of the sequence prediction. LSTMs have internal "memory cells" that allow them to remember important information from earlier words in a sequence and forget less important details. This memory helps them understand long-range dependencies in sentences, which is crucial for predicting the next word accurately.
  * **How it works:** They process the words one by one, constantly updating their internal state based on the current word and their memory of previous words. The output of one LSTM layer can then feed into another (as seen with `return_sequences=True` in the first LSTM layer), allowing the model to learn even more complex patterns.
  * **Dropout Layer:** A `Dropout` layer is used between LSTM layers. This temporarily "turns off" a fraction of neurons during training. This helps prevent the model from becoming too reliant on specific connections, making it more robust and preventing "overfitting" (where the model memorizes the training data too well but performs poorly on new, unseen data).

### **3. Dense (Output) Layer**

  * **What it does:** This is the final layer that takes the information processed by the LSTM layers and converts it into a prediction for the next word.
  * **Activation Function: Softmax**
      * **What it does:** Softmax takes a list of numbers and turns them into a list of probabilities that add up to 1. Each probability corresponds to a word in the model's vocabulary.
      * **Why it's used here:** The model needs to predict which of thousands of possible words is most likely to come next. Softmax gives us a probability distribution over the entire vocabulary, indicating the likelihood of each word being the next one. The word with the highest probability is chosen as the prediction.

### **4. Loss Function (Categorical Cross-Entropy)**

  * **What it does:** During training, this function measures how "wrong" the model's prediction is. For example, if the model predicts "to" with a low probability when the actual next word was "to," the loss will be high. The goal during training is to minimize this loss.
  * **Why it's used:** This loss function is perfect for "multi-class classification" problems, where you're picking one correct answer out of many possible options (in this case, one next word out of your entire vocabulary).

### **Early Stopping**

  * **What it does:** A training technique that monitors the model's performance on a separate validation dataset during training.
  * **Why it's used:** If the model's performance on the validation data stops improving (or starts getting worse) for a set number of training cycles (epochs), early stopping automatically halts the training. This saves computational resources and helps prevent the model from overfitting.

-----

## 📚 Dataset

The model is trained on the full text of **William Shakespeare's "Hamlet"** from the Project Gutenberg corpus. This rich and extensive text provides a large and diverse vocabulary and complex sentence structures, making it an excellent source for training a language model.

-----

## 📂 Project Structure

```
.
├── hamlet.txt                      # The raw text data (Shakespeare's Hamlet)
├── next_word_lstm.h5               # The trained LSTM model for next word prediction
├── tokenizer.pickle                # The saved tokenizer object (maps words to numbers)
├── app.py                          # The Streamlit web application for interactive predictions
├── embedding.ipynb                 # Jupyter Notebook: Demonstrates Word Embedding concepts
├── simple_rnn.ipynb                # Jupyter Notebook: Details the LSTM/GRU model training process
├── requirements.txt                # Python dependencies for the project
└── README.md                       # This README file
```

-----

## ⚙️ Technologies Used

  * **Python 3.9+**
  * **TensorFlow / Keras:** The core deep learning framework used for building, training, and loading the LSTM model.
  * **Streamlit:** For creating the interactive and user-friendly web interface.
  * **Numpy:** Essential for numerical operations and array manipulation.
  * **Pandas:** Used for data handling (though less prominent in the final app, useful for exploration).
  * **NLTK (Natural Language Toolkit):** For accessing the Gutenberg corpus and potentially other text processing tasks.
  * **Matplotlib:** Used for generating plots (e.g., in notebooks for visualization).
  * **WordCloud:** For creating visual word clouds (if integrated into a visualization part of the app).
  * **Scikit-learn:** Provides tools like `train_test_split` for dividing data.
  * **Pickle:** Used for saving and loading the `Tokenizer` object.

-----

## 🚀 How to Run Locally

Follow these steps to get the Next Word Predictor up and running on your local machine:

1.  **Clone the Repository:**

    ```bash
    git clone <your-repository-url>
    cd <your-project-folder>
    ```

2.  **Set up a Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare Data and Train Model (if not already done):**

      * Run the cells in `simple_rnn.ipynb`. This notebook will:
          * Download `hamlet.txt`.
          * Preprocess the text.
          * Train the LSTM model.
          * Save the trained model (`next_word_lstm.h5`) and the `tokenizer.pickle` file.
      * *(Optional: Explore `embedding.ipynb` to understand word embeddings in more detail.)*

5.  **Run the Streamlit Application:**

    ```bash
    streamlit run app.py
    ```

6.  **Access the App:**
    Open your web browser and navigate to the local address provided by Streamlit (usually `http://localhost:8501`).

-----

## 🔮 Future Enhancements

  * **Larger Datasets:** Train the model on more extensive and diverse text corpora for broader vocabulary and improved prediction accuracy.
  * **More Complex Models:** Experiment with Transformer-based models (e.g., BERT, GPT-like architectures) for state-of-the-art text generation.
  * **Temperature Parameter:** Add a "temperature" parameter to the prediction, allowing users to control the randomness/creativity of the generated words.
  * **User Interface Improvements:** Enhance the Streamlit UI with features like dark mode, more styling, or options for different generation strategies.
  * **Deployment:** Containerize the Streamlit application using Docker and deploy it to a cloud platform (e.g., Hugging Face Spaces, Google Cloud Run, AWS EC2, or Azure Web Apps) for public access.
  * **Performance Optimization:** Optimize the model for faster inference if deployment requires low latency.

-----

## 🤝 Credits

  * [Jatin Yadav]
  * [TensorFlow](https://www.tensorflow.org/)
  * [Streamlit](https://streamlit.io/)
  * [Numpy](https://numpy.org/)
  * [NLTK](https://www.nltk.org/)
  * [Project Gutenberg](https://www.gutenberg.org/) (for "Hamlet" text)

-----

## 🙋‍♂️ Let's Connect

* **💼 LinkedIn:** [www.linkedin.com/in/jatin557](https://www.linkedin.com/in/jatin557)
* **📦 GitHub:** [https://github.com/jatinydav557](https://github.com/jatinydav557)
* **📬 Email:** [jatinydav557@gmail.com](mailto:jatinydav557@gmail.com)
* **📱 Contact:** [`+91-7340386035`](tel:+917340386035)
* **🎥 YouTube:** [Checkout my other working projects](https://www.youtube.com/@jatinML/playlists)
