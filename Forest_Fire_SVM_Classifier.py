from PIL import Image
import cv2 as cv
import streamlit as st
import pickle
import numpy as np
import sklearn
import zipfile

zip_file_path = 'Forest_Fire_SVM_classifier_60.zip'

# Extract the zip file
@st.cache_data
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Assuming your model file is named 'model.pkl' inside the zip file
    model_filename = 'Forest_Fire_SVM_classifier_60.pkl'
    
    # Extract the model file from the zip
    zip_ref.extract(model_filename)  # Extract to a specific folder

@st.cache_resource
def load_model():
    with open('Forest_Fire_SVM_classifier_60.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

st.markdown("<h1 style='text-align: center;'>SVM model for Forest Fire Detection from a challenging dataset</h1>", unsafe_allow_html=True)
st.write('')

st.sidebar.title('Navigation')
selection = st.sidebar.radio('',['Home','Model Description','Authors','Acknowledgements'])

if selection == 'Home':
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    st.write('Note: ')
    st.write('(1) The model has been trained solely on Forest Images and Forest Fire Images and so may give absurd results for images that do not belong to either of these two categories.')
    st.write('(2) The images should preferably be sharp and high resolution.')
    st.write('(3) For further information go to Model Description.')
    
    image = []
    input_size = (60,60)
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        image = np.array(image)
        # flipping since opencv returns BGR and PIL returns RGB and model
        # was train on BGR value vectors
        image = np.flip(image)
        image = np.array(image)
        image = cv.resize(image,input_size)
        image = image.reshape(1,-1)
        st.write('')

        
        prediction = model.predict(image)
        st.write('**Prediction:**')
        if prediction == 1:
            st.write('Forest Fire Detected 🔥')
        elif prediction == 0:
            st.write('No Forest Fire Detected 🌲')

elif selection == 'Model Description':
    st.title("Model Description")
    st.header("Support Vector Machines")
    text = 'Support Vector Machines (SVMs) represent a powerful class of supervised machine learning algorithms renowned for their versatility and effectiveness in solving a wide range of classification and regression tasks. Introduced by Vladimir Vapnik and his colleagues in the 1960s, SVMs have since become a cornerstone of modern machine learning.'
    st.write(text)
    text = 'At their core, SVMs excel in finding optimal decision boundaries that separate data points belonging to different classes while maximizing the margin, or distance, between these boundaries. This unique characteristic allows SVMs to perform exceptionally well in scenarios where data may be complex, high-dimensional, or not linearly separable. Moreover, SVMs are known for their ability to generalize from training data to new, unseen examples, which makes them valuable tools for both classification and regression problems.'
    st.write(text)
    st.header('Dataset Used')
    text = 'One of the dataset employed for this study is conveniently accessible in Dataset for Forest Fire Detection in <a href ="https://data.mendeley.com/datasets/gjmr63rz2r/1">Mendeley Data</a>. It is provided as a compressed file “Dataset.rar”. This archive contains two essential components: the training dataset and the test dataset. These datasets consist of images, each having a resolution of 250×250 pixels. The other dataset we used for checking the efficiency of our model is accessible form <a href="https://images.cv/download/forest_fire/948/CALL_FROM_SEARCH/%22forest_fire%22">images.cv</a>. In this dataset all images are of 256×256 pixels and all belong to the category of forest fire. They have been curated to focus specifically on imagery related to forest fires.'
    st.write(text,unsafe_allow_html=True)

    text = 'The training dataset of the Mendeley Data was used for training the model, while the testing data of Mendeley Data and the Forest Fire images from Images.cv were used for testing the model.'
    st.write(text)

    st.header("Implementation")
    text = 'Our dataset comprises meticulously labeled images categorized as "fire" and "no-fire." The primary objective was to employ predictive methods to discern whether unseen test data fell into either the "fire" or "no-fire" category. Given the binary nature of this classification task, several techniques were at our disposal. Ultimately, we opted to train a Support Vector Machine (SVM) model for this purpose. This SVM model was carefully trained to classify images and predict the occurrence of forest fires based on the visual content of the images. This choice was made after careful consideration of its suitability for binary classification tasks involving image data.'
    st.write(text)

    st.header('Procedure')
    text = 'Our dataset presented a challenge in that the number of data samples is notably low in comparison to the multitude of attributes considered. Specifically, we have utilized the RGB pixel values of images as these attributes. To address this issue, we undertook a series of preprocessing steps to enhance the dataset\'s suitability for analysis.'
    st.write(text)

    text = 'First, we resized the images in the training dataset to various resolutions, namely: 10x10, 20x20, 30x30, 40x40, 50x50, 60x60, 70x70, 80x80, 90x90, 100x100, 150x150, 200x200 and 250x250. This resizing was carried out solely on the training datasets to gain a better understanding of the interplay between the number of samples and the number of attributes.'
    st.write(text)

    text = 'Furthermore, in our pursuit of increasing the number of data samples, we applied median blur and horizontal flip to the images in the training data. These augmentation techniques increased the training data by a factor of 4 and were essential in the development of a more robust dataset for subsequent analysis.'
    st.write(text)

    text = 'We tried different kernels for SVM model namely linear, polynomial, sigmoid and Gaussian kernels for each of the resized training dataset to assess their performance. The Gaussian kernel displayed the best performance overall and the gave the best accuracy when the input image resolution was 250x250. The SVM model deployed here is the same model that had the best performance.'
    st.write(text)

elif selection == 'Authors':
    st.header('Authors')
    st.markdown("<u>Ankan Kar</u> and <u>Utpalraj K</u> are the first authors.", unsafe_allow_html=True)
    st.markdown("<u>Aman</u> is the second author.",unsafe_allow_html=True)
    st.header('Author Description')
    st.markdown("Ankan Kar is a Computer Science, Mathematics and Statistics enthusiast. He did his Bachelor's in Mathematics from ISI Bangalore (2020-23) and is currently a M.Sc Computer Science student at CMI (2023-present).")
    st.markdown("Utpalraj K is an aspiring Data Scientist and AI/ML enthusiast. He did his Bachelor's in Mathematics from ISI Bangalore (2020-23) and is currently a M.Sc Data Science student at CMI (2023-present).")
    st.markdown("Aman did his Bachelor's in Mathematics from ISI Bangalore (2019-2022) and is currently a M.Sc Data Science student at CMI (2023-present).")

elif selection == 'Acknowledgements':
    st.write('We would like to convey our heartfelt gratitude to Mr. Chenna Sai Sandeep and Mr. Suneet Nitin Patil for their constructive input and innovative ideas that greatly improved the implementation and analysis of this project. Their insights have significantly elevated the project\'s execution.')
    st.write('We acknowledge that this report would have been notably challenging without the collective commitment and support of all those mentioned above. We sincerely thank everyone for their dedication and contributions, which have transformed this project into a reality.')



    
