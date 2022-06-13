import streamlit as st
from PIL import Image, ImageOps
import os
import time
from keras.models import load_model
import cv2
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_image(image_file):
	img = Image.open(image_file)
	return img
def loadmodel():
    model = load_model("model2.h5")
    return model
def process_image(img):
    data = np.ndarray(shape=(1,128,128,3),dtype=np.float32)
    image = img
    #image sizing
    size = (128,128)
    image = ImageOps.fit(image,size,Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    #Normalize the image
    normalized_image_array = image_array.astype(np.float32)

    #load the image into te array
    data[0] = normalized_image_array
    # st.text(data[0].shape)
    # st.text(img_path)
    # x_val=[]
    # img_arr=cv2.imread(img_path)

    # img_arr=cv2.resize(img_arr,(224,224))
    # x_val.append(img_arr)
    # val_x=np.array(x_val)
    # val_x = val_x/224
    return data
def main():
        st.title("Skin Disease Classification")

        menu = ["Upload Image","All Classifications"]
        choice = st.sidebar.selectbox("Menu",menu)
        st.sidebar.markdown("<h6><i class='fa fa-circle'/>Olutayo Benson-Oladeinbo</h5>", unsafe_allow_html=True)
        if choice == "Upload Image":
            st.subheader("10 Skin Disease Categories")
            st.success("View all categories on the Menu Bar")
            # st.markdown("<h6 style='color: red;'>View all classifications on the Menu Bar</h5>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<h5 style='text-align: center; color: red;'>Eczema</h5>", unsafe_allow_html=True)
                # st.text("Eczema")
                st.image("ui_images/1_17.jpg",width=250)
            with col2:
                st.markdown("<h5 style='text-align: center; color: red;'>Psoriasis</h5>", unsafe_allow_html=True)
                # st.text("Psoriasis pictures Lichen Planus and related diseases")
                st.image("ui_images/0_15.jpg",width=250)

            with col3:
                st.markdown("<h5 style='text-align: center; color: red;'>Melanoma</h5>", unsafe_allow_html=True)
                # st.text("Melanoma")
                st.image("ui_images/ISIC_7600232.jpg",width=187)
            st.subheader("Upload your Picture")
            image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
            if image_file is not None:
                with st.spinner('Image Uploading...'):
                    time.sleep(3)
                # To See details
                file_details = {"filename":image_file.name, "filetype":image_file.type,
                                "filesize":image_file.size}
                st.write(file_details)

                # To View Uploaded Image
                st.image(load_image(image_file),width=250)
                #Saving upload
                with st.spinner('Processing Image...'):
                    time.sleep(5)
                with open(os.path.join("uploadedimages",image_file.name),"wb") as f:
                    f.write((image_file).getbuffer())
                
                model = loadmodel()
                imagepath = Image.open(image_file)
                val_set = process_image(imagepath)
                predicted  = model.predict(val_set)
                y_predicted_labels = [np.argmax(i) for i in predicted]
                classes = ['Atopic Dermatitis','Basal Cell Carcinoma (BCC)',
                           'Benign Keratosis-like Lesions (BKL)','Eczema',
                           'Melanocytic Nevi (NV)','Melanoma',
                           'Psoriasis pictures Lichen Planus and related diseases',
                            'Seborrheic Keratoses and other Benign Tumors',
                            'Tinea Ringworm Candidiasis and other Fungal Infections',
                            'Warts Molluscum and other Viral Infections']
                st.info("Your Diagnosis")
                st.success(classes[y_predicted_labels[0]])
                # st.text(y_predicted_labels[:5])
                st.balloons()
        elif choice == "All Classifications":
            st.subheader("All Classifications")
            diseasesDictionary = [
                {
                    "Name": "Tinea Ringworm Candidiasis and other Fungal Infections",
                    "Image_Url": "IMG_CLASSES/Eczema/1_6.jpg",
                    "Text": """ Ringworm is a common infection of the skin and nails that is caused by fungus
                                The infection is called “ringworm” because it can cause an itchy, red, circular rash. 
                                Ringworm is also called “tinea” or “dermatophytosis.” 
                                The different types of ringworm are usually named for the location of the infection on the body.
                                """,
                    "Text2": """
                                ___________________________________________________________ \n
                                Areas of the body that can be affected by ringworm include: \n
                                ___________________________________________________________ \n
                                Feet (tinea pedis, commonly called “athlete’s foot”)\n
                                Groin, inner thighs, or buttocks (tinea cruris, commonly called “jock itch”)\n
                                Scalp (tinea capitis)\n
                                Beard (tinea barbae)\n
                                Hands (tinea manuum)\n
                                Toenails or fingernails (tinea unguium, also called “onychomycosis”)"""
                },
                {
                    "Name": "Eczema",
                    "Image_Url": "IMG_CLASSES/Eczema/1_6.jpg",
                    "Text": """
                                Atopic eczema causes the skin to become itchy, dry, cracked and sore. \n

                                Some people only have small patches of dry skin, but others may experience widespread inflamed skin all over the body. \n

                                Inflamed skin can become red on lighter skin, and darker brown, purple or grey on darker skin. This can also be more difficult to see on darker skin. \n

                                """,
                    "Text2": """ 
                                Although atopic eczema can affect any part of the body, it most often affects the hands, insides of the elbows, backs of the knees and the face and scalp in children.\n

                                People with atopic eczema usually have periods when symptoms are less noticeable, as well as periods when symptoms become more severe (flare-ups).\n
                            
                             """
                },
                {
                    "Name": "Melanoma",
                    "Image_Url": "IMG_CLASSES/Melanoma/ISIC_6653225.jpg",
                    "Text": "Info about Melanoma",
                    "Text2": ""
                },
                {
                    "Name": "Atopic Dermatitis",
                    "Image_Url": "IMG_CLASSES/Atopic Dermatitis/1_2.jpg",
                    "Text": "Info about Eczema",
                    "Text2": ""
                },
                {
                    "Name": "Basal Cell Carcinoma (BCC)",
                    "Image_Url": "IMG_CLASSES/Melanoma/ISIC_6653225.jpg",
                    "Text": "Info about Melanoma",
                    "Text2": ""
                }
            ]
            for i in diseasesDictionary:
                with st.expander(i['Name']):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(i['Image_Url'])
                    with col2:
                        st.markdown(i["Text"])
                    st.markdown(i["Text2"])
                    
                   

main()