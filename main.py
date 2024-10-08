import streamlit as st
from PIL import Image
import cv2
import numpy as np
import random
from skimage import io
from skimage.metrics import structural_similarity as ssim
from scipy.stats import skew, kurtosis
from skimage.filters import laplace
from skimage.transform import integral_image
import pandas as pd
import glob
import pickle
import joblib
import statistics
from scipy import ndimage
import matplotlib.pyplot as plt

exposure_model = joblib.load('exposure_random_forest_model.joblib')
blur_model = pickle.load(open('rf_regressor_model.sav','rb'))
st.markdown(
    """
    <style>
    body {
        margin: 0;
        padding: 0;
        height: 100vh;
        overflow: hidden;
    }
    .app-container {
        height: 100vh;
        width : 100vw;
        overflow: hidden;
    }
    .block-container{
      padding: 3vh 3vw;
      max-width: 100%;
    }

    </style>
    """,
    unsafe_allow_html=True
)
# st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# st.title("Image Quality Assessment Metric")

st.write("<h3 style='font-weight: bold; text-align : centre'> Image Quality Assessment Metric", unsafe_allow_html=True)

references = []

col1, col2 = st.columns(2)
sub_col1, sub_col2 = col2.columns(2)

progress_max = 3
with col1:
  st.write("<h5 style='font-weight: bold; text-align : centre'> 3 Reference images needed for FOV calculation", unsafe_allow_html=True)
  progress_bar = st.progress(0)
  # st.write("<h3 style='font-weight: bold; text-align : centre'> Upload reference images", unsafe_allow_html=True)
  reference_file = st.file_uploader("Upload reference images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
  # reference_file = st.file_uploader("Upload reference images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)


if(len(reference_file)>3):
  reference_file.pop(1)

for i, uploaded_file in enumerate(reference_file):
    # Display the uploaded image
    image = Image.open(uploaded_file)
    image = np.array(image)
    if (len(references) >= 3) :
       references.pop(1)
    references.append(image)

    # Update progress bar
    progress_bar.progress((i + 1) / progress_max)

if(len(references)>1):
  concatenated_image = np.concatenate(references, axis=1)

  # Display concatenated image with slight spacing
  with col1:
    st.image(concatenated_image, caption="Concatenated Reference Images")

with col2:
  st.write("<h5 style='font-weight: bold; text-align : centre'> Upload image to test", unsafe_allow_html=True)
  uploaded_file = st.file_uploader(label_visibility = "collapsed", label= "Upload image to test", type=["jpg", "jpeg", "png"])

def calculate_overall_score_without_FOV( image, exposure_model, blur_model ):


  exposure_score = cal_exposure(image, exposure_model)

  blur_score = calculate_blur_rf(image, blur_model)

  

  with col2:
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
      st.image(image,caption= 'Uploaded Image')
    with sub_col2:
      # st.write("No references given, FOV score is not calculated")
      fig, ax = plt.subplots(1, 2, figsize=(6, 6))

            # FOV Score

            # Exposure Score
      ax[0].bar(['Exposure Score'], [exposure_score[0]], color='green', alpha=0.7)
      ax[0].set_ylim(0, 1)
      ax[0].set_title('Exposure Score')
          

            # Blur Score
      ax[1].bar(['Blur Score'], [blur_score], color='red', alpha=0.7)
      ax[1].set_ylim(0, 1)
      ax[1].set_title('Blur Score')

      # st.write("Exposure score : {}, Blur score : {} ".format(exposure_score[0], blur_score))
            # Display the plot using st.pyplot
      st.pyplot(fig)
    st.write("Exposure score : {:.5f}, Blur score : {:.5f}".format(exposure_score[0], blur_score))
    


  print(f"Exposure score : {exposure_score}, Blur score : {blur_score} ")

  final_score = min([exposure_score[0], blur_score])
  return final_score

def calculate_histogram_statistics(hist, img):
    total_pixels = np.sum(hist)
    cdf = np.cumsum(hist) / total_pixels


    mean_intensity = np.average(np.arange(256), weights=hist.flatten())
    std_dev_intensity = np.sqrt(np.average((np.arange(256) - mean_intensity)**2, weights=hist.flatten()))


    # Find the intensity values corresponding to the percentiles


    intensity_values = np.arange(256)
    median_intensity = intensity_values[np.argmax(cdf >= 0.5)]

    percentile_10 = intensity_values[np.argmax(cdf >= 0.10)]

    percentile_90 = intensity_values[np.argmax(cdf >= 0.90)]

    kurtosis_intensity = kurtosis(img.flatten())
    skewness_intensity = skew(img.flatten())

    return {
        'mean': mean_intensity,
        'std_dev': std_dev_intensity,
        'median': median_intensity,
        'kurtosis': kurtosis_intensity,
        'skewness': skewness_intensity,
        'percentile_10': percentile_10,
        'percentile_90': percentile_90,
    }
def calculate_image_statistics(image_path):
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = image_path

    # Calculate pixel intensity histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # Calculate histogram statistics
    stats = calculate_histogram_statistics(hist, img)

    return stats

def cal_exposure(file, model):
  stats = calculate_image_statistics(file)
  y_pred = model.predict(np.array(list(stats.values())).reshape(1, -1))
  return y_pred


def calculate_blur_rf(image_file, model):
#   image = cv2.imread(image_file)
  image = image_file
  fourier_spectrum = np.fft.fft2(image)
  psf = np.fft.fftshift(fourier_spectrum)
  magnitude_psf = np.abs(psf)
  mean_magnitude = np.mean(magnitude_psf)

  contrast = np.var(image)
  skewness = skew(image.flatten())
  kurt = kurtosis(image.flatten())


  laplacian = cv2.Laplacian(image, cv2.CV_64F)
  abs_laplacian = np.absolute(laplacian)
  laplacian_mean = np.mean(abs_laplacian)
  laplacian_var = np.var(laplacian)

  x={
     'Contrast' : contrast,
    #  'Skewness' : skewness,
    #  'Kurtosis' : kurt,
     'psf' :	mean_magnitude,
     'Laplacian_Var'	: laplacian_var,
     'Laplacian_mean' :	laplacian_mean
  }

  y=model.predict(np.array(list(x.values())).reshape(1, -1))

  return float(map_value(y, 0.4, 0.89, 0, 1))

def map_value(value, old_min, old_max, new_min, new_max):
    old_range = old_max - old_min
    new_range = new_max - new_min

    # Map the value to the new range
    mapped_value = (((value - old_min) * new_range) / old_range) + new_min

    # Ensure the value is within the specified range
    return max(min(mapped_value, new_max), new_min)

def round_to_nearest_10(degrees):
    # If greater than 180, take 360 - value
      rounded_degrees = round(degrees / 5) * 5

        # Ensure values close to 360 are treated as if they are close to 0
      return rounded_degrees % 360

def calculate_similarity_score(img_folder_path, similar_keypoints, similar_descriptors):

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    img2 = img_folder_path

    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    matches = bf.knnMatch(similar_descriptors, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
      if m.distance <  0.93 * n.distance:
          good_matches.append(m)

    angles1 = [key.angle for key in similar_keypoints]
    angles2 = [key.angle for key in keypoints2]

    # Calculate differences in angles for keypoints in good matches
    angle_diffs = [abs(angles1[m.queryIdx] - angles2[m.trainIdx]) for m in good_matches]

    # Round differences to the nearest multiple of 5
    angle_diffs = [round_to_nearest_10(diff) for diff in angle_diffs]
    try:
      rotation_angle = statistics.mode(angle_diffs)
    except:
      rotation_angle=180

    angles_score = abs(1 - (rotation_angle/180))


  # Calculate a similarity score based on the number of good matches
    similarity_score = len(good_matches)/len(similar_descriptors)

    if similarity_score < 0.1:
      angles_score = 0

    total_score = similarity_score*angles_score

    print(f"Total score : {total_score}, Similarity_score : {similarity_score}, Angle_score : {angles_score} ")

    return total_score, similarity_score, angles_score

def generate_basket_for_similarity(image_filenames):
  sift = cv2.SIFT_create()

  # Store keypoints and descriptors for each image
  keypoints_list = []
  descriptors_list = []

  # Loop through the images and extract keypoints/descriptors
  for filename in image_filenames:
      img = filename # Read the image in grayscale
      keypoints, descriptors = sift.detectAndCompute(img, None)
      keypoints_list.append((filename, keypoints))  # Store keypoints along with the filename
      descriptors_list.append((filename, descriptors))  # Store descriptors along with the filename

  # Find similar keypoints and descriptors among all images
  similar_keypoints = []
  similar_descriptors = []

  for idx, (filename, keypoints) in enumerate(keypoints_list):
      other_keypoints = [kp for i, kp in enumerate(keypoints_list) if i != idx]
      other_descriptors = [desc for i, desc in enumerate(descriptors_list) if i != idx]

      bf = cv2.BFMatcher()
      for i, kp in enumerate(keypoints):
          similar = []
          for o_kp, o_desc in zip(other_keypoints, other_descriptors):
              matches = bf.knnMatch(descriptors_list[idx][1][i:i+1], o_desc[1], k=2)
              for m, n in matches:
                  if m.distance < 0.75 * n.distance:
                      similar.append(kp)
                      break
          if len(similar) == len(other_keypoints):
              similar_keypoints.append(kp)
              similar_descriptors.append(descriptors_list[idx][1][i])

  similar_keypoints = np.array(similar_keypoints)
  similar_descriptors = np.array(similar_descriptors)

  return (similar_keypoints, similar_descriptors )

def calculate_overall_score( image, references , exposure_model, blur_model ):

  similar_keypoints, similar_descriptors = generate_basket_for_similarity(references)
  total_score, similarity_score, angles_score = calculate_similarity_score(image, similar_keypoints, similar_descriptors)

  exposure_score = cal_exposure(image, exposure_model)

  blur_score = calculate_blur_rf(image, blur_model)

  # print(f"FOV score : {total_score}, Exposure score : {exposure_score}, Blur score : {blur_score} ")
  with col2:
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
      st.image(image,caption= 'Uploaded Image')
    with sub_col2:
      fig, ax = plt.subplots(1, 3, figsize=(8, 8))

      # FOV Score

      # Exposure Score
      ax[0].bar(['Exposure'], [exposure_score[0]], color='green', alpha=0.7)
      ax[0].set_ylim(0, 1)
      ax[0].set_title('Exposure')

      # Blur Score
      ax[1].bar(['Blur'], [blur_score], color='red', alpha=0.7)
      ax[1].set_ylim(0, 1)
      ax[1].set_title('Blur')

      ax[2].bar(['FOV'], [total_score], color='blue', alpha=0.7)
      ax[2].set_ylim(0, 1)
      ax[2].set_title('FOV')

      # Adjust layout
      

      # Display the plot using st.pyplot
      st.pyplot(fig)
    
    st.write("Exposure score : {:.5f}, Blur score : {:.5f}, FOV score : {:.5f}".format(exposure_score[0], blur_score, total_score))



  print([total_score, exposure_score[0], blur_score])

  final_score = min([total_score, exposure_score[0], blur_score])
  return final_score

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    resized_image = image.resize((200, 200))

    # with col2:
    #   sub_col1, sub_col2 = st.columns(2)
    #   with sub_col1:
    #     st.image(resized_image,caption= 'Uploaded Image')

    image = np.array(image)
    # Process the image (you can replace this with your own processing logic)
    if (len(references)<3):
      processed_data = calculate_overall_score_without_FOV( image, exposure_model, blur_model )
    else :
      processed_data = calculate_overall_score( image,references, exposure_model, blur_model )

    # Display processed information
    # st.write("Processed Information:", processed_data)