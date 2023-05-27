# importer les librairies
from PIL import Image, ImageStat, ImageFilter, ImageOps
from matplotlib import pyplot as plt
import numpy as np
import streamlit as st


def main():
    selected_box = st.sidebar.selectbox('Sélectionner dans la liste déroulante',
                                        ('Améliorateur d’image sous-marin', 'À propos de l’application'))
    if selected_box == 'À propos de l’application':
        about()
    if selected_box == 'Améliorateur d’image sous-marin':
        image_enhancer()


def about():
    st.title("Welcome!")
    st.caption("Améliorateur d’image sous-marin webb app")
    with st.expander("resume"):
        st.write(
            """Les images sous-marines trouvent des applications dans divers domaines, tels que la recherche marine, l'inspection des habitats aquatiques, la surveillance sous-marine, l'identification des minéraux, et bien plus encore. Cependant, les prises de vue sous-marines sont fortement affectées lors du processus d'acquisition en raison de l'absorption et de la diffusion de la lumière. À mesure que la profondeur augmente, les longueurs d'onde plus longues sont absorbées par l'eau ; par conséquent, les images apparaissent principalement bleu-vert, et le rouge est absorbé en raison de sa plus grande longueur d'onde. Ces phénomènes entraînent une dégradation significative des images, ce qui se traduit par un faible contraste, une distorsion des couleurs et une faible visibilité. Par conséquent, les images sous-marines nécessitent une amélioration pour améliorer la qualité des images afin qu'elles puissent être utilisées dans diverses applications tout en préservant les informations précieuses qu'elles contiennent..""")
    with st.expander("Block Diagram"):
        st.image('./images/block_diagram.png', use_column_width=True)
    with st.expander("Results On Sample Images"):
        st.image('./images/result1.PNG', use_column_width=True)
        st.image('./images/result2.PNG', use_column_width=True)
    with st.expander("Membres de l’équipe"):
        st.write("""saif mohamed marouane - 2018EEB1243
                    \n\nbenzakry aimad - 2018EEB1277""")


def image_enhancer():
    st.header("Améliorateur d’image sous-marin webb app")
    file = st.file_uploader("Veuillez télécharger un fichier image sous-marine RVB", type=["jpg", "png"])
    if file is None:
        st.text("Veuillez télécharger un fichier image")
    else:
        image = Image.open(file)
        if image.mode != 'RGB':
            st.text("Veuillez télécharger l’image RVB")
        else:
            st.text("Image téléchargée")
            st.image(image, use_column_width=True)
            imtype = st.radio("Sélectionnez-en un", ('Image verdâtre', 'Image bleutée'))
            if imtype == "Image verdâtre":
                flag = 0
            else:
                flag = 1
            if (st.button("Améliorer l’image téléchargée")):
                pcafused, averagefused = underwater_image_enhancement(image, flag)
                st.text("Image améliorée à l’aide de la fusion basée sur PCA")
                st.image(pcafused, use_column_width=True)
                st.text("Image améliorée à l’aide de la fusion basée sur la moyenne")
                st.image(averagefused, use_column_width=True)


# # Color Correction

# ## Étape 1 : Compensation des canaux R et B (si nécessaire)

# drapeau = 0 pour rouge, compensation bleue via le canal vert
# drapeau = 1 pour la compensation rouge via le canal vert
def compensate_RB(image, flag):
    # Fractionnement de l’image en composantes R, G et B
    imager, imageg, imageb = image.split()

    # Obtenir une valeur de pixel maximale et minimale
    minR, maxR = imager.getextrema()
    minG, maxG = imageg.getextrema()
    minB, maxB = imageb.getextrema()

    # Convertir en tableau
    imageR = np.array(imager, np.float64)
    imageG = np.array(imageg, np.float64)
    imageB = np.array(imageb, np.float64)

    x, y = image.size

    # Normalisation de la valeur des pixels à la plage (0, 1)
    for i in range(0, y):
        for j in range(0, x):
            imageR[i][j] = (imageR[i][j] - minR) / (maxR - minR)
            imageG[i][j] = (imageG[i][j] - minG) / (maxG - minG)
            imageB[i][j] = (imageB[i][j] - minB) / (maxB - minB)

    # Obtenir la moyenne de chaque canal
    meanR = np.mean(imageR)
    meanG = np.mean(imageG)
    meanB = np.mean(imageB)

    # Compenser le canal rouge et bleu
    if flag == 0:
        for i in range(y):
            for j in range(x):
                imageR[i][j] = int((imageR[i][j] + (meanG - meanR) * (1 - imageR[i][j]) * imageG[i][j]) * maxR)
                imageB[i][j] = int((imageB[i][j] + (meanG - meanB) * (1 - imageB[i][j]) * imageG[i][j]) * maxB)

        # Mise à l’échelle des valeurs de pixels à la plage d’origine
        for i in range(0, y):
            for j in range(0, x):
                imageG[i][j] = int(imageG[i][j] * maxG)

    # Compenser le canal rouge
    if flag == 1:
        for i in range(y):
            for j in range(x):
                imageR[i][j] = int((imageR[i][j] + (meanG - meanR) * (1 - imageR[i][j]) * imageG[i][j]) * maxR)

        # Mise à l’échelle des valeurs de pixels à la plage d’origine
        for i in range(0, y):
            for j in range(0, x):
                imageB[i][j] = int(imageB[i][j] * maxB)
                imageG[i][j] = int(imageG[i][j] * maxG)

    # Créer l’image compensée
    compensateIm = np.zeros((y, x, 3), dtype="uint8")
    compensateIm[:, :, 0] = imageR;
    compensateIm[:, :, 1] = imageG;
    compensateIm[:, :, 2] = imageB;

    compensateIm = Image.fromarray(compensateIm)

    return compensateIm


# ## Étape 2: Équilibrage des blancs à l’aide de l’algorithme du monde gris

def gray_world(image):
    # Fractionnement de l’image en composantes R, G et B
    imager, imageg, imageb = image.split()

    # de grayscale image
    imagegray = image.convert('L')

    # Convertir to array
    imageR = np.array(imager, np.float64)
    imageG = np.array(imageg, np.float64)
    imageB = np.array(imageb, np.float64)
    imageGray = np.array(imagegray, np.float64)

    x, y = image.size

    # Obtenir la valeur moyenne des pixels
    meanR = np.mean(imageR)
    meanG = np.mean(imageG)
    meanB = np.mean(imageB)
    meanGray = np.mean(imageGray)

    # Algorithme du monde gris
    for i in range(0, y):
        for j in range(0, x):
            imageR[i][j] = int(imageR[i][j] * meanGray / meanR)
            imageG[i][j] = int(imageG[i][j] * meanGray / meanG)
            imageB[i][j] = int(imageB[i][j] * meanGray / meanB)

    # Créer l’image blanche équilibrée
    whitebalancedIm = np.zeros((y, x, 3), dtype="uint8")
    whitebalancedIm[:, :, 0] = imageR;
    whitebalancedIm[:, :, 1] = imageG;
    whitebalancedIm[:, :, 2] = imageB;

    # Plotting the compensated image
    # plt.figure(figsize = (20, 20))
    # plt.subplot(1, 2, 1)
    # plt.title("Compensated Image")
    # plt.imshow(image)
    # plt.subplot(1, 2, 2)
    # plt.title("White Balanced Image")
    # plt.imshow(whitebalancedIm)
    # plt.show()

    return Image.fromarray(whitebalancedIm)


# # Netteté de l’image blanche équilibrée

# Effectuer un masquage flou K=1
def sharpen(wbimage, original):
    # Trouvez d’abord l’image lissée à l’aide du filtre gaussien
    smoothed_image = wbimage.filter(ImageFilter.GaussianBlur)

    # Diviser l’image lissée en canaux R, G et B
    smoothedr, smoothedg, smoothedb = smoothed_image.split()

    # Diviser l’image d’entrée
    imager, imageg, imageb = wbimage.split()

    # Convertir une image en tableau
    imageR = np.array(imager, np.float64)
    imageG = np.array(imageg, np.float64)
    imageB = np.array(imageb, np.float64)
    smoothedR = np.array(smoothedr, np.float64)
    smoothedG = np.array(smoothedg, np.float64)
    smoothedB = np.array(smoothedb, np.float64)

    x, y = wbimage.size

    # Perform unsharp masking
    for i in range(y):
        for j in range(x):
            imageR[i][j] = 2 * imageR[i][j] - smoothedR[i][j]
            imageG[i][j] = 2 * imageG[i][j] - smoothedG[i][j]
            imageB[i][j] = 2 * imageB[i][j] - smoothedB[i][j]

    # Créer une image nette
    sharpenIm = np.zeros((y, x, 3), dtype="uint8")
    sharpenIm[:, :, 0] = imageR;
    sharpenIm[:, :, 1] = imageG;
    sharpenIm[:, :, 2] = imageB;

    # Plotting the sharpened image
    # plt.figure(figsize = (20, 20))
    # plt.subplot(1, 3, 1)
    # plt.title("Original Image")
    # plt.imshow(original)
    # plt.subplot(1, 3, 2)
    # plt.title("White Balanced Image")
    # plt.imshow(wbimage)
    # plt.subplot(1, 3, 3)
    # plt.title("Sharpened Image")
    # plt.imshow(sharpenIm)
    # plt.show()

    return Image.fromarray(sharpenIm)


# # Amélioration du contraste de l’image symétrique des blanches par égalisation globale de l’histogramme

def hsv_global_equalization(image):
    # Convert to HSV
    hsvimage = image.convert('HSV')

    # Plot HSV Image
    # plt.figure(figsize = (20, 20))
    # plt.subplot(1, 2, 1)
    # plt.title("White balanced Image")
    # plt.imshow(hsvimage)

    # Fractionnement de la composante teinte, saturation et valeur
    Hue, Saturation, Value = hsvimage.split()
    # Effectuer l’égalisation sur la composante de valeur
    equalizedValue = ImageOps.equalize(Value, mask=None)

    x, y = image.size
    # Créer l’image égalisée
    equalizedIm = np.zeros((y, x, 3), dtype="uint8")
    equalizedIm[:, :, 0] = Hue;
    equalizedIm[:, :, 1] = Saturation;
    equalizedIm[:, :, 2] = equalizedValue;

    # Convertir le tableau en image
    hsvimage = Image.fromarray(equalizedIm, 'HSV')
    # Convertir en RGB
    rgbimage = hsvimage.convert('RGB')

    # Plot equalized image
    # plt.subplot(1, 2, 2)
    # plt.title("Contrast enhanced Image")
    # plt.imshow(rgbimage)

    return rgbimage


# # Fusion de l’image nette et de l’image contrastée

# ## Utilisation de la méthode de calcul de la moyenne

def average_fusion(image1, image2):
    # Diviser les images en composants R, G, B
    image1r, image1g, image1b = image1.split()
    image2r, image2g, image2b = image2.split()

    # Convert en array
    image1R = np.array(image1r, np.float64)
    image1G = np.array(image1g, np.float64)
    image1B = np.array(image1b, np.float64)
    image2R = np.array(image2r, np.float64)
    image2G = np.array(image2g, np.float64)
    image2B = np.array(image2b, np.float64)

    x, y = image1R.shape

    # Effectuer la fusion en faisant la moyenne des valeurs de pixels
    for i in range(x):
        for j in range(y):
            image1R[i][j] = int((image1R[i][j] + image2R[i][j]) / 2)
            image1G[i][j] = int((image1G[i][j] + image2G[i][j]) / 2)
            image1B[i][j] = int((image1B[i][j] + image2B[i][j]) / 2)

    # Créer l’image fusionnée
    fusedIm = np.zeros((x, y, 3), dtype="uint8")
    fusedIm[:, :, 0] = image1R;
    fusedIm[:, :, 1] = image1G;
    fusedIm[:, :, 2] = image1B;

    # Plot the fused image
    # plt.figure(figsize = (20, 20))
    # plt.subplot(1, 3, 1)
    # plt.title("Sharpened Image")
    # plt.imshow(image1)
    # plt.subplot(1, 3, 2)
    # plt.title("Contrast Enhanced Image")
    # plt.imshow(image2)
    # plt.subplot(1, 3, 3)
    # plt.title("Average Fused Image")
    # plt.imshow(fusedIm)
    # plt.show()

    return Image.fromarray(fusedIm)


# ## Utilisation de l’analyse en composantes principales (ACP)
def pca_fusion(image1, image2):
    # Split the images in R, G, B components
    image1r, image1g, image1b = image1.split()
    image2r, image2g, image2b = image2.split()

    # Convertor en column vector
    image1R = np.array(image1r, np.float64).flatten()
    image1G = np.array(image1g, np.float64).flatten()
    image1B = np.array(image1b, np.float64).flatten()
    image2R = np.array(image2r, np.float64).flatten()
    image2G = np.array(image2g, np.float64).flatten()
    image2B = np.array(image2b, np.float64).flatten()

    # Obtenir la moyenne de chaque canal
    mean1R = np.mean(image1R)
    mean1G = np.mean(image1G)
    mean1B = np.mean(image1B)
    mean2R = np.mean(image2R)
    mean2G = np.mean(image2G)
    mean2B = np.mean(image2B)

    # Créer un tableau 2*N où chaque colonne représente chaque canal d’image
    imageR = np.array((image1R, image2R))
    imageG = np.array((image1G, image2G))
    imageB = np.array((image1B, image2B))

    x, y = imageR.shape

    # Soustrayez la moyenne respective de chaque colonne
    for i in range(y):
        imageR[0][i] -= mean1R
        imageR[1][i] -= mean2R
        imageG[0][i] -= mean1G
        imageG[1][i] -= mean2G
        imageB[0][i] -= mean1B
        imageB[1][i] -= mean2B

    # Trouver la matrice de covariance
    covR = np.cov(imageR)
    covG = np.cov(imageG)
    covB = np.cov(imageB)

    # Trouver la valeur eigen et le vecteur eigen
    valueR, vectorR = np.linalg.eig(covR)
    valueG, vectorG = np.linalg.eig(covG)
    valueB, vectorB = np.linalg.eig(covB)

    # Trouvez les coefficients pour chaque canal qui serviront de poids pour les images
    if (valueR[0] >= valueR[1]):
        coefR = vectorR[:, 0] / sum(vectorR[:, 0])
    else:
        coefR = vectorR[:, 1] / sum(vectorR[:, 1])

    if (valueG[0] >= valueG[1]):
        coefG = vectorG[:, 0] / sum(vectorG[:, 0])
    else:
        coefG = vectorG[:, 1] / sum(vectorG[:, 1])

    if (valueB[0] >= valueB[1]):
        coefB = vectorB[:, 0] / sum(vectorB[:, 0])
    else:
        coefB = vectorB[:, 1] / sum(vectorB[:, 1])

    # Convertir en array
    image1R = np.array(image1r, np.float64)
    image1G = np.array(image1g, np.float64)
    image1B = np.array(image1b, np.float64)
    image2R = np.array(image2r, np.float64)
    image2G = np.array(image2g, np.float64)
    image2B = np.array(image2b, np.float64)

    x, y = image1R.shape

    # Calculer la valeur en pixels de l’image fusionnée à partir des coefficients obtenus ci-dessus
    for i in range(x):
        for j in range(y):
            image1R[i][j] = int(coefR[0] * image1R[i][j] + coefR[1] * image2R[i][j])
            image1G[i][j] = int(coefG[0] * image1G[i][j] + coefG[1] * image2G[i][j])
            image1B[i][j] = int(coefB[0] * image1B[i][j] + coefB[1] * image2B[i][j])

    # Créer l’image fusionnée
    fusedIm = np.zeros((x, y, 3), dtype="uint8")
    fusedIm[:, :, 0] = image1R;
    fusedIm[:, :, 1] = image1G;
    fusedIm[:, :, 2] = image1B;

    # Plot the fused image
    # plt.figure(figsize = (20, 20))
    # plt.subplot(1, 3, 1)
    # plt.title("Sharpened Image")
    # plt.imshow(image1)
    # plt.subplot(1, 3, 2)
    # plt.title("Contrast Enhanced Image")
    # plt.imshow(image2)
    # plt.subplot(1, 3, 3)
    # plt.title("PCA Fused Image")
    # plt.imshow(fusedIm)
    # plt.show()

    return Image.fromarray(fusedIm)


# # Fonction d’amélioration de l’image sous-marine

# drapeau = 0 pour rouge, compensation bleue via le canal vert
# drapeau = 1 pour la compensation rouge via le canal vert
def underwater_image_enhancement(image, flag):
    # Compensate image based on flag
    st.text("Compensating Red/Blue Channel Based on Green Channel...")
    compensatedimage = compensate_RB(image, flag)
    # Apply gray world algorithm to complete color correction
    st.text("White Balancing the compensated Image using Grayworld Algorithm...")
    whitebalanced = gray_world(compensatedimage)
    # Perform contrast enhancement using global Histogram Equalization
    st.text("Enhancing Contrast of White Balanced Image using Global Histogram Equalization...")
    contrastenhanced = hsv_global_equalization(whitebalanced)
    # Perform Unsharp Masking to sharpen the color corrected image
    st.text("Sharpening White Balanced Image using Unsharp Masking...")
    sharpenedimage = sharpen(whitebalanced, image)
    # Perform avergaing-based fusion of sharpenend image & contrast enhanced image
    st.text("Performing Average Based Fusion of Sharped Image & Contrast Enhanced Image...")
    averagefused = average_fusion(sharpenedimage, contrastenhanced)
    # Perform PCA-based fusion of sharpenend image & contrast enhanced image
    st.text("Performing PCA Based Fusion of Sharped Image & Contrast Enhanced Image...")
    pcafused = pca_fusion(sharpenedimage, contrastenhanced)

    return pcafused, averagefused


if __name__ == "__main__":
    main()

