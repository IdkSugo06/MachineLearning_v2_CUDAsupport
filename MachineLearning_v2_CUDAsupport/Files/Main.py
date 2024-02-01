from ImageVisualizer import *
ii = ImagesImporter()
im = ImageModifier()

ii.ImportImages("mnist_train.csv")
newImages = im.RotateImages(ii.images)
print("Immagini ruotate")
ii.ExportImages("mnist_train_modifiedImgs.csv", newImages, ii.expectedDigits)
print("Immagini esportate")

if True:
    iv = ImageVisualizer()
    iv.ShowImages(newImages)
print("End")