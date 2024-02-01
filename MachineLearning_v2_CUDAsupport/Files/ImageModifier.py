from ImagesImporter import *
from math import *
import random

class ImageModifier:
    
    def __init__(self):
        self.upScaleMultiplier = 2

    def RotateImages(self, images : list):
        newImages = []
        i = 0
        for image in images:
            print(i, len(images))
            angle = random.randint(-1,1) * pi/9
            offset = (random.randint(-1,1)*1.5, random.randint(-1,1)*1.5)
            noiseFactor = 0.3
            newImages.append(self.RotateImage(image, angle, offset, noiseFactor))
            i += 1
        return newImages

    def RotateImage(self, image : list, angle : float, offset : list, noiseFactor : float):
        upscaledImageDimension = IMAGE_DIM*self.upScaleMultiplier

        #Upscale the image
        upscaledImage = []
        for pixelRow in image:
            newPixelRow = []
            for y in range(self.upScaleMultiplier):
                for pixel in pixelRow:
                    #for x in range(self.upScaleMultiplier):
                    newPixelRow += [pixel] * self.upScaleMultiplier
            upscaledImage += [newPixelRow] * self.upScaleMultiplier
        
        #Rotate the image
        cosA = cos(angle)
        sinA = sin(angle)
        upscaledRotatedImage = []
        for row in upscaledImage:
            upscaledRotatedImage.append(([0] * len(row)))
        y = -upscaledImageDimension / 2 + offset[1] * self.upScaleMultiplier
        for pixelRow in upscaledImage:
            x = -upscaledImageDimension / 2 + offset[0] * self.upScaleMultiplier
            for pixel in pixelRow:
                xf = int((x * cosA - y * sinA) + upscaledImageDimension / 2)
                yf = int((x * sinA + y * cosA) + upscaledImageDimension / 2)
                if not(xf < 0 or xf >= upscaledImageDimension or yf < 0 or yf >= upscaledImageDimension):
                    upscaledRotatedImage[yf][xf] = pixel
                x += 1
            y += 1

        #Downscale the image
        noiseFactor = 1 - noiseFactor
        finalImage = []
        for row in image:
            finalImage.append(row.copy())
        for y in range(len(finalImage)):
            for x in range(len(finalImage[y])):
                xf = x * self.upScaleMultiplier
                yf = y * self.upScaleMultiplier
                averagePixelValue = 0
                for i in range(self.upScaleMultiplier):
                    for j in range(self.upScaleMultiplier):
                        averagePixelValue += upscaledRotatedImage[yf + j][xf + i]
                averagePixelValue = averagePixelValue / (self.upScaleMultiplier*self.upScaleMultiplier)
                finalImage[y][x] = int(averagePixelValue)
                randomValue = random.randint(0,100) / 100
                if(randomValue > noiseFactor):
                    noiseCoefficent = randomValue - noiseFactor
                    finalImage[y][x] = min(255, round(finalImage[y][x] + noiseCoefficent * noiseCoefficent * 255)) 

        return finalImage
