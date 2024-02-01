IMAGE_DIM = 28

class ImagesImporter:

    def __init__(self):
        self.images = [] #list of list of int
        self.expectedDigits = [] #list of int
    
    def ImportImages(self, trainingFile_MNISTcsv : str):
        fileStream = open(trainingFile_MNISTcsv, "r")
        pixelArray = fileStream.readline() #prima riga saltata (contiene solo lables)

        #Allocate the memory needed
        image = [] #list of list of int
        pixelRow = []
        for i in range(IMAGE_DIM):
            pixelRow.append(0)
        for i in range(IMAGE_DIM):
            image.append(pixelRow.copy())

        #Read the images
        j = 0
        while(fileStream):
            pixelArray = fileStream.readline()
            if(len(pixelArray) < 1): break
            expectedDigit = int(pixelArray[0])
            print(j)
            pixelArray = pixelArray[2:] #remove the first digit and the comma

            numRead = ""
            pixelId = 0
            rowId = 0
            for char in pixelArray:
                if char == ',':         
                    image[rowId][pixelId] = int(numRead)
                    numRead = ""

                    pixelId += 1
                    #New row
                    if pixelId == IMAGE_DIM:
                        pixelId = 0
                        rowId += 1
                else:
                    numRead += char
            image[rowId][pixelId] = int(numRead)

            newImage = []
            for i in range(IMAGE_DIM):
                newImage.append(image[i].copy())
            self.images.append(newImage)
            self.expectedDigits.append(expectedDigit)
            j += 1
        print("end")
        fileStream.close()
    
    def ExportImages(self, trainingFile_MNISTcsv : str, images : list, expectedDigits : list):
        fileStream = open(trainingFile_MNISTcsv, "w")
        stringToWrite = "label\n"

        idImage = 0
        for image in images:
            print(idImage)
            stringToWrite += str(expectedDigits[idImage]) + ','
            for row in image:
                stringToWrite += str(row)[1:-1].replace(' ', '') + ','
            stringToWrite = stringToWrite[:-1] + '\n'
            fileStream.write(stringToWrite)
            stringToWrite = ""
            idImage += 1
        
        fileStream.write(stringToWrite)
        fileStream.close()