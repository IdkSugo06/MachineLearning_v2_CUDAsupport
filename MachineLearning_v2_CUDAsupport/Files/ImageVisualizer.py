from ImageModifier import *
import pygame

class ImageVisualizer:
    
    def __init__(self):
        pygame.init()
        self.visualMultiplier = 20
        self.screen = pygame.display.set_mode((IMAGE_DIM*self.visualMultiplier,IMAGE_DIM*self.visualMultiplier))
    
    def ShowImage(self, image):
        y = 0
        for pixelRow in image:
            for i in range(self.visualMultiplier):
                y += 1
                x = 0
                for pixel in pixelRow:
                    for j in range(self.visualMultiplier):
                        x += 1
                        self.screen.set_at((x, y), (pixel,pixel,pixel))
        pygame.display.update()
    
    def ShowImages(self, images : list):
        for image in images:
            y = 0
            for pixelRow in image:
                for i in range(self.visualMultiplier):
                    y += 1
                    x = 0
                    for pixel in pixelRow:
                        for j in range(self.visualMultiplier):
                            x += 1
                            self.screen.set_at((x, y), (pixel,pixel,pixel))
            pygame.display.update()
            input()
