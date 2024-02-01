#pragma once
#include "MyWindow.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define MAX_THREAD_GPU 1024

__global__ void SaveImage_GPU(uint8* p_memoryOfShownImg, uint8* p_memoryOfSavedImg, int shownImgWidth, int savedImgWidth, int scaleMultiplier, int totPixelInSavedImg);

class Drawer {
	//Images
	MyWindow* p_window;
	BitmapImage m_shownImage;
	BitmapImage m_savedImage;
	//Cuda images
	uint8* shownImageMemory_cudaPointer;
	uint8* savedImageMemory_cudaPointer;
	//Other
	int savedImageScaleMultiplier = 1; //The savedImage will be savedImageScaleMultiplier times smaller than the shownImage
	int pencilRadius = 35;
	const int timeBetweenInputs = 100; //ms
	float timeUntilNextInput = 0; //ms
	bool showSavedImg = false;

public:
	Drawer(int savedImgWidth, int savedImgHeight, int scaleMultiplier, MyWindow* _window) {
		p_window = _window;
		savedImageScaleMultiplier = scaleMultiplier;
		m_shownImage.Resize(savedImgWidth * scaleMultiplier, savedImgHeight * scaleMultiplier);
		m_savedImage.Resize(savedImgWidth, savedImgHeight);
		m_shownImage.MyFillMemory(0, 0, 0);

		cudaError_t firstAlloc = cudaMalloc((void**)&shownImageMemory_cudaPointer, m_shownImage.m_numOf_pixels * BMP_IMAGE_BYTEPP);
		cudaError_t secondAlloc = cudaMalloc((void**)&savedImageMemory_cudaPointer, m_savedImage.m_numOf_pixels * BMP_IMAGE_BYTEPP);
		if ((firstAlloc != cudaSuccess) || (secondAlloc != cudaSuccess)) {
			std::cout << "Errore durante allocazione cuda\n"; return;
		}
	}
	~Drawer() {

	}

	void Update(float deltaTime_ms) {
		if (timeUntilNextInput > 0) {
			timeUntilNextInput -= deltaTime_ms;
		}
		else{
			if (InputHandler::inputHandler.isPressed(DC_E)) {
				m_shownImage.MyFillMemory(0, 0, 0);
				timeUntilNextInput = timeBetweenInputs * 4;
			}
			else if (InputHandler::inputHandler.isPressed(DC_S)) {
				SaveImage(); 
				timeUntilNextInput = timeBetweenInputs * 4;
			}
			else if (InputHandler::inputHandler.isPressed(DC_CONTROL)) {
				pencilRadius--; if (pencilRadius <= 0) { pencilRadius = 1; }
				timeUntilNextInput = timeBetweenInputs/2;
			}
			else if (InputHandler::inputHandler.isPressed(DC_SHIFT)) {
				pencilRadius++;
				timeUntilNextInput = timeBetweenInputs/2;
			}
			else if (InputHandler::inputHandler.isPressed(DC_D)) {
				timeUntilNextInput = timeBetweenInputs * 4;	
				showSavedImg = true;
			}
			else if (InputHandler::inputHandler.isPressed(DC_F)) {
				timeUntilNextInput = timeBetweenInputs * 4;
				showSavedImg = false;
			}
		}

		if (InputHandler::inputHandler.isPressed(DC_SPACE)) {
			POINT p;
			if (GetCursorPos(&p)) {
				p.x -= WINDOW_STARTING_POINT_X;
				p.y -= WINDOW_STARTING_POINT_Y;
				if ((p.x > pencilRadius && p.x < SCREEN_WIDTH - pencilRadius && p.y > pencilRadius && p.y < SCREEN_HEIGHT - pencilRadius)) {
					//std::cout << p.x << " " << p.y << "\n";
					for (int x = -pencilRadius; x <= pencilRadius; x++) {
						for (int y = -pencilRadius; y <= pencilRadius; y++) {
							m_shownImage.SetPixel(p.x + x, SCREEN_HEIGHT - (p.y + y), 255, 255, 255);
						}
					}
				}
			}
		}
		if (showSavedImg) {
			p_window->DrawWindowFrame(&m_savedImage);
		}
		else {
			p_window->DrawWindowFrame(&m_shownImage);
		}
	}
	void UpdateImage() {
		p_window->DrawWindowFrame(&m_shownImage);
	}

	BitmapImage& GetSavedImage() {
		return m_savedImage;
	}
	void SaveImage() {
		cudaMemcpy(shownImageMemory_cudaPointer, m_shownImage.p_memory, m_shownImage.m_numOf_pixels * BMP_IMAGE_BYTEPP, cudaMemcpyHostToDevice);
		int numOfBlocks = (m_savedImage.m_numOf_pixels / MAX_THREAD_GPU) + 1;
		SaveImage_GPU<<<numOfBlocks, MAX_THREAD_GPU, 0, 0>>>(shownImageMemory_cudaPointer, savedImageMemory_cudaPointer, m_shownImage.m_width, m_savedImage.m_width, savedImageScaleMultiplier, m_savedImage.m_numOf_pixels);
		cudaMemcpy(m_savedImage.p_memory, savedImageMemory_cudaPointer, m_savedImage.m_numOf_pixels * BMP_IMAGE_BYTEPP, cudaMemcpyDeviceToHost);
	}
};

__global__ void SaveImage_GPU(uint8* p_memoryOfShownImg, uint8* p_memoryOfSavedImg, int shownImgWidth, int savedImgWidth, int scaleMultiplier, int totPixelInSavedImg) {
	if (threadIdx.x > totPixelInSavedImg) { return; }
	int row = threadIdx.x / savedImgWidth;
	int column = threadIdx.x % savedImgWidth;
	float averageValue = 0;
	
	int startingPixelId = row * shownImgWidth * scaleMultiplier + column * scaleMultiplier;
	for (int y = 0; y < scaleMultiplier; y++) {
		for (int x = 0; x < scaleMultiplier; x++) {
			uint8* mem = &(p_memoryOfShownImg)[(startingPixelId + x + y * shownImgWidth) * BMP_IMAGE_BYTEPP];
			averageValue += mem[0];
			averageValue += mem[1];
			averageValue += mem[2];
		}
	}
	uint8 averageColor = averageValue / (3 * scaleMultiplier * scaleMultiplier);

	int savedImgPixelId = row * savedImgWidth + column;
	p_memoryOfSavedImg[savedImgPixelId * BMP_IMAGE_BYTEPP + 2] = averageColor;
	p_memoryOfSavedImg[savedImgPixelId * BMP_IMAGE_BYTEPP + 1] = averageColor;
	p_memoryOfSavedImg[savedImgPixelId * BMP_IMAGE_BYTEPP + 0] = averageColor;
}