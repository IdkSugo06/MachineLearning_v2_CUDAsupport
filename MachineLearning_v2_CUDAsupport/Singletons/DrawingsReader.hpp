#pragma once
#include "Drawer.cuh"
#include "..\AI_structs\NeuralNetwork.hpp"


class DrawingsReader {

	NeuralNetwork* p_neuralNetwork;
	MyArray<neuronValue_t> m_input;

public:
	DrawingsReader(NeuralNetwork& _neuralNetwork) {
		p_neuralNetwork = &_neuralNetwork;
		m_input.Resize(p_neuralNetwork->m_neuronNumbers[0]);
	}
	~DrawingsReader(){

	}

	void ExtractInputFromImage(BitmapImage* image) {

		int pixelId = 0; //Input is inverted
		for (int rowId = (image->m_height - 1) * image->m_width; rowId >= 0 ; rowId -= image->m_width) {
			for (int columnId = 0; columnId < image->m_width; columnId++) {
				int neuronId = rowId + columnId;
				uint8* mem = &(((uint8*)image->p_memory)[pixelId * BMP_IMAGE_BYTEPP]);
				float averageValue = 0;
				averageValue += mem[0];
				averageValue += mem[1];
				averageValue += mem[2];
				m_input[neuronId] = averageValue / (3 * 255);
				pixelId++;
			}
		}
	}
	void ExtractImageFromInput(MyArray<neuronValue_t>* input, BitmapImage* outImage) {
		for (int i = 0; i < input->GetLength(); i++) {
			uint8* mem = &(((uint8*)outImage->p_memory)[i * BMP_IMAGE_BYTEPP]);
			uint8 value = (uint8)(input->p_array[i] * 255);
			mem[0] = value; mem[1] = value; mem[2] = value;
		}
	}
	void GuessImage(BitmapImage* image) {
		ExtractInputFromImage(image);
		p_neuralNetwork->SetInput(m_input);
		p_neuralNetwork->ComputeOutput();
		std::cout << "nn guess: " << p_neuralNetwork->GetOutput() << "\n";
		std::cout << "nn guess: " << p_neuralNetwork->GetIndexOfHeighestOutput() << "\n";
	}
};