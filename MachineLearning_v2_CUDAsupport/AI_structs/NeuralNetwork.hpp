#pragma once 
#include "TrainingBatch.hpp"
#include <string>

struct NeuralNetwork {

public:
	int m_layersNumber = 0;

	MyArray<int> m_neuronNumbers;
	MyArray<neuronValue_t>* p_inputValues{nullptr};
	MyArray<neuronValue_t>* p_outputValues{nullptr};
	MyArray<neuronValue_t>* p_expectedOutput{nullptr};
	MyArray<neuronValue_t> m_cost;
	MyArray<Layer> m_layers;

private:
	ActivationFunctionType m_hl_af = ActivationFunctionType::relu;	//hidden layer activation function
	ActivationFunctionType m_ll_af = ActivationFunctionType::sigmoid; //last layer activation function

public:
	NeuralNetwork(const char* path) {

		// -- -- READ NEURAL NETWORK'S SPECIFICS -- --
		std::fstream fin; char charBuffer[FILE_CHAR_BUFFER_LENGHT];
		fin.open(path, std::ios::in | std::ios::binary);
		if (!fin) {
			std::cout << "Errore nell'apertura del file: " << path << "\n"; return;
		}

		//Check if the neural network is the same
		myFgets(fin, charBuffer, 3);
		if (charBuffer[0] != 'n' || charBuffer[1] != 'n' || charBuffer[2] != ':') {
			std::cout << "File type is not the right one: " << path << "\n"; return;
		}
		m_layersNumber = ReadFloat(fin, charBuffer, '\n');
		//Check if the number of neurons are the same
		myFgets(fin, charBuffer, 7);
		if (charBuffer[0] != 'l' || charBuffer[1] != 'a' || charBuffer[2] != 'y' || charBuffer[3] != 'e' || charBuffer[4] != 'r' || charBuffer[5] != 's' || charBuffer[6] != ':') {
			std::cout << "Error, layer's neurons number weren't found: " << path << "\n"; return;
		}
		m_neuronNumbers.Resize(m_layersNumber + 1);
		for (int layerId = 0; layerId < m_layersNumber + 1; layerId++) {
			m_neuronNumbers[layerId] = ReadFloat(fin, charBuffer, ',');
		}
		myFgets(fin, charBuffer, 3);
		if (charBuffer[0] != 'a' || charBuffer[1] != 'f' || charBuffer[2] != ':') {
			std::cout << "Activation function arent specified, dafault (sigmoid) will be used" << "\n";
			m_hl_af = ActivationFunctionType::sigmoid;
			m_ll_af = ActivationFunctionType::sigmoid;
		}
		else {
			m_hl_af = (ActivationFunctionType)(ReadFloat(fin, charBuffer, ','));
			m_ll_af = (ActivationFunctionType)(ReadFloat(fin, charBuffer, '\n'));
		}
		fin.close();

		// -- -- SET UP THE LAYERS -- --
		//Resize the layers
		m_cost.Resize(GetOutNeuronsNumber());
		m_layers.Resize(m_layersNumber);

		//Set up the layers
		{
			//The input of the first layer is the same as the neuralNetwork input
			m_layers[0].SetNumOfNeurons(m_neuronNumbers[0], m_neuronNumbers[1]);
			m_layers[0].SetInput(p_inputValues);
			m_layers[0].SetActivationFunctionType(m_hl_af);

			//Cycle through all of the layers to resize them and to set the input(MyArray*) array
			for (int i = 1; i < m_layersNumber; i++) {
				m_layers[i].SetNumOfNeurons(m_neuronNumbers[i], m_neuronNumbers[i + 1]); //Resize the layer's weights and biases
				MyArray<neuronValue_t>* output = &(m_layers[i - 1].GetOutput()); //Set the input as the last layer's output
				m_layers[i].SetInput(output);
				m_layers[i].SetActivationFunctionType(m_hl_af); //Set the activation function type
			}
			m_layers[m_layersNumber - 1].SetActivationFunctionType(m_ll_af); //Set the last layer's activation function type 
		}

		//Set the neuralNetwork output as the output of the lastLayer
		p_outputValues = &(GetLastLayer()->GetOutput()); //Set the neural output(*) as the output of the last one

		// -- -- READ THE SAVED WEIGHTS AND BIASES -- --
		if (!ReadWeightsBiases(path)) {
			std::cout << "Something went wrong during the reading of the neural network file, this nn could be broken\n";
		};
	}
	NeuralNetwork(int _numOfLayers, int* _neuronNumbers, ActivationFunctionType _hl_af = ActivationFunctionType::sigmoid, ActivationFunctionType _ll_af = ActivationFunctionType::sigmoid) {

		//Set the neuralNetwork input as the input that were taken as parameter 
		m_neuronNumbers.Resize(_numOfLayers);
		m_neuronNumbers.SetValues(_neuronNumbers);
		m_hl_af = _hl_af; m_ll_af = _ll_af;
		m_cost.Resize(GetOutNeuronsNumber(), 0);

		//Resize the layers
		m_layersNumber = _numOfLayers - 1;
		m_layers.Resize(m_layersNumber);

		//Set up the layers
		{
			//The input of the first layer is the same as the neuralNetwork input
			m_layers[0].SetNumOfNeurons(m_neuronNumbers[0], m_neuronNumbers[1]);
			m_layers[0].SetInput(p_inputValues);
			m_layers[0].SetActivationFunctionType(m_hl_af);

			//Cycle through all of the layers to resize them and to set the input(MyArray*) array
			for (int i = 1; i < m_layersNumber; i++) {
				m_layers[i].SetNumOfNeurons(m_neuronNumbers[i], m_neuronNumbers[i + 1]); //Resize the layer's weights and biases
				MyArray<neuronValue_t>* output = &(m_layers[i - 1].GetOutput()); //Set the input as the last layer's output
				m_layers[i].SetInput(output);
				m_layers[i].SetActivationFunctionType(m_hl_af); //Set the activation function type
			}
			m_layers[m_layersNumber - 1].SetActivationFunctionType(m_ll_af); //Set the last layer's activation function type 
		}

		//Set the neuralNetwork output as the output of the lastLayer
		p_outputValues = &(GetLastLayer()->GetOutput()); //Set the neural output(*) as the output of the last one
	}

	//Methods
	void ComputeOutput__training() {
		for (int i = 0; i < m_layersNumber - 1; i++) {
			m_layers[i].ComputeOutput__training();
		}
		m_layers[m_layersNumber - 1].ComputeOutput(); //last output cant be discarded
	}
	void ComputeOutput() {
		for (int i = 0; i < m_layersNumber; i++) {
			m_layers[i].ComputeOutput();
		}
	}
	void ComputeCost() {
		//layersNumber = neuronNumber - 1, so it would be [neuronNumber - 1] = [layersNumber]
		for (int i = 0; i < m_neuronNumbers[m_layersNumber]; i++) {
			float diff = ((p_outputValues->p_array)[i] - (p_expectedOutput->p_array)[i]);
			m_cost[i] = diff * diff;
		}
	}
	void ComputeGradients(){ //It adds the current gradients with the one before, it has to be avaraged at the end
		m_layers[m_layersNumber - 1].ComputeNeuronValuesGradient__lastLayer(*p_expectedOutput);
		for (int i = m_layersNumber - 2; i >= 0; i--) {
			m_layers[i].ComputeNeuronValuesGradient(m_layers[i + 1]);
		}

		//It adds the current gradients with the one before, it has to be avaraged at the end
		m_layers[0].ComputeWeightsBiasesGradient(*p_inputValues);
		for (int i = 1; i < m_layersNumber; i++) {
			m_layers[i].ComputeWeightsBiasesGradient(m_layers[i-1].GetOutput());
		}
	}
	void ApplyGradients(float learningRate = 0.1f, int trainingDoneNumber = 1) {
		for (int i = 0; i < m_layersNumber; i++) {
			m_layers[i].ApplyGradients(learningRate, trainingDoneNumber);
		}
	}
	float Learn(TrainingBatch& tb, float learningRate = 0.1f) {
		float rightGuesses = 0;
		int outputNeurons = GetOutNeuronsNumber();

		for (int miniBatchId = 0; miniBatchId < tb.m_trainingsNum; miniBatchId += tb.m_miniBatchesSize) {
			for (int specificId = 0; specificId < tb.m_miniBatchesSize; specificId++) {
				int trainingSampleId = miniBatchId + specificId;
				if (trainingSampleId >= tb.m_trainingsNum) { break; }
				SetInput(tb.m_inputs[trainingSampleId]);
				SetExpectedOutput(tb.m_expectedOutputs[trainingSampleId]);
				ComputeOutput__training(); //Some neurons will be deactivated
				ComputeCost();
				ComputeGradients();

				//Find out if the guess was right
				int maxId = 0;
				int expectedId = 0;
				for (int i = 0; i < outputNeurons; i++) {
					if (p_outputValues->p_array[i] > p_outputValues->p_array[maxId]) {
						maxId = i;
					}
					if (p_expectedOutput->p_array[i] > p_expectedOutput->p_array[expectedId]) {
						expectedId = i;
					}
				}
				if (maxId == expectedId) {
					rightGuesses++;
				}
			}
			ApplyGradients(learningRate, tb.m_miniBatchesSize);
		}
		return (rightGuesses / tb.m_trainingsNum);
	}
	float GetCost() {
		float cost = 0;
		for (int i = 0; i < GetOutNeuronsNumber(); i++) {
			cost += m_cost[i];
		}
		return cost / GetOutNeuronsNumber();
	}

	bool SaveNeuralNetwork(const char* path);
	bool ReadWeightsBiases(const char* path);

	//set/get functions
	int GetIndexOfHeighestOutput() {
		int maxId = 0;
		int maxIterations = GetOutNeuronsNumber();
		for (int i = 0; i < maxIterations; i++) {
			if (p_outputValues->p_array[i] > p_outputValues->p_array[maxId]) {
				maxId = i;
			}
		}
		return maxId;
	}
	void SetInput(MyArray<neuronValue_t>& _input) {
		p_inputValues = &_input;
		m_layers[0].SetInput(p_inputValues);
	}
	void SetExpectedOutput(MyArray<neuronValue_t>& _expectedOutput) {
		p_expectedOutput = &_expectedOutput;
	}
	void SetInputExpectedOutput(MyArray<neuronValue_t>& _input, MyArray<neuronValue_t>& _expectedOutput) {
		p_inputValues = &_input;
		p_expectedOutput = &_expectedOutput;
	}
	MyArray<neuronValue_t>& GetOutput() {
		return *p_outputValues;
	}
	Layer* GetLastLayer() {
		return &(m_layers[m_layersNumber - 1]);
	}
	int GetInNeuronsNumber() {
		return m_neuronNumbers[0];
	}
	int GetOutNeuronsNumber() {
		return m_neuronNumbers[m_neuronNumbers.m_length - 1];
	}
};


bool NeuralNetwork::SaveNeuralNetwork(const char* path) {
	//Open the stream
	std::ofstream fout;
	fout.open(path);
	if (!fout) {
		std::cout << "Errore nell'apertura del file: " << path << "\n"; return false;
	}

	//Specify the neuronsNumber
	fout << "nn:" << std::to_string(m_layersNumber) << "\n";
	fout << "layers:" << std::to_string(m_neuronNumbers[0]);
	for (int layerId = 1; layerId < m_layersNumber + 1; layerId++) {
		fout << "," << std::to_string(m_neuronNumbers[layerId]);
	}
	fout << "\n";
	fout << "af:" << std::to_string(m_hl_af) << ',' << std::to_string(m_ll_af) << '\n';

	//Cycle through every layer and write on the file the values
	for (int layerId = 0; layerId < m_layersNumber; layerId++) {
		Layer& layer = m_layers[layerId];

		fout << "w:";
		for (int weightOutId = 0; weightOutId < layer.m_outNeuronsNum; weightOutId++) {
			fout << "{" << layer.m_weights[weightOutId][0];
			for (int weightInId = 1; weightInId < layer.m_inNeuronsNum; weightInId++) {
				fout << "," << layer.m_weights[weightOutId][weightInId];
			}
			fout << "}";
		}
		fout << "\nb:" << layer.m_biases[0];
		for (int biasesOutId = 1; biasesOutId < layer.m_outNeuronsNum; biasesOutId++) {
			fout << "," << layer.m_biases[biasesOutId];
		}
		fout << "\n";
	}
	return true;
}
bool NeuralNetwork::ReadWeightsBiases(const char* path) {
	//Open the stream
	std::fstream fin; char charBuffer[FILE_CHAR_BUFFER_LENGHT];
	fin.open(path, std::ios::in | std::ios::binary);
	if (!fin) {
		std::cout << "Errore nell'apertura del file: " << path << "\n"; return false;
	}
	//Check if the neural network is the same
	//Check if it begins with nn:
	myFgets(fin, charBuffer, 3);
	if (charBuffer[0] != 'n' || charBuffer[1] != 'n' || charBuffer[2] != ':') {
		std::cout << "File type is not the right one: " << path << "\n"; return false;
	}
	//Check if the number of layers are the same
	int _layersNumber = ReadFloat(fin, charBuffer, '\n');
	if (m_layersNumber != _layersNumber) {
		std::cout << "Layers numbers don't match\n"; return false;
	}
	//Check if the number of neurons are the same
	myFgets(fin, charBuffer, 7);
	if (charBuffer[0] != 'l' || charBuffer[1] != 'a' || charBuffer[2] != 'y' || charBuffer[3] != 'e' || charBuffer[4] != 'r' || charBuffer[5] != 's' || charBuffer[6] != ':') {
		std::cout << "Error, layer's neurons number weren't found: " << path << "\n"; return false;
	}
	for (int layerId = 0; layerId < m_layersNumber + 1; layerId++) {
		int neuronsNumber = ReadFloat(fin, charBuffer, ',');
		if (neuronsNumber != m_neuronNumbers[layerId]) {
			std::cout << "NN neurons number " << layerId << " doesn't match\n"; return false;
		}
	}
	//Check if the activation functions (if specified) are the same
	myFgets(fin, charBuffer, 3);
	if (charBuffer[0] != 'a' || charBuffer[1] != 'f' || charBuffer[2] != ':') {
		std::cout << "Activation function aren't specified" << "\n"; return false;
	}
	//Check if the activation functions are the same
	ActivationFunctionType _hl_af = (ActivationFunctionType)(ReadFloat(fin, charBuffer, ','));
	ActivationFunctionType _ll_af = (ActivationFunctionType)(ReadFloat(fin, charBuffer, '\n'));
	if (m_hl_af != _hl_af || m_ll_af != _ll_af) {
		std::cout << "Activation functions in the file aren't the same" << "\n"; return false;
	}

	//Read the values
	for (int layerId = 0; layerId < m_layersNumber; layerId++) {
		Layer& layer = m_layers[layerId];

		//Read weights
		myFgets(fin, charBuffer, 2); //w:
		if (charBuffer[0] != 'w' || charBuffer[1] != ':') { return false; }
		for (int weightOutId = 0; weightOutId < layer.m_outNeuronsNum; weightOutId++) {
			myFgets(fin, charBuffer, 1); //{
			if (charBuffer[0] != '{') { return false; }
			for (int weightInId = 0; weightInId < layer.m_inNeuronsNum; weightInId++) {
				layer.m_weights[weightOutId][weightInId] = ReadFloat(fin, charBuffer, ',', '}');
			}
		}
		//read the biases
		myFgets(fin, charBuffer, 4); //\r\nb: (\r is a carriage return, it's put when fout << "\n")
		if (charBuffer[0] != '\r' || charBuffer[1] != '\n' || charBuffer[2] != 'b' || charBuffer[3] != ':') {return false; }
		for (int biasesOutId = 0; biasesOutId < layer.m_outNeuronsNum; biasesOutId++) {
			layer.m_biases[biasesOutId] = ReadFloat(fin, charBuffer, ',');
		}
	}
	std::cout << "Neural network weights and biases imported correctly\n";
	return true;
}
