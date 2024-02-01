#pragma once 
#include "Layer.hpp"
#include "../Singletons\BasicFileFunctions.hpp"

struct TrainingBatch {
		
private:
	int m_inNeuronsNum = 0, m_outNeuronsNum = 0;
	
public:
	int m_trainingsNum = 0, m_miniBatchesSize = 100;
	float m_testPercentage = 0;
	MyArray<MyArray<neuronValue_t>> m_inputs;
	MyArray<MyArray<neuronValue_t>> m_expectedOutputs;

public:
	TrainingBatch(int _inNeuronsNum = 0, int _outNeuronsNum = 0, int _miniBatchesSize = 100, float m_testPercentage = 0) {
		m_trainingsNum = 0;
		m_miniBatchesSize = _miniBatchesSize;
		m_inNeuronsNum = _inNeuronsNum; 
		m_outNeuronsNum = _outNeuronsNum;
		
		//Resize the inputs and set them up to be long _inNeuronsNum
		m_inputs.Resize(m_trainingsNum, MyArray<neuronValue_t>(m_inNeuronsNum, 0));

		//Resize the expectedOutputs and set them up to be long _outNeuronsNum
		m_expectedOutputs.Resize(m_trainingsNum, MyArray<neuronValue_t>(m_outNeuronsNum, 0));
	}

	//Set methods
	void Resize(int newLength) { //New inputs and expectedOutputs already resized
		m_trainingsNum = newLength;
		//It initialize at 0 ONLY the new arrays
		m_inputs.Resize(m_trainingsNum, MyArray<neuronValue_t>(m_inNeuronsNum, 0));
		m_expectedOutputs.Resize(m_trainingsNum, MyArray<neuronValue_t>(m_outNeuronsNum, 0));
	}
	void ChangeInOutNeuronsNumber(int _inNeuronsNum, int _outNeuronsNum) { //New inputs and expectedOutputs already resized
		m_inNeuronsNum = _inNeuronsNum;
		m_outNeuronsNum = _outNeuronsNum;
		m_inputs.Resize(m_trainingsNum, MyArray<neuronValue_t>(m_inNeuronsNum, 0));
		m_expectedOutputs.Resize(m_trainingsNum, MyArray<neuronValue_t>(m_outNeuronsNum, 0));
	}
	//Get methods
	MyArray<neuronValue_t>& GetLastInput() {
		return m_inputs[m_trainingsNum - 1];
	}
	MyArray<neuronValue_t>& GetLastExpectedOutput() {
		return m_expectedOutputs[m_trainingsNum - 1];
	}

	//Void Methods
	bool ReadCSV__MNISTtraining(const char* csv_path, int dimWidth, int dimHeight);
	bool ReadImage(const char* bmp_path, int expectedDigit);
};

bool TrainingBatch::ReadCSV__MNISTtraining(const char* csv_path, int dimWidth, int dimHeight) {
	if (dimWidth * dimHeight != m_inNeuronsNum) {
		std::cout << "img too big\n"; return false;
	}
	int initialTrainingBatchLength = m_trainingsNum;
	std::cout << "Reading training batch (MNIST_CSV)..." << "\n";


	// -- // FIRST READ - count how big the training batch is // -- // 

	std::fstream csv_fin; char charBuffer[FILE_CHAR_BUFFER_LENGHT];
	csv_fin.open(csv_path, std::ios::in | std::ios::binary);
	if (!csv_fin) {
		std::cout << "Errore nell'apertura del file: " << csv_path << "\n"; return false;
	}

	//The first line is for labels
	myFendline(csv_fin);
	//Count the training examples
	int newTrainingBatchNumber = 0;
	myFgets(csv_fin, charBuffer, 1);
	while (charBuffer[0] != '\n') {
		myFendline(csv_fin); charBuffer[0] = '\n';
		myFgets(csv_fin, charBuffer, 1);
		newTrainingBatchNumber++;
	}
	csv_fin.close();

	// -- // FIRST READ FINISHED // -- // 


	//Resize the trainingBatch
	Resize(m_trainingsNum + newTrainingBatchNumber);


	// -- // SECOND READ - data read // -- // 

	//Reopen the file and read the data
	int currentTrainingId = initialTrainingBatchLength;
	csv_fin.open(csv_path, std::ios::in | std::ios::binary);

	//The first line is for labels
	myFendline(csv_fin);
	myFgets(csv_fin, charBuffer, 1); //read expected digit
	while (charBuffer[0] != '\n') { //If there's no new line, reading expectedDigit and comma wouldnt change the content of the charBuffer, so it would remain '\n'
		int expectedDigit = (int)(charBuffer[0] - '0'); //translate the expected digit
		MyArray<neuronValue_t>& currentInputArray = m_inputs[currentTrainingId];
		m_expectedOutputs[currentTrainingId][expectedDigit] = 1; //The expOut is initialized at 0, outputs should be all 0 except the outputs[expectedDigit]

		//Start Reading the training examples
		myFgets(csv_fin, charBuffer, 1); //comma
		int i = 0;
		while (charBuffer[0] != '\n') {
			float num = ReadFloat(csv_fin, charBuffer, ',') / 255; //In the charBuffer should be the comma or the '\n' after the function
			currentInputArray[i] = num;
			i++;
		}
		currentTrainingId++; 
		
		//New line, new trainingBatch
		myFgets(csv_fin, charBuffer, 2); //expected digit and comma
	}
	std::cout << "Read training batch ended, found " << newTrainingBatchNumber << " trainings\n";

	// -- // SECOND READ FINISHED // -- // 
	return true;
}

//TF is this code
//bool TrainingBatch::ReadImage(const char* bmp_path, int expectedDigit) { //It sets the last input layer to be the image, it has to be ALREADY RESIZED
//
//	//Open the file stream
//	std::fstream bmp_fin;
//	bmp_fin.open(bmp_path, std::ios::in | std::ios::binary);
//	char charBuffer[FILE_CHAR_BUFFER_LENGHT];
//
//	//Check if the stream is opened	
//	if (!bmp_fin) {
//		std::cout << "Errore nella lettura del file mtl specificato (\"" << bmp_path << "\"), attenderà 5 sec" << std::endl; return false;
//	}
//
//	//Resize input and expectedOutput //TO DELETE
//	Resize(m_trainingsNum + 1);
//
//
//	//Set the [expectedDigit] = 1, all the other ones will be 0s
//	GetLastExpectedOutput()[expectedDigit] = 1;
//
//
//	// -- // READ BMP FILE // -- //
//	//HEADERS
//	myFgets(bmp_fin, charBuffer, 2);
//	if (charBuffer[0] != 66 || charBuffer[1] != 77) { std::cout << "no bmp file signature\n"; return false; }
//
//	//uint32 fileSize;
//	myFgets(bmp_fin, charBuffer, 4);
//	//fileSize = (charBuffer[0] | (charBuffer[1] << 8) | (charBuffer[2] << 16) | (charBuffer[3] << 24));
//
//	myFgets(bmp_fin, charBuffer, 4); // reserved bytes (4)
//	myFgets(bmp_fin, charBuffer, 4); // pixelArray offset (4)
//
//	myFgets(bmp_fin, charBuffer, 4);
//	int headerDim = (charBuffer[0] | (charBuffer[1] << 8) | (charBuffer[2] << 16) | (charBuffer[3] << 24));
//
//	//Read width and height and compute padding
//	myFgets(bmp_fin, charBuffer, 4);
//	int imgWidth = (charBuffer[0] | (charBuffer[1] << 8) | (charBuffer[2] << 16) | (charBuffer[3] << 24));
//	myFgets(bmp_fin, charBuffer, 4);
//	int imgHeight = (charBuffer[0] | (charBuffer[1] << 8) | (charBuffer[2] << 16) | (charBuffer[3] << 24));
//	int padding = (4 - ((imgWidth * 3) % 4)) % 4;
//
//	//skip the header (-12 cause of the chars just read)
//	myFgets(bmp_fin, charBuffer, headerDim - 12);
//
//	//PIXEL ARRAY
//	MyArray<neuronValue_t>& newInput = GetLastInput();
//	for (int y = 0; y < imgHeight; y++) {		//In bytes
//		for (int x = 0; x < imgWidth; x++) {	//In bytes
//			myFgets(bmp_fin, charBuffer, 3);
//			newInput[y * imgWidth + x + 0] = ((float)(charBuffer[0])) / 255;
//			newInput[y * imgWidth + x + 1] = ((float)(charBuffer[1])) / 255;
//			newInput[y * imgWidth + x + 2] = ((float)(charBuffer[2])) / 255;
//		}
//		myFgets(bmp_fin, charBuffer, padding);
//	}
//
//	return true;
//}