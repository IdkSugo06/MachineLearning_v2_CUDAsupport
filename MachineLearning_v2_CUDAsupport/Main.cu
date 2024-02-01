#include "AI_structs\NeuralNetwork.hpp"
#include "Singletons\DrawingsReader.hpp"
#include <thread>

#define inputNeurons (28*28)
#define outputNeurons (10)
#define layers (3)
#define autoSaveRate (10)
#define MNIST_trainingCSVPath  "Files\\mnist_train_modifiedImgs.csv"
#define nnFile  "Files\\nn_digitRecognition.myNeuralNetwork"
#define nnFile_autoSaving  "Files\\nn_digitRecognition__AUTOSAVE.myNeuralNetwork"
#define trainingEnabled (false) //Set to true if you want to train

enum TrainingStatus {
	notInitialized,				//No info
	isTraining,					//No stop requested
	trainingStopRequested,		//It's executing the last training session before stopping
	trainingStoppedAccorded,	//Only one iteration, it has just stopped training
	trainingStopped				//Training already stopped
};

TrainingStatus trainingStatus = TrainingStatus::notInitialized;
int trainingToDo = 5000;
void LearningThread(NeuralNetwork& nn, TrainingBatch& tb);
void main(){
	MyWindow window;
	Drawer drawer(28, 28, 40, &window);

	//Creating the neural network and reading saved values
	NeuralNetwork nn(nnFile);

	//Setting up the training variables and objects
	std::thread learningThread;
	if (trainingEnabled) {
		//Creating the training batch
		TrainingBatch tb(inputNeurons, outputNeurons, 100, 10);
		tb.ReadCSV__MNISTtraining(MNIST_trainingCSVPath, 28, 28);
		//Launch the learning thread
		learningThread = std::thread(LearningThread, nn, tb);
	}
	else {
		trainingStatus = TrainingStatus::trainingStopped;
	}

	//Instructions
	std::cout << "Hi user! This should recognise your handwritten digits!\n";
	std::cout << "\tWhen you want to stop the training (if it's enabled), press 'P'\n";
	std::cout << "\tTo draw press 'Space' and move the mouse around, note: do not move around the windows (im lazy i dont want to fix that bug :D)\n";
	std::cout << "\tTo erase press 'E'\n";
	std::cout << "\tUse 'D' and 'F' to switch from saved image view (what the neural network sees) and the normal image view\n";
	std::cout << "\tUse 'S' to save the image (saved image view will only show the last saved image) and 'C' to compute the output\n";
	std::cout << "\tIf you feel like the pencil thikness is too low or too high, change it with 'Shift' and 'Ctrl'\n";
	std::cout << "Few tips:\n";
	std::cout << "=)Try to stay at the center of the image\n";
	std::cout << "=)Try to write disting digits, and not too small, the neural network has fewer resolution than what you see\n";
	std::cout << "=)If you fell like no digit is getting recognized, sry no much I can do ;P\n";
	std::cout << "=)It struggles to recognize 9s, 6s and 8s\n";

	//Drawings
	DrawingsReader drawingsReader(nn);
	bool running = true;
	int timeToNextComputation = 0;
	while (running) {
		if (!window.ProcessMessages()) {
			std::cout << "Closing window\n";
			running = false; break;
		}

		//Action to update every cycle
		{
			//Save progress
			if (trainingStatus == TrainingStatus::trainingStoppedAccorded) { 
				nn.SaveNeuralNetwork(nnFile);
				trainingStatus = TrainingStatus::trainingStopped;
			}
			//Update drawing windows
			else if (trainingStatus == TrainingStatus::trainingStopped) {
				drawer.Update(1);
			}
		}

		//Action on user's will
		if (timeToNextComputation <= 0) {

			//Stop training
			if (trainingStatus == TrainingStatus::isTraining) {
				if (InputHandler::inputHandler.isPressed(DC_P)) {
					trainingStatus = TrainingStatus::trainingStopRequested;
					std::cout << "Requested training stop\n";
					timeToNextComputation = 500;
				}
			}
			//Check if 'guess the img' is been requested
			else if (trainingStatus == TrainingStatus::trainingStopped) {
				//Check the image
				if (InputHandler::inputHandler.isPressed(DC_C)) {
					//drawingsReader.ExtractImageFromInput(&(tb.m_inputs[i]), &(drawer.GetSavedImage()));
					drawingsReader.GuessImage(&(drawer.GetSavedImage()));
					timeToNextComputation = 500;
				}
			}

		} else {
			timeToNextComputation--;
		}
	}
	
	if (trainingEnabled) { 
		learningThread.join(); 
	}
}


void LearningThread(NeuralNetwork& nn, TrainingBatch& tb) {
	if (!trainingEnabled) { 
		trainingStatus = TrainingStatus::trainingStopped;
		return; 
	}
	trainingStatus = TrainingStatus::isTraining;
	//Train
	while (trainingToDo > 0 && trainingStatus != TrainingStatus::trainingStopRequested) {
		std::cout << "\tNew training session\n";
		float accuracy = nn.Learn(tb, 0.7f);
		std::cout << "\tAverage accuracy: " << accuracy * 100 << "%\n";
		trainingToDo--;
		if (trainingToDo % autoSaveRate == 0) {
			nn.SaveNeuralNetwork(nnFile_autoSaving);
			std::cout << "\tProgression saved (autosaved)\n";
		}
	}
	trainingStatus = TrainingStatus::trainingStoppedAccorded;
	std::cout << "Training stopped\n";
}