#pragma once
#include "ActivationFunctions.hpp"


#define INITIAL_BIASES_VALUE (0.0005f)
#define INITIAL_WEIGHTS_VALUE (0.0005f)


__host__ __device__ struct Layer {
public:
	int m_inNeuronsNum, m_outNeuronsNum;

	MyArray<MyArray<weight_t>> m_weights; //[out_id][in_id]
	MyArray<bias_t> m_biases;
	
	MyArray<neuronValue_t>* p_inputValues{nullptr};
	MyArray<neuronValue_t> m_neuronValues;
	MyArray<neuronValue_t> m_activatedNeuronValues; //Output

	MyArray<MyArray<neuronValue_t>> m_gradientWeights; //dC / dw
	MyArray<neuronValue_t> m_gradientBiases; //dC / db 
	MyArray<neuronValue_t> m_gradientNeuronValues; //dC / dz

private:
	ActivationFunctionType m_afType = ActivationFunctionType::null;

public:
	//constructor / deconstructor
	Layer(int _inNeurons = 0, int _outNeurons = 0, MyArray<neuronValue_t>* _inputValues = nullptr, ActivationFunctionType _afType = ActivationFunctionType::sigmoid) {
		m_inNeuronsNum = _inNeurons;
		m_outNeuronsNum = _outNeurons;
		m_afType = _afType;

		//Initialize weights and biases
		m_biases = MyArray<bias_t>(_outNeurons, INITIAL_BIASES_VALUE);
		m_weights = MyArray<MyArray<weight_t>>(_outNeurons, MyArray<weight_t>(_inNeurons, INITIAL_WEIGHTS_VALUE)); //organized that way it has to cycle through all the input to have an output
		InitializeRandomWeights();
		
		//Initialize input and outputs
		p_inputValues = _inputValues;
		m_neuronValues = MyArray<neuronValue_t>(_outNeurons, 0);
		m_activatedNeuronValues = MyArray<neuronValue_t>(_outNeurons, 0);

		//Initialize gradiants
		m_gradientWeights = MyArray<MyArray<weight_t>>(_outNeurons, MyArray<weight_t>(_inNeurons, 0));
		m_gradientBiases = MyArray<neuronValue_t>(_outNeurons, 0);
		m_gradientNeuronValues = MyArray<neuronValue_t>(_outNeurons, 0);
	}
	~Layer() { }

	//Methods
	void ComputeOutput() {
		for (int outId = 0; outId < m_outNeuronsNum; outId++) {
			neuronValue_t _out = 0;

			//Cycle through all the input and multiply them with the right weight
			for (int inId = 0; inId < m_inNeuronsNum; inId++) {
				_out += p_inputValues->p_array[inId] * m_weights[outId][inId];
			}
			_out += m_biases[outId];
			m_neuronValues[outId] = _out; //z_j = (∑_i^(N) w_ij * a_i ) + b_j
			m_activatedNeuronValues[outId] = ActivationFunction(_out, m_afType); //a_j = sig(z_j)
		}
	}
	// TO - DO ; has to be more efficent and generalized
	void ComputeOutput__training(){
		float discartedId1 = (((float)rand()) / RAND_MAX) * m_outNeuronsNum;
		float discartedId2 = (((float)rand()) / RAND_MAX) * m_outNeuronsNum;
		float discartedId3 = (((float)rand()) / RAND_MAX) * m_outNeuronsNum;
		float discartedId4 = (((float)rand()) / RAND_MAX) * m_outNeuronsNum;
		for (int outId = 0; outId < m_outNeuronsNum; outId++) {
			if (outId == discartedId1 || outId == discartedId2 || outId == discartedId3 || outId == discartedId4) {
				continue;
			}
			neuronValue_t _out = 0;
			
			//Cycle through all the input and multiply them with the right weight
			for (int inId = 0; inId < m_inNeuronsNum; inId++) {
				_out += p_inputValues->p_array[inId] * m_weights[outId][inId];
			}
			_out += m_biases[outId];
			m_neuronValues[outId] = _out; //z_j = (∑_i^(N) w_ij * a_i ) + b_j
			m_activatedNeuronValues[outId] = ActivationFunction(_out, m_afType); //a_j = sig(z_j)
		}
	}
	void ComputeNeuronValuesGradient__lastLayer(const MyArray<neuronValue_t>& _expectedOutput) {
		for(int outId = 0; outId < m_outNeuronsNum; outId++){
			//Compute the gradientNeuronValues (gNV) of the last layer
			m_gradientNeuronValues[outId] = 2 * (m_activatedNeuronValues[outId] - _expectedOutput[outId]); //2 * (a - y)
			m_gradientNeuronValues[outId] *= ActivationFunctionDerivative(m_neuronValues[outId], m_afType); //gNV = sigDer(z) * 2 * (a - y)

			//Cycle for all the inputNeurons and compute the gradiantWeights
			for (int inId = 0; inId < m_inNeuronsNum; inId++) {
				m_gradientWeights[outId][inId] += p_inputValues->p_array[inId] * m_gradientNeuronValues[outId]; //w_ij^L = a_i^(L-1) * gNV_j^(L)
			}
			m_gradientBiases[outId] += m_gradientNeuronValues[outId]; //b_j = 1 * gNV_j
		}
	}
	void ComputeNeuronValuesGradient(Layer& nextLayer) {
		for (int outId = 0; outId < m_outNeuronsNum; outId++) {
			neuronValue_t activationDerivative = ActivationFunctionDerivative(m_neuronValues[outId], m_afType);
			
			//Cycle for all the inputNeurons and compute the gradiantWeights (∑_j^N w_ij^(L+1) * gNV_j^(L+1))
			neuronValue_t gradientNeuronValue = 0;
			for (int outNextLayerId = 0; outNextLayerId < nextLayer.m_outNeuronsNum; outNextLayerId++) {
				gradientNeuronValue += (nextLayer.GetWeights())[outNextLayerId][outId] * (nextLayer.GetGradientNeuronValues())[outNextLayerId];
			}

			//gNV_i^L = sigDer(z_i^L) * (∑_j^N w_ij^(L+1) * gNV_j^(L+1))
			m_gradientNeuronValues[outId] = gradientNeuronValue * activationDerivative;
		}
	}
	void ComputeWeightsBiasesGradient(const MyArray<neuronValue_t>& previousOutput) {
		for (int outId = 0; outId < m_outNeuronsNum; outId++) {
			neuronValue_t currentGradientNeuronValue = m_gradientNeuronValues[outId]; //gNV_j^L

			//dC / db_j^L = gNV_j^L
			m_gradientBiases[outId] += currentGradientNeuronValue; 

			//dC / dw_ij^L = a_i^(L-1) * gNV_j^L
			for (int inId = 0; inId < m_inNeuronsNum; inId++) {
				m_gradientWeights[outId][inId] += previousOutput[inId] * currentGradientNeuronValue;
			}
		}
	}
	void ApplyGradients(float learningRate = 0.1f, int trainingsDoneNumber = 1) { //It resets the gradientWeights/gradientBiases 
		float learningCoefficent = learningRate / trainingsDoneNumber;
		for (int outId = 0; outId < m_outNeuronsNum; outId++) {
			m_biases[outId] -= m_gradientBiases[outId] * learningCoefficent;
			m_gradientBiases[outId] = 0;
			for (int inId = 0; inId < m_inNeuronsNum; inId++) {
				m_weights[outId][inId] -= m_gradientWeights[outId][inId] * learningCoefficent;
				m_gradientWeights[outId][inId] = 0;
			}
		}
	}

	//set / get functions
	void InitializeRandomWeights() {
		for (int j = 0; j < m_outNeuronsNum; j++) {
			for (int i = 0; i < m_inNeuronsNum; i++) {
				m_weights[j][i] = (((float)rand()) / RAND_MAX) * INITIAL_WEIGHTS_VALUE;
			}
		}
	}
	void SetNumOfNeurons(int _inNeurons, int _outNeurons) { //Input should be already resized
		m_inNeuronsNum = _inNeurons;
		m_outNeuronsNum = _outNeurons;

		//Resize weights and biases
		m_biases.Resize(_outNeurons, INITIAL_BIASES_VALUE);
		m_weights.Resize(_outNeurons, MyArray<neuronValue_t>(_inNeurons, INITIAL_WEIGHTS_VALUE));
		InitializeRandomWeights();

		//Resize outputs
		m_neuronValues.Resize(_outNeurons, 0);
		m_activatedNeuronValues.Resize(_outNeurons, 0);

		//Resize gradiants
		m_gradientWeights.Resize(_outNeurons, MyArray<neuronValue_t>(_inNeurons, 0));
		m_gradientBiases.Resize(_outNeurons, 0);
		m_gradientNeuronValues.Resize(_outNeurons, 0);
	}
	void SetActivationFunctionType(ActivationFunctionType _afType) {
		m_afType = _afType;
	}
	bool SetInput(MyArray<neuronValue_t>* _inputValues) {
		if (_inputValues == nullptr) { return false; }
		p_inputValues = _inputValues; 
		return true;
	}
	MyArray<neuronValue_t>& GetOutput() {
		return m_activatedNeuronValues;
	}
	MyArray<neuronValue_t>& GetGradientNeuronValues(){
		return m_gradientNeuronValues;
	}
	MyArray<MyArray<neuronValue_t>>& GetWeights() {
		return m_weights;
	}
};
