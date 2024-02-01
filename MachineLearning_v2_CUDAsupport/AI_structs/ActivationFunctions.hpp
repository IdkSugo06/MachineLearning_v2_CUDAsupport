#pragma once
#include "MyArray.hpp"
#include "math.h"

typedef float neuronValue_t;
typedef float weight_t;
typedef float bias_t;


//ActivationFunctions
enum ActivationFunctionType {
	null,
	sigmoid,
	relu
};
//Sigmoid
float Sigmoid(float x) {
	return (1.0f / (1.0f + exp(-x)));
}
float SigmoidDerivative(float x) {
	float af = Sigmoid(x);
	return af * (1 - af);
}
//Relu
float ReLu(float x) {
	if (x > 0) { return x; }
	return 0;
}
float ReLuDerivative(float x) {
	if (x > 0) { return 1; }
	return 0;
}


float ActivationFunction(float x, ActivationFunctionType aft) {
	switch (aft) {
	case ActivationFunctionType::sigmoid: {
		return Sigmoid(x);
		break;
	}
	case ActivationFunctionType::relu: {
		return ReLu(x);
	}
	default: {
		return SigmoidDerivative(x);
		break;
	}
	}
}
float ActivationFunctionDerivative(float x, ActivationFunctionType aft) {
	switch (aft) {
	case ActivationFunctionType::sigmoid: {
		return SigmoidDerivative(x);
		break;
	}
	case ActivationFunctionType::relu: {
		return ReLuDerivative(x);
		break;
	}
	default: {
		return SigmoidDerivative(x);
		break;
	}
	}
}