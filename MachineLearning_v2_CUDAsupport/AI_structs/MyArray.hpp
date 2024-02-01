#pragma once 
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define MAX_THREAD_GPU 1024

template<typename elementType>
struct MyArray {

public:
	int m_length = 0;
	elementType* p_array{nullptr};

public:
	//Constructor / Deconstructors ----------------------------------------------------- Constructor / Deconstructors
	MyArray(int initialLength = 0) { 
		m_length = initialLength;
		if (initialLength != 0) {
			p_array = new elementType[initialLength];
		}
	}
	MyArray(int initialLength, const elementType& initialValue) {
		m_length = initialLength;
		if (initialLength != 0) {
			p_array = new elementType[initialLength];
		}
		for (int i = 0; i < initialLength; i++) {
			p_array[i] = initialValue;
		}
	}
	~MyArray() {
		if (!p_array) { delete[] p_array; }
	}


	//Set functions ----------------------------------------------------- Set functions
	void SetValues(elementType* newValues) {
		for (int i = 0; i < m_length; i++) {
			p_array[i] = newValues[i];
		}
	}
	//bool SetPointer(elementType* _pointer, int _length) {
	//	if (!_pointer) { return false; }
	//	p_array = _pointer; m_length = _length;
	//	return true;
	//}
	//bool SetPointer(elementType* _pointer) { //Length remain the same
	//	if (!_pointer) { return false; }
	//	p_array = _pointer;
	//	return true;
	//}
	bool Resize(const int newlength, elementType initialValue = elementType()) {
		if (m_length >= newlength) { return false; }
		if (!p_array) { p_array = new elementType[newlength]; }//Initial pointer yet to be allocated

		//Allocate the new memory and copy it
		int oldLength = m_length;
		elementType* newPointer = new elementType[newlength];
		for (int i = 0; i < oldLength; i++) {
			newPointer[i] = p_array[i];
		}
		m_length = newlength;
		p_array = newPointer;

		//Initialize the new memory
		for (int i = oldLength; i < newlength; i++) {
			p_array[i] = initialValue;
		}
		return true;
	}
	bool Resize(const int newlength, elementType& initialValue) {
		int oldLength = m_length;
		if (Resize(newlength) == false) { return false; }
		for (int i = m_length; i < newlength; i++) {
			p_array[i] = initialValue;
		}
		return true;
	}


	//Get functions ----------------------------------------------------- Get functions
	static int Length(const MyArray<elementType>& array) {
		return array.m_length;
	}
	int GetLength() const{
		return m_length;
	}
	elementType* GetPointer() {
		return p_array;
	}
	

	//Operator overloads ----------------------------------------------------- Operator overloads
	elementType& operator[] (int elementId) const{
		if (!p_array) { 
			static elementType errorElement;
			return errorElement;
		}
		return p_array[elementId];
	}
	void operator= (const MyArray<elementType>& secondArray){
		//Realloc memory and set it to 0 (IT CALLS THE CONSTRUCTORS)
		delete[] p_array;
		p_array = new elementType[secondArray.m_length];
		m_length = secondArray.m_length;

		//Copy all the data from the second array to the first one
		for (int i = 0; i < m_length; i++) {
			p_array[i] = secondArray[i];
		}
	}
	friend std::ostream& operator<< (std::ostream& os, const MyArray<elementType>& array) {
		int _length = array.GetLength();
		if (_length == 0) { return os; }

		os << "{";
		for (int i = 0; i < _length - 1; i++) {
			os << array[i] << ", ";
		}
		os << array[_length - 1] << "}";
		return os;
	}
};