#pragma once
#include <iostream>
#include "fstream"

#define FILE_CHAR_BUFFER_LENGHT 100
#define FILE_FILENAME_BUFFER_LENGHT 150
#define FILE_FILEDIRECTORY_BUFFER_LENGHT 500 //has to be less than 0xFFFF
#define FILE_STR_TO_FLOAT_BUFFER_LENGHT 50 //has to be less than 0xFF

template<typename T>
void myFgets(std::fstream& fin, T* p_char, unsigned short dim) {
	fin.read((char*)p_char, dim);
}
void myFendline(std::fstream& fin) {
	char c = 0;
	while (c != '\n')  myFgets(fin, &c, 1);
}

template<typename T>
float ReadFloat(std::fstream& fin, T* charBuffer, char separator1 = ' ', char separator2 = ' ') {//The next char read has to be the first digit, the next char to read is the one after the ' ' or '\t' or '\n (charBuffer[0] = '\n')
	char str_number[FILE_STR_TO_FLOAT_BUFFER_LENGHT];

	int i = 0;
	myFgets(fin, (char*)charBuffer, 1);
	while (charBuffer[0] != '\n' && charBuffer[0] != ' ' && charBuffer[0] != '\t' && charBuffer[0] != '\0' && charBuffer[0] != separator1 && charBuffer[0] != separator2) {
		str_number[i] = charBuffer[0];
		myFgets(fin, charBuffer, 1);
		i++;
	}
	str_number[i] = '\0';
	return atof(str_number);
}