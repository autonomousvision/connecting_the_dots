#pragma once

#include <fstream>

class SerializationOut {
public:
  SerializationOut(const std::string& path) : path_(path) {}
  virtual ~SerializationOut() {}

  virtual SerializationOut& operator<<(const bool& v) = 0;
  virtual SerializationOut& operator<<(const char& v) = 0;
  virtual SerializationOut& operator<<(const int& v) = 0;
  virtual SerializationOut& operator<<(const unsigned int& v) = 0;
  virtual SerializationOut& operator<<(const long int& v) = 0;
  virtual SerializationOut& operator<<(const unsigned long int& v) = 0;
  virtual SerializationOut& operator<<(const long long int& v) = 0;
  virtual SerializationOut& operator<<(const unsigned long long int& v) = 0;
  virtual SerializationOut& operator<<(const float& v) = 0;
  virtual SerializationOut& operator<<(const double& v) = 0;

protected:
  const std::string& path_;
};

class SerializationIn {
public:
  SerializationIn(const std::string& path) : path_(path) {}
  virtual ~SerializationIn() {}

  virtual SerializationIn& operator>>(bool& v) = 0;
  virtual SerializationIn& operator>>(char& v) = 0;
  virtual SerializationIn& operator>>(int& v) = 0;
  virtual SerializationIn& operator>>(unsigned int& v) = 0;
  virtual SerializationIn& operator>>(long int& v) = 0;
  virtual SerializationIn& operator>>(unsigned long int& v) = 0;
  virtual SerializationIn& operator>>(long long int& v) = 0;
  virtual SerializationIn& operator>>(unsigned long long int& v) = 0;
  virtual SerializationIn& operator>>(float& v) = 0;
  virtual SerializationIn& operator>>(double& v) = 0;

protected:
  const std::string& path_;
};

class TextSerializationOut : public SerializationOut {
public:
  TextSerializationOut(const std::string& path) : SerializationOut(path), 
    f_(path.c_str()) {}
  virtual ~TextSerializationOut() { 
    f_.close();
  }

  virtual SerializationOut& operator<<(const bool& v) {
    f_ << v << std::endl;
    return (*this);
  }
  virtual SerializationOut& operator<<(const char& v) {
    f_ << v << std::endl;
    return (*this);
  }
  virtual SerializationOut& operator<<(const int& v) {
    f_ << v << std::endl;
    return (*this);
  }
  virtual SerializationOut& operator<<(const unsigned int& v) {
    f_ << v << std::endl;
    return (*this);
  }
  virtual SerializationOut& operator<<(const long int& v) {
    f_ << v << std::endl;
    return (*this);
  }
  virtual SerializationOut& operator<<(const unsigned long int& v) {
    f_ << v << std::endl;
    return (*this);
  }
  virtual SerializationOut& operator<<(const long long int& v) {
    f_ << v << std::endl;
    return (*this);
  }
  virtual SerializationOut& operator<<(const unsigned long long int& v) {
    f_ << v << std::endl;
    return (*this);
  }
  virtual SerializationOut& operator<<(const float& v) {
    f_ << v << std::endl;
    return (*this);
  }
  virtual SerializationOut& operator<<(const double& v) {
    f_ << v << std::endl;
    return (*this);
  }

protected:
  std::ofstream f_;
};

class TextSerializationIn : public SerializationIn {
public:
  TextSerializationIn(const std::string& path) : SerializationIn(path), 
    f_(path.c_str()) {}
  virtual ~TextSerializationIn() {
    f_.close();
  }

  virtual SerializationIn& operator>>(bool& v) {
    f_ >> v;
    return (*this);
  }
  virtual SerializationIn& operator>>(char& v) {
    f_ >> v;
    return (*this);
  }
  virtual SerializationIn& operator>>(int& v) {
    f_ >> v;
    return (*this);
  }
  virtual SerializationIn& operator>>(unsigned int& v) {
    f_ >> v;
    return (*this);
  }
  virtual SerializationIn& operator>>(long int& v) {
    f_ >> v;
    return (*this);
  }
  virtual SerializationIn& operator>>(unsigned long int& v) {
    f_ >> v;
    return (*this);
  }
  virtual SerializationIn& operator>>(long long int& v) {
    f_ >> v;
    return (*this);
  }
  virtual SerializationIn& operator>>(unsigned long long int& v) {
    f_ >> v;
    return (*this);
  }
  virtual SerializationIn& operator>>(float& v) {
    f_ >> v;
    return (*this);
  }
  virtual SerializationIn& operator>>(double& v) {
    f_ >> v;
    return (*this);
  }

protected:
  std::ifstream f_;
};

class BinarySerializationOut : public SerializationOut {
public:
  BinarySerializationOut(const std::string& path) : SerializationOut(path), 
      f_(path.c_str(), std::ios::binary) {}
  virtual ~BinarySerializationOut() {
    f_.close();
  }

  virtual SerializationOut& operator<<(const bool& v) {
    f_.write((char*)&v, sizeof(bool));
    return (*this);
  }
  virtual SerializationOut& operator<<(const char& v) {
    f_.write((char*)&v, sizeof(char));
    return (*this);
  }
  virtual SerializationOut& operator<<(const int& v) {
    f_.write((char*)&v, sizeof(int));
    return (*this);
  }
  virtual SerializationOut& operator<<(const unsigned int& v) {
    f_.write((char*)&v, sizeof(unsigned int));
    return (*this);
  }
  virtual SerializationOut& operator<<(const long int& v) {
    f_.write((char*)&v, sizeof(long int));
    return (*this);
  }
  virtual SerializationOut& operator<<(const unsigned long int& v) {
    f_.write((char*)&v, sizeof(unsigned long int));
    return (*this);
  }
  virtual SerializationOut& operator<<(const long long int& v) {
    f_.write((char*)&v, sizeof(long long int));
    return (*this);
  }
  virtual SerializationOut& operator<<(const unsigned long long int& v) {
    f_.write((char*)&v, sizeof(unsigned long long int));
    return (*this);
  }
  virtual SerializationOut& operator<<(const float& v) {
    f_.write((char*)&v, sizeof(float));
    return (*this);
  }
  virtual SerializationOut& operator<<(const double& v) {
    f_.write((char*)&v, sizeof(double));
    return (*this);
  }

protected:
  std::ofstream f_;
};

class BinarySerializationIn : public SerializationIn {
public:
  BinarySerializationIn(const std::string& path) : SerializationIn(path), 
    f_(path.c_str(), std::ios::binary) {}
  virtual ~BinarySerializationIn() {
    f_.close();
  }

  virtual SerializationIn& operator>>(bool& v) {
    f_.read((char*)&v, sizeof(bool));
    return (*this);
  }
  virtual SerializationIn& operator>>(char& v) {
    f_.read((char*)&v, sizeof(char));
    return (*this);
  }
  virtual SerializationIn& operator>>(int& v) {
    f_.read((char*)&v, sizeof(int));
    return (*this);
  }
  virtual SerializationIn& operator>>(unsigned int& v) {
    f_.read((char*)&v, sizeof(unsigned int));
    return (*this);
  }
  virtual SerializationIn& operator>>(long int& v) {
    f_.read((char*)&v, sizeof(long int));
    return (*this);
  }
  virtual SerializationIn& operator>>(unsigned long int& v) {
    f_.read((char*)&v, sizeof(unsigned long int));
    return (*this);
  }
  virtual SerializationIn& operator>>(long long int& v) {
    f_.read((char*)&v, sizeof(long long int));
    return (*this);
  }
  virtual SerializationIn& operator>>(unsigned long long int& v) {
    f_.read((char*)&v, sizeof(unsigned long long int));
    return (*this);
  }
  virtual SerializationIn& operator>>(float& v) {
    f_.read((char*)&v, sizeof(float));
    return (*this);
  }
  virtual SerializationIn& operator>>(double& v) {
    f_.read((char*)&v, sizeof(double));
    return (*this);
  }

protected:
  std::ifstream f_;
};

