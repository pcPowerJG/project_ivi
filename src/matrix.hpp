#include "tensor.hpp"
#include <vector>
#include <random>
#include <ostream>

class Matrix {
    int rows; // число строк
    int columns; // число столбцов
    std::vector<std::vector<double>> values; // значения

public:
    Matrix(int rows, int columns); // конструктор из заданных размеров

    double& operator()(int i, int j); // индексация
    double operator()(int i, int j) const; // индексация
};

// конструктор из заданных размеров
Matrix::Matrix(int rows, int columns) {
    this->rows = rows; // сохраняем число строк
    this->columns = columns; // сохраняем число столбцов

    values = std::vector<std::vector<double>>(rows, std::vector<double>(columns, 0)); // создаём векторы для значений матрицы
}

// индексация
double& Matrix::operator()(int i, int j) {
    return values[i][j];
}

// индексация
double Matrix::operator()(int i, int j) const {
    return values[i][j];
}

class FullyConnectedLayer {
    // тип активационной функции
    enum class ActivationType {
        None, // без активации
        Sigmoid, // сигмоидальная функция
        Tanh, // гиперболический тангенс
        ReLU, // выпрямитель
        LeakyReLU, // выпрямитель с утечкой
        ELU // экспоненциальный выпрямитель
    };

    TensorSize inputSize; // входой размер
    TensorSize outputSize; // выходной размер

    std::default_random_engine generator;
    std::normal_distribution<double> distribution;

    int inputs; // число входных нейронов
    int outputs; // число выходных нейронов
    
    ActivationType activationType; // тип активационной функции
    Tensor df; // тензор производных функции активации

    Matrix W; // матрица весовых коэффициентов
    Matrix dW; // матрица градиентов весовых коэффициентов

    std::vector<double> b; // смещения
    std::vector<double> db; // градиенты смещений

    ActivationType GetActivationType(const std::string& activationType) const; // получение типа активационной функции по строке

    void InitWeights(); // инициализация весовых коэффициентов
    void Activate(Tensor &output); // применение активационной функции

public:
    FullyConnectedLayer(TensorSize size, int outputs, const std::string& activationType = "none"); // создание слоя

    Tensor Forward(const Tensor &X); // прямое распространение
    Tensor Backward(const Tensor &dout, const Tensor &X); // обратное распространение

    void UpdateWeights(double learningRate); // обновление весовых коэффициентов
};

FullyConnectedLayer::FullyConnectedLayer(TensorSize size, int outputs, const std::string& activationType) : W(outputs, size.height * size.width * size.deep), dW(outputs, size.height * size.width * size.deep), df(1, 1, outputs), distribution(0.0, sqrt(2.0 / (size.height * size.width * size.deep))) {
    // запоминаем входной размер
    inputSize.width = size.width;
    inputSize.height = size.height;
    inputSize.depth = size.depth;

    // вычисляем выходной размер
    outputSize.width = 1;
    outputSize.height = 1;
    outputSize.depth = outputs;

    this->inputs = size.height * size.width * size.deep; // запоминаем число входных нейронов
    this->outputs = outputs; // запоминаем число выходных нейронов

    this->activationType = GetActivationType(activationType); // получаем активационную функцию

    b = std::vector<double>(outputs); // создаём вектор смещений
    db = std::vector<double>(outputs); // создаём вектор градиентов по весам смещения

    InitWeights(); // инициализируем весовые коэффициенты
}

// получение типа активационной функции по строке
FullyConnectedLayer::ActivationType FullyConnectedLayer::GetActivationType(const std::string& activationType) const {
    if (activationType == "sigmoid")
        return ActivationType::Sigmoid;

    if (activationType == "tanh")
        return ActivationType::Tanh;

    if (activationType == "relu")
        return ActivationType::ReLU;

    if (activationType == "leakyrelu")
        return ActivationType::LeakyReLU;

    if (activationType == "elu")
        return ActivationType::ELU;

    if (activationType == "none" || activationType == "")
        return ActivationType::None;

    throw std::runtime_error("Invalid activation function");
}

// инициализация весовых коэффициентов
void FullyConnectedLayer::InitWeights() {
    for (int i = 0; i < outputs; i++) {
        for (int j = 0; j < inputs; j++)
            W(i, j) = distribution(generator); // генерируем очередное случайное число

        b[i] = 0.01; // все смещения делаем равными 0.01
    }
}

// применение активационной функции с вычислением значений её производной
void FullyConnectedLayer::Activate(Tensor &output) {
    if (activationType == ActivationType::None) {
        for (int i = 0; i < outputs; i++) {
            df[i] = 1;
        }
    }
    else if (activationType == ActivationType::Sigmoid) {
        for (int i = 0; i < outputs; i++) {
            output[i] = 1 / (1 + exp(-output[i]));
            df[i] = output[i] * (1 - output[i]);
        }
    }
    else if (activationType == ActivationType::Tanh) {
        for (int i = 0; i < outputs; i++) {
            output[i] = tanh(output[i]);
            df[i] = 1 - output[i] * output[i];
        }
    }
    else if (activationType == ActivationType::ReLU) {
        for (int i = 0; i < outputs; i++) {
            if (output[i] > 0) {
                df[i] = 1;
            }
            else {
                output[i] = 0;
                df[i] = 0;
            }
        }
    }
    else if (activationType == ActivationType::LeakyReLU) {
        for (int i = 0; i < outputs; i++) {
            if (output[i] > 0) {
                df[i] = 1;
            }
            else {
                output[i] *= 0.01;
                df[i] = 0.01;
            }
        }
    }
    else if (activationType == ActivationType::ELU) {
        for (int i = 0; i < outputs; i++) {
            if (value > 0) {
                df[i] = 1;
            }
            else {
                output[i] = exp(output[i]) - 1;
                df[i] = output[i] + 1;
            }
        }
    }
}

// прямое распространение
void FullyConnectedLayer::Forward(const Tensor &X) {
    Tensor output(outputSize); // создаём выходной тензор

    // проходимся по каждому выходному нейрону
    for (int i = 0; i < outputs; i++) {
        double sum = b[i]; // прибавляем смещение

        // умножаем входной тензор на матрицу
        for (int j = 0; j < inputs; j++)
            sum += W(i, j) * X[j];

        output[i] = sum;
    }
    
    Activate(output); // применяем активационную функцию

    return output; // возвращаем выходной тензор
}

// обратное распространение
Tensor FullyConnectedLayer::Backward(const Tensor &dout, const Tensor &X) {
    // домножаем производные на градиенты следующего слоя для сокращения количества умножений
    for (int i = 0; i < outputs; i++)
        df[i] *= dout[i];

    // вычисляем градиенты по весовым коэффициентам
    for (int i = 0; i < outputs; i++) {
        for (int j = 0; j < inputs; j++)
            dW(i, j) = df[i] * X[j];

        db[i] = df[i];
    }

    Tensor dX(inputSize); // создаём тензор для градиентов по входам

    // вычисляем градиенты по входам
    for (int j = 0; j < inputs; j++) {
        double sum = 0;

        for (int i = 0; i < outputs; i++)
            sum += W(i, j) * df[i];

        dX[j] = sum; // записываем результат в тензор градиентов
    }

    return dX; // возвращаем тензор градиентов
}

// обновление весовых коэффициентов
void FullyConnectedLayer::UpdateWeights(double learningRate) {
    for (int i = 0; i < outputs; i++) {
        for (int j = 0; j < inputs; j++)
            W(i, j) -= learningRate * dW(i, j);

        b[i] -= learningRate * db[i]; // обновляем веса смещения
    }
}