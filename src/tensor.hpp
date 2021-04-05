// размерность тензора
struct TensorSize {
    int depth; // глубина
    int height; // высота
    int width; // ширина
};	

// тензор
class Tensor {
    TensorSize size; // размерность тензора
    std::vector<double> values; // значения тензора

    int dw; // произведение глубины на ширину для индексации

    void Init(int width, int height, int depth);

public:
    Tensor(int width, int height, int depth); // создание из размеров
    Tensor(const TensorSize &size); // создание из размера

    double& operator()(int d, int i, int j); // индексация
    double operator()(int d, int i, int j) const; // индексация

    TensorSize GetSize() const; // получение размера

    friend std::ostream& operator<<(std::ostream& os, const Tensor &tensor); // вывод тензора
};

// инициализация по размерам
void Tensor::Init(int width, int height, int depth) {
    size.width = width; // запоминаем ширину
    size.height = height; // запоминаем высоту
    size.depth = depth; // запоминаем глубину

    dw = depth * width; // запоминаем произведение глубины на ширину для индексации

    values = std::vector<double>(width * height * depth, 0); // создаём вектор из width * height * depth нулей
}

// создание из размеров
Tensor::Tensor(int width, int height, int depth) {
    Init(width, height, depth);  
}

// создание из размера
Tensor::Tensor(const TensorSize &size) {
    Init(size.width, size.height, size.depth);
}

// индексация
double& Tensor::operator()(int d, int i, int j) {
    return values[i * dw + j * size.depth + d];
}

// индексация
double Tensor::operator()(int d, int i, int j) const {
    return values[i * dw + j * size.depth + d];
}

// получение размера
TensorSize Tensor::GetSize() const {
    return size;
}

// вывод тензора
std::ostream& operator<<(std::ostream& os, const Tensor &tensor) {
    for (int d = 0; d < tensor.size.depth; d++) {
        for (int i = 0; i < tensor.size.height; i++) {
            for (int j = 0; j < tensor.size.width; j++)
                os << tensor.values[i * tensor.dw + j * tensor.size.depth + d] << " ";
            
            os << std::endl;
        }

        os << std::endl;
    }

    return os;
}

class ConvLayer {
    std::default_random_engine generator; // генератор случайных чисел
    std::normal_distribution<double> distribution; // с нормальным распределением

    TensorSize inputSize; // размер входа
    TensorSize outputSize; // размер выхода

    std::vector<Tensor> W; // фильтры
    std::vector<double> b; // смещения

    std::vector<Tensor> dW; // градиенты фильтров
    std::vector<double> db; // градиенты смещений
    
    int P; // дополнение нулями
    int S; // шаг свёртки

    int fc; // количество фильтров
    int fs; // размер фильтров
    int fd; // глубина фильтров

    void InitWeights(); // инициализация весовых коэффициентов

public:
    ConvLayer(TensorSize size, int fc, int fs, int P, int S); // создание слоя

    Tensor Forward(const Tensor &X); // прямое распространение сигнала
    Tensor Backward(const Tensor &dout, const Tensor &X); // обратное распространение ошибки
    void UpdateWeights(double learningRate); // обновление весовых коэффициентов
    void SetWeight(int index, int i, int j, int k, double weight); // set wght
    void SetBias(int index, double bias); //
};

// создание свёрточного слоя
ConvLayer::ConvLayer(TensorSize size, int fc, int fs, int P, int S) : distribution(0.0, sqrt(2.0 / (fs*fs*size.depth))) {
    // запоминаем входной размер
    inputSize.width = size.width;
    inputSize.height = size.height;
    inputSize.depth = size.depth;

    // вычисляем выходной размер
    outputSize.width = (size.width - fs + 2 * P) / S + 1;
    outputSize.height = (size.height - fs + 2 * P) / S + 1;
    outputSize.depth = fc;

    this->P = P; // сохраняем дополнение нулями
    this->S = S; // сохраняем шаг свёртки

    this->fc = fc; // сохраняем число фильтров
    this->fs = fs; // сохраняем размер фильтров
    this->fd = size.depth; // сохраняем глубину фильтров

    // добавляем fc тензоров для весов фильтров и их градиентов
    W = std::vector<Tensor>(fc, Tensor(fs, fs, fd));
    dW = std::vector<Tensor>(fc, Tensor(fs, fs, fd));
        
    // добавляем fc нулей для весов смещения и их градиентов
    b = std::vector<double>(fc, 0);
    db = std::vector<double>(fc, 0);

    InitWeights(); // инициализируем весовые коэффициенты
}

// инициализация весовых коэффициентов
void ConvLayer::InitWeights() {
    // проходимся по каждому из фильтров
    for (int index = 0; index < fc; index++) {
        for (int i = 0; i < fs; i++)
            for (int j = 0; j < fs; j++)
                for (int k = 0; k < fd; k++)
                    W[index](k, i, j) = distribution(generator); // генерируем случайное число и записываем его в элемент фильтра

        b[index] = 0.01; // все смещения устанавливаем в 0.01
    }
}

// прямое распространение
Tensor ConvLayer::Forward(const Tensor &X) {
    Tensor output(outputSize); // создаём выходной тензор

    // проходимся по каждому из фильтров
    for (int f = 0; f < fc; f++) {
        for (int y = 0; y < outputSize.height; y++) {
            for (int x = 0; x < outputSize.width; x++) {
                double sum = b[f]; // сразу прибавляем смещение

                // проходимся по фильтрам
                for (int i = 0; i < fs; i++) {
                    for (int j = 0; j < fs; j++) {
                        int i0 = S * y + i - P;
                        int j0 = S * x + j - P;

                        // поскольку вне границ входного тензора элементы нулевые, то просто игнорируем их
                        if (i0 < 0 || i0 >= inputSize.height || j0 < 0 || j0 >= inputSize.width)
                            continue;

                        // проходимся по всей глубине тензора и считаем сумму
                        for (int c = 0; c < fd; c++)
                            sum += X(c, i0, j0) * W[f](c, i, j);
                    }
                }

                output(f, y, x) = sum; // записываем результат свёртки в выходной тензор
            }
        }
    }

    return output; // возвращаем выходной тензор
}

// обратное распространение
Tensor ConvLayer::Backward(const Tensor &dout, const Tensor &X) {
    TensorSize size; // размер дельт

    // расчитываем размер для дельт
    size.height = S * (outputSize.height - 1) + 1;
    size.width = S * (outputSize.width - 1) + 1;
    size.depth = outputSize.depth;

    Tensor deltas(size); // создаём тензор для дельт

    // расчитываем значения дельт
    for (int d = 0; d < size.depth; d++)
        for (int i = 0; i < outputSize.height; i++)
            for (int j = 0; j < outputSize.width; j++)
                deltas(d, i * S, j * S) = dout(d, i, j);

    // расчитываем градиенты весов фильтров и смещений
    for (int f = 0; f < fc; f++) {
        for (int y = 0; y < size.height; y++) {
            for (int x = 0; x < size.width; x++) {
                double delta = deltas(f, y, x); // запоминаем значение градиента

                for (int i = 0; i < fs; i++) {
                    for (int j = 0; j < fs; j++) {
                        int i0 = i + y - P;
                        int j0 = j + x - P;
                        
                        // игнорируем выходящие за границы элементы
                        if (i0 < 0 || i0 >= inputSize.height || j0 < 0 || j0 >= inputSize.width)
                            continue;

                        // наращиваем градиент фильтра
                        for (int c = 0; c < fd; c++)
                            dW[f](c, i, j) += delta * X(c, i0, j0);
                    }
                }

                db[f] += delta; // наращиваем градиент смещения
            }
        }
    }

    int pad = fs - 1 - P; // заменяем величину дополнения
    Tensor dX(inputSize); // создаём тензор градиентов по входу

    // расчитываем значения градиента
    for (int y = 0; y < inputSize.height; y++) {
        for (int x = 0; x < inputSize.width; x++) {
            for (int c = 0; c < fd; c++) {
                double sum = 0; // сумма для градиента

                // идём по всем весовым коэффициентам фильтров
                for (int i = 0; i < fs; i++) {
                    for (int j = 0; j < fs; j++) {
                        int i0 = y + i - pad;
                        int j0 = x + j - pad;

                        // игнорируем выходящие за границы элементы
                        if (i0 < 0 || i0 >= size.height || j0 < 0 || j0 >= size.width)
                            continue;

                        // суммируем по всем фильтрам
                        for (int f = 0; f < fc; f++)
                            sum += W[f](c, fs - 1 - i, fs - 1 - j) * deltas(f, i0, j0); // добавляем произведение повёрнутых фильтров на дельты
                    }
                }

                dX(c, y, x) = sum; // записываем результат в тензор градиента
            }
        }
    }

    return dX; // возвращаем тензор градиентов
}

// обновление весовых коэффициентов
void ConvLayer::UpdateWeights(double learningRate) {
    for (int index = 0; index < fc; index++) {
        for (int i = 0; i < fs; i++) {
            for (int j = 0; j < fs; j++) {
                for (int d = 0; d < fd; d++) {
                    W[index](d, i, j) -= learningRate * dW[index](d, i, j); // вычитаем градиент, умноженный на скорость обучения
                    dW[index](d, i, j) = 0; // обнуляем градиент фильтра
                }
            }
        }

        b[index] -= learningRate * db[index]; // вычитаем градиент, умноженный на скорость обучения
        db[index] = 0; // обнуляем градиент веса смещения
    }
}

// установка веса фильтра по индексу
void ConvLayer::SetWeight(int index, int i, int j, int k, double weight) {
    W[index](i, j, k) = weight;
}

// установка веса смещения по индексу
void ConvLayer::SetBias(int index, double bias) {
    b[index] = bias;
}

//--------------------------------------------------

class MaxPoolingLayer {
    TensorSize inputSize; // размер входа
    TensorSize outputSize; // размер выхода
    
    int scale; // во сколько раз уменьшается размерность
    Tensor mask; // маска для максимумов

public:
    MaxPoolingLayer(TensorSize size, int scale = 2); // создание слоя

    Tensor Forward(const Tensor &X); // прямое распространение
    Tensor Backward(const Tensor &dout, const Tensor &X); // обратное распространение
};

// создание слоя
MaxPoolingLayer::MaxPoolingLayer(TensorSize size, int scale) : mask(size) {
    // запоминаем входной размер
    inputSize.width = size.width;
    inputSize.height = size.height;
    inputSize.depth = size.depth;

    // вычисляем выходной размер
    outputSize.width = size.width / scale;
    outputSize.height = size.height / scale;
    outputSize.depth = size.depth;

    this->scale = scale; // запоминаем коэффициент уменьшения
}

// прямое распространение с использованием маски
Tensor MaxPoolingLayer::Forward(const Tensor &X) {
    Tensor output(outputSize); // создаём выходной тензор

    // проходимся по каждому из каналов
    for (int d = 0; d < inputSize.depth; d++) {
        for (int i = 0; i < inputSize.height; i += scale) {
            for (int j = 0; j < inputSize.width; j += scale) {
                int imax = i; // индекс строки максимума
                int jmax = j; // индекс столбца максимума
                double max = X(d, i, j); // начальное значение максимума - значение первой клетки подматрицы

                // проходимся по подматрице и ищем максимум и его координаты
                for (int y = i; y < i + scale; y++) {
                    for (int x = j; x < j + scale; x++) {
                        double value = X(d, y, x); // получаем значение входного тензора
                        mask(d, y, x) = 0; // обнуляем маску

                        // если входное значение больше максимального
                        if (value > max) {
                            max = value; // обновляем максимум
                            imax = y; // обновляем индекс строки максимума
                            jmax = x; // обновляем индекс столбца максимума
                        }
                    }
                }

                output(d, i / scale, j / scale) = max; // записываем в выходной тензор найденный максимум
                mask(d, imax, jmax) = 1; // записываем 1 в маску в месте расположения максимального элемента
            }
        }
    }

    return output; // возвращаем выходной тензор
}

// обратное распространение
Tensor MaxPoolingLayer::Backward(const Tensor &dout, const Tensor &X) {
    Tensor dX(inputSize); // создаём тензор для градиентов

    for (int d = 0; d < inputSize.depth; d++)
        for (int i = 0; i < inputSize.height; i++)
            for (int j = 0; j < inputSize.width; j++)
                dX(d, i, j) = dout(d, i / scale, j / scale) * mask(d, i, j); // умножаем градиенты на маску

    return dX; // возвращаем посчитанные градиенты
}
//------------------------------
class ReLULayer {
    TensorSize size; // размер слоя

public:
    ReLULayer(TensorSize size); // создание слоя

    Tensor Forward(const Tensor &X); // прямое распространение
    Tensor Backward(const Tensor &dout, const Tensor &X); // обратное распространение
};

// создание слоя
ReLULayer::ReLULayer(TensorSize size) {
    this->size = size; // сохраняем размер
}

// прямое распространение
Tensor ReLULayer::Forward(const Tensor &X) {
    Tensor output(size); // создаём выходной тензор

    // проходимся по всем значениям входного тензора
    for (int i = 0; i < size.height; i++)
        for (int j = 0; j < size.width; j++)
            for (int k = 0; k < size.depth; k++)
                output(k, i, j) = X(k, i, j) > 0 ? X(k, i, j) : 0; // вычисляем значение функции активации

    return output; // возвращаем выходной тензор
}

Tensor ReLULayer::Backward(const Tensor &dout, const Tensor &X) {
    Tensor dX(size); // создаём тензор градиентов

    // проходимся по всем значениям тензора градиентов
    for (int i = 0; i < size.height; i++)
        for (int j = 0; j < size.width; j++)
            for (int k = 0; k < size.depth; k++)
                dX(k, i, j) = dout(k, i, j) * (X(k, i, j) > 0 ? 1 : 0); // умножаем градиенты следующего слоя на производную функции активации

    return dX; // возвращаем тензор градиентов
}

//------------------------------