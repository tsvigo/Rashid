#include "dialog.h"

#include <QApplication>
#include "neuralnetwork.h"


#include <fstream>
#include <sstream>
std::vector<std::vector<double>> loadCSV(const std::string& filename) {
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<double> row;

        while (std::getline(lineStream, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        data.push_back(row);
    }
    file.close();
    return data;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<double> loadImage(const QString& filepath) {
    QImage image(filepath);
    if (image.isNull()) {
        throw std::runtime_error("Could not open image file");
    }

    image = image.convertToFormat(QImage::Format_Grayscale8);
    std::vector<double> img_data;
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            int pixelValue = qGray(image.pixel(x, y));
            double normalizedPixel = (pixelValue / 255.0 * 0.99) + 0.01;
            img_data.push_back(normalizedPixel);
        }
    }
    return img_data;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    Dialog w;
    w.show();
  //   return a.exec();
    int inputNodes = 784;
    int hiddenNodes = 200;
    int outputNodes = 10;
    double learningRate = 0.1;

        NeuralNetwork nn(inputNodes, hiddenNodes, outputNodes, learningRate);

    try {
        std::string filename = "/home/viktor/makeyourownneuralnetwork/mnist_dataset/mnist_train.csv";
        std::vector<std::vector<double>> trainingData = loadCSV(filename);

        for (const auto& record : trainingData) {
            std::vector<double> inputs(record.begin() + 1, record.end());
            for (auto& input : inputs) {
                input = (input / 255.0 * 0.99) + 0.01;
            }

            std::vector<double> targets(outputNodes, 0.01);
            targets[static_cast<int>(record[0])] = 0.99;

            nn.train(inputs, targets);
        }

        nn.printWeights();
 //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        QString imagePath = "/home/viktor/makeyourownneuralnetwork/my_own_images/2828_my_own_5.png";
        std::vector<double> image_data = loadImage(imagePath);
        std::vector<double> outputs = nn.query(image_data);

        int recognizedDigit = std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
        std::cout << "Recognized digit: " << recognizedDigit << std::endl;

    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return -1;
    }



   return a.exec();
}
