#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <QImage>
class NeuralNetwork {
public:
    NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, double learningRate);

    void train(const std::vector<double>& inputsList, const std::vector<double>& targetsList);
    std::vector<double> query(const std::vector<double>& inputsList);
    void printWeights();

private:
    int inodes;
    int hnodes;
    int onodes;
    double lr;

    std::vector<std::vector<double>> wih;
    std::vector<std::vector<double>> who;

    double sigmoid(double x);
    std::vector<double> sigmoid(const std::vector<double>& vec);
    std::vector<double> dot(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vec);
    std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& matrix);
};

#endif // NEURALNETWORK_H
