#include "neuralnetwork.h"

NeuralNetwork::NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, double learningRate)
    : inodes(inputNodes), hnodes(hiddenNodes), onodes(outputNodes), lr(learningRate) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0.0, std::pow(inodes, -0.5));

    wih.resize(hnodes, std::vector<double>(inodes));
    for(auto& row : wih)
        for(auto& val : row)
            val = d(gen);

    who.resize(onodes, std::vector<double>(hnodes));
    for(auto& row : who)
        for(auto& val : row)
            val = d(gen);
}

double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

std::vector<double> NeuralNetwork::sigmoid(const std::vector<double>& vec) {
    std::vector<double> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i)
        result[i] = sigmoid(vec[i]);
    return result;
}

std::vector<double> NeuralNetwork::dot(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vec) {
    std::vector<double> result(matrix.size());
    for (size_t i = 0; i < matrix.size(); ++i)
        for (size_t j = 0; j < matrix[0].size(); ++j)
            result[i] += matrix[i][j] * vec[j];
    return result;
}

std::vector<std::vector<double>> NeuralNetwork::transpose(const std::vector<std::vector<double>>& matrix) {
    std::vector<std::vector<double>> result(matrix[0].size(), std::vector<double>(matrix.size()));
    for (size_t i = 0; i < matrix.size(); ++i)
        for (size_t j = 0; j < matrix[0].size(); ++j)
            result[j][i] = matrix[i][j];
    return result;
}

void NeuralNetwork::train(const std::vector<double>& inputsList, const std::vector<double>& targetsList) {
    std::vector<double> inputs = inputsList;
    std::vector<double> targets = targetsList;

    std::vector<double> hidden_inputs = dot(wih, inputs);
    std::vector<double> hidden_outputs = sigmoid(hidden_inputs);

    std::vector<double> final_inputs = dot(who, hidden_outputs);
    std::vector<double> final_outputs = sigmoid(final_inputs);

    std::vector<double> output_errors(targets.size());
    for (size_t i = 0; i < targets.size(); ++i)
        output_errors[i] = targets[i] - final_outputs[i];

    std::vector<std::vector<double>> who_transposed = transpose(who);
    std::vector<double> hidden_errors = dot(who_transposed, output_errors);

    for (size_t i = 0; i < who.size(); ++i)
        for (size_t j = 0; j < who[0].size(); ++j)
            who[i][j] += lr * output_errors[i] * final_outputs[i] * (1.0 - final_outputs[i]) * hidden_outputs[j];

    for (size_t i = 0; i < wih.size(); ++i)
        for (size_t j = 0; j < wih[0].size(); ++j)
            wih[i][j] += lr * hidden_errors[i] * hidden_outputs[i] * (1.0 - hidden_outputs[i]) * inputs[j];
}

std::vector<double> NeuralNetwork::query(const std::vector<double>& inputsList) {
    std::vector<double> inputs = inputsList;
    std::vector<double> hidden_inputs = dot(wih, inputs);
    std::vector<double> hidden_outputs = sigmoid(hidden_inputs);

    std::vector<double> final_inputs = dot(who, hidden_outputs);
    std::vector<double> final_outputs = sigmoid(final_inputs);

    return final_outputs;
}

void NeuralNetwork::printWeights() {
    std::cout << "Weights between input and hidden layers (wih):" << std::endl;
    for(const auto& row : wih) {
        for(const auto& val : row)
            std::cout << val << " ";
        std::cout << std::endl;
    }
}
