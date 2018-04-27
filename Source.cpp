/*
Date: 10/5/2017
NeuralNetwork for learning how torque works
*/

#include <iostream>
#include <time.h>
#include <thread>
#include <chrono>
#include <math.h>
#include <algorithm>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

int absolute(double x) {
	double y = 0;
	if (x >= 0) {
		y = x;
	}
	else {
		y = -x;
	}
	if (y < 0.2) {
		return 0;
	}
	else {
		return 1;
	}
}

class NeuralNetwork {
public:
	NeuralNetwork(std::vector<unsigned>& topology) {
		// Initializing weights vector
		for (unsigned i = 0; i != topology.size() - 1; i++) {
			weights.push_back(std::vector<std::vector<double>>(topology[i + 1], std::vector<double>(topology[i])));
		}

		// Initializing Random weights
		for (unsigned i = 0; i != weights.size(); i++) {
			for (unsigned j = 0; j != weights[i].size(); j++) {
				for (unsigned k = 0; k != weights[i][j].size(); k++) {
					weights[i][j][k] = double(std::rand() % 10000 + 1) / 2500;
					if (std::rand() % 2 == 0)
						weights[i][j][k] *= -1;
						//std::cout << weights[i][j][k] << std::endl;
				}
			}
		}

		// Initializing a vector
		for (unsigned i = 0; i != topology.size(); i++) {
			a.push_back(std::vector<double>(topology[i]));
		}

		// Initializing z vector
		z.push_back(std::vector<double>(0)); // the first layer we won't need
		for (unsigned i = 1; i != topology.size(); i++) {
			z.push_back(std::vector<double>(topology[i]));
		}

		// Initializing bias_weights
		for (unsigned i = 1; i != topology.size(); i++) {
			bias_weights.push_back(std::vector<double>(topology[i]));
		}
		// Randomizing bias_weights
		for (unsigned i = 0; i != bias_weights.size(); i++) {
			for (unsigned j = 0; j != bias_weights[i].size(); j++) {
				bias_weights[i][j] = double(std::rand() % 10000 + 1) / 2500;
				if (std::rand() % 2 == 0)
					bias_weights[i][j] *= -1;
			}
		}

		// Intializing error
		for (unsigned i = 1; i != topology.size(); i++) {
			error.push_back(std::vector<double>(topology[i]));
		}

		// Initializing y vector
		y.resize(topology.back());

		// Initializing learning_rate
		learning_rate = 0.001;

		// Setting bias to 1
		bias = 1;
	}

	void Train(std::vector<std::vector<std::vector<double>>>& train_data) {
		// Loop through train_data examples
		for (unsigned i = 0; i != train_data.size(); i++) {
			// Setting xi to a[0]i for the input layer
			for (unsigned j = 0; j != a[0].size(); j++) {
				a[0][j] = train_data[i][0][j];
				//std::cout << train_data[i][0][j] << std::endl;
			}

			// Setting y vector
			for (unsigned j = 0; j != a.back().size(); j++) {
				y[j] = train_data[i][1][j];
				//std::cout << train_data[i][1][j] << std::endl;
			}

			// FeedForward and Activation(Activate)
			for (unsigned j = 0; j != weights.size(); j++) {
				FeedForward(j);
				Activation(j + 1);
			}

			// Find the Cost
			J();

			// Update the weights
			BackPropagation();
		}
	}

	void Test(std::vector<std::vector<std::vector<double>>>& test_data) {
		// Loop Through the test_data examples
		for (unsigned i = 0; i != test_data.size(); i++) {
			// Setting xi to a[0]i for the input layer
			for (unsigned j = 0; j != a[0].size(); j++) {
				a[0][j] = test_data[i][0][j];
			}

			// Setting y vector
			for (unsigned j = 0; j != a.back().size(); j++) {
				y[j] = test_data[i][1][j];
			}

			// FeedForward and Activation(Activate)
			for (unsigned j = 0; j != weights.size(); j++) {
				FeedForward(j);
				Activation(j + 1);
			}
			/*double guess = 0;
			if (a.back()[0] >= 0.94) {
				guess = 1;
			}
			else if (a.back()[0] <= -0.94) {
				guess = -1;
			}
			else {
				guess = a.back()[0];
			}*/
			//std::cout << "------------" << std::endl;
			std::cout << "Network's answer: " << a.back()[0] << " || " << test_data[i][1][0] << " Real Answer.\tDifference: " << absolute(test_data[i][1][0] - a.back()[0]) << std::endl;
		}
	}


private:

	// Cost Function
	void J() {
		for (unsigned i = 0; i != error.back().size(); i++) {
			error.back()[i] = (y[i] - a.back()[i]) * (1 - std::pow(a.back()[i], 2));
			//std::cout << (y[i] - a.back()[i]) << std::endl;
		}
	}

	// BackPropagation Algorithm
	void BackPropagation() {
		// Compute the rest of the error vector
		for (unsigned i = error.size() - 2; i != -1; i--) {
			for (unsigned j = 0; j != error[i].size(); j++) {
				double sum = 0;
				for (unsigned k = 0; k != error[i + 1].size(); k++) {
					sum += error[i + 1][k] * weights[i + 1][k][j]; // probably wrong
				}
				error[i][j] = sum;
			}
		}

		// Update the weights
		for (unsigned i = 0; i != weights.size(); i++) {
			for (unsigned j = 0; j != weights[i].size(); j++) {
				for (unsigned k = 0; k != weights[i][j].size(); k++) {
					weights[i][j][k] = weights[i][j][k] + learning_rate * a[i][k] * error[i][j];
				}
				bias_weights[i][j] = bias_weights[i][j] + learning_rate * error[i][j];
			}
		}
	}

	// FeedForward simple enough
	void FeedForward(unsigned const& l) {
		std::vector<double> temp_z;
		for (unsigned i = 0; i != weights[l].size(); i++) {
			double sum = 0;
			for (unsigned j = 0; j != weights[l][i].size(); j++) {
				sum += weights[l][i][j] * a[l][j];
			}
			sum += bias_weights[l][i];
			temp_z.push_back(sum);
		}
		z[l + 1] = temp_z;
		a[l + 1] = z[l + 1];
	}

	void Activation(unsigned const& l) {
		// Sigmoid Function
		/*for (unsigned i = 0; i != a[l].size(); i++) {
			a[l][i] = 1 / (1 + std::exp(-z[l][i]));
		}*/
		/*for (unsigned i = 0; i != a[l].size(); i++) {
			a[l][i] = (std::exp(z[l][i]) - std::exp(-z[l][i])) / (std::exp(z[l][i]) + std::exp(-z[l][i]));
			std::cout << a[l][i] << " z: " << z[l][i] << std::endl;
		}*/

		// Tanh
		for (unsigned i = 0; i != a[l].size(); i++) {
			a[l][i] = std::tanh(z[l][i]);
			//std::cout << a[l][i] << " z: " << z[l][i] << std::endl;
		}
	}

	std::vector<std::vector<std::vector<double>>> weights;
	std::vector<std::vector<double>> bias_weights;
	// Activated output of the neurons
	std::vector<std::vector<double>> a;
	// Unactivated output of the neurons
	std::vector<std::vector<double>> z;
	std::vector<std::vector<double>> error;
	std::vector<double> y;
	double learning_rate;
	double bias;
};

void timeLeft(unsigned *i, unsigned *iterations) {
	while (true) {
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		std::cout << "\r%" << double(*i) / double(*iterations) * 100.0;
	}
}


int main() {
	std::srand(time(0));
	//////////////////
	std::ifstream file("dataset.txt");
	std::vector<std::vector<std::vector<double>>> data;
	if (file.is_open()) {
		//std::cout << "File is open." << std::endl;
		std::string line;
		while (std::getline(file, line)) {
			std::stringstream linestream(line);
			std::string element;
			std::vector<std::vector<double>> temp;
			std::vector<double> temp_o;
			std::vector<double> temp_i;
			while (std::getline(linestream, element, ',')) {
				if (element == "B") {
					goto here;
					temp_o.push_back(1);
				}
				else if (element == "R") {
					temp_o.push_back(1);
				}
				else if (element == "L") {
					temp_o.push_back(-1);
				}
				else {
					temp_i.push_back(std::stod(element));
				}
			}
			temp.push_back(temp_i);
			temp.push_back(temp_o);
			//std::cout << temp_i.size() << " " << temp_o.size() << std::endl;
			data.push_back(temp);
			here:;
		}

	}
	else {
		std::cout << "Error opening the file." << std::endl;
	}
	//////////////////////


	//std::vector<std::vector<std::vector<double>>> train_data = { { { 5, 3 }, { -1 } }, { { 12, 5 }, { -1 } }, { { 3, 1 }, { 1 } }, { { 4.5, 2 }, { 1 } }, { { 10, 3 }, { 1 } }, { { 6, 2 }, { 1 } }, { { 3, 2 }, { -1 } }, { { 2.2, 1.1 }, { 1 } }, { { 40, 6 }, { 1 } }, { { 13, 4 }, { -1 } }, { { 17, 4.1 }, { 1 } }, { { 9.5, 3 }, { 1 } }, { { 8, 3.1 }, { -1 } }, { { 25.5, 5 }, { 1 } }, { { 23, 5 }, { -1 } }, { { 50, 7 }, { 1 } }, { { 45, 7 }, { -1 } }, { { 46, 7.1 }, { -1 } }, { { 8.8, 3 }, { -1 } } };
	//std::vector<std::vector<std::vector<double>>> test_data = { { { 4, 1 }, { 1 } }, { { 5.5, 3 }, { -1 } }, { { 8, 7 }, { -1 } }, { { 30, 5.3 }, { 1 } } };
	std::vector<std::vector<std::vector<double>>> train_data;
	std::vector<std::vector<std::vector<double>>> test_data;

	std::cout << data.size() << std::endl;
	for (unsigned i = 0; i != data.size(); i++) {
		int rand = std::rand() % 100;
		if (rand > 95) {
			test_data.push_back(data[i]);
		}
		else {
			train_data.push_back(data[i]);
		}
	}

	std::random_shuffle(train_data.begin(), train_data.end());


	std::vector<unsigned> topology = { train_data[0][0].size(), 50, train_data[0][1].size() };
	NeuralNetwork nn(topology);

	unsigned iterations = 1000;
	unsigned i = 0;
	std::thread t1(timeLeft, &i, &iterations);
	t1.detach();

	while (i != iterations) {
		nn.Train(train_data);
		i++;
	}
	std::cout << "Train-DATA" << std::endl;
	nn.Test(train_data);
	std::cout << "Test-DATA" << std::endl;
	nn.Test(test_data);
}
