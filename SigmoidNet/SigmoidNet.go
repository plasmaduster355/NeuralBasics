package SigmoidNet

import (
	"fmt"
	"math"
	"reflect"

	"github.com/plasmaduster355/nnet"
)

//Sigmoid function
func Sigmoid(value float64) float64 {
	return (1.0 / (1.0 + (math.Exp(-1.0 * (value)))))
}

//Inverse of sigmoid
func SigmoidPrime(value float64) float64 {
	return math.Log(value / (1 - value))
}

//Find the Cost
func Cost(desired float64, output float64) float64 {
	return math.Exp2(output - desired)
}

//Neuron calculating
func Neuro(weights []float64, values []float64, bias float64) float64 {
	var g float64
	if len(weights) == len(values) {
		for x, y := range values {
			g = (y * weights[x]) + g
		}
	} else {
		panic("Not enough/too much input data given")
	}
	return Sigmoid(g + bias)
}

//Neuron calculating
func Neuron2(weights []float64, network nnet.Network, layer int, bias float64) float64 {
	var g float64
	for x, y := range network.Layer[layer-1].Neuron {
		g = (y.Output * weights[x]) + g
	}
	return Sigmoid(g + bias)
}
func Derivative(output float64) float64 {
	return output * (1.0 - output)
}
func ErrorBackpropagationOutput(expected float64, result float64) float64 {
	return (expected - result) * Derivative(result)
}
func ErrorBackpropagation(err float64, output float64) float64 {
	return err * Derivative(output)
}
func SquareError(expexted float64, output float64) float64 {
	return math.Exp2(expexted - output)
}

//Run the Network
func RunNetwork(network nnet.Network, inputs []float64) nnet.Network {
	//Iterate over Layers
	for x, layer := range network.Layer {
		//if it's the first layer use the input data
		if x == 0 {
			//loop through neurons in the first Layer
			for y, neuron := range layer.Neuron {
				//calculate neuron
				neuron.Output = Neuro(neuron.Weights, inputs, neuron.Bias)
				// set output value for backpropagation later
				network.Layer[x].Neuron[y].Output = neuron.Output

			}
		} else {
			//loop through the neurons in the other Layers
			for y, neuron := range layer.Neuron {
				//calculate neuron
				neuron.Output = Neuron2(neuron.Weights, network, x, neuron.Bias)
				// set output value for backpropagation later
				network.Layer[x].Neuron[y].Output = neuron.Output
			}
		}
	}
	//Print Out data
	for _, lastNeuron := range network.Layer[len(network.Layer)-1].Neuron {
		fmt.Print(lastNeuron.ResultTitle + ": ")
		fmt.Println(lastNeuron.Output)
	}
	return network
}

//Generate/creates a new Nureal Network

func BackProp(network nnet.Network, expected []float64) nnet.Network {
	//Itaterate over layers backwards
	for lC := len(network.Layer) - 1; lC >= 0; lC-- {
		//Itertae over neurons
		layer := network.Layer[lC]
		for nC, _ := range layer.Neuron {
			neuron := network.Layer[lC].Neuron[nC]
			//if on first layer
			if lC == len(network.Layer)-1 {
				network.Layer[lC].Neuron[nC].SelfError = ErrorBackpropagationOutput(expected[nC], neuron.Output)
			} else {
				pLayer := network.Layer[lC+1].Neuron
				var err float64
				for _, pL := range pLayer {
					err = err + (pL.Weights[nC] * pL.SelfError)
				}
				network.Layer[lC].Neuron[nC].SelfError = ErrorBackpropagation(err, neuron.Output)
			}
		}
	}
	fmt.Println(network.Layer[len(network.Layer)-1].Neuron[0].SelfError)
	return network
}
func UpdateWeights(network nnet.Network, learningRate float64, inputData []float64) nnet.Network {
	var weight float64
	for lC, layer := range network.Layer {
		if lC != 0 {
			for nC, neuron := range layer.Neuron {
				for iC, i := range network.Layer[lC-1].Neuron {
					weight = network.Layer[lC].Neuron[nC].Weights[iC] + (learningRate * neuron.SelfError * i.Output)
					network.Layer[lC].Neuron[nC].Weights[iC] = weight
				}
			}
		} else {
			for nC, neuron := range layer.Neuron {
				for iC, i := range inputData {
					weight = network.Layer[lC].Neuron[nC].Weights[iC] + (learningRate * neuron.SelfError * i)
					network.Layer[lC].Neuron[nC].Weights[iC] = weight
				}
			}
		}
	}
	fmt.Println(network.Layer[0].Neuron[0].SelfError)
	return network
}
func clear(v interface{}) {
	p := reflect.ValueOf(v).Elem()
	p.Set(reflect.Zero(p.Type()))
}
