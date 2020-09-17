package nnetbasics

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"time"
)

type Network struct {
	Layer     []Layer `json:"layer"`
	NumInputs int     `json:"NumInputs"`
}
type Layer struct {
	Neuron []Neuron `json:"neuron"`
}
type Neuron struct {
	Weights     []float64 `json:"weights"`
	Output      float64   `json:"output"`
	Bias        float64   `json:"bias"`
	Error       []float64 `json:"error"`
	IsOutput    bool      `json:"isOutput"`
	ResultTitle string    `json:"resultTitle"`
	SelfError   float64   `json:"selfError"`
}

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

//Generate random number
func Random() float64 {
	rand.Seed(time.Now().UnixNano())
	return rand.Float64()
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
func Neuron2(weights []float64, network Network, layer int, bias float64) float64 {
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
func RunNetwork(network Network, inputs []float64) Network {
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
func GenerateNetwork(layerCount int, neuron_Array []int, amountOfInputData int, amountOfOutputData int, outputNames []string) (network Network, err bool) {
	if layerCount == len(neuron_Array) {
		err = false
		var x int
		var g int = 0
		var newWeights []float64
		var neuro Neuron
		var lay Layer
		var zeroOut []float64
		//Itaterate over each layer
		for x <= layerCount-1 {
			//Itaterate over neurons to make
			for g <= neuron_Array[x] {
				var h int = 0
				//Test if on layer 1
				if x != 0 {
					//I not on layer 1
					//Make random weights
					for neuron_Array[x-1] >= h {
						newWeights = append(newWeights, Random())
						zeroOut = append(zeroOut, 0.0)
						h++
					}
					zeroOut = append(zeroOut, 0.0)
					neuro.Error = zeroOut
					//Reset loop and set Weight
					h = 0
					neuro.Weights = newWeights
					neuro.Bias = Random()
					lay.Neuron = append(lay.Neuron, neuro)
					newWeights = []float64{}
					zeroOut = []float64{}
				} else {
					//If on layer 1
					//Make random weights for inputs
					for amountOfInputData != h {
						newWeights = append(newWeights, Random())
						zeroOut = append(zeroOut, 0.0)
						h++
					}
					//Reset loop and set Weight in network
					h = 0
					var neuro Neuron
					neuro.Weights = newWeights
					neuro.Bias = Random()
					neuro.Error = zeroOut
					lay.Neuron = append(lay.Neuron, neuro)
					newWeights = []float64{}
				}

				g++
			}
			network.Layer = append(network.Layer, lay)
			network.NumInputs = amountOfInputData
			clear(&lay)
			g = 0
			x++
		}
		newWeights = []float64{}
		x = 0
		var nero Neuron
		//Make output neurons
		for x <= amountOfOutputData-1 {
			nero.IsOutput = true
			zeroOut = []float64{}
			//Randomize new weights
			for g <= len(network.Layer[len(network.Layer)-1].Neuron)-1 {
				newWeights = append(newWeights, Random())
				zeroOut = append(zeroOut, 0.0)
				g++
			}
			nero.Weights = newWeights
			nero.Bias = Random()
			neuro.Error = zeroOut
			nero.ResultTitle = outputNames[x]
			lay.Neuron = append(lay.Neuron, nero)
			x++
		}
		network.Layer = append(network.Layer, lay)
		newWeights = []float64{}
		fmt.Println(newWeights)
	} else {
		err = true
	}
	return network, err
}
func BackProp(network Network, expected []float64) Network {
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
func UpdateWeights(network Network, learningRate float64, inputData []float64) Network {
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
