package nnet

import (
	"fmt"
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
					//If not on layer 1
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
				newWeights = append(newWeights, 0)
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

//Generate random number
func Random() float64 {
	rand.Seed(time.Now().UnixNano())
	return rand.Float64()
}

func clear(v interface{}) {
	p := reflect.ValueOf(v).Elem()
	p.Set(reflect.Zero(p.Type()))
}
