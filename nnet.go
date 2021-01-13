package nnet

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"time"
)

type Network struct {
	NetworkName    string   `json:"network_name"`
	OutputFunction string   `json:"output_function"`
	NumOutputs     int      `json:"num_outputs"`
	OutputNames    []string `json:"output_names"`
	Layer          []Layer  `json:"Layer"`
}
type Layer struct {
	Type     string   `json:"type"`
	IsOutput bool     `json:"is_output"`
	Neuron   []Neuron `json:"Neuron"`
}
type Neuron struct {
	Weights []float64 `json:"weights"`
	Error   float64   `json:"error"`
	Delta   float64   `json:"delta"`
	Output  float64   `json:"output"`
	Bias    float64   `json:"bias"`
}

func main() {

}
func GeneticRun(network Network, survival_rate float64, creature_count int, small_mutation_rate float64, iteration int, input_data [][]float64, expexted_data [][]float64, verbose bool) Network {

	// Creature map
	creatures := make(map[string]float64)
	//ordered creture map
	ordered_cretures := make(map[int][]string)
	// counters
	//----------------
	//iteration counters
	var i int
	//creature counters
	var c int
	//envirement counter
	var e int
	//emvirement 2 counter
	var e2 int
	//check if temp directory exist and create on if not

	if _, err := os.Stat("/temp/1.json"); os.IsNotExist(err) {
		os.MkdirAll("/temp/", 0700)
	}
	//Initate genetic algorithom
	//make inital cretures
	for c <= creature_count {
		tempnet := smallRandomize(network, small_mutation_rate)
		b, _ := json.Marshal(tempnet)
		ioutil.WriteFile("temp/"+strconv.Itoa(c)+"net.json", b, 0644)
		c++
	}
	//reset c
	c = 0
	//run through iterations
	for i <= iteration {
		//run through envirement
		for e <= len(input_data)-1 {
			for c <= creature_count {
				var tempnet Network
				b, _ := ioutil.ReadFile("temp/" + strconv.Itoa(c) + "net.json")
				json.Unmarshal(b, &tempnet)
				tempnet, _ = Run(tempnet, input_data[e])
				_, err := CalcError(tempnet, expexted_data[e])
				creatures[strconv.Itoa(c)] = err
				c++
			}
			//reset c
			c = 0
			//orginize result of errors smallest to largest
			type kv struct {
				Key   string
				Value float64
			}

			var ss []kv
			for k, v := range creatures {
				ss = append(ss, kv{k, v})
			}

			sort.Slice(ss, func(i, j int) bool {
				return ss[i].Value < ss[j].Value
			})
			for c <= len(ss)-1 {
				ordered_cretures[c] = []string{ss[c].Key, fmt.Sprintf("%f", ss[c].Value)}
				c++
			}
			c = 0
			if verbose {
				fmt.Println(ordered_cretures[0][1])
			}
			//Create new creatures
			for c <= creature_count {
				if float64(c) > float64(creature_count)*survival_rate {
					var tempnet Network
					toReplace := ordered_cretures[c]
					parent := ordered_cretures[e2]

					b, _ := ioutil.ReadFile("temp/" + parent[0] + "net.json")
					json.Unmarshal(b, &tempnet)

					tempnet = smallRandomize(tempnet, small_mutation_rate)
					b, _ = json.Marshal(tempnet)
					ioutil.WriteFile("temp/"+toReplace[0]+"net.json", b, 0644)
				}
				e2++
				if float64(e2) > survival_rate*100.0 {
					e2 = 0
				}
				c++
			}
			e2 = 0
			c = 0
			e++
		}

		e = 0
		c = 0
		i++
	}
	var tempnet Network
	b, _ := ioutil.ReadFile("temp/" + ordered_cretures[0][0] + "net.json")
	json.Unmarshal(b, &tempnet)
	return tempnet
}
func smallRandomize(network Network, percentage float64) Network {
	for ly, y := range network.Layer {
		for lx, x := range y.Neuron {
			for lz, _ := range x.Weights {
				isTrue := chance(percentage)
				if isTrue {
					rt := float64(rand.Intn(2 - -2)-2) + rand.Float64()
					network.Layer[ly].Neuron[lx].Weights[lz] = network.Layer[ly].Neuron[lx].Weights[lz] + rt
				}

			}
		}
	}
	return network
}
func largeRandomize() {

}
func chance(percentage float64) bool {
	isTrue := false
	percentage = percentage * 100
	r := float64(rand.Intn(100-0)-0) + rand.Float64()
	if r >= 100-percentage {
		isTrue = true
	}
	return isTrue
}
func CalcError(network Network, expected []float64) (Network, float64) {
	var score float64
	//Itaterate over layers backwards
	for lC := (len(network.Layer) - 1); lC >= 0; lC-- {
		//Itertae over neurons
		layer := network.Layer[lC]
		var nC int
		for nC < len(layer.Neuron)-1 {
			neuron := network.Layer[lC].Neuron[nC]
			//if on last layer
			if lC == (len(network.Layer) - 1) {
				if network.Layer[lC].Type == "relu" {
					network.Layer[lC].Neuron[nC].Delta = network.Layer[lC].Neuron[nC].Delta + reluErrorBackpropagationOutput(expected[nC], neuron.Output)
					score = score + math.Abs(network.Layer[lC].Neuron[nC].Delta)
				} else if network.Layer[lC].Type == "sigmoid" {
					network.Layer[lC].Neuron[nC].Delta = network.Layer[lC].Neuron[nC].Delta + sigmoidErrorBackpropagationOutput(expected[nC], neuron.Output)
					score = score + math.Abs(network.Layer[lC].Neuron[nC].Delta)
				}
			} else {
				pLayer := network.Layer[lC+1].Neuron
				var err float64
				for _, pL := range pLayer {
					err = err + (pL.Weights[nC] * pL.Delta)
				}
				if network.Layer[lC].Type == "relu" {
					network.Layer[lC].Neuron[nC].Delta = network.Layer[lC].Neuron[nC].Delta + reluErrorBackpropagation(err, neuron.Output)
					score = score + math.Abs(network.Layer[lC].Neuron[nC].Delta)
				} else if network.Layer[lC].Type == "sigmoid" {
					network.Layer[lC].Neuron[nC].Delta = network.Layer[lC].Neuron[nC].Delta + sigmoidErrorBackpropagation(expected[nC], neuron.Output)
					score = score + math.Abs(network.Layer[lC].Neuron[nC].Delta)
				}
			}
			nC++
		}
	}
	return network, score
}
func Run(network Network, input_data []float64) (Network, map[string]float64) {
	//local values
	var output float64
	//output map
	out := make(map[string]float64)
	//temp array for calculations
	temparray := []float64{}
	//counters
	//--------------------
	//layer counter
	var l int
	//neuron counter
	var n int
	//input counter
	var i int
	//output counter
	var o int
	//run network
	//iterate layers
	for l <= len(network.Layer)-1 {
		for n < len(network.Layer[l].Neuron)-1 {
			//if on layer 1, use input data
			if l == 0 {
				//loop through input data
				for i < len(network.Layer[l].Neuron[n].Weights)-1 {
					output = output + (input_data[i] * network.Layer[l].Neuron[n].Weights[i])
					i++
				}
			} else {
				//loop through previous neuron data
				for i < len(network.Layer[l].Neuron[n].Weights)-1 {
					output = output + (network.Layer[l-1].Neuron[n].Output * network.Layer[l].Neuron[n].Weights[i])
					i++
				}
			}
			//reset i
			i = 0
			//add bias
			output = output + network.Layer[l].Neuron[n].Bias
			//Preform special function
			if network.Layer[l].Type == "sigmoid" {
				output = sigmoid(output)
			} else if network.Layer[l].Type == "relu" {
				output = relu(output)
			}
			// add output to output of the neuron
			network.Layer[l].Neuron[n].Output = output
			//reset output
			output = 0
			n++
		}
		//reset n
		n = 0
		l++
	}
	for o < len(network.Layer[len(network.Layer)-1].Neuron)-1 {
		temparray = append(temparray, network.Layer[len(network.Layer)-1].Neuron[o].Output)
		o++
	}
	//reset counter
	o = 0
	if network.OutputFunction == "softmax" {
		temparray = softmax(temparray)
	}
	for o <= len(temparray)-1 {
		out[network.OutputNames[o]] = temparray[o]
		o++
	}
	return network, out
}
func relu(number float64) float64 {
	var value float64
	if number >= 0 {
		value = number
	} else {
		value = number * 0.01
	}
	return value
}
func sigmoid(number float64) float64 {
	return 1 / (1 + (math.Exp(-number)))
}
func softmax(numbers []float64) []float64 {
	//denomintor varible
	var denom float64
	//array to hold return
	var toReturn []float64
	//Counter
	var c int
	//add up all vales for denomnator
	for c <= len(numbers)-1 {
		denom = denom + numbers[c]
		c++
	}
	//reset counter
	c = 0
	//divid each number by denom
	for c <= len(numbers)-1 {
		toReturn = append(toReturn, (numbers[c] / denom))
		c++
	}
	return toReturn
}
func Generate(name string, num_inputs int, layer_array []int, layer_types []string, output_names []string, output_function string) Network {
	//load structs
	var network Network
	var layer Layer
	var neuron Neuron
	//Input data
	//check for errors in Input
	if len(layer_array) != len(layer_types) {
		panic("array of layers dont match the array of layer types")
	}
	//counters
	//---------------
	//layer counter
	var l int
	//Neuron counter
	var n int
	//input counter
	var i int
	//iterate over layers and create layers
	for l <= len(layer_array)-1 {
		//iterate over neurons and randomize Weights
		for n <= layer_array[l] {
			//Generate random Weights
			//if on layer 1 setup with inputs instead of previous layer
			if l == 0 {
				for num_inputs >= i {
					neuron.Weights = append(neuron.Weights, random())
					i++
				}
			} else {
				//add weights for each neuron previous layer
				for layer_array[l-1]-1 >= i {
					neuron.Weights = append(neuron.Weights, random())
					i++
				}
			}
			//reset input counter
			i = 0
			//Randomize Bias
			neuron.Bias = random()
			//set zeros
			neuron.Delta = 0
			neuron.Error = 0
			//append neuron to layer
			layer.Neuron = append(layer.Neuron, neuron)
			//null out neuron array
			neuron.Weights = nil
			n++
		}
		layer.Type = layer_types[l]
		if l == len(layer_array)-1 {
			layer.IsOutput = true
		} else {
			layer.IsOutput = false
		}
		//reset neuron counter
		n = 0
		//append layer to network
		network.Layer = append(network.Layer, layer)
		//null out layer
		layer.Neuron = nil
		l++
	}
	//add rest of network info
	network.NetworkName = name
	network.NumOutputs = len(network.Layer[len(network.Layer)-1].Neuron)
	network.OutputNames = output_names
	network.OutputFunction = output_function
	return network
}
func UpdateWeights(network Network, learningRate float64, inputData []float64) Network {
	var weight float64
	for lC, layer := range network.Layer {
		if lC != 0 {
			for nC, neuron := range layer.Neuron {
				for iC, i := range network.Layer[lC-1].Neuron {
					weight = network.Layer[lC].Neuron[nC].Weights[iC] + (learningRate * neuron.Delta * i.Output)
					network.Layer[lC].Neuron[nC].Weights[iC] = weight
				}
			}
		} else {
			for nC, neuron := range layer.Neuron {
				for iC, i := range inputData {
					weight = network.Layer[lC].Neuron[nC].Weights[iC] + (learningRate * neuron.Delta * i)
					network.Layer[lC].Neuron[nC].Weights[iC] = weight
				}
			}
		}
	}
	return network
}
func reluDerivative(value float64) float64 {
	var bitSetVar float64
	if value >= 0 {
		bitSetVar = 1.0
	} else {
		bitSetVar = 0.01
	}
	return bitSetVar
}
func reluErrorBackpropagationOutput(expected float64, result float64) float64 {
	return (expected - result) * reluDerivative(result)
}
func reluErrorBackpropagation(err float64, output float64) float64 {
	return err * reluDerivative(output)
}

func sigmoidDerivative(output float64) float64 {
	return output * (1.0 - output)
}
func sigmoidErrorBackpropagationOutput(expected float64, result float64) float64 {
	return (expected - result) * sigmoidDerivative(result)
}
func sigmoidErrorBackpropagation(err float64, output float64) float64 {
	return err * sigmoidDerivative(output)
}
func random() float64 {
	seed := rand.NewSource(time.Now().Unix())
	r := rand.New(seed)
	return r.Float64()
}
