# NNet
Neural Network package written in go for go!

## Intro
They say Golang is a bad language for Neural Networking. They would be right. So that is why I'm gonna making it a little more bearable.

## Install
Unless you live on the surface and actually have a life, you should know the routine of installing packages. This is how you do it any way if you're new to this(Welcome to the basement world of Go by the way! :) )

```sh
go get github.com/plasmaduster355/NNet
```
## Documentation
#### Getting started
To start, we need to make a network. To do that we do this:

```go
package main
import(
  "fmt"
  "github.com/plasmaduster355/nnet"
)
func main(){
    network,err := nnet.GenerateNetwork(lC,nA,AID,AOD,oN)
    if err != true{
      fmt.Println("Something happened...I blame you ;)")
    }
}
```
Confusing a bit like, "Why doesn't err return a type error". You know what I say to that? Deal with it >:). Anyway, in order to make a network, we need to know some information ABOUT our network. We need to know our layer count (lC), our amount of neurons for each layer (nA), the amount of inputs we will feed it (AID), how much data we want back (AOD), and the names of out output values (oN)(Optional).

###### Finding a good layer Count
The number of layers depend on how quick and accurate you want the network to run. You want it to run really fast? Pick 1-3 layers. Want it to be super accurate? Pick 10-30 layers. JK. Pick like 5-10. You can go as high as you want though if want to have it run more accurately just know, it will be slow.
###### Finding a good neuron Count
The number of neurons also affect on how accurate or quick you want the network to run. If you want high accuracy? Make like 5-10 neurons in each layer. Want it to run quick? pick 2-4 in each layer(See what I did there ;) ). This value has to be in a slice int format or []int. This give control to you more. so if I was making a 3 layer Net work with 2 neurons,3 neurons,2 neurons layout, I would do:
```go
nA := []int{2,3,2}
```
Now that wasn't that hard.
