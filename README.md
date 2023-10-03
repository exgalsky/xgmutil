# scaleran
Scalable random number generator

## Installation
1. git clone https://github.com/marcelo-alvarez/scaleran.git
2. cd scaleran
3. pip install .

## Running
Currently runs on perlmutter in the [xgsmenv](https://github.com/marcelo-alvarez/xgsmenv) enviroment.

Example included here in [scripts/example.py](https://github.com/marcelo-alvarez/halosky/blob/master/scripts/example.py) will produce 20 random numbers from a normal distribution.

```
import scaleran as sr

# create stream
stream = sr.stream.stream(
    seed = 13579,
    nsub = 2
)

seq = stream.generate(start=5,size=20)

print('seq',seq)
print('seq shape:',seq.shape)
print('seq mean:',seq.mean())

```