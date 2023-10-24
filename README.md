# mutil
Math utilities (e.g. FFT, random number generation, convolutions) used for extragalactic sky modeling. 

## Installation
1. git clone https://github.com/exgalsky/mutil.git
2. cd mutils
3. pip install .

## Running
Currently runs on perlmutter in the [xgsmenv](https://github.com/exgalsky/xgsmenv) enviroment.

Example included here in [scripts/example.py](https://github.com/exgalsky/mutil/blob/master/scripts/example.py) will produce 20 random numbers from a normal distribution.

```
import mutils as mu

# create stream
stream = mu.Stream(
    seed = 13579,
    nsub = 2
)

seq = stream.generate(start=5,size=20)

print('seq',seq)
print('seq shape:',seq.shape)
print('seq mean:',seq.mean())

```
