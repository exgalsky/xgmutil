import xgmutil as mu 
import jax.numpy as jnp
import jax

cuda_build = False

# Do jax.distributed.initialize if jaxlib is built with cuda
if cuda_build:
    jax.distributed.initialize()

# create stream
stream = mu.Stream(
    seedkey = 13579,
    nsub    = 2

)

seq = stream.generate(start=5,size=20)

print('seq',seq)
print('seq shape:',seq.shape)
print('seq mean:',seq.mean())

RNG_manager = mu.RNG_manager(seed = 13579)
print(f'master seed      : {RNG_manager.seed}')
print(f'master key       : {RNG_manager.master_key}')
example_rand_stream = RNG_manager.setup_stream('example', component_type='example type', dtype=jnp.float32, nsub=1024, force_no_gpu=False)
print(f'component name   : {example_rand_stream.component}')
print(f'component id     : {example_rand_stream.component_id}')
print(f'component key    : {example_rand_stream.component_key}')


realization = 5
rand_sample = example_rand_stream.generate(start=5,size=20, dist='normal', mc=realization)
print(f'Realization no {realization}:',rand_sample)

realization = 6
rand_sample = example_rand_stream.generate(start=5,size=20, dist='normal', mc=realization)
print(f'Realization no {realization}:',rand_sample)