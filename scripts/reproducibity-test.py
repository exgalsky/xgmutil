import xgmutil as mu
import jax.numpy as jnp

machine_name="ThinkpadX1Carbon_Intel-i7-1270P"
RNG_mgr = mu.RNG_manager()
ran_stream = RNG_mgr.setup_stream('test', nsub=32**3)

print("Using device:", ran_stream.device)

dtypes = [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128]

nsamples = 10
output_file = f'./reproducibity_test_results-{ran_stream.component_id}_{machine_name}-{ran_stream.device}.npz'

sample = []
for dtyp in dtypes:
    sample.append(ran_stream.generate(mc=0, start=0, size=nsamples, dtype=dtyp, dist='normal'))
    print(sample[-1])
jnp.savez(output_file, f32=sample[0], f64=sample[1], c64=sample[2], c128=sample[3])