import xgmutil as mu
import jax.numpy as jnp

machine_name="MacbookPro2021_M1Max"
RNG_mgr = mu.RNG_manager()
ran_stream = RNG_mgr.setup_stream('test')

print("Using device:", ran_stream.device)

dtypes = [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128]

nsamples = 10
output_file = f'./reproducibity_test_results-{ran_stream.component_id}_{machine_name}-{ran_stream.device}.npy'

with open(output_file, 'wb') as f:
    for dtyp in dtypes:
        sample = ran_stream.generate(mc=0, start=0, size=nsamples, dtype=dtyp, dist='normal')
    jnp.save(output_file, sample)

f.close()