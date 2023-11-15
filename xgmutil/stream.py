import jax.numpy as jnp 
import jax.random as rnd
import jax
import os

class Stream:
    '''Stream'''
    def __init__(self, **kwargs):

        self.force_no_gpu = kwargs.get('force_no_gpu',False)
        self.seedkey      = kwargs.get('seedkey', rnd.PRNGKey(123456789))
        self.nsub         = kwargs.get('nsub',1024**3)

    def generate(self,**kwargs):

        if self.force_no_gpu:
            _JAX_PLATFORM_NAME = jax.default_backend()
            jax.default_device("cpu")

        start = kwargs.get('start',0)
        size  = kwargs.get('size' ,1)
        dist  = kwargs.get('dist','normal')
        dtype = kwargs.get('dtype', jnp.float32)
        mc    = kwargs.get('mc', 0)

        if dtype in [jnp.float64, jnp.complex128, jnp.int64]:
            _JAX_X64_INITIAL_STATE = jax.config.read('jax_enable_x64')
            jax.config.update('jax_enable_x64', True)

        key4mc = rnd.fold_in(self.seedkey, mc)

        end = start + size - 1

        start_seqID = start // self.nsub
        end_seqID   = end   // self.nsub

        seqIDs = jnp.arange(start_seqID, end_seqID+1)

        keys = jax.vmap(rnd.fold_in, in_axes=(None, 0), out_axes=0)(key4mc, seqIDs)

        seq = jnp.zeros(0,dtype=jnp.float32)

        for seqID in seqIDs:

            subseq_start = max(seqID * self.nsub, start) - seqID * self.nsub
            subseq_end   = min((seqID+1) * self.nsub -1,end) - seqID * self.nsub

            if dist == 'normal':
                subseq = rnd.normal(keys[seqID], dtype=dtype, shape=(self.nsub,))

            seq = jnp.concatenate((seq,subseq[subseq_start:subseq_end+1]))
        
        if self.force_no_gpu:
            jax.default_device(_JAX_PLATFORM_NAME)

        if dtype in [jnp.float64, jnp.complex128, jnp.int64]:
            jax.config.update('jax_enable_x64', _JAX_X64_INITIAL_STATE)

        return seq





