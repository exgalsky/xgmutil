import jax.numpy as jnp 
import jax.random as rnd
import jax
import sys

class stream:
    '''Stream'''
    def __init__(self, **kwargs):

        self.force_no_gpu = kwargs.get('force_no_gpu',False)
        self.seed         = kwargs.get('seed',123456789)
        self.nsub         = kwargs.get('nsub',1024**3)

    def generate(self,**kwargs):

        start = kwargs.get('start',0)
        size  = kwargs.get('size' ,1)
        dist  = kwargs.get('dist','normal')

        end = start + size - 1

        start_seqID = start // self.nsub
        end_seqID   = end   // self.nsub

        seqIDs = jnp.arange(start_seqID, end_seqID+1)

        seedkey = rnd.PRNGKey(self.seed)

        keys = jax.vmap(rnd.fold_in, in_axes=(None, 0), out_axes=0)(seedkey, seqIDs)

        seq = jnp.zeros(0,dtype=jnp.float32)

        for seqID in seqIDs:

            subseq_start = max(seqID * self.nsub, start) - seqID * self.nsub
            subseq_end   = min((seqID+1) * self.nsub -1,end) - seqID * self.nsub

            subseq = rnd.normal(keys[seqID], dtype=jnp.float32, shape=(self.nsub,))

            seq = jnp.concatenate((seq,subseq[subseq_start:subseq_end+1]))
        
        return seq






