import scaleran as sr

# create stream
stream = sr.Stream(
    seed = 13579,
    nsub = 2
)

seq = stream.generate(start=5,size=20)

print('seq',seq)
print('seq shape:',seq.shape)
print('seq mean:',seq.mean())
