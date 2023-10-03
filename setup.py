from setuptools import setup
pname='scaleran'
setup(name=pname,
      version='0.1',
      description='Scalable random number generator using Jax',
      url='http://github.com/marcelo-alvarez/scaleran',
      author='Shamik Gosh and Marcelo Alvarez',
      license_files = ('LICENSE',),
      packages=['scaleran'],
      zip_safe=False)
