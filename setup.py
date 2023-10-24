from setuptools import setup
pname='xgmutil'
setup(name=pname,
      version='0.1',
      description='Scalable random number generator using Jax',
      url='http://github.com/exgalsky/xgmutil',
      author='exgalsky collaboration',
      license_files = ('LICENSE',),
      packages=[pname],
      zip_safe=False)
