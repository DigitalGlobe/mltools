import sys
from setuptools import setup, find_packages

open_kwds = {}
if sys.version_info > (3,):
    open_kwds['encoding'] = 'utf-8'

# with open('README.md', **open_kwds) as f:
#     readme = f.read()

# long_description=readme,
      
setup(name='mltools',
      version='0.0.3',
      description='Basic lulc, classification and pool detection',
      classifiers=[],
      keywords='',
      author='Kostas Stamatiou',
      author_email='kostas.stamatiou@digitalglobe.com',
      url='https://github.com/kostasthebarbarian/mltools',
      license='MIT',
      packages=find_packages(exclude=['docs']),
      include_package_data=True,
      zip_safe=False,
      install_requires=['numpy','geojson','scikit-learn','shapely','psycopg2']
      )
