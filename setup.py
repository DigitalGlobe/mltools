import sys
from setuptools import setup, find_packages

open_kwds = {}
if sys.version_info > (3,):
    open_kwds['encoding'] = 'utf-8'

# with open('README.md', **open_kwds) as f:
#     readme = f.read()

# long_description=readme,
      
setup(name='mltools',
      version='1.0.7',
      description='Added mask to data extractor.',
      classifiers=[],
      keywords='',
      author='Kostas Stamatiou',
      author_email='kostas.stamatiou@digitalglobe.com',
      url='https://github.com/kostasthebarbarian/mltools',
      license='MIT',
      packages=find_packages(exclude=['docs']),
      include_package_data=True,
      zip_safe=False,
      install_requires=['geojson >= 1.3.2',
                        'psycopg2 >= 2.6.1',
                        'scikit-learn >= 0.17.1',
                        'Shapely >= 1.5.15',
                        'geoio >= 1.1.1',
                        'keras >= 1.0.2']
      )
