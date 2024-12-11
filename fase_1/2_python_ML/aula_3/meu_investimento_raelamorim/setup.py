from setuptools import setup, find_packages

setup(
   name='meu_investimento_raelamorim',
   version='0.2',
   packages=find_packages(),
   install_requires=[],
   author='raelamorim',
   author_email='raelamorim88@gmail.com',
   description='Uma biblioteca para cálculos de investimentos.',
   url='https://github.com/raelamorim/meu_investimento',
   classifiers=[
       'Programming Language :: Python :: 3',
       'License :: OSI Approved :: MIT License',
       'Operating System :: OS Independent',
   ],
   python_requires='>=3.6',
)