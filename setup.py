import setuptools

# with open("README.md", "r") as fh:
    # long_description = fh.read()
long_description = "Pending"

setuptools.setup(
    name='gym_carla',
    version='0.1.1',
    # scripts=['gym_torcs'],
    install_requires=['psutil', 'gym'],
    author="Rousslan F. J. Dossa",
    author_email="dosssman@hotmail.fr",
    description="A pip package for the Gym Carla environment with simplicity in mind",
    long_description=long_description,
    url="https://github.com/dosssman/GymCarla.git",
    packages=setuptools.find_packages(),
    classifiers=[
         "Programming Language :: Python :: 3",
         # "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ]
)
