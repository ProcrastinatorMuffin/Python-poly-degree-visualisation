# Polynomial Degree Visualization

This Python script uses the Manim library to create a visualization of polynomial regression models of varying degrees. The purpose of this script is to provide a visual understanding of how changing the degree of a polynomial regression model affects its fit to a given dataset.

## Capabilities

The script generates a dataset based on a sine function with added noise, then fits polynomial regression models of varying degrees to this data. It visualizes the fitted models, the mean squared error (MSE) for training and validation sets, and the degree of the polynomial. The visualization is animated, showing how the model changes as the degree increases.

## Use Case

This script is useful for understanding the concept of overfitting in machine learning. As the degree of the polynomial increases, the model becomes more complex and starts to fit the noise in the training data, leading to overfitting. This is reflected in the visualization by the increasing MSE for the validation set.

## Implementation

The script uses the Manim library for creating the visualization, and Scikit-learn for generating the dataset and fitting the polynomial regression models. The script is organized into a class `PolyDegreeVisualization` with methods for setting up the scene, generating the data, training the models, and creating the animation.

## Instructions

To run this script, you need to have Python installed along with the Manim and Scikit-learn libraries. 

### Dependencies

#### For Mac:

To install all required dependencies for installing Manim (namely: ffmpeg, Python, and some required Python packages), run:

```bash
brew install py3cairo ffmpeg
```

On Apple Silicon based machines (i.e., devices with the M1 chip or similar; if you are unsure which processor you have check by opening the Apple menu, select About This Mac and check the entry next to Chip), some additional dependencies are required, namely:

```bash
brew install pango pkg-config scipy
```

After all required dependencies are installed, simply run:

```bash
pip3 install manim
```

#### For Windows:

Manim requires a recent version of Python (3.8 or above) and ffmpeg in order to work.

##### Chocolatey

Manim can be installed via Chocolatey simply by running:

```bash
choco install manimce
```

##### Scoop

While there is no recipe for installing Manim with Scoop directly, you can install all requirements by running:

```bash
scoop install python ffmpeg
```

and then Manim can be installed by running:

```bash
python -m pip install manim
```

##### Winget

While there is no recipe for installing Manim with Winget directly, you can install all requirements by running:

```bash
winget install python
winget install ffmpeg
```

and then Manim can be installed by running:

```bash
python -m pip install manim
```

### Virtual Environment

It is recommended to use a virtual environment to avoid conflicts with other packages. You can use Conda for this purpose. To create a new Conda environment, run:

```bash
conda create --name myenv
```

To activate the environment, run:

```bash
conda activate myenv
```

Install dependencies by running:

```bash
pip3 install -r requirements.txt
```

Once the libraries are installed, you can run the script using the following command:

```bash
python poly_degree_visualisation.py
```

This will create a video file in the media directory created in the root directory of script, which you can view to see the visualization.

