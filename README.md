# Single-Bubble Sonoluminescence Dynamics

## Overview
This repository contains the Python codes used to model and analyze the dynamics of single-bubble sonoluminescence (SBSL), an intriguing phenomenon where light is emitted by small gas bubbles in a liquid when excited by sound waves. This project is based on simulations and mathematical modeling techniques to better understand the physical mechanisms behind SBSL, particularly focusing on bubble dynamics under various physical conditions.

## Objectives
- **Simulate Bubble Dynamics:** Using MATLAB and Python to model the behavior of a sonoluminescent bubble under various driving pressures and physical conditions.
- **Analyze Results:** Employ numerical methods to analyze the bubble's response to different parameters such as liquid viscosity, gas type, surface tension, and driving frequency.
- **Visualize Data:** Generate plots and visual representations to better understand the relationship between bubble size, light emission, and acoustic properties.

## Repository Contents
- **`RP_lib.py`** - Python library containing functions and utilities for calculating the physical properties of the bubble and the surrounding fluid.
- **`RP_plot_01.py`** - Python script for plotting the simulation results from the bubble dynamics models.
- **`Bubble.py`** - Main Python script for running bubble dynamics simulations.
- **`etc.py`** - Additional utilities and helper functions used across different simulations.
- **`H2O.yaml`** - Configuration file containing parameters for water, the primary medium in our simulations.
- **`main.py`** - Entry point script that integrates all components of the simulation and runs the complete analysis.
- **`test.ipynb`** - Jupyter notebook used for preliminary tests and experimental analysis.
- **`Modeling the dynamics of single-bubble.pdf`** - Comprehensive study and theoretical background on the dynamics of single-bubble sonoluminescence, including references to key research and findings in the field.

## Running the Simulations
Ensure you have Python 3.8 or later installed, along with necessary libraries such as NumPy, Matplotlib, and SciPy. 


To execute the main simulation:
```bash
python main.py
```

For visualizing results:

```bash
python RP_plot_01.py
```

## Below is an exaple showing the acetone used in simulations:
![err404](/RPeq_acetone.png)
