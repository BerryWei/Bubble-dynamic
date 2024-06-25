from etc import read_yaml
from Bubble import BubbleSimulator
import numpy as np
from pathlib import Path





if __name__ == '__main__':
    configPath = Path(f'./H2O.yaml')
    config={}
    config = read_yaml(configPath)  # Assuming the 'read_yaml' function has been defined
    print(config)

    # Time span
    t_span = (0, 0.4e-6*100)
    t_eval = np.linspace(t_span[0], t_span[1], 10000)

    simulator = BubbleSimulator(**config)
    t, R, dRdt = simulator.simulate(t_span, t_eval)
    simulator.plot(t, R, dRdt)
    print(R[-1])


