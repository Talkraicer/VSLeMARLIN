import os
from xml.etree import ElementTree as ET
import numpy as np
from settings import *

np.random.seed(42)


def set_cfg_file(demand, seed):
    # Load and parse the XML file
    tree = ET.parse(f'{EXP_NAME}.sumocfg')
    root = tree.getroot()

    # set route file
    route_file = root.find("input").find('route-files')
    route_file.set('value', f'{ROOT}/rou_files/{demand}/{seed}/{EXP_NAME}.rou.xml')



    # set net file
    net_file = root.find("input").find('net-file')
    net_file.set('value', f'{ROOT}/cfg_files/{EXP_NAME}.net.xml')

    # set seed
    seed_element = root.find("random_number").find('seed')
    seed_element.set('value', str(seed))

    os.makedirs(f"{demand}/{seed}", exist_ok=True)
    # Save the changes back to the file
    tree.write(f'{demand}/{seed}/{EXP_NAME}.sumocfg')


if __name__ == "__main__":
    seeds = [np.random.randint(0, 10000) for _ in range(10)]
    for demand in DEMANDS:
        for seed in seeds:
            set_cfg_file(demand, seed)