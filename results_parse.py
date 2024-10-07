from settings import *
import xml.etree.ElementTree as ET
import pandas as pd


def output_file_to_df(output_file):
    # Parse the XML file into pd dataframe
    tree = ET.parse(output_file)
    root = tree.getroot()

    dict = {"departDelay": [], "timeLoss": [], "id": [], "depart": []}
    for tripinfo in root.findall('tripinfo'):
        for key in dict.keys():
            dict[key].append(tripinfo.get(key))
    df = pd.DataFrame(dict)
    df["totalDelay"] = df.departDelay.astype(float) + df.timeLoss.astype(float)
    # convert to float except vType
    return df["totalDelay"]


def get_mean_delay(demand,seed, policy_name):

    output_file = RESULTS_FOLDER + f"/results_reps/{demand}/{seed}/{policy_name}_{EXP_NAME}.xml"
    return output_file_to_df(output_file).mean()

if __name__ == '__main__':
    cfg_files = ["cfg_files_VSL/high/860/VSL.sumocfg",
                 "cfg_files_VSL/low/7141/VSL.sumocfg",
                 "cfg_files_VSL/medium/5598/VSL.sumocfg"
                 "cfg_files_VSL/medium/2216/VSL.sumocfg",
                 "cfg_files_VSL/high/8873/VSL.sumocfg",
                 "cfg_files_VSL/medium/1693/VSL.sumocfg",
                 "cfg_files_VSL/low/9660/VSL.sumocfg",
                 "cfg_files_VSL/medium/7879/VSL.sumocfg",
                 "cfg_files_VSL/medium/9256/VSL.sumocfg",
                 "cfg_files_VSL/medium/6994/VSL.sumocfg",
                 ]
    for cfg in cfg_files:
        print(get_mean_delay(cfg, "DQN_distri"))