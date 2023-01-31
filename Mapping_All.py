import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString
import os
from multiprocessing import Pool
from itertools import repeat
from geopy import distance
from datetime import date
import more_itertools as mit
from pyproj import Proj, transform
import sys
from os.path import dirname
from Mapping_Module import Graph_map_infor
from Mapping_Module import Mapping
from Mapping_Module import Identify_rwy_ramp
from Mapping_Module import get_file_name
from scipy.spatial import cKDTree
from tqdm import tqdm

class Mapping_All(object):
    def __init__(self, airport):
        self.airport = airport
        self.dataFileLoc = '../../Data/' + self.airport + '/'
        G = ox.load_graphml(self.dataFileLoc + 'devide_' + self.airport + '.graphml')

        Gp = ox.project_graph(G)
        Gp = Gp.to_undirected()

        identify_rwy_ramp = Identify_rwy_ramp(Gp, airport)
        dict_ramps_polygon = identify_rwy_ramp.identify_runway_and_ramp_polygon()

        # Create ckdtree
        df_G_nodes = pd.DataFrame(
            {"x": nx.get_node_attributes(Gp, "x"), "y": nx.get_node_attributes(Gp, "y")})
        ckdtree_airport = cKDTree(data=df_G_nodes[["x", "y"]], compact_nodes=True, balanced_tree=True)

        df_map = pd.read_csv(self.dataFileLoc + self.airport + '_devide_map_graphml_based.csv')
        # print('df_map_len:{}'.format(len(df_map)))
        df_rwy_edge = df_map[df_map['ref'].str.contains("/", na=False)]
        df_map_index = []

        for i in range(len(df_rwy_edge)):
            df_map_index.append(df_rwy_edge['source'].iloc[i])
            df_map_index.append(df_rwy_edge['target'].iloc[i])

        df_rwy_nodes = df_G_nodes.loc[df_map_index, :].copy()
        ckdtree_rwy = cKDTree(data=df_rwy_nodes[["x", "y"]], compact_nodes=True, balanced_tree=True)

        folder_name = self.dataFileLoc + 'extract_ground_aircraft'
        lst_file_name = get_file_name(folder_name)

        # save file location
        save_folder = 'mapping_result'
        self.save_location = os.path.dirname(__file__) + '/../..//Data/'+self.airport + '/' + str(save_folder) + '/'
        self.createFolder(self.save_location)

        for file_i in range(len(lst_file_name)):
            print('mapping_file_id:{}'.format(file_i))
            file_name = lst_file_name[file_i]

            df_all_tra_ori = pd.read_csv(file_name, low_memory=False)
            save_file_name = lst_file_name[file_i].split('.csv')[0].split('\\')[-1] + '_mapping_result.csv'

            df_all_tra_ori = df_all_tra_ori[(df_all_tra_ori['operation'] == 'A') | (df_all_tra_ori['operation'] == 'D')]
            list_df_all_tra = [g for _, g in df_all_tra_ori.groupby(['reg_num', 'call_sign', 'operation'])]

            list_df_tra = self.split_list(20, list_df_all_tra)

            df_all_mapped_nodes = pd.DataFrame()

            for i in tqdm(range(len(list_df_tra))):

                list_df_tra_i = list_df_tra[i]
                list_df_tra_i = [(Gp, airport, ckdtree_airport, ckdtree_rwy, df_rwy_nodes, df_G_nodes, dict_ramps_polygon, x) for x in list_df_tra_i]

                p = Pool(5)
                list_df_mapped_result = []
                list_df_mapped_result = p.map(self.use_mapping_object, list_df_tra_i)

                p.close()

                df_mapped_nodes = pd.concat(list_df_mapped_result)
                df_all_mapped_nodes = df_all_mapped_nodes.append(df_mapped_nodes)

            df_all_mapped_nodes.to_csv(self.save_location + str(save_file_name), index=False)
    def use_mapping_object(self,mylist):
        G = mylist[0]
        airport = mylist[1]
        ckdtree_airport = mylist[2]
        ckdtree_rwy = mylist[3]
        df_rwy_nodes = mylist[4]
        df_G_nodes = mylist[5]
        dict_ramps_polygon = mylist[6]
        df_tra_ori = mylist[7]

        mapping = Mapping(G, airport, ckdtree_airport, ckdtree_rwy,df_rwy_nodes, df_G_nodes, dict_ramps_polygon,  df_tra_ori)
        df_mapped_df_edge_info = mapping.ordered_process()

        return df_mapped_df_edge_info

    def split_list (self, x, pre_list):
       return [pre_list[i:i+x] for i in range(0, len(pre_list), x)]

    def createFolder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)




def main(airport):
    mapping_all = Mapping_All(airport)

if __name__ == '__main__':
    main('KLAX')











