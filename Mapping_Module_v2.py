import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from shapely import geometry
from shapely.geometry import Point, Polygon, LineString
import os
from multiprocessing import Pool
from itertools import repeat
from geopy import distance
from scipy.spatial import cKDTree
from datetime import date
import more_itertools as mit
from itertools import groupby
from operator import itemgetter
from pyproj import Proj, transform
from rdp import rdp
import time
import sys
from Visualization import visualize_projected_graph_and_ori_tra_with_rdp_results
from Visualization import visulaization_single_tra_and_mapped_tra_for_module
'''Major Updates:
1. Use projected graphml
2. Use K-D Tree for search nearest 
3. Use K-D Tree for runway initial node search
3. Search more than one nearest node '''


class Graph_map_infor(object):

    # In networkx nodes are int;
    # In LAP_map.csv nodes are int.
    def __init__(self, G):
        self.G = G
        self.max_node_osmid = max(list(G.nodes))

    def convert_node_attr_types(self, node_dtypes=None):
        """
        Convert graph nodes' attributes' types from string to numeric.
        Parameters
        ----------
        G : networkx.MultiDiGraph
            input graph
        node_type : type
            convert node ID (osmid) to this type
        node_dtypes : dict of attribute name -> data type
            identifies additional is a numpy.dtype or Python type
            to cast one or more additional node attributes
            defaults to {"elevation":float, "elevation_res":float,
            "lat":float, "lon":float, "x":float, "y":float} if None
        Returns
        -------
        G : networkx.MultiDiGraph
        """
        if node_dtypes is None:
            node_dtypes = {"highway": str, "ref": str, "osmid": str,
                           "x": float, "y": float}
        for _, data in self.G.nodes(data=True):

            # convert numeric node attributes from string to float
            for attr in node_dtypes:
                if attr in data:
                    dtype = node_dtypes[attr]
                    data[attr] = dtype(data[attr])
        return self.G

    def convert_edge_attr_types(self, edge_dtypes=None):
        """
        Convert graph edges' attributes' types from string to numeric.
        Parameters
        ----------
        G : networkx.MultiDiGraph
            input graph
        node_type : type
            convert osmid to this type
        edge_dtypes : dict of attribute name -> data type
            identifies additional is a numpy.dtype or Python
            type to cast one or more additional edge attributes.
        Returns
        -------
        G : networkx.MultiDiGraph
        """
        if edge_dtypes is None:
            edge_dtypes = {"bridge": str, "geometry": LineString, "name": str,
                           "length": float, "oneway": str, "width": str,
                           "ref": str, "osmid": str}
        # convert numeric, bool, and list edge attributes from string
        # to correct data types

        # convert to specfied dtype any possible OSMnx-added edge attributes, which may
        # have multiple values if graph was simplified after they were added
        for _, _, data in self.G.edges(data=True, keys=False):

            # convert to specfied dtype any possible OSMnx-added edge attributes, which may
            # have multiple values if graph was simplified after they were added
            for attr in edge_dtypes:
                if attr in data:
                    dtype = edge_dtypes[attr]
                    data[attr] = dtype(data[attr])
        return self.G

class Identify_rwy_ramp(object):
    def __init__(self, G, airport):
        self.G = G
        self.airport = airport
        if self.airport == 'KLGA':
            self.utm = Proj(proj="utm", zone=18, datum='WGS84')
        elif self.airport == 'KLAX':
            self.utm = Proj(proj="utm", zone=11, datum='WGS84')
        self.lla = Proj(proj='latlong', ellps='WGS84', datum='WGS84')
        self.dict_rwys = ''
        self.dict_ramps = ''
        self.airport = airport

    def identify_runway_and_ramp_polygon_return_rwys_ramps(self):
        list_rwy_polygon = self.create_runway_polygons()
        self.dict_rwys = {k: v for k, v in enumerate(list_rwy_polygon)}

        list_ramp_polygon = self.create_ramp_polygons()
        self.dict_ramps = {k: v for k, v in enumerate(list_ramp_polygon)}

        return self.dict_rwys, self.dict_ramps

    def identify_runway_and_ramp_polygon(self):
        list_ramp_polygon = self.create_ramp_polygons()
        self.dict_ramps = {k: v for k, v in enumerate(list_ramp_polygon)}

        return self.dict_ramps

    def create_runway_polygon(self, df):
        df.loc[:, 'source_osmid_long'] = [self.G.nodes[x]['x'] for x in df['source_osmid'].tolist()]
        df.loc[:, 'source_osmid_lat'] = [self.G.nodes[x]['y'] for x in df['source_osmid'].tolist()]
        df.loc[:, 'target_osmid_long'] = [self.G.nodes[x]['x'] for x in df['target_osmid'].tolist()]
        df.loc[:, 'target_osmid_lat'] = [self.G.nodes[x]['y'] for x in df['target_osmid'].tolist()]

        df.loc[:, 'source_x'] = df.loc[:, 'source_osmid_long']
        df.loc[:, 'source_y'] = df.loc[:, 'source_osmid_lat']
        df.loc[:, 'target_x'] = df.loc[:, 'target_osmid_long']
        df.loc[:, 'target_y'] = df.loc[:, 'target_osmid_lat']

        df_source = df[['source_osmid', 'edge_osmid', 'ref', 'source_osmid_long', 'source_osmid_lat', 'source_x', 'source_y']].copy()
        df_source.columns = ['osmid', 'edge_osmid', 'ref', 'osmid_long', 'osmid_lat', 'x', 'y']
        df_target = df[['target_osmid', 'edge_osmid', 'ref', 'target_osmid_long', 'target_osmid_lat', 'target_x', 'target_y']].copy()
        df_target.columns = ['osmid', 'edge_osmid', 'ref', 'osmid_long', 'osmid_lat', 'x', 'y']
        df_rwy = pd.concat([df_source, df_target])

        df_left = df_rwy[df_rwy['x'] == df_rwy['x'].min()]
        df_right = df_rwy[df_rwy['x'] == df_rwy['x'].max()]

        rwy = geometry.LineString([(float(df_left['x'].iloc[0]), float(df_left['y'].iloc[0])), (float(df_right['x'].iloc[0]), float(df_right['y'].iloc[0]))])
        rwy_buffer = rwy.buffer(50)
        return rwy_buffer

    def create_runway_polygons(self):
        rwy_edges = [k for k, v in nx.get_edge_attributes(self.G, 'ref').items() if "/" in list(v)]
        rwy_source_edges = [x[0] for x in rwy_edges]
        rwy_target_edges = [x[1] for x in rwy_edges]
        rwy_ref = [self.G.edges[x]['ref'] for x in rwy_edges]
        df_rwy = pd.DataFrame({'source_osmid': rwy_source_edges, 'target_osmid': rwy_target_edges, 'edge_osmid': rwy_edges, 'ref': rwy_ref})
        list_df_rwy = [g for _, g in df_rwy.groupby(['ref'])]
        list_rwy_polygon = [[self.create_runway_polygon(x)] for x in list_df_rwy]
        return list_rwy_polygon

    def create_ckdtree_rwys(self):
        rwy_edges = [k for k, v in nx.get_edge_attributes(self.G, 'ref').items() if "/" in list(v)]
        rwy_source_edges = [x[0] for x in rwy_edges]
        rwy_target_edges = [x[1] for x in rwy_edges]
        rwy_nodes = rwy_source_edges.copy()
        rwy_nodes.extend(rwy_target_edges)
        rwy_nodes = list(set(rwy_nodes))
        df_G_nodes = pd.DataFrame(
            {"x": nx.get_node_attributes(self.G, "x"), "y": nx.get_node_attributes(self.G, "y"), "osmid": nx.get_node_attributes(self.G, "osmid")})
        df_rwy = pd.DataFrame()

        for node_id in range(len(rwy_nodes)):
            df_rwy_i = df_G_nodes[df_G_nodes['osmid'] == str(int(rwy_nodes[node_id]))]
            df_rwy = df_rwy.append(df_rwy_i)
        # print(len(df_rwy))

        ckdtree_rwys = cKDTree(data=df_rwy[["x", "y"]], compact_nodes=True, balanced_tree=True)

        return ckdtree_rwys

    def transfrom_lla_to_utm(self, ramp_i):
        ramp_utm_i = [list(transform(self.lla, self.utm, x[0], x[1])) for x in ramp_i]
        return ramp_utm_i

    def create_ramp_polygon(self, ramp_utm_i):
        ramp_polygon_i = geometry.Polygon(ramp_utm_i)
        return ramp_polygon_i

    def create_ramp_polygons(self):
        if self.airport == 'KLGA':
            ramp0 = [[-73.872652, 40.777583], [-73.872678, 40.777596], [-73.874965, 40.777221], [-73.874939, 40.777208],
                     [-73.875386, 40.776806], [-73.875956, 40.776081], [-73.880360, 40.771174], [-73.860003, 40.767544],
                     [-73.857846, 40.769222], [-73.858673, 40.769875], [-73.858776, 40.770634]]
            list_ramps = [ramp0]
            list_ramps_utm = [self.transfrom_lla_to_utm(ramp_i) for ramp_i in list_ramps]
            list_ramps_polygons = [[self.create_ramp_polygon(ramp_utm_i)] for ramp_utm_i in list_ramps_utm]
        elif self.airport == 'KLAX':
            ramp0 = [[-118.412363, 33.946389], [-118.409444, 33.946944], [-118.408611, 33.939167], [-118.411246, 33.938889]]
            ramp1 = [[-118.4027696, 33.9481915], [-118.4024122, 33.94643]]
            ramp9 = [[-118.4005199, 33.9484701], [-118.4001182, 33.9464567]]
            ramp2 = [[-118.4058613, 33.9478394], [-118.4055838, 33.9460534]]
            ramp3 = [[-118.4092477, 33.9474907], [-118.40888, 33.9455592]]
            ramp4 = [[-118.408472, 33.9418635], [-118.4078818, 33.9388006]]
            ramp5 = [[-118.4057779, 33.9420102], [-118.4053421248506, 33.93906292575141]]
            ramp6 = [[-118.403382, 33.9422475], [-118.4029551, 33.9393087]]
            ramp7 = [[-118.4010122, 33.9425965], [-118.400563, 33.9395553]]
            ramp8 = [[-118.3985269, 33.9427359], [-118.3981765, 33.9397906]]
            ramp10 = [[-118.39618449108748, 33.9427359], [-118.3920523, 33.9439285], [-118.390687, 33.9405567], [-118.39618449108748, 33.93999681353312]]
            ramp11 = [[-118.43517862852619, 33.94564207948521], [-118.4277629, 33.9462705], [-118.42714697337702, 33.942129499660616], [-118.4332225845548, 33.94149991249603]]
            ramp12 = [[-118.4321854, 33.9387538], [-118.4283578, 33.9387538], [-118.42764113333334, 33.93608306666667], [-118.42972, 33.93583]]
            ramp13 = [[-118.39520925, 33.934854175], [-118.38125594360439, 33.93625862557506], [-118.3812341, 33.9328789], [-118.3953049, 33.9342275]]
            ramp14 = [[-118.3833651, 33.9429605], [-118.3805072, 33.9431908], [-118.3803402714703, 33.94157083846898], [-118.3834549, 33.9414711]]
            list_ramps_polygons = []
            list_ramps = [ramp0, ramp10, ramp11, ramp12, ramp13, ramp14]
            list_ramps_utm = [self.transfrom_lla_to_utm(ramp_i) for ramp_i in list_ramps]

            list_pre_ramps_polygons = [[self.create_ramp_polygon(ramp_utm_i)] for ramp_utm_i in list_ramps_utm]

            list_pre_ramps_polygons2 = []
            for rampi in [ramp9, ramp1, ramp2, ramp3, ramp4, ramp5, ramp6, ramp7, ramp8]:
                rampi = self.transfrom_lla_to_utm(rampi)
                rampi_utm = geometry.LineString(
                    [(float(rampi[0][0]), float(rampi[0][1])), (float(rampi[1][0]), float(rampi[1][1]))])
                rampi_utm = rampi_utm.buffer(50)
                list_pre_ramps_polygons2.append([rampi_utm])

            list_ramps_polygons.extend(list_pre_ramps_polygons2)
            list_ramps_polygons.extend(list_pre_ramps_polygons)

        return list_ramps_polygons



class Mapping(object):

    def __init__(self, G, airport, ckdtree_airport, ckdtree_rwy, df_rwy_nodes, df_G_nodes, dict_ramps_polygon, df_tra_ori, retrieve_k=5):
        self.G = G # projected
        self.airport = airport
        self.retrieve_k = retrieve_k
        self.ckdtree_airport = ckdtree_airport
        self.ckdtree_rwy = ckdtree_rwy
        self.df_rwy_nodes = df_rwy_nodes
        self.df_G_nodes = df_G_nodes
        self.dict_ramps_polygon = dict_ramps_polygon

        self.df_tra_ori = df_tra_ori
        self.df_tra_pre_processed = df_tra_ori
        self.call_sign = df_tra_ori['call_sign'].iloc[0]
        self.reg_num = df_tra_ori['reg_num'].iloc[0]
        self.operation = df_tra_ori['operation'].iloc[0]
        self.trajectory_id = df_tra_ori['trajectory_id'].iloc[0]
        self.df_tra_rdp = pd.DataFrame()
        self.np_rdp_mapped_nodes = []
        self.lst_mapped_nodes = []
        self.df_mapped_nodes = []
        if self.airport == 'KLGA':
            self.utm = Proj(proj="utm", zone=18, datum='WGS84')
        elif self.airport == 'KLAX':
            self.utm = Proj(proj="utm", zone=11, datum='WGS84')
        else:
            print('Airport not found in Mapping Module')
        self.lla = Proj(proj='latlong', ellps='WGS84', datum='WGS84')

        self.ramp_id = 'none'
        self.rwy_id = 'none'
        self.start_timestamp = 'none'
        self.end_timestamp = 'none'

    def ordered_process(self):
        # All trajectory all processed as Departure
        if self.operation == 'D':
            pass
        if self.operation == 'A':
            self.df_tra_ori = self.df_tra_ori[::-1].reset_index(drop=True)

        self.df_tra_ori = self.preprocessing_trajectory_ramp()  # Remove the trajectory around the gate and added the information about the gate if the trajectory was very close to gate (buffer 50)

        if self.df_tra_ori is None:
            return None
        else:
            self.df_tra_rdp = self.preprocessing_trajectory_rdp()  # Use rdp to simplified the trajectory

            self.np_rdp_mapped_nodes = self.step1_find_all_possible_nodes()  # Locate the most possible nodes of each recorded lat&long positions and shrink to the dataframe with only node

            self.lst_mapped_nodes = self.step2_link_all_nodes()  # find the most possible link route

            # self.lst_mapped_nodes = self.step3_remove_replicate_behavior(self.lst_mapped_nodes)# Remove the node path that replicated

            self.lst_mapped_nodes = self.step4_check_if_no_u_turn()  # Check if there is an u-turn behavior and revise

            self.lst_mapped_nodes = self.step5_double_check_if_linked() # Check if every nodes linked

            self.df_mapped_nodes = pd.DataFrame({'nodes': self.lst_mapped_nodes,
                                                 'call_sign': [self.call_sign] * len(self.lst_mapped_nodes),
                                                 'operation': [self.operation] * len(self.lst_mapped_nodes),
                                                 'reg_num': [self.reg_num] * len(self.lst_mapped_nodes),
                                                 'trajectory_id': [self.trajectory_id] * len(self.lst_mapped_nodes),
                                                 'ramp_id': [self.ramp_id] * len(self.lst_mapped_nodes),
                                                 'rwy_id': [self.rwy_id] * len(self.lst_mapped_nodes),
                                                 'start_time_stamp':[self.start_timestamp]* len(self.lst_mapped_nodes),
                                                 'end_time_stamp':[self.end_timestamp]* len(self.lst_mapped_nodes),
                                                 'time_second': list(range(len(self.lst_mapped_nodes)))})
            return self.df_mapped_nodes

    def preprocessing_trajectory_ramp(self):  # Remove the trajectory around the gate and added the information about the gate if the trajectory was very close to gate (buffer 50)
        ramp_count = 0
        for idx in self.df_tra_ori.index:
            x = self.df_tra_ori.loc[idx, 'longitude']
            y = self.df_tra_ori.loc[idx, 'latitude']
            point_x, point_y = transform(self.lla, self.utm, x, y)
            point = geometry.Point(point_x, point_y)
            list_ramps = list(self.dict_ramps_polygon.values())
            list_ramps_contain = [x[0].contains(point) for x in list_ramps]
            if True in list_ramps_contain:
                ramp_count = ramp_count + 1
                if ramp_count == 1:
                    self.ramp_id = list_ramps_contain.index(True)
        for idx in reversed(self.df_tra_ori.index):
            x = self.df_tra_ori.loc[idx, 'longitude']
            y = self.df_tra_ori.loc[idx, 'latitude']
            point_x, point_y = transform(self.lla, self.utm, x, y)
            point = geometry.Point(point_x, point_y)
            list_ramps = list(self.dict_ramps_polygon.values())
            list_ramps_contain = [x[0].contains(point) for x in list_ramps]
            if True in list_ramps_contain:
                if list_ramps_contain.index(True) == self.ramp_id:
                    self.df_tra_ori = self.df_tra_ori.loc[idx:, :].copy()
                    break

        if not len(self.df_tra_ori) < 50:  # Remove the trajectory too short
            # visualize_map_and_user_defined_nodes(self.df_tra_ori['latitude'].tolist(),self.df_tra_ori['longitude'].tolist(),if_utm=True) # Visualize the removed ramp area's trajectory
            # plt.show()
            return self.df_tra_ori
        else:
            return None

    def angle(self, directions):
        """Return the angle between vectors
        """
        directions = np.array(directions, dtype=np.float64)
        vec2 = directions[1:]
        vec1 = directions[:-1]

        norm1 = np.sqrt((vec1 ** 2).sum(axis=1))
        norm2 = np.sqrt((vec2 ** 2).sum(axis=1))
        cos = (vec1 * vec2).sum(axis=1) / (norm1 * norm2)
        return np.arccos(cos)

    def preprocessing_trajectory_rdp(self):

        trajectory = self.df_tra_ori[['utm_x', 'utm_y']].to_numpy()

        # Build simplified (approximated) trajectory
        # using RDP algorithm.
        mask = rdp(trajectory, epsilon=10, return_mask=True)
        simplified_trajectory = trajectory[mask]
        mask_index = np.where(mask)
        df_select_points = self.df_tra_ori.iloc[mask_index]
        sx = trajectory[mask].T[0]
        sy = trajectory[mask].T[1]

        # Define a minimum angle to treat change in direction
        # as significant (valuable turning point).
        # min_angle = np.pi / 15
        min_angle = np.pi / 36

        # Compute the direction vectors on the simplified_trajectory.
        directions = np.diff(simplified_trajectory, axis=0)
        theta = self.angle(directions)

        # Select the index of the points with the greatest theta.
        # Large theta is associated with greatest change in direction.
        idx = np.where(theta > min_angle)[0] + 1

        # Store the x and y information for the points with the larger angle
        # Add the start point
        # processed_trajectory_x = [sx[0]]
        # processed_trajectory_y = [sy[0]]
        #
        # processed_trajectory_x.extend(sx[idx])
        # processed_trajectory_y.extend(sy[idx])
        #
        # # Add the end point
        # processed_trajectory_x.append(sx[-1])
        # processed_trajectory_y.append(sy[-1])

        self.df_tra_rdp = pd.concat([df_select_points.iloc[[0]], df_select_points.iloc[idx],df_select_points.iloc[[-1]]])

        # # Store the infomation in the df_tra_rdp
        # self.df_tra_rdp = pd.DataFrame({'utm_x': processed_trajectory_x, 'utm_y': processed_trajectory_y})
        #
        # # self.df_tra_rdp = pd.DataFrame({'utm_x':sx,'utm_y':sy})
        #
        # # Store the lat and long for the df_tra_rdp
        # self.df_tra_rdp.loc[:, 'longitude'] = ''
        # self.df_tra_rdp.loc[:, 'latitude'] = ''
        #
        # for i in range(len(self.df_tra_rdp)):
        #     self.df_tra_rdp.iloc[i, self.df_tra_rdp.columns.get_loc('longitude')], self.df_tra_rdp.iloc[i, self.df_tra_rdp.columns.get_loc('latitude')] = transform(
        #         self.utm, self.lla, self.df_tra_rdp.iloc[i, self.df_tra_rdp.columns.get_loc('utm_x')], self.df_tra_rdp.iloc[i, self.df_tra_rdp.columns.get_loc('utm_y')])

        # Visualize_trajectory_preprocessing_result
        # figure, ax = visualize_projected_graph_and_ori_tra_with_rdp_results(self.df_tra_ori, self.G)
        # ax.plot(self.df_tra_rdp['utm_x'].values,self.df_tra_rdp['utm_y'].values, 'b-', label='simplified trajectory')
        # ax.plot(self.df_tra_rdp['utm_x'].values, self.df_tra_rdp['utm_y'].values, 'yo', markersize=7, label='turning points')
        # plt.show()

        start = self.df_tra_rdp['time_stamp'].iloc[0]
        end = self.df_tra_rdp['time_stamp'].iloc[-1]

        self.start_timestamp = min(start, end)
        self.end_timestamp = max(start, end)

        return self.df_tra_rdp

    def step1_find_all_possible_nodes(self):  # retrieve_k is the number of k nearest nodes

        X = self.df_tra_rdp['utm_x'].values
        Y = self.df_tra_rdp['utm_y'].values

        # query the tree for the k nearest node to each point
        points = np.array([X, Y]).T

        dist, k_idx = self.ckdtree_airport.query(points, k=self.retrieve_k)

        nn = np.empty(shape=[X.shape[0], self.retrieve_k])

        for i in range(k_idx.shape[0] - 1):
            nn[i, :] = self.df_G_nodes.iloc[k_idx[i]].index.values

        # rwy process
        rwy_dist, rwy_k_idx = self.ckdtree_rwy.query(points[-1], k=self.retrieve_k)
        rwy_nodes = self.df_rwy_nodes.iloc[rwy_k_idx].index.values
        nn[-1, :] = rwy_nodes

        # find rwy id
        rwy_node_1 = rwy_nodes[0]
        lst_rwy_neighbor = list(self.G.neighbors(rwy_node_1))

        for rwy_i in range(len(lst_rwy_neighbor)):
            rwy_node_2 = lst_rwy_neighbor[rwy_i]
            try:
                ref = self.G[rwy_node_1][rwy_node_2][0]['ref']
            except:
                ref = self.G[rwy_node_2][rwy_node_1][0]['ref']
            if isinstance(ref, list):
                for ref_i in ref:
                    if ref_i.find('/') != -1:
                        self.rwy_id = ref_i
                        break
                break
            if ref.find('/') != -1:
                self.rwy_id = ref
                break
            else:
                continue
        # print(nn)
        return nn

    def step2_link_all_nodes(self):  # Locate and link the most possible nodes of each recorded lat&long positions

        self.lst_mapped_nodes = []

        self.lst_mapped_nodes.append(self.np_rdp_mapped_nodes[0, 0])


        # locate the most most possible node
        for i in range(1, self.np_rdp_mapped_nodes.shape[0] - 1):

            predict_next_node = self.np_rdp_mapped_nodes[i + 1][0]  # use the nearest node for the next node

            previous_node = self.lst_mapped_nodes[-1]

            length_route_1_pre_to_cur, route_1_pre_to_cur = nx.bidirectional_dijkstra(self.G, source=int(previous_node), target=int(self.np_rdp_mapped_nodes[i][0]), weight='length')
            length_route_1_cur_to_next, route_1_cur_to_next = nx.bidirectional_dijkstra(self.G, source=int(self.np_rdp_mapped_nodes[i][0]), target=int(predict_next_node), weight='length')

            length_route_1 = length_route_1_pre_to_cur + length_route_1_cur_to_next

            lst_route_1 = []

            lst_route_1.extend(route_1_pre_to_cur)
            if len(route_1_cur_to_next)>1:
                lst_route_1.extend(route_1_cur_to_next[1:])

            find_diff_route = False
            nearest_k_ind = 1
            while not find_diff_route:
                lst_route_2 = []
                length_route_2_pre_to_cur, route_2_pre_to_cur = nx.bidirectional_dijkstra(self.G, source=self.lst_mapped_nodes[-1], target=int(self.np_rdp_mapped_nodes[i][nearest_k_ind]),
                                                                                          weight='length')
                length_route_2_cur_to_next, route_2_cur_to_next = nx.bidirectional_dijkstra(self.G, source=int(self.np_rdp_mapped_nodes[i][nearest_k_ind]), target=int(predict_next_node),
                                                                                            weight='length')

                length_route_2 = length_route_2_pre_to_cur + length_route_2_cur_to_next

                lst_route_2.extend(route_2_pre_to_cur)
                if len(route_2_cur_to_next) > 1:
                    lst_route_2.extend(route_2_cur_to_next[1:])


                if length_route_2 < length_route_1:
                    if lst_route_1 == lst_route_2: # prevent float error
                        nearest_k_ind = nearest_k_ind + 1
                        if nearest_k_ind == self.retrieve_k:
                            find_diff_route = True
                            self.lst_mapped_nodes.append(self.np_rdp_mapped_nodes[i][0])
                    else:
                        self.lst_mapped_nodes.append(self.np_rdp_mapped_nodes[i][nearest_k_ind])
                        find_diff_route = True
                elif length_route_1 < length_route_2:
                    self.lst_mapped_nodes.append(self.np_rdp_mapped_nodes[i][0])
                    find_diff_route = True
                else:
                    nearest_k_ind = nearest_k_ind + 1
                    if nearest_k_ind == self.retrieve_k:
                        find_diff_route = True
                        self.lst_mapped_nodes.append(self.np_rdp_mapped_nodes[i][0])

        self.lst_mapped_nodes.append(self.np_rdp_mapped_nodes[-1, 0])  # append the end node's nearest node
        # print(self.lst_mapped_nodes)
        self.lst_mapped_nodes = self.step3_remove_replicate_behavior(self.lst_mapped_nodes)
        # print(self.lst_mapped_nodes)
        lst_mapped_route_nodes = self.lst_mapped_nodes.copy()
        self.lst_mapped_nodes = []

        for i in range(len(lst_mapped_route_nodes) - 1):
            shortest_route = nx.shortest_path(self.G, source=lst_mapped_route_nodes[i], target=lst_mapped_route_nodes[i + 1], weight='length')
            if len(shortest_route) > 1:
                self.lst_mapped_nodes.extend(shortest_route[:-1])
        self.lst_mapped_nodes.append(lst_mapped_route_nodes[-1])
        # print(self.lst_mapped_nodes)
        return self.lst_mapped_nodes

    def removeDuplicates(self, S):
        st = []
        record_remove = []
        i = 0
        while i < len(S):
            if len(st) != 0 and st[-1] == S[i]:
                record_remove.append(i)
                i += 1
                st.pop(-1)
            else:
                st.append(S[i])
                i += 1
        return record_remove

    def identify_remove(self, lst):
        if len(lst) == 1:
            return lst
        else:
            lst_remove_idx = []

            for i in reversed(range(1, len(lst))):
                lst_remove_idx.append(lst[0] - i)
            lst_remove_idx.extend(lst)
            return lst_remove_idx

    def step3_remove_replicate_behavior(self, lst):  # Remove the node path that replicated
        # print(self.lst_mapped_nodes)
        remove_record = self.removeDuplicates(lst)
        remove_record_idx = []
        for k, g in groupby(enumerate(remove_record), lambda ix: ix[0] - ix[1]):
            remove_record_idx.append(list(map(itemgetter(1), g)))
        remove_record_idx = [self.identify_remove(x) for x in remove_record_idx]
        remove_record_idx = [item for sublist in remove_record_idx for item in sublist]

        for i in range(len(remove_record_idx)):
            self.lst_mapped_nodes[remove_record_idx[i]] = ''
        self.lst_mapped_nodes = [x for x in self.lst_mapped_nodes if not x == '']
        # print(self.lst_mapped_nodes)
        return self.lst_mapped_nodes

    def step4_check_if_no_u_turn(self):
        # print('step4:')
        # print(self.lst_mapped_nodes)

        remove_record_idx = []
        for i in range(len(self.lst_mapped_nodes)):
            # print('len:{}'.format(len(self.lst_mapped_nodes)))
            # odd num of u-turn
            if i > 0:
                for k in range(1, i):
                    if ((i - k) >= 0) and ((i + k) < (len(self.lst_mapped_nodes) - 1)):
                        # print('i:{}'.format(i))
                        # print('k:{}'.format(k))

                        if self.lst_mapped_nodes[(i - k)] == self.lst_mapped_nodes[(i + k)]:
                            count_check_i = 0
                            remove = True
                            for remove_check_ind in range(2 * k - 1):
                                count_check_i = count_check_i + 1
                                if self.lst_mapped_nodes[(i - k + count_check_i)] == self.lst_mapped_nodes[(i + k - count_check_i)]:
                                    continue
                                else:
                                    remove = False
                                    break
                            if remove == True:
                                remove_record_idx.extend(list(range((i - k), (i + k))))

            # even num of u-turn
            if i > 1:
                if self.lst_mapped_nodes[i - 1] == self.lst_mapped_nodes[i]:
                    for k in range(0, i):
                        if i - k - 1 >= 0 and i + k < len(self.lst_mapped_nodes):
                            if self.lst_mapped_nodes[i - k - 1] == self.lst_mapped_nodes[i + k]:
                                remove_record_idx.extend(list(range((i - k - 1), (i + k))))

        for i in range(len(remove_record_idx)):
            self.lst_mapped_nodes[remove_record_idx[i]] = ''

        # print(self.lst_mapped_nodes)
        self.lst_mapped_nodes = [x for x in self.lst_mapped_nodes if not x == '']

        # print(self.lst_mapped_nodes)

        if self.operation == 'A':
            self.lst_mapped_nodes = self.lst_mapped_nodes[::-1]
        # print(self.lst_mapped_nodes)
        return self.lst_mapped_nodes

    def step5_double_check_if_linked(self):
        lst_final_nodes = []
        lst_final_nodes.append(self.lst_mapped_nodes[0])
        for i in range(len(self.lst_mapped_nodes)-1):
            node_1 = self.lst_mapped_nodes[i]
            node_2 = self.lst_mapped_nodes[i+1]
            if node_2 in self.G.neighbors(node_1):
                lst_final_nodes.append(node_2)
            else:
                shortest_route = nx.shortest_path(self.G, source=node_1, target=node_2, weight='length')
                lst_final_nodes.extend(shortest_route[1:])
        return lst_final_nodes





def get_file_name(folder_name):
    files_location = os.path.join(os.getcwd(), folder_name)
    lst_file_name = []

    for file_name in os.listdir(files_location):
        file = os.path.join(files_location, file_name)
        lst_file_name.append(file)

    return lst_file_name



if __name__ == '__main__':
    airport = 'KLAX'
    # call_sign = 'EVA015'
    # reg_num = '8990d7'
    # operation = 'D'
    # trajectory_id = 1374
    trajectory_id = 1175



    start_time = time.time()

    run_name = os.path.basename(__file__).split('.')[0]
    today = date.today().strftime("_%d%m%Y_")

    '''Load Map, Nodes'''
    print(os.getcwd())
    G = ox.load_graphml('../../Data/' + airport+'/devide_' + airport + '.graphml')

    Gp = ox.project_graph(G)
    Gp = Gp.to_undirected()

    G_info = Graph_map_infor(Gp)

    '''Revise the graph's dtypes to make length is float'''
    Gp = G_info.convert_node_attr_types()
    Gp = G_info.convert_edge_attr_types()

    identify_rwy_ramp = Identify_rwy_ramp(Gp, airport)
    dict_ramps_polygon = identify_rwy_ramp.identify_runway_and_ramp_polygon()

    # Create ckdtree
    df_G_nodes = pd.DataFrame(
        {"x": nx.get_node_attributes(Gp, "x"), "y": nx.get_node_attributes(Gp, "y")})
    ckdtree_airport = cKDTree(data=df_G_nodes[["x", "y"]], compact_nodes=True, balanced_tree=True)

    df_map = pd.read_csv('../../Data/' + airport + '/' + airport + '_devide_map_graphml_based.csv')
    df_rwy_edge = df_map[df_map['ref'].str.contains("/", na=False)]
    df_map_index = []

    for i in range(len(df_rwy_edge)):
        df_map_index.append(df_rwy_edge['source'].iloc[i])
        df_map_index.append(df_rwy_edge['target'].iloc[i])

    df_rwy_nodes = df_G_nodes.loc[df_map_index, :].copy()
    ckdtree_rwy = cKDTree(data=df_rwy_nodes[["x", "y"]], compact_nodes=True, balanced_tree=True)

    # folder_name = 'Data\\rewrite_20190905_20190919'
    # lst_file_name = get_file_name(folder_name)
    #
    # for i in range(len(lst_file_name)):
    #     file_name = lst_file_name[i]
    # file_name = '../../Data/' + airport + '/extract_ground_aircraft/IFF_LAX+ASDEX_20170910_080102_86328_id.csv'
    file_name = '../../Data/' + airport + '/extract_ground_aircraft/IFF_LAX+ASDEX_20170905_080008_86356_id.csv'

    df_all_tra_ori = pd.read_csv(file_name, low_memory=False)
    # df_tra = df_all_tra_ori[(df_all_tra_ori['operation'] == 'D') & (df_all_tra_ori['reg_num'] == 'ad646e') & (df_all_tra_ori['call_sign'] =='JBU524')]
    df_tra = df_all_tra_ori[(df_all_tra_ori['trajectory_id'] == trajectory_id)]


    mapping = Mapping(Gp, airport, ckdtree_airport, ckdtree_rwy, df_rwy_nodes, df_G_nodes, dict_ramps_polygon, df_tra, retrieve_k=5)
    df_mapped_df_edge_info = mapping.ordered_process()
    # print(len(df_tra))
    print(df_mapped_df_edge_info)
    print("--- %s seconds ---" % (time.time() - start_time))

    visulaization_single_tra_and_mapped_tra_for_module(df_tra, Gp, df_mapped_df_edge_info)
    plt.show()
    print(df_mapped_df_edge_info['nodes'].tolist())






