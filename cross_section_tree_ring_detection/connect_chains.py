#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2023 Author(s) Henry Marichal (hmarichal93@gmail.com

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
from pathlib import Path
from typing import List, Tuple

import cross_section_tree_ring_detection.chain as ch
from cross_section_tree_ring_detection.interpolation_nodes import compute_interpolation_domain, \
    domain_interpolation
from cross_section_tree_ring_detection.basic_properties import similarity_conditions, \
    exist_chain_overlapping, InfoVirtualBand


DEBUG = False

NOT_REPETING_CHAIN = -1
MODULE_NAME = 'connect_chains'


def extract_border_chain_from_list(ch_s: List[ch.Chain], nodes_s: List[ch.Node]):
    """
    Extract border chain from chain and nodes list
    @param ch_s: chain list.
    @param nodes_s: node list
    @return:
    """
    ch_s_without_border = [chain for chain in ch_s if chain.type != ch.TypeChains.border]
    try:
        border_chain = next(chain for chain in ch_s if (chain.type == ch.TypeChains.border))
    except StopIteration:
        border_chain = None
    nodes_s_without_border = [node for node in nodes_s if node.chain_id != border_chain.id] if border_chain is not None \
        else []
    return border_chain, ch_s_without_border, nodes_s_without_border


class ConnectParameters:
    """Class for grouping all the parameter from table 1 in the paper."""
    iterations = 9
    params = {'th_radial_tolerance': [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.2],
              'neighbourhood_size': [10, 10, 22, 22, 45, 45, 22, 45, 45],
              'th_regular_derivative': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2, 2, 2],
              'th_distribution_size': [2, 2, 3, 3, 3, 3, 2, 3, 3],
              'derivative_from_center': [False, False, False, False, False, False, True, True, True]}

    def __init__(self, ch_s, nodes_s):
        self.border_chain, self.ch_s_without_border, self.nodes_s_without_border = extract_border_chain_from_list(ch_s,
                                                                                                                  nodes_s)


    def get_iteration_parameters(self, counter):
        iteration_params = {'th_radial_tolerance': self.params['th_radial_tolerance'][counter],
                            'th_distribution_size': self.params['th_distribution_size'][counter],
                            'neighbourhood_size': self.params['neighbourhood_size'][counter],
                            'th_regular_derivative': self.params['th_regular_derivative'][counter],
                            'derivative_from_center': self.params['derivative_from_center'][counter],
                            'l_ch_s': self.ch_s_without_border if counter < self.iterations - 1 else self.ch_s_without_border + [self.border_chain],
                            'l_nodes_s': self.nodes_s_without_border if counter < self.iterations - 1 else self.nodes_s_without_border + self.border_chain.l_nodes
                            }

        if counter == self.iterations - 1:
            self.border_chain.change_id(len(self.ch_s_without_border))



        return iteration_params

    def update_list_for_next_iteration(self, ch_c, nodes_c):
        self.ch_s_without_border, self.nodes_s_without_border = ch_c, nodes_c

def copy_chains_and_nodes(ch_s):
    ch_s = [ch.copy_chain(chain) for chain in ch_s]
    nodes_s = []
    for chain in ch_s:
        nodes_s += chain.l_nodes

    return  ch_s, nodes_s
def connect_chains(l_ch_s, cy, cx, nr, debug, debut_im_pre, output_dir):
    """
    Logic to connect chains. Same logic to connect chains is applied several times, smoothing restriction.
    Implements Algorithm 3 in the paper
    @param l_ch_s: chain list
    @param cy: pith y's coordinate
    @param cx: pith x's coordinate
    @param nr: total number of ray
    @param debut_im_pre: segmented gray image
    @return:
    l_ch_s: connected chains
    l_nodes_s: list nodes
    """
    # Copy ch_i and nodes
    l_ch_s, l_nodes_s = copy_chains_and_nodes(l_ch_s)
    # Paper Table 1 parameters initialization
    parameters = ConnectParameters(l_ch_s, l_nodes_s)

    # Line 1
    M = compute_intersection_matrix(l_ch_s, l_nodes_s, Nr=nr)

    # Line 2. Iteration over the parameters, 9 iterations in total
    for i in range(parameters.iterations):
        # Line 3. Get parameters for iteration i
        iteration_params = parameters.get_iteration_parameters(i)

        # Line 4.
        # debug_im_pre is a copy of the input image. It is used to visualize the results of the current iteration for debugging purposes.
        # Algorithm 4 in the paper
        l_ch_c, l_nodes_c, M = connect_chains_main_logic(M=M, cy=cy, cx = cx, nr=nr, debug_imgs=debug, im_pre= debut_im_pre,
                                                         save=f"{output_dir}/output_{i}_", **iteration_params)

        # Line 5. Update chain list and chain node list for next iteration
        parameters.update_list_for_next_iteration(l_ch_c, l_nodes_c)

    # Line 7
    return l_ch_c, l_nodes_c


class SystemStatus:
    def __init__(self, l_ch, l_nodes, M, cy, cx, Nr=360, th_radial_tolerance=0.1, th_distribution_size=2,
                 th_regular_derivative=1.5, neighbourhood_size=45, derivative_from_center=False, debug=False, counter=0,
                 save=None, img=None):
        #initialization
        self.l_nodes_s = l_nodes
        self.l_ch_s = l_ch
        self.__sort_chain_list_and_update_relative_position()

        #system parameters
        self.Nr = Nr
        self.derivative_from_center = derivative_from_center
        self.th_distribution_size = th_distribution_size
        self.debug = debug
        self.neighbourhood_size = neighbourhood_size
        self.M = M
        self.center = [cy, cx]
        self.img = img
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.next_chain_index = 0
        self.iterations_since_last_change = 0
        self.th_radial_tolerance = th_radial_tolerance
        self.label = "system_status"
        self.counter = counter
        self.th_regular_derivative = th_regular_derivative
        self.path = save
        if self.path is not None and self.debug:
            Path(self.path).mkdir(exist_ok=True)

    def get_common_chain_to_both_borders(self, chain: ch.Chain):
        chain_angle_domain = chain.get_dot_angle_values()
        angles_where_there_is_no_nodes = [angle for angle in np.arange(0, 360, 360 / self.Nr) if
                               angle not in chain_angle_domain]
        angles_where_there_is_no_nodes += [chain.extA.angle, chain.extB.angle]
        chains_where_there_is_no_nodes = []
        for ch_i in self.l_ch_s:
            ch_i_angles = ch_i.get_dot_angle_values()
            if np.intersect1d(ch_i_angles, angles_where_there_is_no_nodes).shape[0] == len(angles_where_there_is_no_nodes):
                chains_where_there_is_no_nodes.append(ch_i)

        if len(chains_where_there_is_no_nodes) == 0:
            return None

        nodes_in_ray_a = [cad.get_node_by_angle(chain.extA.angle) for cad in chains_where_there_is_no_nodes]
        nodes_in_ray_a.sort(key=lambda x: ch.euclidean_distance_between_nodes(x, chain.extA))
        id_closest = nodes_in_ray_a[0].chain_id
        return ch.get_chain_from_list_by_id(self.l_ch_s, id_closest)


    def compute_all_elements_needed_to_check_if_exist_chain_overlapping(self, chain):
        ch_i = self.get_common_chain_to_both_borders(chain)
        ch_j_endpoint_node = chain.extB
        ch_k_endpoint_node = chain.extA
        ch_j_endpoint_type = ch.EndPoints.B

        interpolated_nodes = [] #domain interpolation function output
        chain_copy = ch.copy_chain(chain)
        domain_interpolation(ch_i, ch_j_endpoint_node, ch_k_endpoint_node, ch_j_endpoint_type, chain_copy,
                             interpolated_nodes)
        interpolated_nodes_plus_endpoints = [ch_j_endpoint_node] + interpolated_nodes + [ch_k_endpoint_node]

        return ch_i, interpolated_nodes_plus_endpoints, ch_j_endpoint_type

    def fill_chain_if_there_is_no_overlapping(self, chain):
        """
        Algorithm 10 in the supplementary material. Complete chain if there is no overlapping.
        :param chain: chain to be completed if conditions are met
        :return: if conditions are met, chain is completed. Otherwise, nothing is done. Complete means that new nodes
        are added to the chain.
        """
        # Line 2 to 5. Check if chain is too small or is full. If it is, return.
        threshold = 0.9
        if chain.size >= chain.Nr or chain.size < threshold * chain.Nr:
            return

        # Line 6
        ch_i, l_nodes, endpoint_type = \
            self.compute_all_elements_needed_to_check_if_exist_chain_overlapping(chain)

        # Line 7. Algorithm 13 in the paper. Check if there is an overlapping chain.
        exist_chain = exist_chain_overlapping(self.l_ch_s, l_nodes, chain, chain, endpoint_type, ch_i)

        # Line 8
        if exist_chain:
            return

        # Line 11
        self.add_nodes_list_to_system(chain, l_nodes)

        return

    def continue_in_loop(self):
        # If iteration_sin_last_chain is equal to the number of chains, the algorithm stops. This means that
        # all chains have been iterated at least one time and no chains have been connected or interpolated.
        return self.iterations_since_last_change < len(self.l_ch_s)

    def get_next_chain(self):
        """
        Algorithm 13 in the paper. Get next chain to be processed.
        :return: next supported chain
        """
        # Line 2
        ch_i = self.l_ch_s[self.next_chain_index]

        # Line 3
        self.size_l_chain_init = len(self.l_ch_s)

        # Line 4. Algorithm 12 in the paper
        self.fill_chain_if_there_is_no_overlapping(ch_i)

        # Line 5
        return ch_i



    def is_new_dot_valid(self, new_dot):
        if new_dot in self.l_nodes_s:
            return False
        if new_dot.x >= self.height or new_dot.y >= self.width or new_dot.x < 0 or new_dot.y < 0:
            return False

        return True

    def update_chain_neighbourhood(self, l_chains_to_update_neighborhood):
        dummy_chain = None
        for chain_p in l_chains_to_update_neighborhood:
            border = ch.EndPoints.A
            inward_chain, outward_chain, dot_border = get_inward_and_outward_visible_chains(self.l_ch_s, chain_p, border)

            chain_p.A_outward = outward_chain if outward_chain is not None else dummy_chain
            chain_p.A_inward = inward_chain if inward_chain is not None else dummy_chain
            border = ch.EndPoints.B
            inward_chain, outward_chain, dot_border = get_inward_and_outward_visible_chains(self.l_ch_s, chain_p, border)
            chain_p.B_outward = outward_chain if outward_chain is not None else dummy_chain
            chain_p.B_inward = inward_chain if inward_chain is not None else dummy_chain

        return


    def add_nodes_list_to_system(self, chain, l_nodes: List[ch.Node]):
        processed_node_list = []
        for new_dot in l_nodes:
            if chain.id != new_dot.chain_id:
                raise

            if new_dot in self.l_nodes_s:
                exist_at_least_one_endpoint_in_the_list = any([new_dot == chain.extA, new_dot == chain.extB])
                if not exist_at_least_one_endpoint_in_the_list:
                    raise Exception(f"Node {new_dot} is already in the list and it is not an endpoint of the chain")
                continue

            processed_node_list.append(new_dot)


            self.l_nodes_s.append(new_dot)
            # 1.0 Update ch_i list intersection
            chain_id_intersecting, chains_over_radial_direction = self._chains_id_over_radial_direction(
                new_dot.angle)
            self.M[chain.id, chain_id_intersecting] = 1
            self.M[chain_id_intersecting, chain.id] = 1

            # 2.0 Update boundary chains above and below.
            dots_over_direction = [dot for chain in chains_over_radial_direction for dot in chain.l_nodes if
                                   dot.angle == new_dot.angle]
            dots_over_direction.append(new_dot)
            dots_over_direction.sort(key=lambda x: x.radial_distance)
            idx_new_dot = dots_over_direction.index(new_dot)

            up_dot = dots_over_direction[idx_new_dot + 1] if idx_new_dot < len(dots_over_direction) - 1 else None
            if up_dot is not None:
                up_chain = ch.get_chain_from_list_by_id( chain_list= chains_over_radial_direction, chain_id = up_dot.chain_id)
                if up_dot == up_chain.extA:
                    up_chain.A_inward = chain
                elif up_dot == up_chain.extB:
                    up_chain.B_inward = chain

            down_dot = dots_over_direction[idx_new_dot - 1] if idx_new_dot > 0 else None
            if down_dot is not None:
                down_chain = ch.get_chain_from_list_by_id(chain_list=chains_over_radial_direction,
                                                        chain_id=down_dot.chain_id)
                if down_dot == down_chain.extA:
                    down_chain.A_outward = chain
                elif down_dot == down_chain.extB:
                    down_chain.B_outward = chain


        change_border = chain.add_nodes_list(processed_node_list)
        self.update_chain_neighbourhood([chain])

    @staticmethod
    def get_next_chain_index_in_list(chains_list, support_chain):
        return (chains_list.index(support_chain) + 1) % len(chains_list)

    def update_system_status(self, ch_i, l_s_outward, l_s_inward):
        """
        Algorithm 12 in the paper. Update the system state after the iteration.
        @param ch_i: support chain in current iteration
        @param l_s_outward: outward list of chains
        @param l_s_inward: inward list of chains
        @return: self.next_chain_index: index of the next support chain to be processed
        """
        # Line 1. self.l_ch_s is an attribute of the self object.
        # Line 2
        if self._system_status_change():
            # Line 3
            self.l_ch_s.sort(key=lambda x: x.size, reverse=True)
            # Variable used to check if system status has changed in current iteration. If it has (variable is set to 0),
            # the algorithm continues one more iteration. If it has not (variable is increased by 1). Variable is used as
            # exit condition in the while loop (Algorithm 2, line 3, continue_in_loop method).
            self.iterations_since_last_change = 0

            # Line 4
            l_current_iteration = [ch_i] + l_s_outward + l_s_inward

            # Line 5
            l_current_iteration.sort(key=lambda x: x.size, reverse=True)

            # Line 6
            longest_chain = l_current_iteration[0]

            # Line 7 to 12
            if longest_chain.id == ch_i.id:
                next_chain_index = self.get_next_chain_index_in_list(self.l_ch_s, ch_i)
            else:
                next_chain_index = self.l_ch_s.index(longest_chain)


        else:
            # Line 15
            next_chain_index = self.get_next_chain_index_in_list(self.l_ch_s, ch_i)
            # System status has not changed. Increase variable by 1.
            self.iterations_since_last_change += 1

        # Line 17
        self.next_chain_index = next_chain_index

        return 0

    def _chains_id_over_radial_direction(self, angle):
        chains_in_radial_direction = ch.get_chains_within_angle(angle, self.l_ch_s)
        chains_id_over_radial_direction = [cad.id for cad in chains_in_radial_direction]

        return chains_id_over_radial_direction, chains_in_radial_direction

    def __sort_chain_list_and_update_relative_position(self):
        self.l_ch_s = sorted(self.l_ch_s, key=lambda x: x.size, reverse=True)
        self.update_chain_neighbourhood(self.l_ch_s)

    def _system_status_change(self):
        self.chain_size_at_the_end_of_iteration = len(self.l_ch_s)
        return self.size_l_chain_init > self.chain_size_at_the_end_of_iteration


def update_pointer(ch_j, closest, l_candidates_chi):
    ch_j_index = l_candidates_chi.index(ch_j)
    j_pointer = ch_j_index if closest is not None else ch_j_index + 1
    return j_pointer



def iterate_over_chains_list_and_complete_them_if_met_conditions(state):
    for chain in state.l_ch_s:
        state.fill_chain_if_there_is_no_overlapping(chain)

    return state.l_ch_s, state.l_nodes_s, state.M


def debugging_chains(state, chains_to_debug, filename):
    if state.debug:

        ch.visualize_selected_ch_and_chains_over_image_([ch for ch in chains_to_debug if ch is not None], state.l_ch_s,
                                                        img=state.img, filename=filename)
        state.counter += 1


def connect_chains_main_logic(M, cy, cx, nr, l_ch_s, l_nodes_s, th_radial_tolerance=2, th_distribution_size=2,
                              th_regular_derivative=1.5, neighbourhood_size=22, derivative_from_center=False,
                              debug_imgs=False, im_pre=None, save=None):
    """
    Logic for connecting chains based on similarity conditions. Implements Algorithm 4 from the paper.
    @param l_ch_s: list of chains
    @param l_nodes_s: list of nodes belonging to chains
    @param M: matrix of intersections between chains
    @param cy: y coordinate of pith (disk center)
    @param cx: x coordinate of pith (disk center)
    @param th_radial_tolerance: threshold for radial tolerance
    @param th_distribution_size: threshold for distribution size
    @param th_regular_derivative: threshold for regular derivative
    @param neighbourhood_size: size of neighbourhood in which we search for similar chains
    @param derivative_from_center: if true, derivative is calculated from cy, otherwise from support chain
    @param im_pre: image for debug
    @param nr: number of rays
    @param debug_imgs: debug parameter
    @param save: image save locating. Debug only
    @return: nodes and chain list after connecting
    """
    # Line 1. Initialization of the state object. It contains all the information needed to connect chains.
    state = SystemStatus(l_ch_s, l_nodes_s, M, cy, cx, Nr=nr, th_radial_tolerance=th_radial_tolerance,
                         th_distribution_size=th_distribution_size, th_regular_derivative=th_regular_derivative,
                         neighbourhood_size=neighbourhood_size, derivative_from_center=derivative_from_center,
                         debug=debug_imgs, save=save, img=im_pre)

    # Line 2. State.continue_in_loop() check if current state is equal to the previous one. If it is, the algorithm stops.
    # If some chains have been connected in the current iteration, the algorithm continues one more iteration.
    # Additionaly, if nodes have been added to some chain, the algorithm continues one more iteration.
    while state.continue_in_loop():
        # Line 3. Get next chain to be processed. Algorithm 13 in the paper.
        ch_i = state.get_next_chain()

        # Line 4. Get chains in ch_i neighbourhood
        l_s_outward, l_s_inward = get_chains_in_and_out_wards(state.l_ch_s, ch_i)

        # Line 7 to 10 is implemented within the for loop statement.
        for location, l_candidates_chi in zip([ch.ChainLocation.inwards, ch.ChainLocation.outwards], [l_s_inward, l_s_outward]):
            j_pointer = 0
            # Line 11
            while len(l_candidates_chi) > j_pointer:
                # Line 12
                debugging_chains(state, [ch_i] + l_candidates_chi, f'{state.path}/{state.counter}_0_{ch_i.label_id}_{location}.png')
                ch_j = l_candidates_chi[j_pointer]

                # Line 13
                debugging_chains(state, [ch_i, ch_j], f'{state.path}/{state.counter}_1.png')
                l_no_intersection_j = get_non_intersection_chains(state.M, l_candidates_chi, ch_j)

                # Line 14. Algorithm 14 in the paper
                ch_k_b = get_closest_chain_logic(state, ch_j, l_candidates_chi, l_no_intersection_j, ch_i, location,
                                                 ch.EndPoints.B)
                debugging_chains(state, [ch_i, ch_j, ch_k_b], f'{state.path}/{state.counter}_2.png')

                # Line 15. Algorithm 14 in the paper
                ch_k_a = get_closest_chain_logic(state, ch_j, l_candidates_chi, l_no_intersection_j, ch_i, location,
                                                 ch.EndPoints.A)
                debugging_chains(state, [ch_i, ch_j, ch_k_a], f'{state.path}/{state.counter}_3.png')

                # Line 16
                ch_k, endpoint = select_closest_chain(ch_j, ch_k_a, ch_k_b)
                debugging_chains(state, [ch_i, ch_j, ch_k], f'{state.path}/{state.counter}_4.png')

                # Line 17. Algorithm 5 in the paper
                connect_two_chains(state, ch_j, ch_k, l_candidates_chi, endpoint, ch_i)
                debugging_chains(state, [ch_i, ch_j], f'{state.path}/{state.counter}_5.png')

                # Line 18
                j_pointer = update_pointer(ch_j, ch_k, l_candidates_chi)

        # Line 19. Implementing the logic of Algorithm 12
        state.update_system_status(ch_i, l_s_outward, l_s_inward)

    # Line 20
    l_ch_c, l_nodes_c, intersection_matrix = iterate_over_chains_list_and_complete_them_if_met_conditions(state)
    debugging_chains(state, l_ch_c, f'{state.path}/{state.counter}.png')

    # Line 21
    return l_ch_c, l_nodes_c, intersection_matrix


def intersection_chains(M, candidate_chain: ch.Chain, l_sorted_chains_in_neighbourhood):
    inter_next_chain = np.where(M[candidate_chain.id] == 1)[0]
    l_intersections_candidate = [set.cad for set in l_sorted_chains_in_neighbourhood if
                                set.cad.id in inter_next_chain and candidate_chain.id != set.cad.id]

    return l_intersections_candidate
def get_all_chain_in_subset_that_satisfy_condition(state: SystemStatus, ch_j: ch.Chain, ch_i: ch.Chain,
                                                   endpoint: int, radial_distance: float, candidate_chain: ch.Chain,
                                                   l_intersections_candidate):
    l_intersection_candidate_set = [Set(radial_distance, candidate_chain)]
    for chain_inter in l_intersections_candidate:
        pass_control, radial_distance = connectivity_goodness_condition(state, ch_j, chain_inter, ch_i,
                                                                        endpoint)
        if pass_control:
            l_intersection_candidate_set.append(Set(radial_distance, chain_inter))
    return l_intersection_candidate_set

def get_the_closest_chain_by_radial_distance_that_does_not_intersect(state: SystemStatus, ch_j: ch.Chain,
                                                                     ch_i: ch.Chain, endpoint: int,
                                                                     candidate_chain_radial_distance: float,
                                                                     candidate_chain: ch.Chain, M,
                                                                     l_sorted_chains_in_neighbourhood):
    """
    Implements Algorithm 14 in the supplementary material.
    @param state: Data structure with all the information of the system
    @param ch_j: current chain
    @param ch_i: support chain
    @param endpoint: Chj endpoint
    @param candidate_chain_radial_distance: radial distance between Chj and candidate chain
    @param candidate_chain: angular closer chain to Chj
    @param M: intersection matrix
    @param l_sorted_chains_in_neighbourhood: chains in Chj endpoint neighbourhood sorted by angular distance
    @return: closest chain to Chj that satisfies connectivity goodness conditions
    """
    # Line 1 Get all the chains that intersect to candidate_chain
    l_intersections_candidate = intersection_chains(M, candidate_chain, l_sorted_chains_in_neighbourhood)

    # Line 2 Get all the chains that intersect to candidate_chain and satisfy connectivity_goodness_condition with ch_j
    l_intersections_candidate_set = get_all_chain_in_subset_that_satisfy_condition(state, ch_j, ch_i,
                                                                                 endpoint, candidate_chain_radial_distance,
                                                                                 candidate_chain, l_intersections_candidate)
    # Line 3 Sort them by proximity to ch_j
    l_intersections_candidate_set.sort(key=lambda x: x.distance)

    # Line 4 Return ch_k ch_i
    ch_k = l_intersections_candidate_set[0].cad

    # Line 5
    return ch_k

def get_closest_chain(state: SystemStatus, ch_j: ch.Chain, l_no_intersection_j: List[ch.Chain], ch_i: ch.Chain,
                      location: int, endpoint: int, M):
    """
    Search for the closest chain to ch_j that does not intersect with ch_j and met conditions.
    Implements Algorithm 14 from the paper
    @param state: SystemStatus
    @param ch_j: source chain
    @param l_no_intersection_j: list of chains that do not intersect with ch_j. Set of candidate chains to connect with
                                ch_j
    @param ch_i: support chain of ch_j
    @param location: inward or outward ch_j location regarding ch_i
    @param endpoint: ch_j endpoint
    @param M: intersection matrix
    @return: the closest chain to ch_j
    """
    # Line 1 and 2. Sort chains by proximity
    neighbourhood_size = state.neighbourhood_size
    l_sorted_chains_in_neighbourhood = get_chains_in_neighbourhood(neighbourhood_size, l_no_intersection_j, ch_j,
                                                                   ch_i, endpoint, location)

    # Line 3 and 4
    next_id = 0
    ch_k = None

    # Line 5. Search for closest chain to ch_i
    length_chains = len(l_sorted_chains_in_neighbourhood)
    while next_id < length_chains:
        # Line 6
        candidate_chain = l_sorted_chains_in_neighbourhood[next_id].cad

        # Line 7 Algorithm 15 from paper.
        pass_control, radial_distance = connectivity_goodness_condition(state, ch_j, candidate_chain, ch_i,
                                                                        endpoint)

        # Line 8
        if pass_control:
            # Line 9. Check that do not exist other chains that intersect next ch_i that is radially ch_k to ch_j
            # Get chains that intersect next ch_i.
            ch_k = get_the_closest_chain_by_radial_distance_that_does_not_intersect(state, ch_j, ch_i,
                                                                                       endpoint, radial_distance,
                                                                                       candidate_chain, M,
                                                                                       l_sorted_chains_in_neighbourhood)

            break

        # Line 12
        next_id += 1

    # Line 14
    return ch_k


def get_closest_chain_logic(state, ch_j, l_candidates_chi, l_no_intersection_j, ch_i, location, endpoint):
    """
    Get the ch_k chain tha met condition  if it is symmetric. If it is not symmetric return None.
    @param state: System status instance. It contains all the information about the system.
    @param l_candidates_chi: List of chains that can be candidates to be connected to ch_j
    @param ch_j: Chain that is going to be connected to another chain
    @param l_no_intersection_j: List of chains that do not intersect with ch_j
    @param ch_i: Chain that support ch_j
    @param location: Location of ch_j regard to support chain (inward/outward)
    @param endpoint: Endpoint of ch_j that is going to be connected
    @return: closest chain, ch_k, to ch_j that met condition
    """
    # Line 2. Algorithm 14 in the paper
    ch_k = get_closest_chain(state, ch_j, l_no_intersection_j, ch_i, location, endpoint, state.M)

    # Line 3
    if ch_k is None:
        return ch_k

    # Line 6
    l_no_intersection_k = get_non_intersection_chains(state.M, l_candidates_chi, ch_k)

    # Line 7 to 12
    endpoint_k = ch.EndPoints.A if endpoint == ch.EndPoints.B else ch.EndPoints.B

    # Line 13
    symmetric_chain = get_closest_chain(state, ch_k, l_no_intersection_k, ch_i, location, endpoint_k, state.M)

    # Line 14
    ch_k = None if symmetric_chain != ch_j else ch_k
    if ch_k is not None and (ch_k.size + ch_j.size) > ch_k.Nr:
        ch_k = None

    # Line 17
    return ch_k

def move_nodes_from_one_chain_to_another(ch_j, ch_k):
    for node in ch_k.l_nodes:
        node.chain_id = ch_j.id

    change_border = ch_j.add_nodes_list(ch_k.l_nodes)
    return change_border
def generate_new_nodes(state, ch_j, ch_k, endpoint, ch_i):
    ch_j_endpoint = ch_j.extA if endpoint == ch.EndPoints.A else ch_j.extB
    ch_k_endpoint = ch_k.extB if endpoint == ch.EndPoints.A else ch_k.extA

    l_new_nodes = []
    domain_interpolation(ch_i, ch_j_endpoint, ch_k_endpoint, endpoint, ch_j, l_new_nodes)
    state.add_nodes_list_to_system(ch_j, l_new_nodes)
    return

def updating_chain_nodes(state, ch_j, ch_k):
    change_border = move_nodes_from_one_chain_to_another(ch_j, ch_k)
    if change_border:
        state.update_chain_neighbourhood([ch_j])

    return
def delete_closest_chain(state, ch_k, l_candidates_chi):
    cad_2_index = state.l_ch_s.index(ch_k)
    del state.l_ch_s[cad_2_index]
    id_connected_chain = l_candidates_chi.index(ch_k)
    del l_candidates_chi[id_connected_chain]
    return

def update_intersection_matrix(state, ch_j, ch_k):
    inter_cad_1 = state.M[ch_j.id]
    inter_cad_2 = state.M[ch_k.id]
    or_inter_cad1_cad2 = np.logical_or(inter_cad_1, inter_cad_2)
    state.M[ch_j.id] = or_inter_cad1_cad2
    state.M[:, ch_j.id] = or_inter_cad1_cad2
    state.M = np.delete(state.M, ch_k.id, 1)
    state.M = np.delete(state.M, ch_k.id, 0)
    return
def update_chains_ids(state, ch_k):
    for ch_old in state.l_ch_s:
        if ch_old.id > ch_k.id:
            new_id = ch_old.id - 1
            ch_old.change_id(new_id)
    return
def connect_two_chains(state: SystemStatus, ch_j, ch_k, l_candidates_chi, endpoint, ch_i):
    """
    Algorithm 5 in the paper. Connect chains ch_j and ch_k updating all the information about the system.
    @param state: class object that contains all the information about the system.
    @param ch_j: chain j to connect
    @param ch_k: chain k to connect
    @param l_candidates_chi: list of chains that have support chain ch_i
    @param endpoint: endpoint of ch_j that is going to be connected
    @param ch_i: support chain
    @return: None
    """
    # Line 1
    if endpoint is None:
        return

    # Line 4
    if ch_j == ch_k:
        return

    # Line 7 Generate new dots
    generate_new_nodes(state, ch_j, ch_k, endpoint, ch_i)

    # Line 8 move node from one ch_i to another
    updating_chain_nodes(state, ch_j, ch_k)

    # Line 9 update chains
    update_chain_after_connect(state, ch_j, ch_k)

    # Line 10 delete ch_k from  list l_candidate_chi  and state.l_ch_s
    delete_closest_chain(state, ch_k, l_candidates_chi)

    # Line 11 update intersection matrix
    update_intersection_matrix(state, ch_j, ch_k)

    # Line 12 update ch_i ids
    update_chains_ids(state, ch_k)

    return


def get_inward_and_outward_list_chains_via_pointers(l_ch_s:List[ch.Chain], support_chain: ch.Chain):
    """
    Get the inward and outward chains of  ch_i
    @param l_ch_s: list of chains
    @param support_chain: support chain, ch_i,  to get the inward and outward chains
    @return: inward and outward list chains
    """
    l_s_outward = []
    l_s_inward = []
    for ch_cand in l_ch_s:
        if ch_cand == support_chain:
            continue
        a_outward, b_outward, a_inward, b_inward = ch_cand.A_outward, ch_cand.B_outward, ch_cand.A_inward, ch_cand.B_inward

        if (ch_cand not in l_s_outward) and ((a_inward is not None and support_chain is a_inward) or
                                             (b_inward is not None and support_chain is b_inward)):
            l_s_outward.append(ch_cand)

        if (ch_cand not in l_s_inward) and ((a_outward is not None and support_chain is a_outward) or
                                            (b_outward is not None and support_chain is b_outward)):
            l_s_inward.append(ch_cand)

    return l_s_outward, l_s_inward


def get_non_intersection_chains(M, l_candidates_chi, ch_j):
    """
    Get the list of chains that not intersect with ch_j
    @param M: intersection matrix
    @param l_candidates_chi: list of chain
    @param ch_j: chain j
    @return:return the list of chains that not intersect with ch_j
    """
    try:
        id_inter = np.where(M[ch_j.id] == 1)[0]
        candidates_chi_non_chj_intersection = [cad for cad in l_candidates_chi if cad.id not in id_inter]
    except IndexError:
        candidates_chi_non_chj_intersection = []
        print(IndexError)
    return candidates_chi_non_chj_intersection

def get_intersection_chains(M, l_candidates_chi, ch_j):
    """
    Get the list of chain that intersect with ch_j
    @param M: intersection matrix
    @param l_candidates_chi: list of chain
    @param ch_j: chain j
    @return: return the list of chain that intersect with ch_j
    """
    id_inter = np.where(M[ch_j.id] == 1)[0]
    candidates_chi_non_chj_intersection = [cad for cad in l_candidates_chi if cad.id in id_inter]
    return candidates_chi_non_chj_intersection


def remove_chains_if_present_at_both_groups(S_up, S_down):
    up_down = [cad for cad in S_up if cad in S_down]
    for cad in up_down:
        S_up.remove(cad)
    return up_down



def get_chains_in_and_out_wards(l_ch_s, support_chain):
    """
    Get chains inwards and outwards from l_ch_s given support chain, ch_i
    @param l_ch_s: list of chains
    @param support_chain: support chain, ch_i
    @return: list of chains inwards and list of chains outwards
    """
    l_s_outward, l_s_inward = get_inward_and_outward_list_chains_via_pointers(l_ch_s, support_chain)
    remove_chains_if_present_at_both_groups(l_s_outward, l_s_inward)
    return l_s_outward, l_s_inward


def select_closest_chain(chain, a_neighbour_chain, b_neighbour_chain):
    if a_neighbour_chain is not None:
        d_a = distance_between_border(chain, a_neighbour_chain, ch.EndPoints.A)
    else:
        d_a = -1

    if b_neighbour_chain is not None:
        d_b = distance_between_border(chain, b_neighbour_chain,  ch.EndPoints.B)
    else:
        d_b = -1

    if d_a == d_b == -1:
        closest_chain = None
        endpoint = None

    elif d_a >= d_b:
        closest_chain = a_neighbour_chain
        endpoint = ch.EndPoints.A

    elif d_b > d_a:
        closest_chain = b_neighbour_chain
        endpoint = ch.EndPoints.B

    else:
        raise

    return closest_chain, endpoint



class Set:
    def __init__(self, angular_distance, cad):
        self.distance = angular_distance
        self.cad = cad


def get_chains_in_neighbourhood(neighbourhood_size: float, l_no_intersection_j: List[ch.Chain], ch_j: ch.Chain,
                                ch_i: ch.Chain, endpoint: int, location: int) ->List[Set]:
    """
    Get all the chains in the neighbourhood of the chain ch_j included in the list no_intersection_j
    @param neighbourhood_size: angular neighbourhood size
    @param l_no_intersection_j: list of chains that do not intersect with ch_j
    @param ch_j: chain j
    @param ch_i: support chain, ch_i
    @param endpoint: ch_j endpoint
    @param location: inward or outward location
    @return: list of chains in the neighbourhood of ch_j
    """
    l_chains_in_neighbourhood = []
    for cand_chain in l_no_intersection_j:
        angular_distance = ch.angular_distance_between_chains(ch_j, cand_chain, endpoint)
        if angular_distance < neighbourhood_size and cand_chain.id != ch_j.id:
            l_chains_in_neighbourhood.append(Set(angular_distance, cand_chain))

    if endpoint == ch.EndPoints.A and location == ch.ChainLocation.inwards:
        l_chains_in_neighbourhood = [element for element in l_chains_in_neighbourhood if element.cad.B_outward == ch_i]

    elif endpoint == ch.EndPoints.A and location == ch.ChainLocation.outwards:
        l_chains_in_neighbourhood = [element for element in l_chains_in_neighbourhood if
                                   element.cad.B_inward == ch_i]
    elif endpoint == ch.EndPoints.B and location == ch.ChainLocation.inwards:
        l_chains_in_neighbourhood = [element for element in l_chains_in_neighbourhood if element.cad.A_outward == ch_i]

    elif endpoint == ch.EndPoints.B and location == ch.ChainLocation.outwards:
        l_chains_in_neighbourhood = [element for element in l_chains_in_neighbourhood if
                                   element.cad.A_inward == ch_i]

    l_sorted_chains_in_neighbourhood = sort_chains_in_neighbourhood(l_chains_in_neighbourhood, ch_j)

    return l_sorted_chains_in_neighbourhood


def sort_chains_in_neighbourhood(chains_in_neighbourhood: List[Set], ch_j: ch.Chain):
    """
    Sort chains by angular distance. Set of chains with same angular distance, are sorted by euclidean distance to ch_j
    @param chains_in_neighbourhood: list of Sets. A set elements is composed by a chain and a distance between support
     chain and ch_j
    @param ch_j: chain j
    @return: sorted list List[Set]
    """
    sorted_chains_in_neighbourhood = []
    unique_angular_distances = np.unique([conj.distance for conj in chains_in_neighbourhood])
    for d in unique_angular_distances:
        chains_same_angular_distance = [conj.cad for conj in chains_in_neighbourhood if conj.distance == d]
        euclidean_distance_set = [Set(ch.minimum_euclidean_distance_between_chains_endpoints(ch_d, ch_j), ch_d)
                                  for ch_d in chains_same_angular_distance]
        euclidean_distance_set.sort(key=lambda x: x.distance)
        sorted_chains_in_neighbourhood += [Set(d, set.cad) for set in euclidean_distance_set]
    return sorted_chains_in_neighbourhood


def check_endpoints(support_chain: ch.Chain, ch_j: ch.Chain, candidate_chain: ch.Chain, endpoint: int) -> bool:
    """
    Check if the endpoints of the chain ch_j are in the interpolation domain of the support chain, ch_i
    @param support_chain: support chain of ch_j and candidate_chain
    @param ch_j: chain j
    @param candidate_chain:  candidate chain
    @param endpoint: ch_j endpoint
    @return: boolean
    """
    support_chain_angular_domain = support_chain.get_dot_angle_values()
    ext_cad_1 = ch_j.extA if endpoint == ch.EndPoints.A else ch_j.extB
    ext_cad_2 = candidate_chain.extB if endpoint == ch.EndPoints.A else candidate_chain.extA
    interpolation_domain = compute_interpolation_domain(endpoint, ext_cad_1, ext_cad_2, support_chain.Nr)
    intersection = np.intersect1d(interpolation_domain, support_chain_angular_domain)
    return True if len(intersection) == len(interpolation_domain) else False


def connectivity_goodness_condition(state: SystemStatus, ch_j: ch.Chain, candidate_chain: ch.Chain, ch_i: ch.Chain,
                                    endpoint: int) -> Tuple[bool, float]:
    """
    Check if the chain candidate_chain can be connected to the chain ch_j.
    Implements Algorithm 15 of the paper
    @param state: system status
    @param ch_j: chain j
    @param candidate_chain: candidate chain
    @param ch_i: support chain
    @param endpoint: ch_j endpoint
    @return: True if the chain candidate_chain can be connected to the chain ch_j
    """
    # Line 6. Size criterion
    if ch_j.size + candidate_chain.size > ch_j.Nr:
        return (False, -1)

    # Line 7. Connect chains by correct endpoint
    check_pass = check_endpoints(ch_i, ch_j, candidate_chain, endpoint)
    if not check_pass:
        return (False, -1)

    # Line 8. Radial check
    check_pass, distribution_distance = similarity_conditions(state, state.th_radial_tolerance,
                                                              state.th_distribution_size, state.th_regular_derivative,
                                                              state.derivative_from_center, ch_i, ch_j, candidate_chain,
                                                              endpoint)
    # Line 9
    return (check_pass, distribution_distance)


def get_ids_chain_intersection(state, chain_id):
    ids_interseccion = list(np.where(state.M[chain_id] == 1)[0])
    ids_interseccion.remove(chain_id)
    return ids_interseccion


def distance_between_border(chain_1, chain_2, border_1):
    node1 = chain_1.extA if border_1 == ch.EndPoints.A else chain_2.extB
    node2 = chain_2.extB if border_1 == ch.EndPoints.A else chain_2.extA
    d = ch.euclidean_distance_between_nodes(node1, node2)
    return d



def get_inward_and_outward_visible_chains(chain_list: List[ch.Chain], chain: ch.Chain, endpoint: str):
    node_direction = chain.extA if endpoint == ch.EndPoints.A else chain.extB
    inward_chain = None
    outward_chain = None
    dot_chain_index, dots_over_ray_direction = get_dots_in_radial_direction(node_direction, chain_list)
    if dot_chain_index < 0:
        return None, None, node_direction

    if dot_chain_index > 0:
        down_dot = dots_over_ray_direction[dot_chain_index - 1]
        inward_chain = ch.get_chain_from_list_by_id(chain_list, down_dot.chain_id)

    if len(dots_over_ray_direction) - 1 > dot_chain_index:
        up_dot = dots_over_ray_direction[dot_chain_index + 1]
        outward_chain = ch.get_chain_from_list_by_id(chain_list, up_dot.chain_id)

    return inward_chain, outward_chain, node_direction


def get_dots_in_radial_direction(node_direction: ch.Node, chain_list: List[ch.Chain]):
    chains_in_radial_direction = ch.get_chains_within_angle(node_direction.angle, chain_list)
    nodes_over_ray = ch.get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
        chains_in_radial_direction, node_direction.angle)

    list_dot_chain_index = [idx for idx, node in enumerate(nodes_over_ray) if
                            node.chain_id == node_direction.chain_id]
    if len(list_dot_chain_index) > 0:
        dot_chain_index = list_dot_chain_index[0]
    else:
        nodes_over_ray = []
        dot_chain_index = -1

    return dot_chain_index, nodes_over_ray


def update_chain_after_connect(state, ch_j, ch_k):

    for chain in state.l_ch_s:
        if chain.A_outward is not None:
            if chain.A_outward.id == ch_k.id:
                chain.A_outward = ch_j
        if chain.A_inward is not None:
            if chain.A_inward.id == ch_k.id:
                chain.A_inward = ch_j

        if chain.B_outward is not None:
            if chain.B_outward.id == ch_k.id:
                chain.B_outward = ch_j

        if chain.B_inward is not None:
            if chain.B_inward.id == ch_k.id:
                chain.B_inward = ch_j
    return 0


def intersection_between_chains(chain1: ch.Chain, chain2: ch.Chain):
    angle_intersection = [node.angle for node in chain1.l_nodes if chain2.get_node_by_angle(node.angle)]
    return True if len(angle_intersection) > 0 else False

def compute_intersection_matrix(chains_list: List[ch.Chain], nodes_list: List[ch.Node], Nr: int):
    """
    Compute intersection matrix. If chain_i intersection chain_j then img_height[i,j] == img_height[j,i] == 1 else 0
    @param chains_list: chains list
    @param nodes_list: nodes list
    @param Nr: total rays in disk
    @return: img_height: Square matrix of lenght len(l_ch_s).
    """
    M = np.eye(len(chains_list))
    for angle in np.arange(0, 360, 360 / Nr):
        chains_id_over_direction = np.unique([node.chain_id for node in nodes_list if node.angle == angle])
        x, y = np.meshgrid(chains_id_over_direction, chains_id_over_direction)
        M[x, y] = 1

    return M

