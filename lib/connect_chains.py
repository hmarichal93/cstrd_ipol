#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: henry
"""
import numpy as np
from pathlib import Path
from typing import List, Tuple

import lib.chain as ch
from lib.interpolation_nodes import compute_interpolation_domain, \
    domain_interpolation
from lib.basic_properties import similarity_conditions, \
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
    border_chain = next(chain for chain in ch_s if (chain.type == ch.TypeChains.border))
    nodes_s_without_border = [node for node in nodes_s if node.chain_id != border_chain.id]
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
                            'chain_list': self.ch_s_without_border if counter < self.iterations - 1 else self.ch_s_without_border + [self.border_chain],
                            'nodes_list': self.nodes_s_without_border if counter < self.iterations - 1 else self.nodes_s_without_border + self.border_chain.nodes_list
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
        nodes_s += chain.nodes_list

    return  ch_s, nodes_s
def connect_chains(ch_s, nodes_s, cy, cx, nr, im_pre, debug, output_dir):
    """
    Logic to connect chains. Same logic to connect chains is applied several times, smoothing restriction
    @param ch_s: chain list
    @param nodes_s: nodes list.
    @param cy: pith y's coordinate
    @param cx: pith x's coordinate
    @param nr: total number of ray
    @param im_pre: segmented gray image
    @return:
    ch_c: connected chains
    nodes_c: list nodes
    """
    # Copy chain and nodes
    ch_s, nodes_s = copy_chains_and_nodes(ch_s)

    parameters = ConnectParameters(ch_s, nodes_s)

    intersection_matrix = compute_intersection_matrix(ch_s, nodes_s, Nr=nr)

    for counter in range(parameters.iterations):
        iteration_params = parameters.get_iteration_parameters(counter)

        ch_c, nodes_c, intersection_matrix = connect_chains_main_logic(intersections_matrix=intersection_matrix,
                                                                       img_center=[cy, cx], debug_imgs=debug,
                                                                       save = f"{output_dir}/output_{counter}_", nr=nr,
                                                                       img=im_pre, **iteration_params)

        parameters.update_list_for_next_iteration(ch_c, nodes_c)


    return ch_c, nodes_c


class SystemStatus:
    def __init__(self, dots_list, chains_list, intersection_matrix, center, img, neighbourhood_size=45,
                 th_radial_tolerance=0.1, debug=False, th_distribution_size=2, derivative_from_center=False,
                 th_regular_derivative=1.5, Nr=360, save=None, counter=0):
        #initialization
        self.dots_list = dots_list
        self.chains_list = chains_list
        self.__sort_chain_list_and_update_relative_position()

        #system parameters
        self.Nr = Nr
        self.derivative_from_center = derivative_from_center
        self.th_distribution_size = th_distribution_size
        self.debug = debug
        self.neighbourhood_size = neighbourhood_size
        self.intersections_matrix = intersection_matrix
        self.center = center
        self.img = img
        self.M = img.shape[0]
        self.N = img.shape[1]
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
        for ch_i in self.chains_list:
            ch_i_angles = ch_i.get_dot_angle_values()
            if np.intersect1d(ch_i_angles, angles_where_there_is_no_nodes).shape[0] == len(angles_where_there_is_no_nodes):
                chains_where_there_is_no_nodes.append(ch_i)

        if len(chains_where_there_is_no_nodes) == 0:
            return None

        nodes_in_ray_a = [cad.get_node_by_angle(chain.extA.angle) for cad in chains_where_there_is_no_nodes]
        nodes_in_ray_a.sort(key=lambda x: ch.euclidean_distance_between_nodes(x, chain.extA))
        id_closest = nodes_in_ray_a[0].chain_id
        return ch.get_chain_from_list_by_id(self.chains_list, id_closest)

    def fill_chain_using_support_chain(self, support_chain, ch1):
        ch1_endpoint = ch1.extB
        ch2_endpoint = ch1.extA
        endpoint = ch.EndPoints.B
        generated_list_nodes = []
        domain_interpolation(support_chain, ch1_endpoint, ch2_endpoint, endpoint, ch1, generated_list_nodes)
        # assert len([dot for dot in ch1.nodes_list if dot in generated_list_nodes]) == 0
        # assert len([node.angle for node in generated_list_nodes if ch1.get_node_by_angle(node.angle)]) == 0

        self.add_list_to_system(ch1, generated_list_nodes)

        return

    def fill_chain_if_there_is_no_intersection(self, chain):
        chain_border = self.get_common_chain_to_both_borders(chain)
        ch1_border = chain.extB
        ch2_border = chain.extA
        enpoint = ch.EndPoints.B
        virtual_nodes = []
        chain_copy = ch.copy_chain(chain)
        domain_interpolation(chain_border, ch1_border, ch2_border, enpoint, chain_copy, virtual_nodes)
        virtual_node_plus_endpoints = [ch1_border] + virtual_nodes + [ch2_border]

        exit_chain = exist_chain_overlapping(self.chains_list, virtual_node_plus_endpoints, chain, chain, enpoint,
                                             chain_border)
        if not exit_chain:
            chain_border = None
            self.fill_chain_using_support_chain(chain_border, chain)


        return 0

    def continue_in_loop(self):
        return self.iterations_since_last_change < len(self.chains_list)

    def get_next_chain(self):
        chain = self.chains_list[self.next_chain_index]
        self.chain_size_at_the_begining_of_iteration = len(self.chains_list)
        if chain.is_full(regions_count=8) and chain.size < chain.Nr:
            self.fill_chain_if_there_is_no_intersection(chain)

        return chain



    def is_new_dot_valid(self, new_dot):
        if new_dot in self.dots_list:
            return False
        if new_dot.x >= self.M or new_dot.y >= self.N or new_dot.x < 0 or new_dot.y < 0:
            return False

        return True

    def update_chain_neighbourhood(self, chains_list_to_update_neighborhood):
        dummy_chain = None
        for chain_p in chains_list_to_update_neighborhood:
            border = ch.EndPoints.A
            inward_chain, outward_chain, dot_border = get_inward_and_outward_visible_chains(self.chains_list, chain_p, border)

            chain_p.A_up = outward_chain if outward_chain is not None else dummy_chain
            chain_p.A_down = inward_chain if inward_chain is not None else dummy_chain
            border = ch.EndPoints.B
            inward_chain, outward_chain, dot_border = get_inward_and_outward_visible_chains(self.chains_list, chain_p, border)
            chain_p.B_up = outward_chain if outward_chain is not None else dummy_chain
            chain_p.B_down = inward_chain if inward_chain is not None else dummy_chain

        return


    def add_list_to_system(self, chain, node_list: List[ch.Node]):
        processed_node_list = []
        for new_dot in node_list:
            if chain.id != new_dot.chain_id:
                raise

            processed_node_list.append(new_dot)
            if new_dot in self.dots_list:
                continue

            self.dots_list.append(new_dot)
            # 1.0 Update chain list intersection
            chain_id_intersecting, chains_over_radial_direction = self._chains_id_over_radial_direction(
                new_dot.angle)
            self.intersections_matrix[chain.id, chain_id_intersecting] = 1
            self.intersections_matrix[chain_id_intersecting, chain.id] = 1

            # 2.0 Update boundary chains above and below.
            dots_over_direction = [dot for chain in chains_over_radial_direction for dot in chain.nodes_list if
                                   dot.angle == new_dot.angle]
            dots_over_direction.append(new_dot)
            dots_over_direction.sort(key=lambda x: x.radial_distance)
            idx_new_dot = dots_over_direction.index(new_dot)

            up_dot = dots_over_direction[idx_new_dot + 1] if idx_new_dot < len(dots_over_direction) - 1 else None
            if up_dot is not None:
                up_chain = ch.get_chain_from_list_by_id( chain_list= chains_over_radial_direction, chain_id = up_dot.chain_id)
                if up_dot == up_chain.extA:
                    up_chain.A_down = chain
                elif up_dot == up_chain.extB:
                    up_chain.B_down = chain

            down_dot = dots_over_direction[idx_new_dot - 1] if idx_new_dot > 0 else None
            if down_dot is not None:
                down_chain = ch.get_chain_from_list_by_id(chain_list=chains_over_radial_direction,
                                                        chain_id=down_dot.chain_id)
                if down_dot == down_chain.extA:
                    down_chain.A_up = chain
                elif down_dot == down_chain.extB:
                    down_chain.B_up = chain


        change_border = chain.add_nodes_list(processed_node_list)
        self.update_chain_neighbourhood([chain])

    @staticmethod
    def _next_chain_index_in_list(chains_list, support_chain):
        return (chains_list.index(support_chain) + 1) % len(chains_list)

    def update_system_state(self, support_chain, S_outward, S_inward):
        self.chain_size_at_the_end_of_iteration = len(self.chains_list)

        if self._state_changes_in_this_iteration():
            self.chains_list = sorted(self.chains_list, key=lambda x: x.size, reverse=True)
            self.iterations_since_last_change = 0

            S_current_iteration = [support_chain] + S_outward + S_inward
            chains_sorted = sorted(S_current_iteration, key=lambda x: x.size, reverse=True)
            if chains_sorted[0].id == support_chain.id:
                self.next_chain_index = self._next_chain_index_in_list(self.chains_list, support_chain)
            else:
                self.next_chain_index = self.chains_list.index(chains_sorted[0])


        else:
            self.iterations_since_last_change += 1
            self.next_chain_index = self._next_chain_index_in_list(self.chains_list, support_chain)

    def _chains_id_over_radial_direction(self, angle):
        chains_in_radial_direction = ch.get_chains_within_angle(angle, self.chains_list)
        chains_id_over_radial_direction = [cad.id for cad in chains_in_radial_direction]

        return chains_id_over_radial_direction, chains_in_radial_direction

    def __sort_chain_list_and_update_relative_position(self):
        self.chains_list = sorted(self.chains_list, key=lambda x: x.size, reverse=True)
        self.update_chain_neighbourhood(self.chains_list)

    def _state_changes_in_this_iteration(self):
        return self.chain_size_at_the_begining_of_iteration > self.chain_size_at_the_end_of_iteration


def update_pointer(ch_j, closest, candidates_chi):
    ch_j_index = candidates_chi.index(ch_j)
    j_pointer = ch_j_index if closest is not None else ch_j_index + 1
    return j_pointer


def interpolate_between_chain_endpoints_if_met_conditions(state):
    for chain_support_ch_i in state.chains_list:
        if chain_support_ch_i.is_full(regions_count=8) and chain_support_ch_i.size < chain_support_ch_i.Nr:
            state.fill_chain_if_there_is_no_intersection(chain_support_ch_i)
    return state.chains_list, state.dots_list, state.intersections_matrix


def connect_chains_main_logic(chain_list, nodes_list, intersections_matrix, img_center, th_radial_tolerance=2,
                              th_distribution_size=2, th_regular_derivative=1.5, neighbourhood_size=22,
                              derivative_from_center=False, debug_imgs=False, nr=360, img=None, save=None):
    """

    @param chain_list:
    @param nodes_list:
    @param intersections_matrix:
    @param img_center:
    @param th_radial_tolerance:
    @param th_distribution_size:
    @param th_regular_derivative:
    @param neighbourhood_size:
    @param derivative_from_center:
    @param debug_imgs:
    @param nr:
    @param img:
    @param save:
    @return:
    """
    state = SystemStatus(nodes_list, chain_list, intersections_matrix, img_center, img,
                         th_radial_tolerance=th_radial_tolerance, th_distribution_size=th_distribution_size,
                         th_regular_derivative=th_regular_derivative, neighbourhood_size=neighbourhood_size,
                         derivative_from_center=derivative_from_center, debug=debug_imgs, Nr=nr, save=save)

    while state.continue_in_loop():
        chain_support_ch_i = state.get_next_chain()
        s_outward, s_inward = get_chains_in_and_out_wards(state.chains_list, chain_support_ch_i)

        for location, candidates_chi in zip([ch.ChainLocation.inwards, ch.ChainLocation.outwards], [s_inward, s_outward]):
            j_pointer = 0
            while len(candidates_chi) > j_pointer:
                if state.debug:
                    ch.visualize_selected_ch_and_chains_over_image_([chain_support_ch_i]+candidates_chi, state.chains_list,
                                                        img=state.img, filename=f'{state.path}/{state.counter}_0_{chain_support_ch_i.label_id}_{location}.png')
                    state.counter += 1

                ch_j = candidates_chi[j_pointer]
                if state.debug:
                    ch.visualize_selected_ch_and_chains_over_image_([chain_support_ch_i, ch_j], state.chains_list, im_pre = state.im_pre,
                                                                     filename = f'{state.path}/{state.counter}_1.png')
                    state.counter += 1

                no_intersection_j = get_non_intersection_chains(state.intersections_matrix, candidates_chi, ch_j)
                closest_b = get_closest_chain_logic(state, candidates_chi, ch_j, no_intersection_j, chain_support_ch_i,
                                                    location, ch.EndPoints.B)
                if state.debug and closest_b is not None:
                    # ch.visualize_selected_ch_and_chains_over_image_([chain_support_ch_i, ch_j, closest_b], state.chains_list, im_pre = state.im_pre,
                    #                                                 filename = f'{state.path}/{state.counter}_2.png')
                    state.counter += 1

                closest_a = get_closest_chain_logic(state, candidates_chi, ch_j, no_intersection_j, chain_support_ch_i,
                                                    location, ch.EndPoints.A)
                if state.debug and closest_a is not None:
                    # ch.visualize_selected_ch_and_chains_over_image_([chain_support_ch_i, ch_j, closest_a], state.chains_list, im_pre = state.im_pre,
                    #                                                 filename = f'{state.path}/{state.counter}_3.png')
                    state.counter += 1

                closest, endpoint = select_closest_chain(ch_j, closest_a, closest_b)
                if state.debug and closest is not None:
                    # ch.visualize_selected_ch_and_chains_over_image_([chain_support_ch_i, ch_j, closest], state.chains_list, im_pre = state.im_pre,
                    #                                                 filename=f'{state.path}/{state.counter}_4.png')
                    state.counter += 1

                connect_two_chains(state, ch_j, closest, candidates_chi, endpoint, chain_support_ch_i)
                if state.debug:
                    # ch.visualize_selected_ch_and_chains_over_image_([chain_support_ch_i, ch_j], state.chains_list, im_pre = state.im_pre,
                    #                                                 filename=f'{state.path}/{state.counter}_5.png')
                    state.counter += 1

                j_pointer = update_pointer(ch_j, closest, candidates_chi)

        state.update_system_state(chain_support_ch_i, s_outward, s_inward)

    ch_f, nodes_s, intersection_matrix = interpolate_between_chain_endpoints_if_met_conditions(state)
    if state.debug:
        ch.visualize_selected_ch_and_chains_over_image_(ch_f, state.chains_list,
                                                        img=state.img, filename=f'{state.path}/{state.counter}.png')
    return ch_f, nodes_s, intersection_matrix


def intersection_chains(intersections_matrix, next_chain: ch.Chain, sorted_chains_in_neighbourhood):
    inter_next_chain = np.where(intersections_matrix[next_chain.id] == 1)[0]
    next_chain_intersections = [set.cad for set in sorted_chains_in_neighbourhood if
                                set.cad.id in inter_next_chain and next_chain.id != set.cad.id]

    return next_chain_intersections
def get_all_chain_in_subset_that_satisfy_condition(state: SystemStatus, ch_j: ch.Chain, chain_support_ch_i: ch.Chain,
                                                   endpoint: int, radial_distance: float, next_chain: ch.Chain,
                                                   next_chain_intersections):
    next_chain_intersection_set = [Set(radial_distance, next_chain)]
    for cad_inter in next_chain_intersections:
        pass_control, radial_distance = connectivity_goodness_condition(state, ch_j, cad_inter, chain_support_ch_i,
                                                                        endpoint)
        if pass_control:
            next_chain_intersection_set.append(Set(radial_distance, cad_inter))
    return next_chain_intersection_set

def get_the_closest_chain_by_radial_distance_that_does_not_intersect(state: SystemStatus, ch_j: ch.Chain,
                                                                     chain_support_ch_i: ch.Chain, endpoint: int,
                                                                     next_chain_radial_distance: float,
                                                                     next_chain: ch.Chain, intersections_matrix,
                                                                     sorted_chains_in_neighbourhood):
    # 1.0 Get all the chains that intersect to next_chain
    next_chain_intersections = intersection_chains(intersections_matrix, next_chain, sorted_chains_in_neighbourhood)
    # 2.0 Get all the chains that intersect to next_chain and satisfy connectivity_goodness_condition with ch_j
    next_chain_intersection_set = get_all_chain_in_subset_that_satisfy_condition(state, ch_j, chain_support_ch_i,
                                                                                 endpoint, next_chain_radial_distance,
                                                                                 next_chain, next_chain_intersections)
    # 3.0 Sort them by proximity to ch_j
    next_chain_intersection_set.sort(key=lambda x: x.distance)

    # Return closest chain
    closest = next_chain_intersection_set[0].cad
    return closest

def get_closest_chain(state: SystemStatus, ch_j: ch.Chain, no_intersection_j: List[ch.Chain],
                      chain_support_ch_i: ch.Chain, location: int, endpoint: int, intersections_matrix):
    """
    Return closest chain in neighbourhood
    @param state: SystemStatus
    @param ch_j: source chain
    @param no_intersection_j: list of chains that do not intersect with ch_j
    @param chain_support_ch_i: support chain of ch_j
    @param location: inward or outward ch_j location
    @param endpoint: ch_j endpoint
    @return: closest chain
    """
    closest = None
    # 1.0 sort chains by proximity
    sorted_chains_in_neighbourhood = get_chains_in_neighbourhood(state.neighbourhood_size, no_intersection_j, ch_j,
                                                                 chain_support_ch_i, endpoint, location)
    lenght_chains = len(sorted_chains_in_neighbourhood)
    if lenght_chains == 0:
        return closest

    # 2.0 search for closest chain
    next_id = 0
    while next_id < lenght_chains:
        next_chain = sorted_chains_in_neighbourhood[next_id].cad
        pass_control, radial_distance = connectivity_goodness_condition(state, ch_j, next_chain, chain_support_ch_i,
                                                                        endpoint)
        if pass_control:
            # 3.0 Check that do not exist other chains that intersect next chain that is radially closest to ch_j
            # Get chains that intersect next chain
            closest = get_the_closest_chain_by_radial_distance_that_does_not_intersect(state, ch_j, chain_support_ch_i,
                                                                                       endpoint, radial_distance,
                                                                                       next_chain, intersections_matrix,
                                                                                       sorted_chains_in_neighbourhood)

            break

        next_id += 1

    return closest


def get_closest_chain_logic(state, candidates_chi, ch_j, no_intersection_j, chain_support_ch_i, location, endpoint):
    """
    Get the closest chain tha met condition  if it is symmetric. If it is not symmetric return None.
    @param state: System status instance. It contains all the information about the system.
    @param candidates_chi: List of chains that can be candidates to be connected to ch_j
    @param ch_j: Chain that is going to be connected to another chain
    @param no_intersection_j: List of chains that do not intersect with ch_j
    @param chain_support_ch_i: Chain that support ch_j
    @param location: Location of ch_j regard to support chain (inward/outward)
    @param endpoint: Endpoint of ch_j that is going to be connected
    @return: closest chain to ch_j that met condition
    """
    closest = get_closest_chain(state, ch_j, no_intersection_j, chain_support_ch_i, location, endpoint,
                                state.intersections_matrix)
    if closest is None:
        return closest
    no_intersection_closest = get_non_intersection_chains(state.intersections_matrix, candidates_chi, closest)
    symmetric_chain = get_closest_chain(state, closest, no_intersection_closest, chain_support_ch_i, location,
                                        ch.EndPoints.A if endpoint == ch.EndPoints.B else ch.EndPoints.B,
                                        state.intersections_matrix)

    closest = None if symmetric_chain != ch_j else closest

    if closest is not None and (closest.size + ch_j.size) > closest.Nr:
        closest = None

    return closest

def move_nodes_from_one_chain_to_another(src, dst):
    for node in dst.nodes_list:
        node.chain_id = src.id

    change_border = src.add_nodes_list(dst.nodes_list)
    return change_border
def generate_new_nodes(state, ch_j, closest, endpoint, support_chain):
    ch1_endpoint = ch_j.extA if endpoint == ch.EndPoints.A else ch_j.extB
    ch2_endpoint = closest.extB if endpoint == ch.EndPoints.A else closest.extA

    new_nodes_list = []
    domain_interpolation(support_chain, ch1_endpoint, ch2_endpoint, endpoint, ch_j, new_nodes_list)
    state.add_list_to_system(ch_j, new_nodes_list)
    return

def updating_chain_nodes(state, ch_j, closest):
    change_border = move_nodes_from_one_chain_to_another(ch_j, closest)
    if change_border:
        state.update_chain_neighbourhood([ch_j])

    return
def delete_closest_chain(state, ch_j, closest, candidates_chi):
    cad_2_index = state.chains_list.index(closest)
    del state.chains_list[cad_2_index]
    id_connected_chain = candidates_chi.index(closest)
    del candidates_chi[id_connected_chain]
    return

def update_intersection_matrix(state, ch_j, closest):
    inter_cad_1 = state.intersections_matrix[ch_j.id]
    inter_cad_2 = state.intersections_matrix[closest.id]
    or_inter_cad1_cad2 = np.logical_or(inter_cad_1, inter_cad_2)
    state.intersections_matrix[ch_j.id] = or_inter_cad1_cad2
    state.intersections_matrix[:, ch_j.id] = or_inter_cad1_cad2
    state.intersections_matrix = np.delete(state.intersections_matrix, closest.id, 1)
    state.intersections_matrix = np.delete(state.intersections_matrix, closest.id, 0)
    return
def update_chains_ids(state, closest):
    for cad_old in state.chains_list:
        if cad_old.id > closest.id:
            new_id = cad_old.id - 1
            cad_old.change_id(new_id)
    return
def connect_two_chains(state: SystemStatus, ch_j, closest, candidates_chi, endpoint, support_chain):
    if endpoint is None:
        return None

    if ch_j == closest:
        return None

    # 1.0 Generate new dots
    generate_new_nodes(state, ch_j, closest, endpoint, support_chain)
    # 2.0 move node from one chain to another
    updating_chain_nodes(state, ch_j, closest)
    # 3.0 update chains
    update_chain_after_connect(state,  ch_j, closest)
    # 4.0 delete closest chain from chain list
    delete_closest_chain(state, ch_j, closest, candidates_chi)
    # 5.0 update intersection matrix
    update_intersection_matrix(state, ch_j, closest)
    # 6.0 update chain ids
    update_chains_ids(state, closest)

    return ch_j


def get_in_and_out_wards_chain_via_pointers(chains_list, chain):
    S_up = []
    S_down = []
    for ch_cand in chains_list:
        if ch_cand == chain:
            continue
        a_up, b_up, a_down, b_down = ch_cand.A_up, ch_cand.B_up, ch_cand.A_down, ch_cand.B_down

        if (ch_cand not in S_up) and ((a_down is not None and chain is a_down) or
                                      (b_down is not None and chain is b_down)):
            S_up.append(ch_cand)

        if (ch_cand not in S_down) and ((a_up is not None and chain is a_up) or
                                        (b_up is not None and chain is b_up)):
            S_down.append(ch_cand)

    return S_up, S_down


def get_non_intersection_chains(intersection_matrix, candidates_chi, ch_j):
    """

    @param intersection_matrix:
    @param candidates_chi:
    @param ch_j:
    @return:
    """
    id_inter = np.where(intersection_matrix[ch_j.id] == 1)[0]
    candidates_chi_non_chj_intersection = [cad for cad in candidates_chi if cad.id not in id_inter]
    return candidates_chi_non_chj_intersection

def get_intersection_chains(intersection_matrix, candidates_chi, ch_j):
    """

    @param intersection_matrix:
    @param candidates_chi:
    @param ch_j:
    @return:
    """
    id_inter = np.where(intersection_matrix[ch_j.id] == 1)[0]
    candidates_chi_non_chj_intersection = [cad for cad in candidates_chi if cad.id  in id_inter]
    return candidates_chi_non_chj_intersection


def remove_chains_if_present_at_both_groups(S_up, S_down):
    up_down = [cad for cad in S_up if cad in S_down]
    for cad in up_down:
        S_up.remove(cad)
    return up_down



def get_chains_in_and_out_wards(chains_list, chain):
    """

    @param chains_list:
    @param chain:
    @return:
    """
    S_up, S_down = get_in_and_out_wards_chain_via_pointers(chains_list, chain)
    remove_chains_if_present_at_both_groups(S_up, S_down)
    return S_up, S_down


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


def get_chains_in_neighbourhood(neighbourhood_size: float, no_intersection_j: List[ch.Chain], ch_j:ch.Chain,
                                chain_support_ch_i:ch.Chain, endpoint:int, location:int)->List[Set]:
    """
    Get all the chains in the neighbourhood of the chain ch_j included in the list no_intersection_j
    @param neighbourhood_size: angular neighbourhood size
    @param no_intersection_j: list of chains that do not intersect with ch_j
    @param ch_j: source chain
    @param chain_support_ch_i: support chain
    @param endpoint: ch_j endpoint
    @param location: inward or outward location
    @return: list of chains in the neighbourhood of ch_j
    """
    chains_in_neighbourhood = []
    for cad in no_intersection_j:
        angular_distance = ch.angular_distance_between_chains_endpoints(ch_j, cad, endpoint)
        if angular_distance < neighbourhood_size and cad.id != ch_j.id:
            chains_in_neighbourhood.append(Set(angular_distance, cad))

    if endpoint == ch.EndPoints.A and location == ch.ChainLocation.inwards:
        chains_in_neighbourhood = [element for element in chains_in_neighbourhood if element.cad.B_up == chain_support_ch_i]

    elif endpoint == ch.EndPoints.A and location == ch.ChainLocation.outwards:
        chains_in_neighbourhood = [element for element in chains_in_neighbourhood if
                                   element.cad.B_down == chain_support_ch_i]
    elif endpoint == ch.EndPoints.B and location == ch.ChainLocation.inwards:
        chains_in_neighbourhood = [element for element in chains_in_neighbourhood if element.cad.A_up == chain_support_ch_i]

    elif endpoint == ch.EndPoints.B and location == ch.ChainLocation.outwards:
        chains_in_neighbourhood = [element for element in chains_in_neighbourhood if
                                   element.cad.A_down == chain_support_ch_i]

    sorted_chains_in_neighbourhood = sort_chains_in_neighbourhood(chains_in_neighbourhood, ch_j)

    return sorted_chains_in_neighbourhood


def sort_chains_in_neighbourhood(chains_in_neighbourhood: List[Set], ch_j: ch.Chain):
    """
    Sort chains by angular distance. Set of chains with same angular distance, are sorted by euclidean distance to ch_j
    @param chains_in_neighbourhood: list of Sets. A set elements is composed by a chain and a distance between chain and ch_j
    @param ch_j: source chain
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


def check_endpoints(support_chain: ch.Chain, ch_j: ch.Chain, next_chain: ch.Chain, endpoint: int) -> bool:
    """
    Check if the endpoints of the chain ch_j are in the interpolation domain of the support chain
    @param support_chain: support chain of ch_j and next_chain
    @param ch_j: source chain
    @param next_chain:  destination chain
    @param endpoint: ch_j endpoint
    @return: boolean
    """
    support_chain_angular_domain = support_chain.get_dot_angle_values()
    ext_cad_1 = ch_j.extA if endpoint == ch.EndPoints.A else ch_j.extB
    ext_cad_2 = next_chain.extB if endpoint == ch.EndPoints.A else next_chain.extA
    interpolation_domain = compute_interpolation_domain(endpoint, ext_cad_1, ext_cad_2, support_chain.Nr)
    intersection = np.intersect1d(interpolation_domain, support_chain_angular_domain)
    return True if len(intersection) == len(interpolation_domain) else False


def connectivity_goodness_condition(state: SystemStatus, ch_j: ch.Chain, next_chain: ch.Chain, support_chain: ch.Chain,
                                    endpoint: int) -> Tuple[bool, float]:
    """
    Check if the chain next_chain can be connected to the chain ch_j
    @param state: system status
    @param ch_j: source chain
    @param next_chain: destination chain
    @param support_chain: support chain
    @param endpoint: ch_j endpoint
    @return: True if the chain next_chain can be connected to the chain ch_j
    """
    # 0. Size criterion
    if ch_j.size + next_chain.size > ch_j.Nr:
        return (False, -1)

    # 1. Connect chains by correct endpoint
    check_pass = check_endpoints(support_chain, ch_j, next_chain, endpoint)
    if not check_pass:
        return (False, -1)

    # 2. Radial check
    check_pass, distribution_distance = similarity_conditions(state, state.th_radial_tolerance, state.th_distribution_size,
                                                              state.th_regular_derivative, state.derivative_from_center,
                                                              support_chain, ch_j, next_chain, endpoint)

    return (check_pass, distribution_distance)


def get_ids_chain_intersection(state, chain_id):
    ids_interseccion = list(np.where(state.intersections_matrix[chain_id] == 1)[0])
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


def update_chain_after_connect(state, cad_1, cad_2):

    for cad in state.chains_list:
        if cad.A_up is not None:
            if cad.A_up.id == cad_2.id:
                cad.A_up = cad_1
        if cad.A_down is not None:
            if cad.A_down.id == cad_2.id:
                cad.A_down = cad_1

        if cad.B_up is not None:
            if cad.B_up.id == cad_2.id:
                cad.B_up = cad_1

        if cad.B_down is not None:
            if cad.B_down.id == cad_2.id:
                cad.B_down = cad_1
    return 0


def intersection_between_chains(chain1: ch.Chain, chain2: ch.Chain):
    angle_intersection = [node.angle for node in chain1.nodes_list if chain2.get_node_by_angle(node.angle)]
    return True if len(angle_intersection) > 0 else False

def compute_intersection_matrix(chains_list: List[ch.Chain], nodes_list: List[ch.Node], Nr: int):
    """
    Compute intersection matrix. If chain_i intersection chain_j then M[i,j] == M[j,i] == 1 else 0
    @param chains_list: chains list
    @param nodes_list: nodes list
    @param Nr: total rays in disk
    @return: M_int: Square matrix of lenght len(chains_list).
    """
    M_int = np.eye(len(chains_list))
    for angle in np.arange(0, 360, 360 / Nr):
        chains_id_over_direction = np.unique([node.chain_id for node in nodes_list if node.angle == angle])
        x, y = np.meshgrid(chains_id_over_direction, chains_id_over_direction)
        M_int[x, y] = 1

    return M_int

