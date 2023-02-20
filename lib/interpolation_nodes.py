import numpy as np
from typing import List

import lib.chain as ch

def compute_interpolation_domain(endpoint:str, endpoint_cad1: ch.Node, endpoint_cad2: ch.Node, Nr):
    interpolation_domain = []

    step = 360 / Nr if endpoint == ch.EndPoints.B else - 360 /  Nr
    current_angle = endpoint_cad1.angle
    while current_angle % 360 != endpoint_cad2.angle:
        current_angle += step
        current_angle = current_angle % 360
        interpolation_domain.append(current_angle)

    return interpolation_domain[:-1]

def from_polar_to_cartesian(r,angulo,centro):
    y = centro[0] + r * np.cos(angulo * np.pi / 180)
    x = centro[1] + r * np.sin(angulo * np.pi / 180)
    return (y,x)

def generate_node_list_between_two_support_chains_and_two_radial_distances(r2_ratio, r1_ratio, total_nodes,
                                                                           interpolation_angle_domain, center,
                                                                           inward_chain, outward_chain, chain):
    cad_id = chain.id
    generated_node_list = []
    m = (r2_ratio - r1_ratio) / total_nodes
    n = r1_ratio
    for idx_current_node, angle in enumerate(interpolation_angle_domain):
        dot_list_in_radial_direction = ch.get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
            [inward_chain], angle % 360)
        support_node = dot_list_in_radial_direction[0]
        radio_init = support_node.radial_distance
        dot_list_in_radial_direction = ch.get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
            [outward_chain], angle % 360)
        support_node = dot_list_in_radial_direction[0]
        radio_superior = support_node.radial_distance
        radial_distance_between_chains = radio_superior - radio_init

        radio_inter =  (m * (idx_current_node) + n)*radial_distance_between_chains + radio_init
        i, j = from_polar_to_cartesian(radio_inter, angle % 360, center)
        i = i if i < chain.M else chain.M - 1
        j = j if j < chain.N else chain.N - 1

        # radios.append(radio_inter)
        params = {
            "x": j,
            "y": i,
            "angle": angle % 360,
            "radial_distance": radio_inter,
            "chain_id": cad_id
        }

        node = ch.Node(**params)
        generated_node_list.append(node)

    return generated_node_list
def generate_nodes_list_between_two_radial_distances(r2, r1, total_nodes, interpolation_angular_domain, center, sign,
                                                     support_chain, chain):
    cad_id = chain.id
    generated_node_list = []
    m = (r2 - r1) / (total_nodes - 0)
    n = r1 - m * 0
    for current_idx_node, angle in enumerate(interpolation_angular_domain):
        if support_chain is not None:
            dot_list_in_radial_direction = ch.get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
                [support_chain], angle % 360)

            support_node = dot_list_in_radial_direction[0]
            radio_init = support_node.radial_distance
            #radio_init = r1
        else:
            radio_init = 0

        radio_inter = sign * (m * (current_idx_node) + n) + radio_init
        i, j = from_polar_to_cartesian(radio_inter, angle % 360, center)
        i = i if i < chain.M else chain.M - 1
        j = j if j < chain.N else chain.N - 1

        generated_node_list.append(ch.Node(**{'x': j, 'y': i, 'angle': angle % 360 , 'radial_distance': radio_inter,
                  'chain_id': cad_id}) )

    return generated_node_list

def get_radial_distance_to_chain(chain, dot):
    dot_list_in_radial_direction = ch.get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
        [chain], dot.angle)
    soporte_pto1 = dot_list_in_radial_direction[0]
    rii = ch.euclidean_distance_between_nodes(soporte_pto1,dot)
    return rii

def compute_radial_ratio(cadena_inferior,cadena_superior, dot):
    r1_inferior = get_radial_distance_to_chain(cadena_inferior, dot)
    r1_superior = get_radial_distance_to_chain(cadena_superior, dot)
    return  r1_inferior / (r1_superior+r1_inferior)
def interpolate_in_angular_domain_via_2_chains(inward_support_chain, outward_support_chain, ch1_endpoint, ch2_endpoint,
                                               endpoint, ch1, node_list):

    domian_angle_interpolation = compute_interpolation_domain(endpoint, ch1_endpoint, ch2_endpoint, ch1.Nr)
    center = ch1.center

    r1_ratio = compute_radial_ratio(inward_support_chain, outward_support_chain, ch1_endpoint)
    r2_ratio = compute_radial_ratio(inward_support_chain, outward_support_chain, ch2_endpoint)



    ###
    total_nodes = len(domian_angle_interpolation)
    if total_nodes == 0:
        return

    generated_nodes = generate_node_list_between_two_support_chains_and_two_radial_distances(r2_ratio, r1_ratio,
                                                                                             total_nodes,
                                                                                             domian_angle_interpolation,
                                                                                             center,
                                                                                             inward_support_chain,
                                                                                             outward_support_chain, ch1)

    node_list += generated_nodes

    return
def domain_interpolation(support_chain: ch.Chain, endpoint_cad1: ch.Node, endpoint_cad2: ch.Node, endpoint,
                         chain: ch.Chain, node_list: List[ch.Node]):

    domain_angles = compute_interpolation_domain(endpoint, endpoint_cad1, endpoint_cad2, chain.Nr)
    center = chain.center

    if support_chain is not None:
        dot_list_in_radial_direction = ch.get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
            [support_chain], endpoint_cad1.angle)
        node1_support = dot_list_in_radial_direction[0]
        r1 = ch.euclidean_distance_between_nodes(node1_support, endpoint_cad1)
        dot_list_in_radial_direction = ch.get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
            [support_chain], endpoint_cad2.angle)
        node2_support = dot_list_in_radial_direction[0]
        sign = -1 if node2_support.radial_distance > endpoint_cad2.radial_distance else +1
        r2 = ch.euclidean_distance_between_nodes(node2_support, endpoint_cad2)
    else:
        r1 = endpoint_cad1.radial_distance
        r2 = endpoint_cad2.radial_distance
        sign = 1
    ###
    total_nodes = len(domain_angles)
    if total_nodes == 0:
        return

    generated_nodes = generate_nodes_list_between_two_radial_distances(r2, r1, total_nodes, domain_angles, center, sign,
                                                                       support_chain, chain)

    node_list += generated_nodes

    return

def complete_chain_using_2_support_ring(inward_chain, outward_chain, ch1):
    ch1_endpoint = ch1.extB
    ch2_endpoint = ch1.extA
    endpoint = ch.EndPoints.B
    generated_nodes = []
    interpolate_in_angular_domain_via_2_chains(inward_chain, outward_chain, ch1_endpoint, ch2_endpoint, endpoint, ch1,
                                               generated_nodes)

    change_border = ch1.add_nodes_list(generated_nodes)

    return change_border

def complete_chain_using_support_ring(support_chain, ch1):

    ch1_endpoint = ch1.extB
    ch2_endpoint = ch1.extA
    endpoint = ch.EndPoints.B
    generated_list_nodes = []
    domain_interpolation(support_chain, ch1_endpoint, ch2_endpoint, endpoint, ch1, generated_list_nodes)
    change_border = ch1.add_nodes_list(generated_list_nodes)

    return change_border

def connect_2_chain_via_inward_and_outward_ring(outward_chain, inward_chain, chain1, chain2, node_list, endpoint,
                                                add=True, chain_list=None):
    ch1_endpoint = chain1.extA if endpoint == ch.EndPoints.A else chain1.extB
    ch2_endpoint = chain2.extB if endpoint == ch.EndPoints.A else chain2.extA

    #1.0
    generated_node_list = []
    interpolate_in_angular_domain_via_2_chains(inward_chain, outward_chain, ch1_endpoint, ch2_endpoint, endpoint,
                                               chain1, generated_node_list)
    node_list += generated_node_list


    #2.0
    nodes = []
    nodes += chain2.nodes_list
    for node in nodes:
        node.chain_id = chain1.id

    # assert not chain1.check_if_nodes_are_missing()
    # assert len([node.angle for node in nodes if chain1.get_node_by_angle(node.angle)]) == 0

    if add:
        change_border = chain1.add_nodes_list(nodes + generated_node_list)
    else:
        change_border = chain1.add_nodes_list(nodes)


    # ##########################################################

    return generated_node_list, change_border


