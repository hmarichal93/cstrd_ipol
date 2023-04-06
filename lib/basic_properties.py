import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import List

from lib.drawing import Color, Drawing
from lib.interpolation_nodes import generate_nodes_list_between_two_radial_distances
import lib.chain as ch
from lib.interpolation_nodes import domain_interpolation

def draw_segment_between_nodes(pto1, pto2, img, color=(0, 255, 0), thickness=2):
    pts = np.array([[pto1.y, pto1.x], [pto2.y, pto2.x]], dtype=int)
    isClosed = False
    img = cv2.polylines(img, [pts],
                        isClosed, color, thickness)

    return img


class InfoVirtualBand:
    DOWN = 0
    INSIDE = 1
    UP = 2

    def __init__(self, virtual_nodes, src_chain, dst_chain, endpoint, support_chain=None, band_width=None):
        if band_width is None:
            band_width = 0.05 if support_chain.type == ch.TypeChains.center else 0.1

        self.virtual_nodes = virtual_nodes
        self.src_chain = src_chain
        self.dst_chain = dst_chain
        self.endpoint = endpoint
        self.support_chain = support_chain
        params = {'y': self.src_chain.center[1], 'x': self.src_chain.center[0], 'angle': 0, 'radial_distance': 0,
                  'chain_id': -1}
        self.center = ch.Node(**params)


        ext1 = self.src_chain.extB if endpoint == ch.EndPoints.B else self.src_chain.extA
        ext1_support = self.support_chain.get_node_by_angle(ext1.angle) if self.support_chain is not None else self.center
        ext2 = self.dst_chain.extB if endpoint == ch.EndPoints.A else self.dst_chain.extA
        ext2_support = self.support_chain.get_node_by_angle(ext2.angle) if self.support_chain is not None else self.center
        delta_r1 = ch.euclidean_distance_between_nodes(ext1, ext1_support)
        delta_r2 = ch.euclidean_distance_between_nodes(ext2, ext2_support)
        self.inf_cand = delta_r2 * (1 - band_width)
        self.sup_cand = delta_r2 * (1 + band_width)
        self.inf_orig = delta_r1 * (1 - band_width)
        self.sup_orig = delta_r1 * (1 + band_width)

        self.generate_band()

    def generate_band_limit(self, r2, r1, total_nodes):
        interpolation_domain = [node.angle for node in self.virtual_nodes]
        endpoint_cad2 = self.virtual_nodes[-1]
        support_node2 = self.support_chain.get_node_by_angle(endpoint_cad2.angle) if self.support_chain is not None else self.center
        sign = -1 if support_node2.radial_distance > endpoint_cad2.radial_distance else +1
        generated_dots = generate_nodes_list_between_two_radial_distances(r2, r1, total_nodes, interpolation_domain,
                                                                          self.dst_chain.center, sign,
                                                                          self.support_chain, self.dst_chain)
        self.interpolation_domain = interpolation_domain
        return generated_dots

    def generate_band(self):
        total_nodes = len(self.virtual_nodes)
        r1 = self.sup_orig
        r2 = self.sup_cand
        self.sup_band = self.generate_band_limit(r2, r1, total_nodes)
        r1 = self.inf_orig
        r2 = self.inf_cand
        self.inf_band = self.generate_band_limit(r2, r1, total_nodes)

        return

    @staticmethod
    def mean_radial_in_node_list(node_list: List[ch.Node]):
        return np.mean([node.radial_distance for node in node_list])

    def is_dot_in_band(self, node: ch.Node):
        inf_mean_radii = self.mean_radial_in_node_list(self.inf_band)
        sup_mean_radii = self.mean_radial_in_node_list(self.sup_band)
        inner_band = self.inf_band if inf_mean_radii < sup_mean_radii else self.sup_band
        outter_band = self.sup_band if inf_mean_radii < sup_mean_radii else self.inf_band
        lowest = [inf for inf in inner_band if inf.angle == node.angle][0]
        highest = [inf for inf in outter_band if inf.angle == node.angle][0]

        if node.radial_distance <= lowest.radial_distance:
            relative_position = InfoVirtualBand.DOWN

        elif highest.radial_distance >= node.radial_distance >= lowest.radial_distance:
            relative_position = InfoVirtualBand.INSIDE

        else:
            relative_position = InfoVirtualBand.UP

        return relative_position

    def is_chain_in_band(self, chain: ch.Chain):
        node_chain_in_interval = [node for node in chain.nodes_list if
                                      node.angle in self.interpolation_domain]
        res = False
        prev_status = None
        for node in node_chain_in_interval:
            res = self.is_dot_in_band(node)
            if res == InfoVirtualBand.INSIDE:
                break

            if prev_status and prev_status != res:
                break

            prev_status = res
            res = False

        return res

    def generate_chain_from_node_list(self, node_list: List[ch.Node]):
        chain = ch.Chain(node_list[0].chain_id, self.src_chain.center, self.src_chain.M,
                           self.src_chain.N, Nr = self.src_chain.Nr)
        chain.add_nodes_list(node_list)

        return chain

    def draw_band(self, img, overlapping_chain: List[ch.Chain]):
        img = Drawing.chain(self.generate_chain_from_node_list(self.inf_band), img, color=Color.orange)
        img = Drawing.chain(self.generate_chain_from_node_list(self.sup_band), img, color=Color.maroon)
        img = Drawing.chain(self.src_chain, img, color=Color.blue)
        img = Drawing.chain(self.dst_chain, img, color=Color.yellow)
        if self.support_chain is not None:
            img = Drawing.chain(self.support_chain, img, color=Color.red)

        for chain in overlapping_chain:
            img = Drawing.chain(chain, img, color=Color.purple)

        return img



def vector_derivative(f, Nr):
    """center derivative of a vector f"""
    return np.gradient(f)



def regularity_of_the_derivative_condition(state, Nr, ch_j_nodes, ch_i_nodes, ch_k_nodes, endpoint_i, th_deriv=1):
    """
    Compute radial derivative of the serie formed by src nodes + virtual nodes + dst nodes
    @param Nr: number of rays
    @param ch_j_nodes: all nodes involved. src nodes + virtual nodes + dst nodes
    @param ch_i_nodes: src nodes
    @param ch_k_nodes: dst nodes
    @param th_deriv: threshold of the derivative
    @param endpoint_i: endpoint of the src chain
    @return: boolean indicative of the regularity of the derivative
    """
    ch_j_radials = [node.radial_distance for node in ch_j_nodes]
    nodes_radial_distance_src_chain = [node.radial_distance for node in ch_i_nodes]
    nodes_radial_distance_dst_chain = [node.radial_distance for node in ch_k_nodes]

    abs_der_1 = np.abs(vector_derivative(nodes_radial_distance_src_chain, Nr))
    abs_der_2 = np.abs(vector_derivative(nodes_radial_distance_dst_chain, Nr))
    abs_der_3 = np.abs(vector_derivative(ch_j_radials, Nr))
    maximum_derivative_chains = np.maximum(abs_der_1.max(), abs_der_2.max())

    max_derivative_end = np.max(abs_der_3)
    RegularDeriv = max_derivative_end <= maximum_derivative_chains * th_deriv

    if state is not None and state.debug:
        f, (ax2, ax1) = plt.subplots(2, 1)
        ax1.plot(abs_der_3)
        if endpoint_i == ch.EndPoints.A :
            ax1.plot(np.arange(0,len(abs_der_2)), abs_der_2[::-1])
            ax1.plot(np.arange(len(ch_j_radials)-len(nodes_radial_distance_src_chain), len(ch_j_radials)), abs_der_1)
        else:
            ax1.plot(np.arange(0, len(abs_der_1)), abs_der_1[::-1])
            ax1.plot(np.arange(len(ch_j_radials) - len(nodes_radial_distance_dst_chain), len(ch_j_radials)), abs_der_2)

        ax1.hlines(y=max_derivative_end, xmin=0, xmax= np.maximum(len(nodes_radial_distance_src_chain), len(nodes_radial_distance_dst_chain)), label='Salto')
        ax1.hlines(y=th_deriv * maximum_derivative_chains, xmin=0, xmax=np.maximum(len(nodes_radial_distance_dst_chain), len(nodes_radial_distance_src_chain)), colors='r', label='umbral')
        ax1.legend()

        ax2.plot(ch_j_radials)
        if endpoint_i == ch.EndPoints.A :
            ax2.plot( np.arange( 0, len(abs_der_2)), nodes_radial_distance_dst_chain[::-1] , 'r')
            ax2.plot(np.arange(len(ch_j_radials)-len(nodes_radial_distance_src_chain), len(ch_j_radials)),
                     nodes_radial_distance_src_chain)
        else:
            ax2.plot(np.arange(0, len(abs_der_1)), nodes_radial_distance_src_chain[::-1], 'r')
            ax2.plot(np.arange(len(ch_j_radials) - len(nodes_radial_distance_dst_chain), len(ch_j_radials)),
                     nodes_radial_distance_dst_chain)

        plt.title(f'{RegularDeriv}')
        plt.savefig(f'{str(state.path)}/{state.counter}_derivada_{RegularDeriv}.png')
        plt.close()
        state.counter += 1

    return RegularDeriv


def generate_virtual_nodes_without_support_chain(ch_1, ch_2, endpoint):
    ch1_border = ch_1.extA if endpoint == ch.EndPoints.A else ch_1.extB
    ch2_border = ch_2.extB if endpoint == ch.EndPoints.A else ch_2.extA

    virtual_nodes = []
    support_chain = None
    domain_interpolation(support_chain, ch1_border, ch2_border, endpoint, ch_1, virtual_nodes)
    return virtual_nodes

def regularity_of_the_derivative(state, ch_i, ch_k, endpoint_i, node_list, ch_i_nodes, ch_k_nodes, th_deriv=1,
                                 derivative_from_center=False):
    """
    Regularity of the derivative for the virtual nodes generated between the two chains.
    @param state: at this moment is used only for debug
    @param ch_i: source chain to be connected
    @param ch_k: destination chain to be connected
    @param endpoint_i: endpoint of ch_i to be connected
    @param node_list: all the nodes involved in the connection, including the virtual nodes
    @param ch_i_nodes: nodes of ch_i
    @param ch_k_nodes: nodes of ch_k
    @param th_deriv: derivative threshold
    @param derivative_from_center: bool for regenerating the virtual nodes interpolating from the center of the chain.
    @return: boolean indicative of the regularity of the derivative
    """
    if derivative_from_center:
        new_list = []
        virtual_nodes = generate_virtual_nodes_without_support_chain(ch_i, ch_k, endpoint_i)
        angles = [n.angle for n in virtual_nodes]
        for node in node_list:
            if node.angle not in angles:
                new_list.append(node)
            else:
                new_list.append(ch.get_node_from_list_by_angle(virtual_nodes, node.angle))

        node_list = new_list

    RegularDeriv = regularity_of_the_derivative_condition(state, ch_i.Nr, node_list, ch_i_nodes, ch_k_nodes, endpoint_i,
                                                          th_deriv=th_deriv)

    return RegularDeriv



def generate_virtual_nodes_between_two_chains(src_chain, dst_chain, support_chain, endpoint):
    virtual_nodes = []

    cad1_endpoint = src_chain.extA if endpoint == ch.EndPoints.A else src_chain.extB
    cad2_endpoint = dst_chain.extB if endpoint == ch.EndPoints.A else dst_chain.extA

    domain_interpolation(support_chain, cad1_endpoint, cad2_endpoint, endpoint, src_chain, virtual_nodes)

    return virtual_nodes


def radial_tolerance_for_connecting_chains(state, th_radial_tolerance, endpoints_radial):
    """
    Check maximum radial distance allowed to connect chains
    @param state: state of the algorithm. At this point, it is used to debug.
    @param th_radial_tolerance: radial tolerance threshold
    @param endpoints_radial: radial distance between endpoints and support chain
    @return: bool indicating if radial distance is within tolerance
    """
    delta_ri = endpoints_radial[0]
    delta_ri_plus_i = endpoints_radial[1]
    inf_delta_ri = delta_ri * (1 - th_radial_tolerance)
    sup_delta_ri = delta_ri * (1 + th_radial_tolerance)
    RadialTol = inf_delta_ri <= delta_ri_plus_i <= sup_delta_ri

    #debug info
    if state is not None and state.debug:
        plt.figure()
        plt.axvline(x=delta_ri, color='b', label=f'delta_ri')
        plt.axvline(x=delta_ri_plus_i, color='r', label=f'delta_ri_plus_i')
        plt.axvline(x=inf_delta_ri, color='k', label=f'inf radial')
        plt.axvline(x=sup_delta_ri, color='k', label=f'sup radial')
        plt.title(f"{RadialTol}: Th {state.radio_limit}.")
        plt.savefig(f'{str(state.path)}/{state.counter}_max_radial_condition.png')
        plt.close()
        state.counter += 1

    return RadialTol


class Neighbourhood:
    """
    Class to compute and store the total_nodes of a chain and the candidate chains to connect to it. It generates the virtual nodes
    to compute the similarity condition.
    """
    def __init__(self, src_chain, dst_chain, support_chain, endpoint, n_nodes=20):
        """

        @param src_chain: src chain to be connected
        @param dst_chain: dst chain to be connected
        @param support_chain: support chain to be used to generate the virtual nodes
        @param endpoint: src chain endpoint to be connected
        @param n_nodes: number of nodes to be considered in the total_nodes
        """
        self.virtual_nodes = generate_virtual_nodes_between_two_chains(src_chain, dst_chain, support_chain, endpoint)
        self.src_endpoint = src_chain.extB if endpoint == ch.EndPoints.B else src_chain.extA
        self.dst_endpoint = dst_chain.extA if endpoint == ch.EndPoints.B else dst_chain.extB
        self.endpoint_and_virtual_nodes = [self.src_endpoint] + self.virtual_nodes + [self.dst_endpoint]
        self.radial_distance_endpoints_to_support = radial_distance_between_nodes_belonging_to_same_ray(
            [self.src_endpoint, self.dst_endpoint],
            support_chain)
        self.radial_distance_virtual_nodes_to_support = radial_distance_between_nodes_belonging_to_same_ray(
            self.virtual_nodes, support_chain)

        self.src_chain_nodes = src_chain.sort_dots(direction=ch.ClockDirection.anti_clockwise)[
                               :n_nodes] if endpoint == ch.EndPoints.A \
            else src_chain.sort_dots(direction=ch.ClockDirection.clockwise)[:n_nodes]
        self.dst_chain_nodes = dst_chain.sort_dots(direction=ch.ClockDirection.clockwise)[
                               :n_nodes] if endpoint == ch.EndPoints.A \
            else dst_chain.sort_dots(direction=ch.ClockDirection.anti_clockwise)[:n_nodes]
        self.set_i = radial_distance_between_nodes_belonging_to_same_ray(self.src_chain_nodes,
                                                                         support_chain)
        self.set_k = radial_distance_between_nodes_belonging_to_same_ray(
            self.dst_chain_nodes, support_chain)

        # sort all nodes, src + virtuales + dst
        if endpoint == ch.EndPoints.B:
            self.neighbourhood_nodes = self.src_chain_nodes[::-1] + self.virtual_nodes + self.dst_chain_nodes
        else:
            self.neighbourhood_nodes = self.dst_chain_nodes[::-1] + self.virtual_nodes[::-1] + self.src_chain_nodes

    def draw_neighbourhood(self, name):
        r = [node.radial_distance for node in self.neighbourhood_nodes]
        plt.figure()
        plt.plot(r, 'b')
        plt.savefig(name)
        plt.close()


def similar_radial_distances_of_nodes_in_both_chains(state, distribution_th, set_i, set_k):
    """
    Check if the radial distances of the nodes in both chains are similar via distribution of the radial distances
    @param state: state of the algorithm. At this point, it is used to debug.
    @param distribution_th: size of the distributions to check if they are similar
    @param set_i: radial distance between nodes of the first chain and the support chain
    @param set_k: radial distances between nodes of the second chain and the support chain
    @return:
    + bool indicating if the radial distances are similar
    + distance between the mean of the distributions
    """
    mean_i = np.mean(set_i)
    sigma_i = np.std(set_i)
    inf_range_i = mean_i - distribution_th * sigma_i
    sup_range_i = mean_i + distribution_th * sigma_i

    mean_k = np.mean(set_k)
    std_k = np.std(set_k)
    inf_range_k = mean_k - distribution_th * std_k
    sup_range_k = mean_k + distribution_th * std_k
    # check intersection between intervals
    SimilarRadialDist = inf_range_k <= sup_range_i and inf_range_i <= sup_range_k

    if state is not None and state.debug:
        plt.figure()
        plt.hist(set_i, bins=10, alpha=0.3, color='r', label='src radial')
        plt.hist(set_k, bins=10, alpha=0.3, color='b', label='dst radial')
        plt.axvline(x=inf_range_i, color='r', label=f'inf src')
        plt.axvline(x=sup_range_i, color='r', label=f'sup src')
        plt.axvline(x=inf_range_k, color='b', label=f'inf dst')
        plt.axvline(x=sup_range_k, color='b', label=f'sup dst')
        plt.legend()
        plt.title(f"{SimilarRadialDist}: Th {distribution_th}.")
        plt.savefig(f'{str(state.path)}/{state.counter}_distribution_condition.png')
        plt.close()
        state.counter += 1

    return SimilarRadialDist, np.abs(mean_i - mean_k)


def similarity_conditions(state, th_radial_tolerance, th_distribution_size, th_regular_derivative,
                          derivative_from_center, support_chain, src_chain, dst_chain, endpoint, check_overlapping=True,
                          chain_list=None):
    """
    Similarity conditions defined in equation 6 in the paper
    @param state: state of the algorithm. At this point is used to debug.
    @param th_radial_tolerance:  Described at Table1 in the paper
    @param th_distribution_size:  Described at Table1 in the paper
    @param th_regular_derivative:  Described at Table1 in the paper
    @param derivative_from_center: Described at Table1 in the paper
    @param support_chain: support chain
    @param src_chain: source chain to connect
    @param dst_chain: destination chain to connect
    @param endpoint: endpoint of the source chain
    @param check_overlapping: check if the chains are overlapping
    @param chain_list: list of chains to check if the chains are overlapping
    @return: return a bool indicating if the similarity conditions are satisfied and the radial distance between
    the chains radial distributions
    """
    neighbourhood = Neighbourhood(src_chain, dst_chain, support_chain, endpoint)
    if state is not None and state.debug:
        ch.visualize_selected_ch_and_chains_over_image_([support_chain, src_chain, dst_chain], state.chains_list,
                                                        img=state.img, filename=f'{state.path}/{state.counter}_radial_conditions_{endpoint}.png')
        state.counter += 1
        neighbourhood.draw_neighbourhood(f'{state.path}/{state.counter}_radials_conditions_neighbourdhood_{src_chain.label_id}_{dst_chain.label_id}.png')
        state.counter += 1

    if len(neighbourhood.set_i) <= 1 or len(
            neighbourhood.set_k) <= 1:
        return False, -1

    # 1. Radial tolerance for connecting chains
    RadialTol = radial_tolerance_for_connecting_chains(state, th_radial_tolerance,
                                                                   neighbourhood.radial_distance_endpoints_to_support)



    # 2. Similar radial distances of nodes in both chains
    SimilarRadialDist, distribution_distance = similar_radial_distances_of_nodes_in_both_chains(state, th_distribution_size,
                                                                                                neighbourhood.set_i,
                                                                                                neighbourhood.set_k)

    check_pass = RadialTol or SimilarRadialDist
    if not check_pass:
        return (False, distribution_distance)

    # 3. Derivative condition
    RegularDeriv = regularity_of_the_derivative(state, src_chain, dst_chain, endpoint,
                                                neighbourhood.neighbourhood_nodes,
                                                ch_i_nodes=neighbourhood.src_chain_nodes,
                                                ch_k_nodes=neighbourhood.dst_chain_nodes, th_deriv=th_regular_derivative,
                                                derivative_from_center=derivative_from_center)

    if not RegularDeriv:
        return (False, distribution_distance)

    # 4.0 Check there is not chains in region
    if check_overlapping:
        exist_chain = exist_chain_overlapping(state.ch_s_list if chain_list is None else chain_list,
                    neighbourhood.endpoint_and_virtual_nodes, src_chain, dst_chain, endpoint, support_chain)

        if exist_chain:
            return (False, distribution_distance)

    return (True, distribution_distance)


def radial_distance_between_nodes_belonging_to_same_ray(node_list, support_chain):
    """
    Compute radial distance between nodes belonging to same ray
    @param node_list:
    @param support_chain:
    @return:
    """
    radial_distances = []
    for node in node_list:
        support_node = support_chain.get_node_by_angle(node.angle)
        if support_node is None:
            break
        radial_distances.append(ch.euclidean_distance_between_nodes(support_node, node))

    return radial_distances


def exist_chain_in_band_logic(chain_list, band_info):
    chain_of_interest = [band_info.dst_chain, band_info.src_chain]
    if band_info.support_chain is not None:
        chain_of_interest.append(band_info.support_chain)

    try:
        fist_chain_in_region = next(chain for chain in chain_list if
                                           chain not in chain_of_interest and band_info.is_chain_in_band(chain))
    except StopIteration:
        fist_chain_in_region = None

    return [fist_chain_in_region] if fist_chain_in_region is not None else []


def exist_chain_overlapping(chains_list, endpoints_and_virtual_nodes, src_chain, dst_chain, endpoint, support_chain):
    """
    Check if there is a chain in the region defined by the band
    @param chains_list: full chain list
    @param endpoints_and_virtual_nodes: both endpoints and virtual nodes
    @param src_chain: source chain
    @param dst_chain: destination chain
    @param endpoint: source chain endpoint
    @param support_chain: support chain
    @return: boolean indicating if exist chain in band
    """
    info_band = InfoVirtualBand(endpoints_and_virtual_nodes, src_chain, dst_chain, endpoint,
                                support_chain)
    chains_in_band = exist_chain_in_band_logic(chains_list, info_band)
    exist_chain = len(chains_in_band) > 0


    return exist_chain
