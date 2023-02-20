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

    def __init__(self, virtual_nodes, src_chain, dst_chain, endpoint, support_chain=None, inf_orig=None, sup_orig=None,
                 inf_cand=None, sup_cand=None, band_width=0.1):
        self.virtual_nodes = virtual_nodes
        self.src_chain = src_chain
        self.dst_chain = dst_chain
        self.endpoint = endpoint
        self.support_chain = support_chain
        params = {'y': self.src_chain.center[1], 'x': self.src_chain.center[0], 'angle': 0, 'radial_distance': 0,
                  'chain_id': -1}
        self.center = ch.Node(**params)

        if band_width:
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

        else:
            self.inf_orig = inf_orig
            self.inf_cand = inf_cand
            self.sup_orig = sup_orig
            self.sup_cand = sup_cand

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



def vector_derivative(f, Nr, paso=2):
    return np.gradient(f)



def max_derivative(state, Nr, nodes_list, src_chain_nodes, dst_chain_nodes, endpoint, threshold=1):
    """
    Compute radial derivative of the serie formed by src nodes + virtual nodes + dst nodes
    @param Nr:
    @param nodes_list:
    @param nodes_radial_distance_src_chain:
    @param nodes_radial_distance_dst_chain:
    @param threshold:
    @return:
    """
    virtual_radials = [node.radial_distance for node in nodes_list]
    nodes_radial_distance_src_chain = [node.radial_distance for node in src_chain_nodes]
    nodes_radial_distance_dst_chain = [node.radial_distance for node in dst_chain_nodes]

    abs_der_1 = np.abs(vector_derivative(nodes_radial_distance_src_chain, Nr))
    abs_der_2 = np.abs(vector_derivative(nodes_radial_distance_dst_chain, Nr))
    abs_der_3 = np.abs(vector_derivative(virtual_radials, Nr))
    max_derivative_init = np.maximum(abs_der_1.max(), abs_der_2.max())

    max_derivative_end = np.max(abs_der_3)
    res = max_derivative_end <= threshold * max_derivative_init

    if state is not None and state.debug:
        f, (ax2, ax1) = plt.subplots(2, 1)
        ax1.plot(abs_der_3)
        if endpoint == ch.EndPoints.A :
            ax1.plot(np.arange(0,len(abs_der_2)), abs_der_2[::-1])
            ax1.plot(np.arange(len(virtual_radials)-len(nodes_radial_distance_src_chain), len(virtual_radials)), abs_der_1)
        else:
            ax1.plot(np.arange(0, len(abs_der_1)), abs_der_1[::-1])
            ax1.plot(np.arange(len(virtual_radials) - len(nodes_radial_distance_dst_chain), len(virtual_radials)), abs_der_2)

        ax1.hlines(y=max_derivative_end, xmin=0, xmax= np.maximum(len(nodes_radial_distance_src_chain), len(nodes_radial_distance_dst_chain)), label='Salto')
        ax1.hlines(y=threshold * max_derivative_init, xmin=0, xmax=np.maximum(len(nodes_radial_distance_dst_chain), len(nodes_radial_distance_src_chain)),colors='r', label='umbral')
        ax1.legend()

        ax2.plot(virtual_radials)
        if endpoint == ch.EndPoints.A :
            ax2.plot( np.arange( 0, len(abs_der_2)), nodes_radial_distance_dst_chain[::-1] , 'r')
            ax2.plot(np.arange(len(virtual_radials)-len(nodes_radial_distance_src_chain), len(virtual_radials)),
                     nodes_radial_distance_src_chain)
        else:
            ax2.plot(np.arange(0, len(abs_der_1)), nodes_radial_distance_src_chain[::-1], 'r')
            ax2.plot(np.arange(len(virtual_radials) - len(nodes_radial_distance_dst_chain), len(virtual_radials)),
                     nodes_radial_distance_dst_chain)

        plt.title(f'{res}')
        plt.savefig(f'{str(state.path)}/{state.counter}_derivada_{res}.png')
        plt.close()
        state.counter += 1

    return res


def generate_virtual_nodes_without_support_chain(ch_1, ch_2, endpoint):
    ch1_border = ch_1.extA if endpoint == ch.EndPoints.A else ch_1.extB
    ch2_border = ch_2.extB if endpoint == ch.EndPoints.A else ch_2.extA

    virtual_nodes = []
    support_chain = None
    domain_interpolation(support_chain, ch1_border, ch2_border, endpoint, ch_1, virtual_nodes)
    return virtual_nodes

def derivative_condition(state, cad_1, cad_2, endpoint, node_list, src_chain_nodes,
                         dst_chain_nodes, threshold=None, derivative_from_center=False):
    if derivative_from_center:
        new_list = []
        virtual_nodes = generate_virtual_nodes_without_support_chain(cad_1, cad_2, endpoint)
        angles = [n.angle for n in virtual_nodes]
        for node in node_list:
            if node.angle not in angles:
                new_list.append(node)
            else:
                new_list.append(ch.get_node_from_list_by_angle(virtual_nodes, node.angle))

        node_list = new_list

    res = max_derivative(state, cad_1.Nr, node_list, src_chain_nodes,
                         dst_chain_nodes, endpoint, threshold=threshold)

    return res



def generate_virtual_nodes_between_two_chains(src_chain, dst_chain, support_chain, endpoint):
    virtual_nodes = []

    cad1_endpoint = src_chain.extA if endpoint == ch.EndPoints.A else src_chain.extB
    cad2_endpoint = dst_chain.extB if endpoint == ch.EndPoints.A else dst_chain.extA

    domain_interpolation(support_chain, cad1_endpoint, cad2_endpoint, endpoint, src_chain, virtual_nodes)

    return virtual_nodes


def max_radial_condition(state, threshold, endpoints_radial):
    """
    Check maximum radial distance allowed to connect chains
    @param threshold:
    @param endpoints_radial:
    @return:
    """
    src_radial = endpoints_radial[0]
    dst_radial = endpoints_radial[1]
    inf_src_radial = src_radial * (1 - threshold)
    sup_src_radial = src_radial * (1 + threshold)
    pass_check = inf_src_radial <= dst_radial <= sup_src_radial
    if state is not None and state.debug:
        plt.figure()
        plt.axvline(x=src_radial, color='b', label=f'src_radial')
        plt.axvline(x=dst_radial, color='r', label=f'dst_radial')
        plt.axvline(x=inf_src_radial, color='k', label=f'inf radial')
        plt.axvline(x=sup_src_radial, color='k', label=f'sup radial')
        plt.title(f"{pass_check}: Th {state.radio_limit}.")
        plt.savefig(f'{str(state.path)}/{state.counter}_max_radial_condition.png')
        plt.close()
        state.counter += 1
    return pass_check, src_radial, dst_radial, inf_src_radial, sup_src_radial


class Neighbourhood:
    def __init__(self, src_chain, dst_chain, support_chain, endpoint, chain_neighbourhoood=20):
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
                          :chain_neighbourhoood] if endpoint == ch.EndPoints.A \
            else src_chain.sort_dots(direction=ch.ClockDirection.clockwise)[:chain_neighbourhoood]
        self.dst_chain_nodes = dst_chain.sort_dots(direction=ch.ClockDirection.clockwise)[
                          :chain_neighbourhoood] if endpoint == ch.EndPoints.A \
            else dst_chain.sort_dots(direction=ch.ClockDirection.anti_clockwise)[:chain_neighbourhoood]
        self.radial_distance_src_nodes_to_support = radial_distance_between_nodes_belonging_to_same_ray(self.src_chain_nodes,
                                                                                                        support_chain)
        self.radial_distance_dst_nodes_to_support = radial_distance_between_nodes_belonging_to_same_ray(
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


def distribution_condition(state, width_std, src_radial, dst_radial):
    mean_src = np.mean(src_radial)
    std_src = np.std(src_radial)
    inf_limit = mean_src - width_std * std_src
    sup_limit = mean_src + width_std * std_src

    mean_dst = np.mean(dst_radial)
    inf_dst = mean_dst - width_std * np.std(dst_radial)
    sup_dst = mean_dst + width_std * np.std(dst_radial)
    pass_check = inf_dst <= sup_limit and inf_limit <= sup_dst

    if state is not None and state.debug:
        plt.figure()
        plt.hist(src_radial, bins=10, alpha=0.3, color='r', label='src radial')
        plt.hist(dst_radial, bins=10, alpha=0.3, color='b', label='dst radial')
        plt.axvline(x=inf_limit, color='r', label=f'inf src')
        plt.axvline(x=sup_limit, color='r', label=f'sup src')
        plt.axvline(x=inf_dst, color='b', label=f'inf dst')
        plt.axvline(x=sup_dst, color='b', label=f'sup dst')
        plt.legend()
        plt.title(f"{pass_check}: Th {width_std}.")
        plt.savefig(f'{str(state.path)}/{state.counter}_distribution_condition.png')
        plt.close()
        state.counter += 1
    return pass_check, np.abs(mean_src - mean_dst)


def radials_conditions(state, radial_th, distribution_th,derivative_th, derivative_from_center, support_chain, src_chain,
                       dst_chain, endpoint, check_overlapping=True, chain_list=None):
    """
    Check radials conditions.
    @param state:
    @param support_chain:
    @param src_chain:
    @param dst_chain:
    @param endpoint:
    @return:
    """
    assert support_chain is not None
    neighbourhood = Neighbourhood(src_chain, dst_chain, support_chain, endpoint)
    if state is not None and state.debug:
        ch.visualize_selected_ch_and_chains_over_image_([support_chain, src_chain, dst_chain], state.chains_list,
                                                        img=state.img, filename=f'{state.path}/{state.counter}_radial_conditions_{endpoint}.png')
        state.counter += 1
        neighbourhood.draw_neighbourhood(f'{state.path}/{state.counter}_radials_conditions_neighbourdhood_{src_chain.label_id}_{dst_chain.label_id}.png')
        state.counter += 1

    if len(neighbourhood.radial_distance_src_nodes_to_support) <= 1 or len(
            neighbourhood.radial_distance_dst_nodes_to_support) <= 1:
        return False, -1

    # 1. Max radial condition
    pass_check_max_radial_condition, src_radial, dst_radial, inf_src_radial, sup_src_radial = max_radial_condition( state, radial_th,
                                                                                neighbourhood.radial_distance_endpoints_to_support)



    # 2. Distribution Condition
    pass_check_distribution_condition, distribution_distance = distribution_condition(state, distribution_th,
                                                                                      neighbourhood.radial_distance_src_nodes_to_support,
                                                                                      neighbourhood.radial_distance_dst_nodes_to_support)

    check_pass = pass_check_max_radial_condition or pass_check_distribution_condition
    if not check_pass:
        return (check_pass, distribution_distance)

    # 3. Derivative condition
    pass_check_derivative_condition = derivative_condition(state, src_chain, dst_chain, endpoint,
                                                           neighbourhood.neighbourhood_nodes,
                                                           src_chain_nodes = neighbourhood.src_chain_nodes,
                                                           dst_chain_nodes = neighbourhood.dst_chain_nodes,
                                                           threshold = derivative_th,
                                                           derivative_from_center= derivative_from_center)

    if not pass_check_derivative_condition:
        return (pass_check_derivative_condition, distribution_distance)

    # 4.0 Check there is not chains in region
    if check_overlapping:
        info_band = InfoVirtualBand(neighbourhood.endpoint_and_virtual_nodes, src_chain, dst_chain, endpoint,
                                    support_chain, band_width=0.05 if support_chain.type == ch.TypeChains.center else 0.1)
        chain_list = state.chains_list if chain_list is None else chain_list
        exist_chain = exist_chain_in_band(chain_list, info_band)
        if exist_chain:
            return (not exist_chain, distribution_distance)

    return (pass_check_derivative_condition, distribution_distance)


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


def exist_chain_in_band(chains_list, info_band):
    """

    @param state:
    @param info_band:
    @return:
    """

    chains_in_band = exist_chain_in_band_logic(chains_list, info_band)
    exist_chain = len(chains_in_band) > 0


    return exist_chain
