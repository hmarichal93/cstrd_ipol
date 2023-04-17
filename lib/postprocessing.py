import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, Polygon
import cv2
from typing import List

import lib.chain as ch
from lib.drawing import Drawing
from lib.interpolation_nodes import complete_chain_using_2_support_ring, connect_2_chain_via_inward_and_outward_ring, \
    complete_chain_using_support_ring
from lib.basic_properties import similarity_conditions
from lib.connect_chains import intersection_between_chains, get_inward_and_outward_visible_chains, SystemStatus, debugging_chains


def build_boundary_poly(outward_ring, inward_ring):
    """
    Convert shaeply poly to region poly
    @rtype: object
    @param outward_ring: outward shapely polygon
    @param inward_ring: inward shapely poylgon
    @return: poly region
    """
    if outward_ring is None and inward_ring is None:
        return None
    if outward_ring is not None and inward_ring is not None:
        x, y = outward_ring.exterior.coords.xy
        pts_ext = [[j, i] for i, j in zip(y, x)]
        x, y = inward_ring.exterior.coords.xy
        pts_int = [[j, i] for i, j in zip(y, x)]
        poly = Polygon(pts_ext, [pts_int])

    else:
        if outward_ring is None:
            x, y = inward_ring.exterior.coords.xy
        else:
            x, y = outward_ring.exterior.coords.xy

        pts_ext = [[j, i] for i, j in zip(y, x)]
        poly = Polygon(pts_ext)

    return poly


def search_shapely_inward_chain(shapley_incomplete_chain, outward_ring, inward_ring):
    """
    Search for shapely polygon inside region delimitad for outward and inward rings
    @param shapley_incomplete_chain: shapely polygon chains not completed. Not nr nodes
    @param outward_ring: shapely polygon ch_i completed.nr nodes
    @param inward_ring: shapely polygon ch_i completed.nr nodes
    @return:
    """
    poly = build_boundary_poly(outward_ring, inward_ring)
    if poly is None:
        return []
    contains = np.vectorize(lambda p: poly.contains(Point(p)), signature='(n)->()')
    shapely_inward_chains_subset = []
    for cadena in shapley_incomplete_chain:
        x, y = cadena.xy
        pts = [[i, j] for i, j in zip(y, x)]
        if len(pts) == 0:
            continue
        try:
            vector = contains(np.array(pts))
        except Exception as e:
            continue
        if outward_ring is not None:
            if vector.sum() == vector.shape[0]:
                shapely_inward_chains_subset.append(cadena)
        else:
            if vector.sum() == 0:
                shapely_inward_chains_subset.append(cadena)

    return shapely_inward_chains_subset


def from_shapely_to_chain(uncompleted_shapely_chain, uncomplete_chain, shapely_inward_chains_subset):
    inward_chain_subset = [uncomplete_chain[uncompleted_shapely_chain.index(cad_shapely)]
                           for cad_shapely in shapely_inward_chains_subset]
    inward_chain_subset.sort(key=lambda x: x.size, reverse=True)
    return inward_chain_subset


class Ring(Polygon):
    def __init__(self, chain: ch.Chain, id: int):
        lista_pts = [[node.x, node.y] for node in chain.sort_dots()]
        self.id = id
        super(self.__class__, self).__init__(lista_pts)

    def draw(self, image):
        x, y = self.exterior.coords.xy
        lista_pts = [[i, j] for i, j in zip(y, x)]
        pts = np.array(lista_pts,
                       np.int32)

        pts = pts.reshape((-1, 1, 2))
        is_closed = True
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 1
        # Using cv2.polylines() method
        # Draw a Blue polygon with
        # thickness of 1 px
        image = cv2.polylines(image, [pts],
                              is_closed, color, thickness)

        image = Drawing.put_text(f'{self.id}', image, (int(y[0]), int(x[0])))

        return image


class DiskContext:
    def __init__(self, l_ch_c, idx_start, save_path=None, img=None, debug=True):
        self.l_within_chains_subset = []
        self.neighbourhood_size = None
        self.debug = debug
        self.save_path = save_path
        self.img = img
        self.completed_chains = [cad for cad in l_ch_c if cad.size >= cad.Nr]
        self.completed_chains, self.poly_completed_chains = self._from_completed_chain_to_poly(
            self.completed_chains)

        self.uncompleted_chains = [cad for cad in l_ch_c if cad.size < cad.Nr]
        self.uncompleted_chains_poly = self._from_uncompleted_chains_to_poly(self.uncompleted_chains)
        self.idx = 1 if idx_start is None else idx_start

    def get_inward_outward_ring(self, idx):
        self.neighbourhood_size = 45
        outward_ring = None
        inward_ring = None
        if len(self.poly_completed_chains) > idx > 0:
            inward_ring = self.poly_completed_chains[idx - 1]
            outward_ring = self.poly_completed_chains[idx]

        return inward_ring, outward_ring

    def update(self):
        inward_poly_ring, outward_poly_ring = self.get_inward_outward_ring(self.idx)
        shapely_inward_chain_subset = search_shapely_inward_chain(self.uncompleted_chains_poly, outward_poly_ring,
                                                                  inward_poly_ring)
        self.l_within_chains_subset = from_shapely_to_chain(self.uncompleted_chains_poly,
                                                            self.uncompleted_chains,
                                                            shapely_inward_chain_subset)

        self.inward_ring, self.outward_ring = self._from_shapely_ring_to_chain(inward_poly_ring,
                                                                               outward_poly_ring)

    def exit(self):
        self.idx += 1
        if self.idx >= len(self.completed_chains):
            return True

        return False

    def drawing(self, iteration):
        ch.visualize_selected_ch_and_chains_over_image_(
            self.l_within_chains_subset + [chain for chain in [self.inward_ring, self.outward_ring] if chain is not None],
            [], img=self.img, filename=f'{self.save_path}/{iteration}_0.png')

    def _from_shapely_ring_to_chain(self, poly_ring_inward, poly_ring_outward):
        inward_chain_ring = None
        outward_chain_ring = None
        if poly_ring_inward is not None:
            inward_chain_ring = self.completed_chains[self.poly_completed_chains.index(
                poly_ring_inward)]

        if poly_ring_outward is not None:
            outward_chain_ring = self.completed_chains[
                self.poly_completed_chains.index(poly_ring_outward)]
        return inward_chain_ring, outward_chain_ring

    def sort_list_by_index_array(self, indexes, list_position):
        Z = [list_position[position] for position in indexes]
        return Z

    def sort_shapely_list_and_chain_list(self, cadena_list, shapely_list):
        idx_sort = [i[0] for i in sorted(enumerate(shapely_list), key=lambda x: x[1].area)]
        shapely_list = self.sort_list_by_index_array(idx_sort, shapely_list)
        cadena_list = self.sort_list_by_index_array(idx_sort, cadena_list)
        return cadena_list, shapely_list

    def _from_completed_chain_to_poly(self, completed_chain):
        poly_completed_chains = []
        for chain in completed_chain:
            ring = Ring(chain, id=chain.id)
            poly_completed_chains.append(ring)

        completed_chain, poly_completed_chains = self.sort_shapely_list_and_chain_list(completed_chain,
                                                                                       poly_completed_chains)

        return completed_chain, poly_completed_chains

    def _from_uncompleted_chains_to_poly(self, uncompleted_chain):
        uncompleted_chain_shapely = []
        for chain in uncompleted_chain:
            lista_pts = [Point(punto.y, punto.x) for punto in chain.sort_dots()]
            uncompleted_chain_shapely.append(LineString(lista_pts))

        return uncompleted_chain_shapely


class ChainsBag:
    def __init__(self, inward_chain_set):
        """
        Iterate over chains in a region.
        @param inward_chain_set: set of inward ch_i inside region sorted by size
        """
        self.chain_set = inward_chain_set
        self.chains_id_already_selected = []

    def get_next_chain(self):
        next = None
        for chain in self.chain_set:
            if chain.id not in self.chains_id_already_selected:
                next = chain
                self.chains_id_already_selected.append(next.id)
                break

        return next


def select_support_chain(outward_chain_ring, inward_chain_ring, endpoint):
    """
    Select the ch_k support ch_i to the endpoint. Over endpoint ray direction, the support ch_i with the smallest
    distance between nodes is selected
    @param outward_chain_ring: outward support ch_i
    @param inward_chain_ring:  inward support ch_i
    @param endpoint: source ch_i endpoint
    @return: ch_k support ch_i
    """
    chains_in_radial_direction = []
    if outward_chain_ring is not None:
        chains_in_radial_direction.append(outward_chain_ring)

    if inward_chain_ring is not None:
        chains_in_radial_direction.append(inward_chain_ring)

    dot_list_in_radial_direction = ch.get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
        chains_in_radial_direction, endpoint.angle)

    distance = [ch.euclidean_distance_between_nodes(endpoint, completed_chain_node) for completed_chain_node in
                dot_list_in_radial_direction]

    if len(distance) < 2:
        support_chain = outward_chain_ring if outward_chain_ring is not None else inward_chain_ring
    else:
        support_chain = ch.get_chain_from_list_by_id(chains_in_radial_direction,
                                                     dot_list_in_radial_direction[np.argmin(distance)].chain_id)

    return support_chain


def exist_angular_intersection_with_src_chain(chain: ch.Chain, src_chain_angular_domain: List[int]):
    domain = [node.angle for node in chain.l_nodes]
    if len(np.intersect1d(domain, src_chain_angular_domain)) == 0:
        return False
    return True


def angular_domain_overlapping_higher_than_threshold(src_chain_angular_domain: List[int], inter_chain: ch.Chain,
                                                     overlapping_threshold: int = 45):
    """
    Check if overlapping angular domain between two chains is higher than a threshold
    @param src_chain_angular_domain: src ch_i
    @param inter_chain: another ch_i
    @param overlapping_threshold: overlapping threshold
    @return: boolean value
    """
    inter_domain = [node.angle for node in inter_chain.l_nodes]
    inter = np.intersect1d(inter_domain, src_chain_angular_domain)
    if (len(inter) >= len(src_chain_angular_domain)) or (len(inter) > overlapping_threshold):
        return True
    else:
        return False


def split_chain(chain: ch.Chain, node: ch.Node, id_new=10000000):
    """
    Split a ch_i in two chains
    @param chain: Parent ch_i. Chain to be split
    @param node: node element where the ch_i will be split
    @param id_new: new id
    @return: tuple of child chains
    """
    node_list = chain.sort_dots()
    idx_split = node_list.index(node)
    ch1_node_list = [ch.copy_node(node) for node in node_list[:idx_split]]
    if idx_split < len(node_list) - 1:
        ch2_node_list = [ch.copy_node(node) for node in node_list[idx_split + 1:]]
    else:
        ch2_node_list = []

    if len(ch1_node_list) > 1:
        ch1_sub = ch.Chain(id_new, chain.Nr, center=chain.center, img_height=chain.img_height,
                           img_width=chain.img_width)
        for node_ch in ch1_node_list:
            node_ch.chain_id = ch1_sub.id

        ch1_sub.add_nodes_list(ch1_node_list)
    else:
        ch1_sub = None

    if len(ch2_node_list) > 1:
        ch2_sub = ch.Chain(id_new, chain.Nr, center=chain.center, img_height=chain.img_height,
                           img_width=chain.img_width)
        for node_ch in ch2_node_list:
            node_ch.chain_id = ch2_sub.id
        ch2_sub.add_nodes_list(ch2_node_list)
    else:
        ch2_sub = None

    return (ch1_sub, ch2_sub)


def select_no_intersection_chain_at_endpoint(ch1_sub: ch.Chain, ch2_sub: ch.Chain, src_chain: ch.Chain,
                                             ray_direction: int, total_nodes=10):
    """
    Select the ch_i that does not intersect with the source ch_i at endpoint
    @param ch1_sub: child ch_i 1
    @param ch2_sub:  child ch_i 2
    @param src_chain: source ch_i
    @param ray_direction: ray direction source ch_i
    @param total_nodes:
    @return: ch_i that does not intersect with the source ch_i at endpoint
    """
    if ch1_sub is None and ch2_sub is None:
        return None
    if ch1_sub is None and ch2_sub is not None:
        return ch2_sub
    if ch2_sub is None and ch1_sub is not None:
        return ch1_sub

    endpoint = ch.EndPoints.A if ray_direction == src_chain.extA.angle else ch.EndPoints.B
    direction = ch.ClockDirection.clockwise if endpoint == ch.EndPoints.B else ch.ClockDirection.anti_clockwise
    nodes_neighbourhood = src_chain.sort_dots(direction=direction)[:total_nodes]
    src_nodes = ch.get_nodes_angles_from_list_nodes(nodes_neighbourhood)

    domain1 = ch.get_nodes_angles_from_list_nodes(ch1_sub.l_nodes) if ch1_sub.size > 0 else src_nodes
    domain2 = ch.get_nodes_angles_from_list_nodes(ch2_sub.l_nodes) if ch2_sub.size > 0 else src_nodes
    if np.intersect1d(domain1, src_nodes).shape[0] == 0:
        return ch1_sub
    elif np.intersect1d(domain2, src_nodes).shape[0] == 0:
        return ch2_sub
    else:
        return None


def split_intersecting_chains(direction, l_filtered_chains, ch_j):
    """
    Split intersecting chains
    @param direction: endpoint direction for split chains
    @param l_filtered_chains: list of chains to be split
    @param ch_j: source chain
    @return: split chains list
    """
    l_search_chains = []
    for inter_chain in l_filtered_chains:
        split_node = inter_chain.get_node_by_angle(direction)
        if split_node is None:
            continue
        sub_ch1, sub_ch2 = split_chain(inter_chain, split_node)
        # 1.0 Found what ch_i intersect the longest one
        ch_k = select_no_intersection_chain_at_endpoint(sub_ch1, sub_ch2, ch_j, direction)
        if ch_k is None:
            continue

        # 2.0 Longest ch_i intersect two times
        if intersection_between_chains(ch_k, ch_j):
            node_direction_2 = ch_j.extB.angle if split_node.angle == ch_j.extA.angle else ch_j.extA.angle
            split_node_2 = ch_k.get_node_by_angle(node_direction_2)
            if split_node_2 is None:
                continue
            sub_ch1, sub_ch2 = split_chain(ch_k, split_node_2)
            ch_k = select_no_intersection_chain_at_endpoint(sub_ch1, sub_ch2, ch_j, node_direction_2)
            if ch_k is None:
                continue

        ch_k.change_id(inter_chain.id)
        ch_k.label_id = inter_chain.label_id

        l_search_chains.append(ch_k)

    return l_search_chains


def split_intersecting_chain_in_other_endpoint(endpoint, src_chain, within_chain_set, within_nodes, chain_search_set):
    """
    Split intersecting ch_i in other endpoint
    @param endpoint:
    @param src_chain: source ch_i
    @param within_chain_set: chains within the region
    @param within_nodes: nodes within the region
    @param chain_search_set: chains to be split
    @return:
    """
    node_other_endpoint = src_chain.extB if endpoint == ch.EndPoints.A else src_chain.extA
    direction = node_other_endpoint.angle
    node_direction = [node for node in within_nodes if
                      ((node.angle == direction) and not (node.chain_id == src_chain.id))]
    direction_cad_id = set([node.chain_id for node in node_direction])
    intersect_chain_id = [cad.id for cad in within_chain_set if
                          cad.id in direction_cad_id]
    intersecting_chains_in_other_endpoint = [chain for chain in chain_search_set if
                                             chain.id in intersect_chain_id]
    chain_search_set = [chain for chain in chain_search_set if
                        chain not in intersecting_chains_in_other_endpoint]
    chain_search_set_in_other_endpoint = split_intersecting_chains(direction, intersecting_chains_in_other_endpoint,
                                                                   src_chain)
    chain_search_set += chain_search_set_in_other_endpoint

    return chain_search_set


def filter_no_intersected_chain_far(no_intersecting_chains, src_chain, endpoint, neighbourhood_size=45):
    """
    Filter the chains that are not intersected with the ch_j and are far from the endpoint
    @param no_intersecting_chains: list of no intersecting ch_i with src ch_i
    @param src_chain: source ch_i
    @param endpoint: endpoint of source ch_i
    @param neighbourhood_size: max total_nodes size
    @return: list of chains that are not intersected with the ch_j and are not far from the endpoint
    """
    closest_chains_set = []
    for chain in no_intersecting_chains:
        distance = ch.angular_distance_between_chains_endpoints(src_chain, chain, endpoint)
        if distance < neighbourhood_size:
            closest_chains_set.append((distance, chain))

    # sort by proximity to endpoint and return
    closest_chains_set.sort(key=lambda x: x[0])
    no_intersecting_chain_set = [chain for distance, chain in
                                 closest_chains_set]

    return no_intersecting_chain_set


def add_chains_that_intersect_in_other_endpoint(within_chain_set, no_intersections_chain, search_chain_set, src_chain,
                                                neighbourhood_size, endpoint):
    """
    Add chains that intersect in other endpoint
    @param within_chain_set: chains in region
    @param no_intersections_chain: chains that do not intersect with ch_j. Also ch_i that can be connected by this
    endpoint is added
    @param search_chain_set: candidate chains to be connected
    @param src_chain: source chan
    @param neighbourhood_size:
    @param endpoint: source ch_i endpoint
    @return:
    """
    for in_chain in within_chain_set:
        if in_chain in no_intersections_chain + search_chain_set:
            continue
        if ch.angular_distance_between_chains_endpoints(src_chain, in_chain, endpoint) < neighbourhood_size:
            endpoint_in_chain = in_chain.extA if ch.EndPoints.A == endpoint else in_chain.extB
            exist_intersection_in_other_endpoint = src_chain.get_node_by_angle(endpoint_in_chain.angle) is not None
            if exist_intersection_in_other_endpoint:
                # Check that there no intersection between both src endpoints
                sorted_order = ch.ClockDirection.clockwise if endpoint == ch.EndPoints.A else ch.ClockDirection.anti_clockwise
                in_chain_neighbourhood_nodes = in_chain.sort_dots(sorted_order)[:neighbourhood_size]
                if src_chain.get_node_by_angle(in_chain_neighbourhood_nodes[0].angle) is None:
                    search_chain_set.append(in_chain)
    return 0


def get_chains_that_satisfy_similarity_conditions(state, support_chain, src_chain, search_chain_set,
                                                  endpoint):
    """
    Get chains that satisfy similarity conditions
    @param state: debugging variable
    @param support_chain: support ch_i
    @param src_chain: source ch_i
    @param search_chain_set: list of candidate chains
    @param endpoint: source ch_i endpoint
    @return: list of ch_i  that satisfy similarity conditions
    """
    candidate_chains = []
    radial_distance_candidate_chains = []
    candidate_chain_euclidean_distance = []
    candidate_chain_idx = 0
    while True:
        if len(search_chain_set) <= candidate_chain_idx:
            break

        candidate_chain = search_chain_set[candidate_chain_idx]
        candidate_chain_idx += 1

        check_pass, distribution_distance = similarity_conditions(state=state, th_radial_tolerance=0.2,
                                                                  th_distribution_size=3, th_regular_derivative=2,
                                                                  derivative_from_center=False, ch_i=support_chain,
                                                                  ch_j=src_chain, candidate_chain=candidate_chain,
                                                                  endpoint=endpoint, check_overlapping=False)
        if check_pass:
            candidate_chains.append(candidate_chain)
            radial_distance_candidate_chains.append(distribution_distance)

            endpoint_src = src_chain.extA if endpoint == ch.EndPoints.A else src_chain.extB
            endpoint_candidate_chain = candidate_chain.extB if endpoint == ch.EndPoints.A else candidate_chain.extA
            endpoint_distance = ch.euclidean_distance_between_nodes(endpoint_src, endpoint_candidate_chain)
            candidate_chain_euclidean_distance.append(endpoint_distance)

    return candidate_chain_euclidean_distance, radial_distance_candidate_chains, candidate_chains


def select_closest_candidate_chain(candidate_chains, candidate_chain_euclidean_distance,
                                   radial_distance_candidate_chains, within_chains_set, aux_chain):
    """
    Select the ch_k candidate ch_i to src ch_i
    @param candidate_chains: list of ch_i candidate
    @param candidate_chain_euclidean_distance: euclidean distance of list of candidate ch_i
    @param radial_distance_candidate_chains: radial distance of list of candidaate ch_i
    @param within_chains_set: full list of ch_i within region
    @param aux_chain: check if the candidate ch_i is the same as aux_chain
    @return: candidate ch_i to be connected
    """
    candidate_chain = None
    diff = -1
    if len(candidate_chains) > 0:
        candidate_chain = candidate_chains[np.argmin(candidate_chain_euclidean_distance)]
        diff = np.min(radial_distance_candidate_chains)

    if aux_chain is not None and candidate_chain == aux_chain:
        if candidate_chain in within_chains_set:
            within_chains_set.remove(aux_chain)
            candidate_chain = None
            diff = -1

    return candidate_chain, diff


def select_nodes_within_region_over_ray(src_chain, endpoint_node, within_node_list):
    return [node for node in within_node_list if
            ((node.angle == endpoint_node.angle) and not (node.chain_id == src_chain.id))]


def extract_chains_ids_from_nodes(nodes_ray):
    return set([node.chain_id for node in nodes_ray])


def get_chains_from_ids(within_chains_set, chain_id_ray):
    return [chain for chain in within_chains_set if chain.id in chain_id_ray]


def get_chains_that_no_intersect_src_chain(src_chain, src_chain_angle_domain, within_chains_set,
                                           endpoint_chain_intersections):
    return [cad for cad in within_chains_set if
            cad not in endpoint_chain_intersections and cad != src_chain and
            not exist_angular_intersection_with_src_chain(cad, src_chain_angle_domain)]


def remove_chains_with_higher_overlapping_threshold(src_chain_angle_domain, endpoint_chain_intersections,
                                                    neighbourhood_size):
    return [inter_chain for inter_chain in endpoint_chain_intersections if
            not angular_domain_overlapping_higher_than_threshold(src_chain_angle_domain,
                                                                 inter_chain, overlapping_threshold=neighbourhood_size)]


def remove_none_elements_from_list(list):
    return [element for element in list if element is not None]


def split_and_connect_neighbouring_chains(l_within_nodes: List[ch.Node], l_within_chains, ch_j: ch.Chain,
                                          endpoint: int, outward_ring, inward_ring, neighbourhood_size,
                                          debug_params, save_path, aux_chain=None):
    """
    Logic for split and connect chains within region
    @param l_within_nodes: nodes within region
    @param l_within_chains: chains within region
    @param ch_j: source chain. The one that is being to connect if condition are met.
    @param endpoint: endpoint of source ch_j to find candidate chains to connect.
    @param outward_ring: outward support ch_i ring
    @param inward_ring: inward support ch_i ring
    @param neighbourhood_size: size of total_nodes to search for candidate chains
    @param debug_params: debug param
    @param save_path: debug param. Path to save debug images
    @param aux_chain: ch_i candidate to be connected by other endpoint. It is used to check that it is not connected by this endpoint
    @return: candidate ch_i to connect
    """
    img, iteration, debug = debug_params
    # 1.1 Get angle domain for source ch_j
    ch_j_angle_domain = ch.get_nodes_angles_from_list_nodes(ch_j.l_nodes)

    # 1.2 Get endpoint node
    ch_j_node = ch_j.extA if endpoint == ch.EndPoints.A else ch_j.extB

    # 2.0 Select ch_j  support chain over endpoint
    ch_i = select_support_chain(outward_ring, inward_ring, ch_j_node)

    # 2.1 Select within nodes over endpoint ray
    l_nodes_ray = select_nodes_within_region_over_ray(ch_j, ch_j_node, l_within_nodes)

    # 2.2 Select within chains id over endpoint ray
    l_chain_id_ray = extract_chains_ids_from_nodes(l_nodes_ray)

    # 2.3 Select within chains over endpoint ray by chain id
    l_endpoint_chains = get_chains_from_ids(l_within_chains, l_chain_id_ray)



    # 3.1 filter chains that intersect with an overlapping threshold higher than 45 degrees. If overlapping threshold is
    # so big, it is not a good candidate to connect
    l_filtered_chains = remove_chains_with_higher_overlapping_threshold(ch_j_angle_domain,
                                                                                  l_endpoint_chains,
                                                                                  neighbourhood_size)

    if debug:
        boundary_ring_list = remove_none_elements_from_list([outward_ring, inward_ring])
        ch.visualize_selected_ch_and_chains_over_image_(
            [ch_i, ch_j] + l_filtered_chains + boundary_ring_list, l_within_chains
            , img,
            f'{save_path}/{iteration[0]}_split_chains_{ch_j.label_id}_2_1_{endpoint}_{ch_i.label_id}.png')
        iteration[0] += 1

    # 4.0 Split intersection chains by endpoint
    l_candidates = split_intersecting_chains(ch_j_node.angle, l_filtered_chains, ch_j)
    if debug:
        ch.visualize_selected_ch_and_chains_over_image_(
            [ch_i, ch_j] + l_candidates + boundary_ring_list, l_within_chains
            , img,
            f'{save_path}/{iteration[0]}_split_chains_{ch_j.label_id}_2_2_{endpoint}.png')
        iteration[0] += 1

    # 5.0 Select chains that do not intersect to ch_j
    l_no_intersection_j = get_chains_that_no_intersect_src_chain(ch_j, ch_j_angle_domain, l_within_chains, l_endpoint_chains)

    if aux_chain is not None:
        l_no_intersection_j += [aux_chain]
    # 5.1 Add ch_i that intersect in other endpoint
    add_chains_that_intersect_in_other_endpoint(l_within_chains, l_no_intersection_j, l_candidates, ch_j,
                                                neighbourhood_size, endpoint)
    if debug:
        ch.visualize_selected_ch_and_chains_over_image_(
            [ch_i, ch_j] + l_candidates + boundary_ring_list, l_within_chains
            , img,
            f'{save_path}/{iteration[0]}_split_chains_{ch_j.label_id}_2_3_{endpoint}.png')
        iteration[0] += 1
    # 5.1 Split intersection chains by other endpoint
    l_candidates = split_intersecting_chain_in_other_endpoint(endpoint, ch_j, l_within_chains,
                                                              l_within_nodes,
                                                              l_candidates)
    if debug:
        ch.visualize_selected_ch_and_chains_over_image_(
            [ch_i, ch_j] + l_candidates + boundary_ring_list, l_within_chains
            , img,
            f'{save_path}/{iteration[0]}_split_chains_{ch_j.label_id}_2_4_{endpoint}.png')
        iteration[0] += 1

    # 6.0 Filter no intersected chains that are far from endpoint
    l_candidates += filter_no_intersected_chain_far(l_no_intersection_j, ch_j, endpoint, neighbourhood_size)
    if debug:
        ch.visualize_selected_ch_and_chains_over_image_(
            [ch_i, ch_j] + l_candidates + boundary_ring_list, l_within_chains
            , img,
            f'{save_path}/{iteration[0]}_split_chains_{ch_j.label_id}_2_5_{endpoint}.png')
        iteration[0] += 1
        counter_init = iteration[0]
        state = SystemStatus([ch_j.l_nodes], [ch_j], np.zeros((2, 2)), ch_j.center, img, debug=debug,
                             save=f"{save_path}", counter=iteration[0])

    else:
        state = None

    # 7.0 Get chains that satisfy similarity conditions
    l_ch_k_euclidean_distance, l_ch_k_radial_distance, l_ch_k = \
        get_chains_that_satisfy_similarity_conditions(state, ch_i, ch_j, l_candidates, endpoint)
    # 7.1  Select ch_k candidate ch_i that satisfy similarity conditions
    ch_k, diff = select_closest_candidate_chain(l_ch_k, l_ch_k_euclidean_distance,
                                                l_ch_k_radial_distance, l_within_chains,
                                                aux_chain)
    if debug:
        iteration[0] += state.counter - counter_init
        if ch_k is not None:
            ch.visualize_selected_ch_and_chains_over_image_(
                [ch_i, ch_j] + [ch_k] + boundary_ring_list, l_within_chains
                , img,
                f'{save_path}/{iteration[0]}_split_chains_{ch_j.label_id}_2_6_2_{endpoint}.png')
            iteration[0] += 1

    return ch_k, diff, ch_i


def debugging_postprocessing(debug, l_ch, img, l_within_chain_subset, filename, iteration):
    if debug:
        ch.visualize_selected_ch_and_chains_over_image_(l_ch, l_within_chain_subset
                                                        , img,
                                                        filename)
        iteration[0] += 1


def split_and_connect_chains(l_within_chain_subset: List[ch.Chain], inward_chain, outward_chain, l_ch_p, l_nodes_c,
                             neighbourhood_size=45, debug=False, img=None, save_path=None, iteration=None):
    """
    Split chains that intersect in other endpoint and connect them if connectivity goodness conditions are met
    @param l_within_chain_subset: uncompleted chains delimitated by inward_ring and outward_ring
    @param inward_chain: inward ring of the region.
    @param outward_chain: outward ring of the region.
    @param l_ch_p: full ch_i list
    @param l_nodes_c: full nodes list
    @param neighbourhood_size: total_nodes size to search for chains that intersect in other endpoint
    @param debug: Set to true if debugging is allowed
    @param img: debug parameter. Image matrix
    @param save_path: debug parameter. Path to save debugging images
    @param iteration: debug parameter. Iteration counter
    @return: boolean value indicating if a ch_i was completed over region

    """
    # Initialization step
    l_within_chain_subset.sort(key=lambda x: x.size, reverse=True)
    connected = False
    completed_chain = False
    ch_j = None
    debug_params = img, iteration, debug

    # Get inward nodes
    l_inward_nodes = ch.get_nodes_from_chain_list(l_within_chain_subset)
    # Main loop to split chains that intersect over endpoints
    generator = ChainsBag(l_within_chain_subset)
    while True:
        if not connected:
            if ch_j is not None and ch_j.is_full(regions_count=4):
                complete_chain_using_2_support_ring(inward_chain, outward_chain, ch_j)
                completed_chain = True
                debugging_postprocessing(debug, [ch_j], l_within_chain_subset, img ,
                                         f'{save_path}/{iteration[0]}_split_chains_{ch_j.label_id}.png', iteration)
                ch_j = None

            else:
                ch_j = generator.get_next_chain()

        if ch_j is None:
            break
        debugging_postprocessing(debug, [ch_j, inward_chain, outward_chain], l_within_chain_subset, img,
                                 f'{save_path}/{iteration[0]}_split_chains_{ch_j.label_id}_init.png', iteration)
        # 2.0 Split chains in endpoint A and get candidate ch_i
        endpoint = ch.EndPoints.A
        ch_k_a, diff_a, ch_i_a = split_and_connect_neighbouring_chains(l_inward_nodes, l_within_chain_subset, ch_j,
                                                                       endpoint, outward_chain, inward_chain,
                                                                       neighbourhood_size, debug_params,
                                                                       save_path=save_path)
        # 3.0 Split chains in endpoint B and get candidate ch_i
        endpoint = ch.EndPoints.B
        ch_k_b, diff_b, ch_i_b = split_and_connect_neighbouring_chains(l_inward_nodes, l_within_chain_subset, ch_j,
                                                                       endpoint, outward_chain, inward_chain,
                                                                       neighbourhood_size, debug_params,
                                                                       save_path=save_path, aux_chain=ch_k_a)

        if debug:
            candidates_set = []
            if ch_k_b is not None:
                candidates_set.append(ch_k_b)
                candidates_set.append(ch_i_b)

            if ch_k_a is not None:
                candidates_set.append(ch_k_a)
                candidates_set.append(ch_i_a)
            debugging_postprocessing(debug, [ch_j] + candidates_set, l_within_chain_subset, img,
                                     f'{save_path}/{iteration[0]}_split_chains_{ch_j.label_id}_candidate.png', iteration)

        connected, ch_i, endpoint = connect_radially_closest_chain(ch_j, ch_k_a, diff_a,
                                                                   ch_i_a, ch_k_b, diff_b,
                                                                   ch_i_b, l_ch_p,
                                                                   l_within_chain_subset, l_nodes_c,
                                                                   inward_chain, outward_chain)

        debugging_postprocessing(debug, [ch_i, ch_j], l_within_chain_subset, img,
                                 f'{save_path}/{iteration[0]}_split_chains_{ch_j.label_id}_end.png', iteration)

    return completed_chain


def connect_2_chain_via_support_chain(outward_chain, inward_chain, src_chain, candidate_chain, nodes_list, endpoint,
                                      chain_list, inner_chain_list):
    """
    Connect 2 chains using outward and inward ch_i as support chains
    @param outward_chain: outward support ch_i
    @param inward_chain: inward support ch_i
    @param src_chain: source ch_i
    @param candidate_chain: candidate ch_i
    @param nodes_list: full node list
    @param endpoint: source ch_i endpoint
    @param chain_list: full ch_i list
    @param inner_chain_list: ch_i list delimitated by inward_ring and outward_ring
    @return: None. Chains are modified in place. Candidate ch_i is removed from chain_list and inner_chain_list and
     ch_j is modified
    """
    connect_2_chain_via_inward_and_outward_ring(outward_chain, inward_chain, src_chain, candidate_chain, nodes_list,
                                                endpoint)

    # Remove ch_i from ch_i lists. Candidate ch_i must be removed from inner_chain_list(region) and chain_list(global)
    inner_candidate_chain = ch.get_chain_from_list_by_id(inner_chain_list, candidate_chain.id)
    if inner_candidate_chain is not None:
        cadena_ref_lista_original = inner_candidate_chain
        inner_chain_list.remove(cadena_ref_lista_original)
        chain_list.remove(cadena_ref_lista_original)

    global_candidate_chain = ch.get_chain_from_list_by_id(chain_list, candidate_chain.id)
    if global_candidate_chain is not None:
        chain_list.remove(global_candidate_chain)

    return


def connect_radially_closest_chain(src_chain, candidate_chain_a, diff_a, support_chain_a, candidate_chain_b, diff_b,
                                   support_chain_b, ch_p_list, within_chains_subset, node_c_list, inward_ring,
                                   outward_ring):
    """
    Given 2 candidate chains, connect the one that is radially closer to the source ch_i
    @param src_chain: source ch_i
    @param candidate_chain_a: candidate ch_i at endpoint A
    @param diff_a: difference between source ch_i and candidate ch_i at endpoint A
    @param support_chain_a: support ch_i at endpoint A
    @param candidate_chain_b: candidate ch_i at endpoint B
    @param diff_b: difference between source ch_i and candidate ch_i at endpoint B
    @param support_chain_b: support ch_i at endpoint B
    @param ch_p_list: full ch_i list over disk
    @param within_chains_subset: chains within the region of interest
    @param node_c_list: full node list over disk
    @param inward_ring: inward ring delimiting region of interest
    @param outward_ring: outward ring delimiting region of interest
    @return:
    """
    if (0 <= diff_a <= diff_b) or (diff_b < 0 and diff_a >= 0):
        candidate_chain = candidate_chain_a
        support_chain = support_chain_a
        endpoint = ch.EndPoints.A

    elif (0 <= diff_b < diff_a) or (diff_a < 0 and diff_b >= 0):
        candidate_chain = candidate_chain_b
        support_chain = support_chain_b
        endpoint = ch.EndPoints.B

    else:
        return False, support_chain_a, ''

    if candidate_chain.size + src_chain.size > candidate_chain.Nr:
        return False, support_chain_a, ''

    connect_2_chain_via_support_chain(outward_ring, inward_ring, src_chain, candidate_chain, node_c_list, endpoint,
                                      ch_p_list, within_chains_subset)

    return True, support_chain, endpoint


def postprocessing(l_ch_c, l_nodes_c, cy, cx, img_pre, save_path, debug):
    """
    Posprocessing chain list. Conditions are relaxed in order to re-fine ch_i connections
    @param l_ch_c: chain list
    @param l_nodes_c: node list
    @param cy: pith y's coordinate
    @param cx: pith x's coordinate
    @param save_path: debug locations
    @param img_pre: input image
    @param debug: debug flag
    @return:
    - l_ch_p: connected chains  list
    """
    # initialization
    l_ch_p = [ch.copy_chain(chain) for chain in l_ch_c]
    chain_was_completed = False
    idx_start = None
    # debug parameter
    iteracion = [0]
    # end initialization

    while True:
        ctx = DiskContext(l_ch_p, idx_start, save_path=save_path, img=img_pre)
        while len(ctx.completed_chains) > 0:
            ctx.update()
            if debug:
                ctx.drawing(iteracion[0])
                iteracion[0] += 1

            ############################################################################################################
            # First Postprocessing. Split all chains and connect them if it possible
            chain_was_completed = split_and_connect_chains(ctx.l_within_chains_subset, ctx.inward_ring,
                                                           ctx.outward_ring, l_ch_p, l_nodes_c,
                                                           neighbourhood_size=ctx.neighbourhood_size, debug=debug,
                                                           img=img_pre, save_path=save_path, iteration=iteracion)
            # If ch_i was completed, restart iteration
            if chain_was_completed:
                idx_start = ctx.idx
                break
            ############################################################################################################
            # Second posproccessing
            connect_chains_if_there_is_enough_data(ctx, l_nodes_c, l_ch_p)

            ############################################################################################################

            if ctx.exit():
                break

        if not chain_was_completed:
            break

    # Finale Step, fill ch_i
    complete_chains_if_required(l_ch_p)

    return l_ch_p


def connect_chains_if_there_is_enough_data(ctx, l_nodes_c, l_ch_p):
    """
    Connect chains if there is enough data. This is the last step of the postprocessing
    @param ctx: context object
    @param l_nodes_c: full node list in disk
    @param l_ch_p: full chain list in disk
    @return:
    """
    there_is_chain = len(ctx.l_within_chains_subset) == 1
    if there_is_chain:
        l_inward_chains = ctx.l_within_chains_subset[0]
        postprocessing_unique_chain(l_inward_chains, ctx.inward_ring, ctx.outward_ring, l_nodes_c)
        return

    more_than_1_chain = len(ctx.l_within_chains_subset) > 1
    if more_than_1_chain:
        postprocessing_more_than_one_chain_without_intersection(ctx.l_within_chains_subset, ctx.inward_ring,
                                                                ctx.outward_ring, l_nodes_c, l_ch_p)

    return 0


def complete_chains_if_required(ch_p):
    """
    Complete chains if full and size is less than Nr
    @param ch_p: ch_i list to complete
    @return:
    """
    chain_list = [chain for chain in ch_p if chain.type not in [ch.TypeChains.border]]
    for chain in chain_list:
        if chain.is_full() and chain.size < chain.Nr:
            inward_chain, outward_chain, _ = get_inward_and_outward_visible_chains(chain_list, chain, ch.EndPoints.A)
            if inward_chain is not None and outward_chain is not None:
                complete_chain_using_2_support_ring(inward_chain, outward_chain, chain)

            elif inward_chain is not None or outward_chain is not None:
                support_chain = None
                complete_chain_using_support_ring(support_chain, chain)

    return 0


def postprocessing_unique_chain(within_chain, inward_ring_chain, outward_ring_chain, node_list,
                                information_threshold=180):
    """
    Postprocessing for unique ch_i if ch_i size is greater than information threshold
    @param within_chain: ch_i in region
    @param inward_ring_chain: inward ring ch_i
    @param outward_ring_chain: outward ring ch_i
    @param node_list: full node list in disk
    @param information_threshold: data threshold
    @return:
    """
    if within_chain.size > information_threshold:
        complete_chain_using_2_support_ring(inward_ring_chain, outward_ring_chain, within_chain)

    return


def build_no_intersecting_chain_set(chains_subset):
    """
    Build a set of chains that do not intersect with each other.
    @param chains_subset: all chains within region
    @return: subset of chains that do not intersect with each other
    """
    chains_subset.sort(key=lambda x: x.size)
    chains_subset = [chain for chain in chains_subset if not chain.is_full()]
    no_intersecting_subset = []
    while len(chains_subset) > 0:
        longest_chain = chains_subset[-1]
        longest_chain_intersect_already_added_chain = len([chain for chain in no_intersecting_subset
                                                           if intersection_between_chains(chain, longest_chain)]) > 0

        chains_subset.remove(longest_chain)
        if longest_chain_intersect_already_added_chain:
            continue

        no_intersecting_subset.append(longest_chain)

    return no_intersecting_subset


def postprocessing_more_than_one_chain_without_intersection(chain_subset, outward_ring_chain, inward_ring_chain,
                                                            node_list, chain_list, information_threshold=180):
    """
    Postprocessing for more than one ch_i without intersection. If we have more than one ch_i in region that not intersect
    each other. This ch_i subset also have to have an angular domain higher than information_threshold. Then we iterate over
     the chains and if satisfy similarity condition, we can connect them.
    @param chain_subset: chains in region defined by outward and inward ring
    @param outward_ring_chain: outward ring ch_i
    @param inward_ring_chain: inward ring ch_i
    @param node_list: full node list in all the disk
    @param chain_list: full ch_i list in all the disk, not only the region
    @param information_threshold:
    @return: connect chains if it possible
    """
    # get all the chains that not intersect each other
    no_intersecting_subset = build_no_intersecting_chain_set(chain_subset)
    enough_information = np.sum([cad.size for cad in no_intersecting_subset]) > information_threshold
    if not enough_information:
        return 0

    no_intersecting_subset.sort(key=lambda x: x.extA.angle)

    # Fist ch_i. All the nodes of ch_i that satisfy similarity condition will be added to this ch_i
    src_chain = no_intersecting_subset.pop(0)
    endpoint_node = src_chain.extB
    endpoint = ch.EndPoints.B
    # Select radially closer ch_i to ch_j endpoint
    support_chain = select_support_chain(outward_ring_chain, inward_ring_chain, endpoint_node)

    # Iterate over the rest of chains
    while len(no_intersecting_subset) > 0:
        next_chain = no_intersecting_subset[0]
        check_pass, distribution_distance = similarity_conditions(None, 0.2, 3, 2, False, support_chain, src_chain,
                                                                  next_chain, endpoint, check_overlapping=True,
                                                                  chain_list=chain_subset)

        if check_pass:
            #connect candidate_chain to ch_j
            connect_2_chain_via_support_chain(outward_ring_chain, inward_ring_chain, src_chain,
                                              next_chain, node_list, endpoint, chain_list,
                                              no_intersecting_subset)
        else:
            no_intersecting_subset.remove(next_chain)

    complete_chain_using_2_support_ring(inward_ring_chain, outward_ring_chain, src_chain)

    return 0
