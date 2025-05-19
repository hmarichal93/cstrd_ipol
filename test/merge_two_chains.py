import numpy as np

import lib.chain as ch
from lib.connect_chains import merge_two_chains, update_chain_list, SystemStatus, compute_intersection_matrix
from lib.interpolation_nodes import from_polar_to_cartesian
def create_chain(ch_j_id, Nr, center, img_height, img_width, chain_type, nodes_list):
    ch_j = ch.Chain(ch_j_id, Nr, center, img_height, img_width, type=chain_type)
    ch_j.add_nodes_list([node for node in nodes_list if node.chain_id == ch_j_id])
    return ch_j

def set_up(angular_range = 120, y_position = 10, Nr=360, img_height=150, img_width=150, chain_type=ch.TypeChains.normal,
           support_chain_pith_distance = 5, angle_ch_j = 51, angle_ch_k = 69):
    image = np.zeros((img_height, img_width), dtype=np.uint8)
    center = ch.Node(x=img_width // 2, y= img_height // 2, chain_id=1, radial_distance=0, angle=0)
    nodes_list = []
    ch_j_id = 1
    ch_k_id = 2
    ch_i_id = 0
    for angle in range(angular_range):
        if angle < angle_ch_j:
            cad_id = ch_j_id
        elif angle>angle_ch_k:
            cad_id = ch_k_id
        else:
            continue
        i, j = from_polar_to_cartesian(y_position, angle, (center.x, center.y))
        node = ch.Node(**{'y': i, 'x': j, 'angle': angle % 360, 'radial_distance': y_position,
                   'chain_id': cad_id})
        nodes_list.append(node)

    ch_j = create_chain(ch_j_id, Nr, (center.y, center.x), img_height, img_width, chain_type, nodes_list)
    ch_k = create_chain(ch_k_id, Nr, (center.y, center.x), img_height, img_width, chain_type, nodes_list)

    for angle in range(angular_range):
        cad_id = ch_i_id
        i, j = from_polar_to_cartesian(support_chain_pith_distance, angle, (center.x, center.y))
        node = ch.Node(**{'y': i, 'x': j, 'angle': angle % 360, 'radial_distance': support_chain_pith_distance,
                   'chain_id': cad_id})
        nodes_list.append(node)

    ch_i = create_chain(ch_i_id, Nr, (center.y, center.x), img_height, img_width, chain_type, nodes_list)

    endpoint = ch.EndPoints.B
    return ch_j, ch_k, ch_i, endpoint, image

def set_up_case_2(angular_range = 120, y_position = 30, Nr=360, img_height=150, img_width=150,
                  chain_type=ch.TypeChains.normal):
    ch_j, ch_k, ch_i, endpoint, image = set_up(angular_range = angular_range, y_position = y_position, Nr=Nr,
                                               img_height=img_height,
                                               img_width=img_width, chain_type=chain_type,
                                               angle_ch_j=20, angle_ch_k=100)
    center = ch.Node(x=img_width // 2, y= img_height // 2, chain_id=1, radial_distance=0, angle=0)
    nodes_list = []
    ch_i_upper_id = ch_k.label_id + 1
    for angle in range(angular_range):
        cad_id = ch_i_upper_id
        radial_distance = y_position*1.2 #*  np.cos(np.deg2rad(angle))
        #gaussian nosie to radial_distance
        radial_distance +=0.35*angle
        radial_distance += np.random.normal(0, 0.5)

        i, j = from_polar_to_cartesian(radial_distance, angle, (center.x, center.y))
        node = ch.Node(**{'y': i, 'x': j, 'angle': angle % 360, 'radial_distance': radial_distance,
                   'chain_id': cad_id})
        nodes_list.append(node)

    ch_i_upper = create_chain(ch_i_upper_id, Nr, (center.y, center.x), img_height, img_width, chain_type, nodes_list)

    return ch_j, ch_k, ch_i, endpoint, image, ch_i_upper

def test_merge_two_chains_case_1():
    ch_j, ch_k, ch_i, endpoint, image = set_up()
    print("ch_k", ch_k)
    print("ch_i", ch_i)
    print("ch_j", ch_j)
    ch.visualize_selected_ch_and_chains_over_image_([ch_i,ch_j, ch_k], [],
                                                    img=image, filename="test/merge_two_chain_case1_start.png")
    interpolated = merge_two_chains(ch_j, ch_k, endpoint, ch_i)
    print("interpolated", interpolated)
    ch.visualize_selected_ch_and_chains_over_image_([ch_i,ch_j, ch_k], [],
                                                    img=image, filename="test/merge_two_chain_case1_end.png")
    #check
    assert len([node for node in interpolated if node.chain_id != ch_j.label_id]) == 0
    assert ch_j.extB == ch_k.extB

def test_merge_two_chains_case_2():
    ch_j, ch_k, ch_i, endpoint, image, ch_i_upper = set_up_case_2()
    print("ch_k", ch_k)
    print("ch_i", ch_i)
    print("ch_j", ch_j)
    print("ch_i_upper", ch_i_upper)
    ch.visualize_selected_ch_and_chains_over_image_([ch_i,ch_j, ch_k, ch_i_upper], [],
                                                    img=image, filename="test/merge_two_chain_case2_start.png")
    interpolated = merge_two_chains(ch_j, ch_k, endpoint, ch_i, support2=ch_i_upper)
    print("interpolated", interpolated)
    ch.visualize_selected_ch_and_chains_over_image_([ch_i,ch_j, ch_k, ch_i_upper], [],
                                                    img=image, filename="test/merge_two_chain_case2_end.png")
    #check
    assert len([node for node in interpolated if node.chain_id != ch_j.label_id]) == 0
    assert ch_j.extB == ch_k.extB

    ch_j, ch_k, ch_i, endpoint, image, ch_i_upper = set_up_case_2()
    interpolated = merge_two_chains(ch_j, ch_k, endpoint, ch_i_upper)
    print("interpolated", interpolated)
    ch.visualize_selected_ch_and_chains_over_image_([ch_j, ch_k, ch_i_upper],
                                                    [ch_i,ch_j, ch_k, ch_i_upper],
                                                    img=image, filename="test/merge_two_chain_case3_end.png")
def test_update_chain_list():
    ch_j, ch_k, ch_i, endpoint, image = set_up()
    l_ch_s = [ch_j, ch_k, ch_i]
    l_nodes_s = [node for chain in l_ch_s for node in chain.l_nodes]
    nr = 360
    cy, cx = ch_i.center
    M = compute_intersection_matrix(l_ch_s, l_nodes_s, Nr=nr)
    state = SystemStatus(l_ch_s, l_nodes_s, M, cy, cx, Nr=nr, save='test/', img=image)
    print("M", M)
    print("ch_k", ch_k)
    print("ch_i", ch_i)
    print("ch_j", ch_j)
    ch.visualize_selected_ch_and_chains_over_image_([ch_i,ch_j, ch_k], [],
                                                    img=image, filename="test/update_chain_list_case1_start.png")
    interpolated = merge_two_chains(ch_j, ch_k, endpoint, ch_i)
    print("interpolated", interpolated)

    l_candidates = [ch_k]
    # Line 11
    update_chain_list(state, ch_j, ch_k, l_candidates, interpolated)
    print("M", M)
    print("ch_k", ch_k)
    print("ch_i", ch_i)
    print("ch_j", ch_j)
    print("l_candidates", l_candidates)
    print("l_ch_s", state.l_ch_s)
    assert len(l_candidates) == 0
    assert ch_k not in state.l_ch_s

    ch.visualize_selected_ch_and_chains_over_image_(state.l_ch_s, [],
                                                    img=image, filename="test/update_chain_list_case1_end.png")



if __name__ == "__main__":
    test_merge_two_chains_case_1()
    test_merge_two_chains_case_2()
    test_update_chain_list()
