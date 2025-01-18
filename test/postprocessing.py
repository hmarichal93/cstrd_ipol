import lib.postprocessing as post # import the module
from lib.utils import load_pickle
import lib.chain as ch

from pathlib import Path
from shapely.geometry import Polygon, LineString

def test_split_and_connect_neighbouring_chains(data):
    l_within_chains = data['l_within_chains']
    l_within_nodes = data['l_within_nodes']
    ch_j = data['ch_j']
    endpoint = data['endpoint']
    outward_ring = data['outward_ring']
    inward_ring = data['inward_ring']
    neighbourhood_size = data['neighbourhood_size']
    debug_params = data['debug_params']
    aux_chain = data['aux_chain']

    save_path = "./output/split_and_connect_neighbouring_chains"
    Path(save_path).mkdir(exist_ok=True, parents=True)
    ch_k = post.split_and_connect_neighbouring_chains(l_within_nodes, l_within_chains, ch_j, endpoint, outward_ring,
                                                      inward_ring, neighbourhood_size,debug_params, save_path, aux_chain)

    return

def from_chain_to_shapelly(chain,type=1):
    points,_,_ = chain.to_array()
    if type == 1:
        return LineString(points)
    return Polygon(points)

def test_split_and_connect(data):
    l_within_chains = data['l_within_chains']
    #l_within_nodes = data['l_within_nodes']
    #ch_j = data['ch_j']
    #endpoint = data['endpoint']
    outward_ring = data['outward_ring']
    inward_ring = data['inward_ring']
    neighbourhood_size = data['neighbourhood_size']
    debug_params = data['debug_params']
    #aux_chain = data['aux_chain']
    l_ch_p = data['l_ch_p']
    l_nodes_c = data['l_nodes_c']
    img, iteration, debug = debug_params
    debug_img_pre = img.copy()
    save_path = "./output/split_and_connect"
    if Path(save_path).exists():
        import os
        os.system(f"rm -r {save_path}")
    Path(save_path).mkdir(exist_ok=True, parents=True)

    ###
    iteration[0] += 0
    ch.visualize_selected_ch_and_chains_over_image_(
        l_within_chains, [outward_ring, inward_ring], img, f'{save_path}/{iteration[0]}_init.png')
    iteration[0] += 1


    res = post.split_and_connect_chains(l_within_chains, inward_ring, outward_ring,
                                        l_ch_p, l_nodes_c, neighbourhood_size=neighbourhood_size,
                                        debug=debug, img=debug_img_pre, save_path=save_path,
                                        iteration=iteration)
    assert  ch.get_chain_from_list_by_id(l_ch_p, 13) is None
    ch.visualize_selected_ch_and_chains_over_image_(
        l_ch_p, [outward_ring, inward_ring], img, f'{save_path}/{iteration[0]}_end.png')
    iteration[0] += 1
    return

def test_split_and_connect_2(data):
    l_within_chains = data['l_within_chains']
    #l_within_nodes = data['l_within_nodes']
    #ch_j = data['ch_j']
    #endpoint = data['endpoint']
    outward_ring = data['outward_ring']
    inward_ring = data['inward_ring']
    neighbourhood_size = data['neighbourhood_size']
    debug_params = data['debug_params']
    #aux_chain = data['aux_chain']
    l_ch_p = data['l_ch_p']
    l_nodes_c = data['l_nodes_c']
    img, iteration, debug = debug_params
    debug_img_pre = img.copy()
    save_path = "./output/split_and_connect"
    if Path(save_path).exists():
        import os
        os.system(f"rm -r {save_path}")

    Path(save_path).mkdir(exist_ok=True, parents=True)

    ###
    iteration[0] += 0
    ch.visualize_selected_ch_and_chains_over_image_(
        l_within_chains, [outward_ring, inward_ring], img, f'{save_path}/{iteration[0]}_init.png')
    iteration[0] += 1


    res = post.split_and_connect_chains(l_within_chains, inward_ring, outward_ring,
                                        l_ch_p, l_nodes_c, neighbourhood_size=neighbourhood_size,
                                        debug=debug, img=debug_img_pre, save_path=save_path,
                                        iteration=iteration)
    ch.visualize_selected_ch_and_chains_over_image_(
        l_ch_p, [outward_ring, inward_ring], img, f'{save_path}/{iteration[0]}_end.png')
    iteration[0] += 1
    return

def main():
    print("Test 1")
    #test_split_and_connect(load_pickle("test/input/obj_to_save_599.pkl"))
    print("Test 2")

    test_split_and_connect_2(load_pickle("test/input/global_chain_is_not_none.pkl"))
    return

if __name__ == "__main__":
    main()