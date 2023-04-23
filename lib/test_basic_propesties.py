import numpy as np
import cv2

import lib.chain as ch
from lib.drawing import Drawing
import lib.basic_properties as bp

def add_chain(x, y, center, l_nodes, l_chains, x_max, y_max, chain_id):
    l_chain_nodes = []
    for i in range(len(x)):
        l_chain_nodes.append(ch.Node(x[i], y, radial_distance=ch.euclidean_distance(np.array([x[i], y]), center), angle=x[i],
                               chain_id=chain_id))

    chain = ch.Chain(chain_id=chain_id, Nr=360, center=center, img_width=x_max, img_height=y_max)
    chain.add_nodes_list(l_chain_nodes)
    l_chains.append(chain)
    l_nodes += l_chain_nodes


def generate_img():
    y = 1
    y_max = 10
    x_max = 10
    img = np.zeros((y_max, x_max, 3), dtype=np.uint8) + 255
    center = np.array([y_max / 2, x_max / 2])
    x = np.arange(1, 5, 1)
    l_nodes = []
    l_chains = []
    add_chain(x, y, center, l_nodes, l_chains, x_max, y_max, 1)

    ch_id = 2
    y = 2
    x = np.arange(1, 9, 1)
    add_chain(x, y, center, l_nodes, l_chains, x_max, y_max, ch_id)

    ch_id = 3
    y = 3
    x = np.arange(5, 9, 1)
    add_chain(x, y, center, l_nodes, l_chains, x_max, y_max, ch_id)

    ch_id = 0
    y = 0
    x = np.arange(0, 10, 1)
    add_chain(x, y, center, l_nodes, l_chains, x_max, y_max, ch_id)

    ch.visualize_selected_ch_and_chains_over_image_(l_chains, [], img=img, filename="test.png")

    #################################################################################################
    sup_band = []
    inf_band = []
    sup_band.append(ch.Node( x=4, y=0.5, chain_id=5, radial_distance= ch.euclidean_distance(np.array([4, 0.5]), center), angle=4))
    sup_band.append(
        ch.Node(x=5, y=2.5, chain_id=5, radial_distance=ch.euclidean_distance(np.array([5, 2.5]), center), angle=5))

    inf_band.append(ch.Node(x=4, y=1.5, chain_id=6, radial_distance= ch.euclidean_distance(np.array([4, 1.5]), center), angle=4))
    inf_band.append(
        ch.Node(x=5, y=3.5, chain_id=6, radial_distance=ch.euclidean_distance(np.array([5, 3.5]), center), angle=5))
    dominio = [4,5]
    ch_j = l_chains[0]
    ch_k = l_chains[2]
    ch_i = l_chains[3]
    endpoint_type = ch.EndPoints.A
    info_band = bp.InfoVirtualBand(l_nodes= l_nodes, ch_j = ch_j, ch_k= ch_k,endpoint= endpoint_type,
                                ch_i = ch_i, debug=True, domain=dominio, sup_band=sup_band, inf_band=inf_band)

    img_band = info_band.draw_band(img.copy(),l_chains)
    cv2.imwrite("test_band.png", img_band)

    ch_l = l_chains[1]
    exist_chain_in_band = info_band.is_chain_in_band(ch_l)
    print(exist_chain_in_band)
    #l_chains_in_band = bp.exist_chain_in_band_logic(l_chains, info_band)
    return img,l_chains,l_nodes

def main():
    im_in, l_chains, l_nodes = generate_img()




    return


if __name__=='__main__':
    main()