import numpy as np

import lib.chain_v4 as ch
from lib.utils import write_log

MODULE_NAME = 'interpolacion'



def calcular_dominio_de_interpolacion(extremo, extremo_cad1, extremo_cad2):
    dominio_de_interpolacion = []

    step = 360 / extremo_cad1.Nr if extremo in 'B' else -360 / extremo_cad1.Nr
    angulo_actual = extremo_cad1.angulo
    while angulo_actual %360 != extremo_cad2.angulo:
        angulo_actual += step
        angulo_actual = angulo_actual % 360
        dominio_de_interpolacion.append(angulo_actual)

    return dominio_de_interpolacion[:-1]

def from_polar_to_cartesian(r,angulo,centro):
    y = centro[0] + r * np.cos(angulo * np.pi / 180)
    x = centro[1] + r * np.sin(angulo * np.pi / 180)
    return (y,x)

def generar_lista_puntos_entre_dos_distancias_radiales_y_dos_cadenas_soporte(r2_ratio, r1_ratio, total_de_puntos, dominio_de_interpolacion, centro,
                                                       cadena_inferior, cadena_superior, cad):
    cad_id = cad.id
    lista_puntos_generados = []
    m = (r2_ratio - r1_ratio) / total_de_puntos
    n = r1_ratio
    for idx_pto_actual, angulo in enumerate(dominio_de_interpolacion):
        dot_list_in_radial_direction = ch.get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
            [cadena_inferior], angulo % 360)
        punto_soporte = dot_list_in_radial_direction[0]
        radio_init = punto_soporte.radio
        dot_list_in_radial_direction = ch.get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
            [cadena_superior], angulo % 360)
        punto_soporte = dot_list_in_radial_direction[0]
        radio_superior = punto_soporte.radio
        radial_distance_between_chains = radio_superior - radio_init

        radio_inter =  (m * (idx_pto_actual) + n)*radial_distance_between_chains + radio_init
        i, j = from_polar_to_cartesian(radio_inter, angulo % 360, centro)
        i = i if i < cad.M else cad.M - 1
        j = j if j < cad.N else cad.N - 1

        # radios.append(radio_inter)
        params = {
            "x": i,
            "y": j,
            "angulo": angulo % 360,
            "radio": radio_inter,
            "gradFase": -1,
            "cadenaId": cad_id,
            "Nr": cad.Nr
        }

        punto = ch.Punto(**params)
        lista_puntos_generados.append(punto)

    return lista_puntos_generados
def generar_lista_puntos_entre_dos_distancias_radiales(r2, r1, total_de_puntos, dominio_de_interpolacion, centro, signo,
                                                       cadena_anillo_soporte, cad):
    cad_id = cad.id
    lista_puntos_generados = []
    m = (r2 - r1) / (total_de_puntos - 0)
    n = r1 - m * 0
    for idx_pto_actual, angulo in enumerate(dominio_de_interpolacion):
        if cadena_anillo_soporte is not None:
            dot_list_in_radial_direction = ch.get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
                [cadena_anillo_soporte], angulo % 360)

            punto_soporte = dot_list_in_radial_direction[0]
            radio_init = punto_soporte.radio
            #radio_init = r1
        else:
            radio_init = 0

        radio_inter = signo * (m * (idx_pto_actual) + n) + radio_init
        i, j = from_polar_to_cartesian(radio_inter, angulo % 360, centro)
        i = i if i < cad.M else cad.M - 1
        j = j if j < cad.N else cad.N - 1

        # radios.append(radio_inter)
        params = {
            "x": i,
            "y": j,
            "angulo": angulo % 360,
            "radio": radio_inter,
            "gradFase": -1,
            "cadenaId": cad_id,
            "Nr": cad.Nr
        }

        punto = ch.Punto(**params)
        lista_puntos_generados.append(punto)

    return lista_puntos_generados

def get_radial_distance_to_chain(chain, dot):
    dot_list_in_radial_direction = ch.get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
        [chain], dot.angulo)
    soporte_pto1 = dot_list_in_radial_direction[0]
    rii = ch.distancia_entre_puntos(soporte_pto1,dot)
    return rii

def compute_radial_ratio(cadena_inferior,cadena_superior, dot):
    r1_inferior = get_radial_distance_to_chain(cadena_inferior, dot)
    r1_superior = get_radial_distance_to_chain(cadena_superior, dot)
    return  r1_inferior / (r1_superior+r1_inferior)
def interpolar_en_domino_via_dos_cadenas(cadena_anillo_soporte_inferior,cadena_anillo_soporte_superior,
                                         extremo_cad1, extremo_cad2, extremo, cad1, listaPuntos):

    dominio_de_interpolacion = calcular_dominio_de_interpolacion(extremo, extremo_cad1, extremo_cad2)
    centro = cad1.centro

    r1_ratio = compute_radial_ratio(cadena_anillo_soporte_inferior, cadena_anillo_soporte_superior, extremo_cad1)
    r2_ratio = compute_radial_ratio(cadena_anillo_soporte_inferior, cadena_anillo_soporte_superior, extremo_cad2)



    ###
    total_de_puntos = len(dominio_de_interpolacion)
    if total_de_puntos == 0:
        return

    lista_puntos_generados = generar_lista_puntos_entre_dos_distancias_radiales_y_dos_cadenas_soporte(r2_ratio, r1_ratio,
                    total_de_puntos, dominio_de_interpolacion, centro, cadena_anillo_soporte_inferior,
                    cadena_anillo_soporte_superior, cad1 )

    listaPuntos += lista_puntos_generados

    return
def interpolar_en_domino(cadena_anillo_soporte, extremo_cad1, extremo_cad2, extremo, cad1, listaPuntos):

    dominio_de_interpolacion = calcular_dominio_de_interpolacion(extremo, extremo_cad1,extremo_cad2)
    centro = cad1.centro
    if cadena_anillo_soporte is not None:
        dot_list_in_radial_direction = ch.get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
            [cadena_anillo_soporte], extremo_cad1.angulo)
        soporte_pto1 = dot_list_in_radial_direction[0]
        r1 = ch.distancia_entre_puntos(soporte_pto1,extremo_cad1)
        dot_list_in_radial_direction = ch.get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
            [cadena_anillo_soporte], extremo_cad2.angulo)
        soporte_pto2 = dot_list_in_radial_direction[0]
        signo = -1 if soporte_pto2.radio > extremo_cad2.radio else +1
        r2 = ch.distancia_entre_puntos(soporte_pto2, extremo_cad2)
    else:
        r1 = extremo_cad1.radio
        r2 = extremo_cad2.radio
        signo = 1
    ###
    total_de_puntos = len(dominio_de_interpolacion)
    if total_de_puntos == 0:
        return

    lista_puntos_generados = generar_lista_puntos_entre_dos_distancias_radiales(r2, r1, total_de_puntos, dominio_de_interpolacion, centro, signo,
                                                       cadena_anillo_soporte, cad1)

    listaPuntos += lista_puntos_generados

    return

def completar_cadena_via_anillo_soporte(cadena_inferior, cadena_superior,  cad1, listaPuntos):
    extremo_cad1 = cad1.extB
    extremo_cad2 = cad1.extA
    #ordenar puntos dominio de interpolacion de soporte_pto1 a soporte_pto2
    #dominio_de_interpolacion = [angulo for angulo in rango_angular if angulo not in dominio_interior]
    #ordenar puntos dominio de interpolacion de soporte_pto1 a soporte_pto2
    extremo = 'B'
    #assert len(cad1.lista) == len([punto for punto in listaPuntos if punto.cadenaId == cad1.id])
    lista_puntos_generados = []
    ###check part
    dominio_angular_cad1 = cad1._completar_dominio_angular(cad1)
    assert len(dominio_angular_cad1) == cad1.size

    interpolar_en_domino_via_dos_cadenas(cadena_inferior, cadena_superior,
                                         extremo_cad1, extremo_cad2, extremo, cad1, lista_puntos_generados)
    assert len([dot for dot in cad1.lista if dot in lista_puntos_generados]) == 0
    change_border = cad1.add_lista_puntos(lista_puntos_generados)

    #assert len(cad1.lista) == len([punto for punto in listaPuntos if punto.cadenaId == cad1.id])
    ###check part
    dominio_angular_cad1 = cad1._completar_dominio_angular(cad1)
    assert len(dominio_angular_cad1) == cad1.size
    return change_border

def pegar_dos_cadenas_interpolando_via_cadena_superior_e_inferior(cadena_superior, cadena_inferior, cad1, cad2,  listaPuntos, extremo,add=True, listaCadenas=None):
    label = 'pegar_dos_cadenas_interpolando_via_cadena_soporte'
    write_log(MODULE_NAME, label, f'cad1_{cad1.label_id}_cand_{cad2.label_id}_extremo_{extremo}')
    extremo_cad1 = cad1.extA if extremo in 'A' else cad1.extB
    extremo_cad2 = cad2.extB if extremo in 'A' else cad2.extA
    ###check part############################################
    dominio_angular_cad1 = cad1._completar_dominio_angular(cad1)
    assert len(dominio_angular_cad1) == cad1.size
    ##########################################################
    #1.0 Generar puntos de interpolacion
    lista_puntos_generados = []
    interpolar_en_domino_via_dos_cadenas(cadena_inferior,cadena_superior,
                                         extremo_cad1, extremo_cad2, extremo, cad1, lista_puntos_generados)
    listaPuntos += lista_puntos_generados
    assert len([dot for dot in cad1.lista if dot in lista_puntos_generados]) == 0


    #2.0 Pasar puntos de cadena 2 a 1.
    puntos = []
    puntos += cad2.lista
    for punto in puntos:
        punto.cadenaId = cad1.id

    change_border = cad1.add_lista_puntos(puntos)
    if add:
        cad1.add_lista_puntos(lista_puntos_generados)
        ###########################check part###########################################################################
        dominio_angular_cad1 = cad1._completar_dominio_angular(cad1)
        assert len(dominio_angular_cad1) == cad1.size
        ################################################################################################################


    # ##########################################################

    return lista_puntos_generados, change_border

def pegar_dos_cadenas_interpolando_via_cadena_soporte(cadena_anillo_soporte, cad1, cad2,  listaPuntos, extremo,add=True, listaCadenas=None):
    label = 'pegar_dos_cadenas_interpolando_via_cadena_soporte'
    write_log(MODULE_NAME, label, f'cad1_{cad1.label_id}_cand_{cad2.label_id}_extremo_{extremo}')
    extremo_cad1 = cad1.extA if extremo in 'A' else cad1.extB
    extremo_cad2 = cad2.extB if extremo in 'A' else cad2.extA
    ###check part############################################
    dominio_angular_cad1 = cad1._completar_dominio_angular(cad1)
    assert len(dominio_angular_cad1) == cad1.size
    ##########################################################
    #1.0 Generar puntos de interpolacion
    lista_puntos_generados = []
    interpolar_en_domino(cadena_anillo_soporte, extremo_cad1, extremo_cad2, extremo, cad1, lista_puntos_generados)
    listaPuntos += lista_puntos_generados
    assert len([dot for dot in cad1.lista if dot in lista_puntos_generados]) == 0


    #2.0 Pasar puntos de cadena 2 a 1.
    puntos = []
    puntos += cad2.lista
    for punto in puntos:
        punto.cadenaId = cad1.id

    change_border = cad1.add_lista_puntos(puntos)
    if add:
        cad1.add_lista_puntos(lista_puntos_generados)
        ###########################check part###########################################################################
        dominio_angular_cad1 = cad1._completar_dominio_angular(cad1)
        assert len(dominio_angular_cad1) == cad1.size
        ################################################################################################################


    # ##########################################################

    return lista_puntos_generados, change_border