import numpy as np
import lib.chain_v4 as ch
from lib.io import Nr
from lib.utils import write_log

MODULE_NAME = 'interpolacion'
def get_closest_chain_dot_to_angle(chain,angle):
    label='get_closest_chain_dot_to_angle'
    chain_dots = chain.sort_dots(sentido='antihorario')
    A = chain.extA.angulo
    B = chain.extB.angulo
    dominio = chain._completar_dominio_angular(chain)
    if angle not in dominio:
        if np.abs(A-angle)>np.abs(B-angle):
            return chain.extB
        else:
            return chain.extA
    closest_dot = None
    if A<=B:
        for dot in chain_dots:
            if dot.angulo>=angle:
                closest_dot = dot
                break

    else:
        for dot in chain_dots:
            if ((A<=dot.angulo and angle>=A) or (B>= dot.angulo and angle<=B)):
                if dot.angulo>=angle:
                    closest_dot = dot
                    break
            elif B>= dot.angulo and angle>B:
                closest_dot = dot
                break
    if closest_dot is None:
        d1 = np.abs(B-angle)
        d2 = np.abs(A-angle)
        if d1> d2:
            closest_dot = chain.extA
        else:
            closest_dot = chain.extB
    #write_log(MODULE_NAME, label, f"cad.id {chain.id} angle {angle} A {A} B {B} closest {closest_dot}")
    return closest_dot

def get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(chains_list,angle):
    label = 'get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center'
    lista_puntos_perfil = []
    for chain in chains_list:
        dot = get_closest_chain_dot_to_angle(chain, angle)
        if dot is not None:
            lista_puntos_perfil.append(dot)
    #write_log(MODULE_NAME,label,f"{lista_puntos_perfil}")
    if len(lista_puntos_perfil)>0:
        lista_puntos_perfil= sorted(lista_puntos_perfil, key=lambda x: x.radio, reverse=False)
    return lista_puntos_perfil


def calcular_dominio_de_interpolacion(extremo, extremo_cad1, extremo_cad2):
    dominio_de_interpolacion = []

    step = 360 / Nr if extremo in 'B' else -360 / Nr
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


def generar_lista_puntos_entre_dos_distancias_radiales(r2, r1, total_de_puntos, dominio_de_interpolacion, centro, signo,
                                                       cadena_anillo_soporte, cad):
    cad_id = cad.id
    lista_puntos_generados = []
    m = (r2 - r1) / (total_de_puntos - 0)
    n = r1 - m * 0
    for idx_pto_actual, angulo in enumerate(dominio_de_interpolacion):
        if cadena_anillo_soporte is not None:
            dot_list_in_radial_direction = get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
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
        }

        punto = ch.Punto(**params)
        lista_puntos_generados.append(punto)

    return lista_puntos_generados


def interpolar_en_domino(cadena_anillo_soporte, extremo_cad1, extremo_cad2, extremo, cad1, listaPuntos):

    dominio_de_interpolacion = calcular_dominio_de_interpolacion(extremo, extremo_cad1,extremo_cad2)
    centro = cad1.centro
    if cadena_anillo_soporte is not None:
        dot_list_in_radial_direction = get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
            [cadena_anillo_soporte], extremo_cad1.angulo)
        soporte_pto1 = dot_list_in_radial_direction[0]
        r1 = ch.distancia_entre_puntos(soporte_pto1,extremo_cad1)
        dot_list_in_radial_direction = get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
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

def completar_cadena_via_anillo_soporte(cadena_anillo_soporte, cad1, listaPuntos):
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

    interpolar_en_domino(cadena_anillo_soporte, extremo_cad1, extremo_cad2, extremo, cad1, lista_puntos_generados)
    cad1.add_lista_puntos(lista_puntos_generados)
    #assert len(cad1.lista) == len([punto for punto in listaPuntos if punto.cadenaId == cad1.id])
    ###check part
    dominio_angular_cad1 = cad1._completar_dominio_angular(cad1)
    assert len(dominio_angular_cad1) == cad1.size
    return

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


    #2.0 Pasar puntos de cadena 2 a 1.
    puntos = []
    puntos += cad2.lista
    for punto in puntos:
        punto.cadenaId = cad1.id

    cad1.add_lista_puntos(puntos)
    if add:
        cad1.add_lista_puntos(lista_puntos_generados)
        ###########################check part###########################################################################
        dominio_angular_cad1 = cad1._completar_dominio_angular(cad1)
        assert len(dominio_angular_cad1) == cad1.size
        ################################################################################################################


    # ##########################################################

    return lista_puntos_generados