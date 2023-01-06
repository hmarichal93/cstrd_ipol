
from pathlib import Path
from shapely.geometry import LineString, Point, Polygon
import numpy as np
import matplotlib.pyplot as plt
import cv2

import lib.chain_v4 as ch
import lib.union_puntos_prolijo_performance_3 as union
from lib.interpolacion import completar_cadena_via_anillo_soporte, interpolar_en_domino_via_dos_cadenas, pegar_dos_cadenas_interpolando_via_cadena_superior_e_inferior
from lib.propiedades_fundamentales import InfoBandaVirtual, hay_cadenas_superpuestas_en_banda, criterio_distancia_radial_no_debugging,\
    derivada_maxima, generar_puntos_virtuales_sin_cadena_soporte
from lib.dibujar import Dibujar

def put_text(text, image, org, color=(0, 0, 0), fontScale=1 / 4):
    # font
    font = cv2.FONT_HERSHEY_DUPLEX
    # fontScale

    # Blue color in BGR

    # Line thickness of 2 px
    thickness = 1

    # Using cv2.putText() method
    image = cv2.putText(image, text, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)

    return image

class Anillo(Polygon):
    def __init__(self,cadena,id):
        lista_pts = [[punto.x, punto.y] for punto in cadena.sort_dots()]
        self.id = id
        super(self.__class__, self).__init__(lista_pts)

    def dibujar(self,image):
        x,y = self.exterior.coords.xy
        lista_pts = [[i,j] for i,j in zip(y,x)]
        pts = np.array(lista_pts,
                       np.int32)

        pts = pts.reshape((-1, 1, 2))
        isClosed = True
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 1
        # Using cv2.polylines() method
        # Draw a Blue polygon with
        # thickness of 1 px
        image = cv2.polylines(image, [pts],
                              isClosed, color, thickness)

        image = put_text(f'{self.id}',image,(int(y[0]),int(x[0])))

        return image

def construir_poligono_limite(anillo_externo, anillo_interno):
    if anillo_externo is not None and anillo_interno is not None:
        x, y = anillo_externo.exterior.coords.xy
        pts_ext = [[j, i] for i, j in zip(y, x)]
        x, y = anillo_interno.exterior.coords.xy
        pts_int = [[j, i] for i, j in zip(y, x)]
        poligono = Polygon(pts_ext, [pts_int])

    else:
        if anillo_externo is None:
            x, y = anillo_interno.exterior.coords.xy
        else:
            x, y = anillo_externo.exterior.coords.xy
        pts_ext = [[j, i] for i, j in zip(y, x)]
        poligono = Polygon(pts_ext)

    return poligono

def buscar_cadenas_interiores_shapely(cadenas_incompletas_shapely, anillo_externo, anillo_interno):
    poligono = construir_poligono_limite( anillo_externo, anillo_interno)
    contains = np.vectorize(lambda p: poligono.contains(Point(p)), signature='(n)->()')
    subconjunto_cadenas_interiores_shapely = []
    for cadena in cadenas_incompletas_shapely:
        x, y = cadena.xy
        pts = [[i, j] for i, j in zip(y, x)]
        if len(pts)==0:
            continue
        try:
            vector = contains(np.array(pts))
        except Exception as e:
            continue
        if anillo_externo is not None:
            if vector.sum() == vector.shape[0]:
                subconjunto_cadenas_interiores_shapely.append(cadena)
        else:
            if vector.sum() == 0:
                subconjunto_cadenas_interiores_shapely.append(cadena)

    return subconjunto_cadenas_interiores_shapely


def convertir_cadena_shapely_a_cadena(cadenas_incompletas_shapely, cadenas_incompletas, subconjunto_cadenas_interiores_shapely):
    conjunto_cadenas_interiores = [cadenas_incompletas[cadenas_incompletas_shapely.index(cad_shapely)]
                                   for cad_shapely in subconjunto_cadenas_interiores_shapely]
    conjunto_cadenas_interiores.sort(key=lambda x: x.size, reverse=True)
    return conjunto_cadenas_interiores


class ContextoDisco:
    def __init__(self, listaCadenas, save_path, img, debug=True, regions_cadena_completa= 8,idx_start = None):
        self.debug = debug
        self.save_path = save_path
        self.img = img
        self.cadenas_completas = [cad for cad in listaCadenas if cad.size>= cad.Nr]
        self.cadenas_completas, self.cadenas_completas_poligonos = self._convertir_cadenas_completas_a_poligonos(
            self.cadenas_completas)

        #self.cadenas_incompletas = [cad for cad in listaCadenas if not cad.esta_completa(regiones=regions_cadena_completa)]
        self.cadenas_incompletas = [cad for cad in listaCadenas if  cad.size < cad.Nr]
        self.cadenas_incompletas_poligonos = self._convertir_cadenas_incompletas_a_poligonos(self.cadenas_incompletas)
        self.idx = 1 if idx_start is None else idx_start

    def obtener_anillo_interior_y_exterior(self,idx):
        self.vecindario_amplitud = 45
        if len(self.cadenas_completas_poligonos) > idx > 0:
            anillo_interior = self.cadenas_completas_poligonos[idx - 1]
            anillo_exterior = self.cadenas_completas_poligonos[idx]

        if self.idx>10:
            self.vecindario_amplitud = self.vecindario_amplitud // 2
        #TODO: agregar decremento de amplitud vecindario segun distancia al centro y al borde.

        return anillo_interior, anillo_exterior

    def update(self):

        anillo_interior_poligono, anillo_exterior_poligono = self.obtener_anillo_interior_y_exterior(self.idx)
        subconjunto_cadenas_interiores_shapely = buscar_cadenas_interiores_shapely(self.cadenas_incompletas_poligonos,
                                                                                   anillo_exterior_poligono,
                                                                                   anillo_interior_poligono)
        self.subconjunto_cadenas_interiores = convertir_cadena_shapely_a_cadena(self.cadenas_incompletas_poligonos,
                                                                           self.cadenas_incompletas,
                                                                           subconjunto_cadenas_interiores_shapely)

        self.anillo_interior, self.anillo_exterior = self._convertir_anillo_shapely_a_cadena(anillo_interior_poligono,
                                                                                             anillo_exterior_poligono)

    def exit(self):
        self.idx += 1
        if self.idx >= len(self.cadenas_completas):
            return True

        return False

    def dibujar(self, iteracion):
        ch.visualizarCadenasSobreDisco( self.subconjunto_cadenas_interiores +
            [cadena for cadena in [self.anillo_interior, self.anillo_exterior] if
                                           cadena is not None], self.img,
            f'{iteracion}_picar_cadenas_cadenas_interiores_{self.idx}', labels=False,gris=True,
            save=f"{self.save_path}")

    def _convertir_anillo_shapely_a_cadena(self, anillo_interior_poligono, anillo_exterior_poligono):
        cadena_anillo_interior = None
        cadena_anillo_exterior = None
        if anillo_interior_poligono is not None:
            cadena_anillo_interior = self.cadenas_completas[self.cadenas_completas_poligonos.index(
                anillo_interior_poligono)]

        if anillo_exterior_poligono is not None:
            cadena_anillo_exterior = self.cadenas_completas[
                self.cadenas_completas_poligonos.index(anillo_exterior_poligono)]
        return cadena_anillo_interior, cadena_anillo_exterior

    def sort_list_by_index_array(self, indexes, lista):
        Z = []
        for position in indexes:
            Z.append(lista[position])
        return Z

    def sort_shapely_list_and_cadena_list(self, cadena_list, shapely_list):
        idx_sort = [i[0] for i in sorted(enumerate(shapely_list), key=lambda x: x[1].area)]
        shapely_list = self.sort_list_by_index_array(idx_sort, shapely_list)
        cadena_list = self.sort_list_by_index_array(idx_sort, cadena_list)
        return cadena_list, shapely_list

    def _convertir_cadenas_completas_a_poligonos(self, cadenas_completas):
        cadenas_completas_poligonos = []
        for cadena in cadenas_completas:
            anillo = Anillo(cadena, id=cadena.id)
            cadenas_completas_poligonos.append(anillo)

        cadenas_completas, cadenas_completas_poligonos = self.sort_shapely_list_and_cadena_list(cadenas_completas,
                                                                                                cadenas_completas_poligonos)

        return cadenas_completas, cadenas_completas_poligonos

    def _convertir_cadenas_incompletas_a_poligonos(self,cadenas_incompletas):
        cadenas_incompletas_shapely = []
        for cadena in cadenas_incompletas:
            lista_pts = [Point(punto.y, punto.x) for punto in cadena.sort_dots()]
            cadenas_incompletas_shapely.append(LineString(lista_pts))

        return cadenas_incompletas_shapely


def postprocesamiento_cuando_hay_una_unica_cadena_grande(cadena_interior, cadena_anillo_interior, cadena_anillo_exterior,
                                                        listaPuntos, UMBRAL_INFORMACION=180):
    if cadena_interior.size > UMBRAL_INFORMACION:
        #punto_extremo = cadena_interior.extA
        # cadena_soporte = seleccionar_cadena_soporte(cadena_anillo_exterior, cadena_anillo_interior,
        #                                             punto_extremo)

        completar_cadena_via_anillo_soporte(cadena_anillo_interior, cadena_anillo_exterior, cadena_interior,
                                            listaPuntos)

    return

def main_postprocesamiento(results, debug=False):
    image = results['img'].copy()
    img = results['img'].copy()
    listaPuntos = results['listaPuntos']
    iteracion = [0]
    listaCadenas = results['listaCadenas']

    save_path = Path(results['save_path']) / "post_debug_1"
    #save_path.mkdir(exist_ok=True)
    if debug:
        save_path.mkdir(exist_ok=True)
        save_path = str(save_path)
        ch.visualizarCadenasSobreDisco(listaCadenas, image, f'picar_cadenas_inicio', labels=True, save=f"{save_path}")

    se_completo_cadena = False
    idx_start = None
    while True:
        ctx = ContextoDisco(listaCadenas, save_path, img,idx_start = idx_start)
        while len(ctx.cadenas_completas) > 0:
            ctx.update()
            if debug:
                ctx.dibujar(iteracion[0])
                iteracion[0] += 1

            ############################################################################################################
            #postprocesamiento 1
            se_completo_cadena = picar_y_unir_cadenas_region(iteracion, img, ctx.subconjunto_cadenas_interiores,
                                ctx.anillo_interior, ctx.anillo_exterior, listaCadenas, debug, save_path, listaPuntos,
                                                             ctx.idx, vecindario= ctx.vecindario_amplitud)



            ############################################################################################################
            # postprocesamiento 3: completar cadenas

            if se_completo_cadena:
                #hay que iniciar de nuevo
                idx_start = ctx.idx
                break
            ############################################################################################################
            #postprocesamiento 2
            ### LOGICA algoritmo
            hay_una_cadena = len(ctx.subconjunto_cadenas_interiores) == 1
            if hay_una_cadena:
                cadena_interior = ctx.subconjunto_cadenas_interiores[0]
                postprocesamiento_cuando_hay_una_unica_cadena_grande(cadena_interior,
                                                                     ctx.anillo_interior, ctx.anillo_exterior,
                                                                     listaPuntos)

            mas_de_1_cadena = len(ctx.subconjunto_cadenas_interiores) > 1
            if mas_de_1_cadena:
                if debug:
                    ctx.dibujar(iteracion[0])
                    iteracion[0] += 1
                postprocesamiento_cuando_hay_mas_de_1_cadena_sin_interseccion(ctx,ctx.subconjunto_cadenas_interiores,
                                                    ctx.anillo_interior, ctx.anillo_exterior, listaPuntos, listaCadenas)


            ############################################################################################################



            if ctx.exit():
                break


        if not se_completo_cadena:
            break

    # if debug:
    #     generate_pdf(save_path)
    if debug:
        ch.visualizarCadenasSobreDisco(listaCadenas, results['img'],
                                       f'final', labels=False, gris=True,
                                       save=f"{save_path}")
    results['listaCadenas'] = listaCadenas

    return

def armar_subconjunto_de_cadenas_no_intersectantes(subconjunto_clase_cadenas):
    subconjunto_clase_cadenas.sort(key=lambda x: x.size)
    subconjunto_clase_cadenas = [cad for cad in subconjunto_clase_cadenas if not cad.esta_completa()]
    subconjunto_cadenas_no_intersectantes = []
    while len(subconjunto_clase_cadenas) > 0:
        cadena_mayor = subconjunto_clase_cadenas[-1]
        cadena_mayor_intersecta_cadenas_ya_agregadas = len([cad for cad in subconjunto_cadenas_no_intersectantes
                                                        if union.cadenas_se_intersectan(cad, cadena_mayor)]) > 0

        subconjunto_clase_cadenas.remove(cadena_mayor)
        if cadena_mayor_intersecta_cadenas_ya_agregadas:
            continue


        subconjunto_cadenas_no_intersectantes.append(cadena_mayor)

    return subconjunto_cadenas_no_intersectantes

def postprocesamiento_cuando_hay_mas_de_1_cadena_sin_interseccion(ctx,subconjunto_clase_cadenas, cadena_anillo_exterior, cadena_anillo_interior,listaPuntos,
                                                        listaCadenas, UMBRAL_INFORMACION = 180):

    subconjunto_cadenas_no_intersectantes = armar_subconjunto_de_cadenas_no_intersectantes(subconjunto_clase_cadenas)
    hay_suficiente_informacion_del_anillo = np.sum([cad.size for cad in subconjunto_cadenas_no_intersectantes]) > UMBRAL_INFORMACION
    if not hay_suficiente_informacion_del_anillo:
        return 0

    #pegar subconjunto de cadenas
    ##########################################################################ordenar subconjunto de cadenas por extremo
    subconjunto_cadenas_no_intersectantes.sort(key=lambda x: x.extA.angulo)
    ####################################################################################################################
    cadena_origen = subconjunto_cadenas_no_intersectantes.pop(0)
    punto_extremo = cadena_origen.extB
    extremo = 'B'
    cadena_soporte = seleccionar_cadena_soporte(cadena_anillo_exterior, cadena_anillo_interior, punto_extremo)

    while len(subconjunto_cadenas_no_intersectantes) > 0:
        cadena_siguiente = subconjunto_cadenas_no_intersectantes[0]
        valida_acumulado, diferencia_radial, inf_banda = criterio_distancia_radial_no_debugging(0.2, cadena_soporte,
                                                        cadena_origen, cadena_siguiente, extremo)

        cadenas_superpuestas_cruzadas = hay_cadenas_superpuestas_en_banda(subconjunto_clase_cadenas, inf_banda)
        hay_cadena = len(cadenas_superpuestas_cruzadas) > 0
        if not hay_cadena:
            pegar_2_cadenas_via_cadena_soporte(cadena_anillo_exterior,cadena_anillo_interior, cadena_origen, cadena_siguiente, listaPuntos, extremo,
                                               listaCadenas, subconjunto_cadenas_no_intersectantes)
        else:
            subconjunto_cadenas_no_intersectantes.remove(cadena_siguiente)

    # completar_cadena_via_anillo_soporte(cadena_soporte, cadena_origen,
    #                                     listaPuntos)
    completar_cadena_si_no_hay_interseccion_con_otras(cadena_origen, cadena_anillo_interior, cadena_anillo_exterior, listaCadenas, listaPuntos)

    return 0



def lista_puntos(conjunto_cadenas_interiores):
    puntos_interiores = []
    for cadena in conjunto_cadenas_interiores:
        puntos_interiores += cadena.lista
    return puntos_interiores

def completar_cadena_si_no_hay_interseccion_con_otras(chain, cadena_inferior, cadena_superior, lista_cadenas, lista_puntos):
    #construir bandas de busqueda
    completar_cadena_via_anillo_soporte(cadena_inferior, cadena_superior, chain, lista_puntos)

    return True
def picar_y_unir_cadenas_region(iteracion,img, conjunto_cadenas_interiores,cadena_anillo_interior,
                            cadena_anillo_exterior, listaCadenas, debug, save_path,  listaPuntos,idx, vecindario = 90):
    label = 'picar_y_unir_cadenas_region'
    conjunto_cadenas_interiores.sort(key=lambda x: x.size, reverse=True)
    puntos_interiores = lista_puntos(conjunto_cadenas_interiores)
    debug_params = {'img':img, 'state':debug, 'iteracion':iteracion,'save_path':save_path , 'idx':idx}
    generator = BolsaCadenas(conjunto_cadenas_interiores)
    if debug:
        ch.visualizarCadenasSobreDisco(
            conjunto_cadenas_interiores,
            img,
            f'{iteracion[0]}_{label}', labels=True,
            save=f"{save_path}")
        iteracion[0] += 1
    se_pego = False
    se_completo_cadena = False
    cadena_origen = None
    while True:
        #1.0 obtener cadena origen
        if not se_pego:
            if cadena_origen is not None and cadena_origen.esta_completa(regiones=4):
                se_completo_cadena = completar_cadena_si_no_hay_interseccion_con_otras( cadena_origen,
                                cadena_anillo_interior, cadena_anillo_exterior, listaCadenas, listaPuntos)
                if not se_completo_cadena :
                    se_completo_cadena = False

                else:
                    if debug:
                        ch.visualizarCadenasSobreDisco(
                            [cadena_soporte, cadena_origen],
                            img,
                            f'{iteracion[0]}_picar_cadenas_{cadena_origen.label_id}_ext_{extremo}_2-2.png', labels=True,
                            save=f"{save_path}")
                        iteracion[0] += 1

                    se_completo_cadena = True
                cadena_origen = None
                #completar_cadena_via_anillo_soporte( cadena_soporte, cadena_origen, listaPuntos)


            else:
                cadena_origen = generator.get_siguiente_cadena()


        if cadena_origen is None:
                break
        else:
            if debug:
                ch.visualizarCadenasSobreDisco(
                    [cadena_origen],
                    img,
                    f'{iteracion[0]}_picar_cadenas_{cadena_origen.label_id}.png', labels=True,
                    save=f"{save_path}")
                iteracion[0] += 1
        #2.0 por extremo
        extremo = 'A'
        cadena_candidata_a, diff_a, cadena_soporte_a = picar_y_unir_cadenas_region_extremo_cadena(puntos_interiores, conjunto_cadenas_interiores, cadena_origen,
                                                   extremo,
                                                   cadena_anillo_exterior, cadena_anillo_interior, vecindario,
                                                   debug_params,
                                                   listaCadenas, listaPuntos)

        extremo = 'B'
        cadena_candidata_b, diff_b, cadena_soporte_b = picar_y_unir_cadenas_region_extremo_cadena(puntos_interiores, conjunto_cadenas_interiores, cadena_origen,
                                                   extremo,
                                                   cadena_anillo_exterior, cadena_anillo_interior, vecindario,
                                                   debug_params,
                                                   listaCadenas, listaPuntos, cadena_auxiliar=cadena_candidata_a)


        se_pego, cadena_soporte, extremo = pegar_cadena_mas_cercana_si_amerita(cadena_origen,  cadena_candidata_a, diff_a,
            cadena_soporte_a, cadena_candidata_b, diff_b, cadena_soporte_b, listaCadenas, conjunto_cadenas_interiores,
            listaPuntos, cadena_anillo_interior, cadena_anillo_exterior )

        if se_pego and debug:
            ch.visualizarCadenasSobreDisco(
                [cadena_soporte, cadena_origen],
                img,
                f'{iteracion[0]}_picar_cadenas_{cadena_origen.label_id}_ext_{extremo}_2-2.png', labels=True,
                save=f"{save_path}")
            iteracion[0] += 1





    return se_completo_cadena


def pegar_cadena_mas_cercana_si_amerita(cadena_origen, cadena_candidata_a, diff_a, cadena_soporte_a,cadena_candidata_b,
    diff_b, cadena_soporte_b, listaCadenas, conjunto_cadenas_interiores, listaPuntos, cadena_interior, cadena_exterior ):

    if (0 <= diff_a <= diff_b) or (diff_b < 0 and diff_a>=0):
        cadena_candidata = cadena_candidata_a
        cadena_soporte = cadena_soporte_a
        extremo = 'A'

    elif (0 <= diff_b < diff_a) or (diff_a < 0 and diff_b>=0):
        cadena_candidata = cadena_candidata_b
        cadena_soporte = cadena_soporte_b
        extremo = 'B'

    else:
        return False, cadena_soporte_a,''

    if cadena_candidata.size + cadena_origen.size > cadena_candidata.Nr:
        return False, cadena_soporte_a,''
    pegar_2_cadenas_via_cadena_soporte(cadena_exterior, cadena_interior, cadena_origen, cadena_candidata, listaPuntos, extremo,
                                       listaCadenas, conjunto_cadenas_interiores)

    return True, cadena_soporte, extremo

def pegar_2_cadenas_via_cadena_soporte(cadena_superior, cadena_inferior, cadena_origen, cadena_candidata, listaPuntos, extremo,listaCadenas, subconjunto_cadenas_interiores):
    pegar_dos_cadenas_interpolando_via_cadena_superior_e_inferior(cadena_superior,cadena_inferior, cadena_origen, cadena_candidata,  listaPuntos, extremo)
    # eliminar cadena candidata
    cadena_candidata_original = [cadena for cadena in subconjunto_cadenas_interiores if
                                 cadena.id == cadena_candidata.id]
    if len(cadena_candidata_original) > 0:
        cadena_ref_lista_original = cadena_candidata_original[0]
        subconjunto_cadenas_interiores.remove(cadena_ref_lista_original)
        listaCadenas.remove(cadena_ref_lista_original)
    cad_original = [cad for cad in listaCadenas if cad.id == cadena_candidata.id]
    if len(cad_original) > 0:
        listaCadenas.remove(cad_original[0])

    return

def hay_interseccion_angular_cadena_origen(cadena, dominio_angular_cadena_origen):
    dominio = cadena._completar_dominio_angular(cadena)
    if len(np.intersect1d(dominio, dominio_angular_cadena_origen)) == 0:
        return False
    return True


def filtrar_cadenas_no_intersectantes_lejanas(cadenas_no_intersectantes, cadena_origen, extremo):
    amplitud_vecindario = 45
    # Quedarme unicamente con las cadenas cercanas segun cierta distancia maxima
    conjunto_de_cadenas_candidatas_cercanas = []
    for cad in cadenas_no_intersectantes:
        distancia = union.distancia_angular_entre_cadenas(cadena_origen, cad, extremo)
        if distancia < amplitud_vecindario:
            conjunto_de_cadenas_candidatas_cercanas.append((distancia, cad))

    # ordenar por cercania al extremo de la cadena (menor a mayor)
    conjunto_de_cadenas_candidatas_cercanas.sort(key=lambda x: x[0])
    cojunto_de_cadenas_de_busqueda_no_intersectantes = [cadena for distancia, cadena in
                                                        conjunto_de_cadenas_candidatas_cercanas]

    return cojunto_de_cadenas_de_busqueda_no_intersectantes


def cortar_cadenas_intersectantes_en_el_otro_extremo(extremo, cadena_origen, conjunto_cadenas_interiores, puntos_interiores, conjunto_cadenas_busqueda):
    punto_otro_extremo = cadena_origen.extB if extremo in 'A' else cadena_origen.extA
    direccion = punto_otro_extremo.angulo
    puntos_direccion = [punto for punto in puntos_interiores if
                        ((punto.angulo == direccion) and not (punto.cadenaId == cadena_origen.id))]
    cad_id_direccion = set([punto.cadenaId for punto in puntos_direccion])
    cadenas_intersectantes_id = [cad.id for cad in conjunto_cadenas_interiores if
                                 cad.id in cad_id_direccion]
    cadenas_intersectantes_en_otro_extremo = [cad for cad in conjunto_cadenas_busqueda if
                                              cad.id in cadenas_intersectantes_id]
    conjunto_cadenas_busqueda = [cad for cad in conjunto_cadenas_busqueda if
                                 cad not in cadenas_intersectantes_en_otro_extremo]
    conjunto_cadenas_busqueda_en_otro_extremo_cortadas = cortar_cadenas_intersectantes(direccion,
                                                                                       cadenas_intersectantes_en_otro_extremo,
                                                                                       cadena_origen)
    conjunto_cadenas_busqueda += conjunto_cadenas_busqueda_en_otro_extremo_cortadas


    return 0


def dividir_cadena(cadena, punto, nuevo_id):
    # dividir cadena a la mitad
    lista_puntos = cadena.sort_dots()
    idx_split = lista_puntos.index(punto)
    lista_puntos_cad1 = lista_puntos[:idx_split]
    if idx_split < len(lista_puntos)-1:
        lista_puntos_cad2 = lista_puntos[idx_split+1:]
    else:
        lista_puntos_cad2 = []

    if len(lista_puntos_cad1) > 1:
        sub_cadena1 = ch.Cadena(cadena.id, cadena.centro, cadena.M, cadena.N, cadena.Nr)
        for cad_punto in lista_puntos_cad1:
            cad_punto.cadenaId = sub_cadena1.id
        sub_cadena1.add_lista_puntos(lista_puntos_cad1)
    else:
        sub_cadena1 = None
    if len(lista_puntos_cad2)>1:
        sub_cadena2 = ch.Cadena(nuevo_id, cadena.centro, cadena.M, cadena.N, cadena.Nr)
        for cad_punto in lista_puntos_cad2:
            cad_punto.cadenaId = sub_cadena2.id
        sub_cadena2.add_lista_puntos(lista_puntos_cad2)
    else:
        sub_cadena2 = None

    return (sub_cadena1, sub_cadena2)


# def interpolar_en_domino(cadena_anillo_soporte, extremo_cad1, extremo_cad2, extremo, cad1, listaPuntos):
#     dominio_de_interpolacion = calcular_dominio_de_interpolacion(extremo, extremo_cad1,extremo_cad2)
#     centro = cad1.centro
#     dot_list_in_radial_direction = union.get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
#         [cadena_anillo_soporte], extremo_cad1.angulo)
#     soporte_pto1 = dot_list_in_radial_direction[0]
#     r1 = ch.distancia_entre_puntos(soporte_pto1,extremo_cad1)
#     dot_list_in_radial_direction = union.get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
#         [cadena_anillo_soporte], extremo_cad2.angulo)
#     soporte_pto2 = dot_list_in_radial_direction[0]
#     signo = -1 if soporte_pto2.radio > extremo_cad2.radio else +1
#     r2 = ch.distancia_entre_puntos(soporte_pto2, extremo_cad2)
#
#     ###
#     radios = []
#     total_de_puntos = len(dominio_de_interpolacion)
#     if total_de_puntos == 0:
#         return
#     m = (r2-r1) / (total_de_puntos - 0)
#     n = r1 - m*0
#     lista_puntos_generados = []
#     for idx_pto_actual, angulo in enumerate(dominio_de_interpolacion):
#         dot_list_in_radial_direction = union.get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
#             [cadena_anillo_soporte], angulo % 360)
#
#         punto_soporte = dot_list_in_radial_direction[0]
#
#         radio_inter = signo*(m* (idx_pto_actual) + n) + punto_soporte.radio
#         i, j = from_polar_to_cartesian(radio_inter, angulo%360, centro)
#         radios.append(radio_inter)
#         params = {
#             "x": i,
#             "y": j,
#             "angulo": angulo % 360,
#             "radio": radio_inter,
#             "gradFase": -1,
#             "cadenaId": cad1.id,
#         }
#
#         punto = ch.Punto(**params)
#         if punto not in listaPuntos:
#             lista_puntos_generados.append(punto)
#             listaPuntos.append(punto)
#
#     cad1.add_lista_puntos(lista_puntos_generados)
#
#     return

def from_polar_to_cartesian(r,angulo,centro):
    y = centro[0] + r * np.cos(angulo * np.pi / 180)
    x = centro[1] + r * np.sin(angulo * np.pi / 180)
    return (y,x)

def seleccionar_cadena_que_no_intersecta_en_extremo(sub_cad1, sub_cad2, cadena_origen, direccion,vecindad=10):
    if sub_cad1 is None and sub_cad2 is None:
        return None
    if sub_cad1 is None and sub_cad2 is not None:
        return sub_cad2
    if sub_cad2 is None and sub_cad1 is not None:
        return sub_cad1

    extremo_corte = 'A' if direccion == cadena_origen.extA.angulo else 'B'
    sentido = 'horario' if extremo_corte in 'B' else 'anti'
    puntos_vecindad = cadena_origen.sort_dots(sentido=sentido)[:vecindad]
    dominio_angular_origen = [dot.angulo for dot in puntos_vecindad]

    dominio_1 = sub_cad1._completar_dominio_angular(sub_cad1) if sub_cad1.size>0 else dominio_angular_origen
    dominio_2 = sub_cad2._completar_dominio_angular(sub_cad2) if sub_cad2.size>0 else dominio_angular_origen
    if np.intersect1d(dominio_1,dominio_angular_origen).shape[0]==0:
        return sub_cad1
    elif np.intersect1d(dominio_2,dominio_angular_origen).shape[0]==0:
        return sub_cad2
    else:
        return None


def cortar_cadenas_intersectantes(direccion, cadenas_intersectantes_filtradas, cadena_origen, cadena_id_auxiliar=10000000):
    conjunto_cadenas_busqueda = []
    for cadena_inter in cadenas_intersectantes_filtradas:
        punto_divicion = cadena_inter.getDotByAngle(direccion)
        if len(punto_divicion) == 0:
            continue
        punto_divicion = punto_divicion[0]
        sub_cad1, sub_cad2 = dividir_cadena(cadena_inter, punto_divicion, cadena_id_auxiliar)
        # 1.0 Determinar que cadena intersecta a cadena larga.
        #cadena_candidata = seleccionar_cadena_que_no_intersecta(sub_cad1, sub_cad2, cadena_origen)
        cadena_candidata = seleccionar_cadena_que_no_intersecta_en_extremo(sub_cad1, sub_cad2, cadena_origen, direccion)
        if cadena_candidata is None:
            continue

        #2.0 Puede suceder que cadena intersecte dos veces, ambos extremos
        if union.cadenas_se_intersectan(cadena_candidata, cadena_origen):
            direccion_2 = cadena_origen.extB.angulo if punto_divicion.angulo == cadena_origen.extA.angulo else cadena_origen.extA.angulo
            punto_divicion_2 = cadena_candidata.getDotByAngle(direccion_2)
            if len(punto_divicion_2)==0:
                continue
            punto_divicion_2 = punto_divicion_2[0]
            sub_cad1, sub_cad2 = dividir_cadena(cadena_candidata, punto_divicion_2, cadena_id_auxiliar)
            cadena_candidata = seleccionar_cadena_que_no_intersecta_en_extremo(sub_cad1, sub_cad2, cadena_origen,
                                                                               direccion_2)
            if cadena_candidata is None:
                continue

        cadena_candidata.changeId(cadena_inter.id)
        cadena_candidata.label_id = cadena_inter.label_id

        conjunto_cadenas_busqueda.append(cadena_candidata)

    return conjunto_cadenas_busqueda

def seleccionar_cadena_que_no_intersecta(sub_cad1, sub_cad2, cadena_larga):
    if sub_cad1.size > 0 and not union.cadenas_se_intersectan(cadena_larga, sub_cad1):
        cadena_candidata = sub_cad1

    elif sub_cad2.size > 0 and not union.cadenas_se_intersectan(cadena_larga, sub_cad2):
        cadena_candidata = sub_cad2
    else:
        cadena_candidata = None

    return cadena_candidata

class BolsaCadenas:
    def __init__(self, conjunto_cadenas_interiores):
        self.conjunto_cadenas = conjunto_cadenas_interiores
        self.conjunto_id_cadenas_recorridas = []

    def get_siguiente_cadena(self):
        siguiente = None
        for cadena in self.conjunto_cadenas:
            if cadena.id not in self.conjunto_id_cadenas_recorridas:
                siguiente = cadena
                self.conjunto_id_cadenas_recorridas.append(siguiente.id)
                break

        return siguiente

def picar_y_unir_cadenas_region_extremo_cadena(puntos_interiores, conjunto_cadenas_interiores, cadena_origen, extremo,
                                               cadena_anillo_exterior, cadena_anillo_interior, vecindad_amplitud, debug_params,
                                               listaCadenas, listaPuntos, cadena_auxiliar=None):
    img, save_path, iteracion, debug = debug_params['img'],debug_params['save_path'],debug_params['iteracion'], debug_params['state']
    se_pego = False
    debug_lista_anillos_limites = [cadena for cadena in [cadena_anillo_exterior, cadena_anillo_interior] if cadena is not None]
    dominio_angular_cadena_origen = cadena_origen._completar_dominio_angular(cadena_origen)
    punto_extremo = cadena_origen.extA if extremo in 'A' else cadena_origen.extB

    # 1.0 Selecciono la cadena soporte mas cercana
    cadena_soporte = seleccionar_cadena_soporte(cadena_anillo_exterior, cadena_anillo_interior, punto_extremo)

    # 2.1 Busco cadenas intersectantes por extremo
    puntos_direccion = [punto for punto in puntos_interiores if
                        ((punto.angulo == punto_extremo.angulo) and not (punto.cadenaId == cadena_origen.id))]
    cad_id_direccion = set([punto.cadenaId for punto in puntos_direccion])
    cadenas_intersectantes_por_extremo = [cad for cad in conjunto_cadenas_interiores if cad.id in cad_id_direccion]
    cadenas_no_intersectantes = [cad for cad in conjunto_cadenas_interiores if
                                 cad not in cadenas_intersectantes_por_extremo and cad != cadena_origen and
                                 not hay_interseccion_angular_cadena_origen(cad, dominio_angular_cadena_origen)]

    if cadena_auxiliar is not None:
        cadenas_no_intersectantes += [cadena_auxiliar]
    # 2.2 Filtro aquellas cadenas cuyo dominio angular soporte el dominio angular de la cadena origen
    cadenas_intersectantes_filtradas = [cadena_inter for cadena_inter in cadenas_intersectantes_por_extremo if
                                        not soporte_angular_de_cadena_incluido_en_intervalo(
                                            dominio_angular_cadena_origen, cadena_inter)]

    if debug:
        ch.visualizarCadenasSobreDisco(
            [cadena_soporte, cadena_origen] + cadenas_intersectantes_filtradas +  debug_lista_anillos_limites, img,
            f'{iteracion[0]}_picar_cadenas_{cadena_origen.label_id}_ext_{extremo}_2-2', labels=True, save=f"{save_path}")
        iteracion[0] += 1

    # 2.3 Corto cadenas intersectantes en un extremo
    conjunto_cadenas_busqueda = cortar_cadenas_intersectantes(punto_extremo.angulo, cadenas_intersectantes_filtradas,
                                                              cadena_origen)
    if debug:
        ch.visualizarCadenasSobreDisco(
            [cadena_soporte, cadena_origen] + conjunto_cadenas_busqueda + debug_lista_anillos_limites, img,
            f'{iteracion[0]}_picar_cadenas_{cadena_origen.label_id}_ext_{extremo}_2-3', labels=True, save=f"{save_path}")
        iteracion[0] += 1

    # 2.4 Corto cadenas intersectantes en el otro extremo
    cortar_cadenas_intersectantes_en_el_otro_extremo(extremo, cadena_origen, conjunto_cadenas_interiores,
                                                     puntos_interiores, conjunto_cadenas_busqueda)
    if debug:
        ch.visualizarCadenasSobreDisco(
            [cadena_soporte, cadena_origen] + conjunto_cadenas_busqueda + debug_lista_anillos_limites, img,
            f'{iteracion[0]}_picar_cadenas_{cadena_origen.label_id}_ext_{extremo}_2-4', labels=True,
            save=f"{save_path}")
        iteracion[0] += 1

    # 2.5 Buscar cadenas no intersectantes en vecindad
    conjunto_cadenas_busqueda += filtrar_cadenas_no_intersectantes_lejanas(cadenas_no_intersectantes, cadena_origen, extremo)
    if debug:
        ch.visualizarCadenasSobreDisco(
            [cadena_soporte, cadena_origen] + conjunto_cadenas_busqueda + debug_lista_anillos_limites , img,
            f'{iteracion[0]}_picar_cadenas_{cadena_origen.label_id}_ext_{extremo}_2-5', labels=True,
            save=f"{save_path}")
        iteracion[0] += 1

    # 2.6 Buscar Candidata
    # 2.6.1 ordenar cadenas candidatas
    conjunto_de_cadenas_candidatas = []
    distancia_radial_cadenas_candidatas = []
    distancia_euclidea_cadenas_candidatas = []
    idx_cadena_candidata = 0
    while True:
        if len(conjunto_cadenas_busqueda) <= idx_cadena_candidata:
            break
        cadena_candidata = conjunto_cadenas_busqueda[idx_cadena_candidata]
        idx_cadena_candidata += 1

        valida_kmeans, distancia_entre_bordes = criterio_kmeans(cadena_candidata, cadena_origen, extremo, cadena_soporte, vecindad_amplitud)
        #valida_acumulado, diferencia_radial = union.check_cumulative_radio(cadena_origen, cadena_candidata, cadena_soporte, extremo,
        #                                                 umbral=0.2)
        valida_acumulado, diferencia_radial, inf_banda = criterio_distancia_radial_no_debugging(0.2, cadena_soporte,
                                                        cadena_origen, cadena_candidata, extremo)

        cadenas_superpuestas_cruzadas = hay_cadenas_superpuestas_en_banda(conjunto_cadenas_interiores, inf_banda)
        hay_cadena = len(cadenas_superpuestas_cruzadas) > 0
        hay_cadena = False
        if ( valida_kmeans or valida_acumulado ) and not hay_cadena:
        #if  not hay_cadena and (cadena_origen.size  + cadena_candidata.size <= Nr):
            if debug:
                img_debug = Dibujar.lista_cadenas(conjunto_cadenas_interiores, img.copy())
                img_debug = inf_banda.dibujar_bandas(img_debug, cadenas_superpuestas_cruzadas)
                cv2.imwrite(f'{str(save_path)}/{iteracion[0]}_superpuestas_{hay_cadena}_orig_{cadena_origen.label_id}_ext_{extremo}.png',
                            img_debug)

                iteracion[0] += 1

            puntos_virtuales = generar_puntos_virtuales_sin_cadena_soporte(cadena_origen, cadena_candidata, extremo)
            umbral = 1.5
            N = 20
            paso = 2
            res, abs_der_1, abs_der_2, abs_der_3, salto, radios_1, radios_2, radios_virtuales, coeficiente, derivada_maxima_var = \
                derivada_maxima(cadena_origen, cadena_candidata, extremo, puntos_virtuales, umbral, N, paso)

            if res:
                conjunto_de_cadenas_candidatas.append(cadena_candidata)
                distancia_radial_cadenas_candidatas.append(diferencia_radial)
                distancia_euclidea_cadenas_candidatas.append(distancia_entre_bordes)

    # 2.6.2 Pegar cadenas si amerita
    cadena_candidata = None
    diff = -1
    if len(conjunto_de_cadenas_candidatas) > 0:
        cadena_candidata = conjunto_de_cadenas_candidatas[np.argmin(distancia_euclidea_cadenas_candidatas)]
        diff = np.min(distancia_radial_cadenas_candidatas)

    if cadena_auxiliar is not None and cadena_candidata == cadena_auxiliar:
        if cadena_candidata in conjunto_cadenas_interiores:
            conjunto_cadenas_interiores.remove(cadena_auxiliar)
            cadena_candidata = None
            diff = -1


    return cadena_candidata,diff, cadena_soporte

def soporte_angular_de_cadena_incluido_en_intervalo(dominio_angular_cadena_origen, cadena_inter, umbral_solapamiento=45):
    dominio_inter = cadena_inter._completar_dominio_angular(cadena_inter)
    inter = np.intersect1d(dominio_inter, dominio_angular_cadena_origen)
    if (len(inter) >= len(dominio_angular_cadena_origen)) or (len(inter) > umbral_solapamiento):
        return True
    else:
        return False


def criterio_kmeans(cadena_candidata, cadena_origen, extremo, cadena_soporte, vecindad_amplitud):
    distancias, cadena_ids, distancia_entre_bordes = calcular_distancia_acumulada_vecindad(
        cadena_origen,
        cadena_candidata,
        extremo,
        cadena_soporte, vecindad_amplitud=vecindad_amplitud)

    return se_cumple_condicion_agrupamiento(distancias, cadena_ids,
                                            debug=None,
                                            histograma_test=False), distancia_entre_bordes



def calcular_distancia_acumulada_vecindad(cadena_larga, cadena_candidata, extremo, cadena_soporte, vecindad_amplitud = 90):
    sentido = 'anti' if extremo in 'A' else 'horario'
    vecindad = cadena_larga.sort_dots(sentido=sentido)[:vecindad_amplitud]
    vecindad += cadena_candidata.sort_dots(sentido='anti' if sentido in 'horario' else 'horario')[:vecindad_amplitud]

    distancias = []
    cadena_ids = []
    for dot in vecindad:
        dot_list_in_radial_direction = ch.get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
            [cadena_soporte], dot.angulo)
        distancias.append(np.abs(dot_list_in_radial_direction[0].radio - dot.radio))
        cadena_ids.append(dot.cadenaId)

    cadena_origen_extremo = cadena_larga.extA if extremo in 'A' else cadena_larga.extB
    cadena_candidata_extremo =  cadena_candidata.extB if extremo in 'A' else cadena_candidata.extA
    return distancias, cadena_ids,  ch.distancia_entre_puntos(cadena_origen_extremo, cadena_candidata_extremo)



def se_cumple_condicion_agrupamiento(distancias, cadena_ids, debug=None,histograma_test = False):
    from sklearn.cluster import KMeans
    distancias = np.array(distancias).reshape((-1, 1))
    kmeans = KMeans(n_clusters=2, random_state=0).fit(distancias)
    km_labels = kmeans.labels_
    clase_0 = np.where(km_labels == 0)[0]
    clase_1 = np.where(km_labels == 1)[0]
    if histograma_test:
        hist, bins, _ = plt.hist(distancias)
        hist_0,_,_ = plt.hist(distancias[clase_0], bins=bins)
        hist_1,_,_ = plt.hist(distancias[clase_1], bins=bins)
        interseccion = [value0+value1 for value0, value1 in zip( hist_0, hist_1) if value1>0 and value0>0]
        condicion = len(interseccion) > 0

    else:
        condicion = (np.unique(np.array(cadena_ids)[clase_1]).shape[0] > 1 and np.unique(np.array(cadena_ids)[clase_0]).shape[0] > 1)
        hist, bins, _ = plt.hist(distancias)
        #hist_0,_,_ = plt.hist(distancias[clase_0], bins=bins)
        #hist_1,_,_ = plt.hist(distancias[clase_1], bins=bins)
        cadena_ids_unique = np.unique(cadena_ids)

        cadena_id_1 = np.where(np.array(cadena_ids) == cadena_ids_unique[0])[0]
        cadena_id_2 = np.where(np.array(cadena_ids) == cadena_ids_unique[1])[0]
        media_0 = np.mean(distancias[cadena_id_1])
        media_1 = np.mean(distancias[cadena_id_2])
        ancho_bin = np.mean(np.gradient(bins))
        if media_0>media_1:
            infimo = np.max(distancias[cadena_id_2])
            supremo = np.min(distancias[cadena_id_1])
        else:
            infimo = np.max(distancias[cadena_id_1])
            supremo = np.min(distancias[cadena_id_2])

        condicion |= np.abs(supremo-infimo) < ancho_bin
        #interseccion = [value0+value1 for value0, value1 in zip( hist_0, hist_1) if value1>0 and value0>0]
        #condicion|= len(interseccion) > 0
        #condicion = (np.unique(np.array(cadena_ids)[clase_1]).shape[0] > 1 or np.unique(np.array(cadena_ids)[clase_0]).shape[0] > 1)

    if debug is not None:
        print(debug)
        plt.figure()
        plt.subplot(121)
        hist, bins, _ = plt.hist(distancias)
        plt.hist(distancias[clase_0], bins=bins)
        plt.hist(distancias[clase_1], bins=bins)
        plt.title(f"kmeans clusters: {condicion}")

        plt.subplot(122)
        cadena_ids_unique = np.unique(cadena_ids)

        cadena_id_1 = np.where(np.array(cadena_ids) == cadena_ids_unique[0])[0]
        cadena_id_2 = np.where(np.array(cadena_ids) == cadena_ids_unique[1])[0]

        hist, bins, _ = plt.hist(distancias)
        plt.hist(distancias[cadena_id_1], bins=bins)
        plt.hist(distancias[cadena_id_2], bins = bins)
        plt.show()
        plt.savefig(debug)
        plt.close()


    return condicion


def seleccionar_cadena_soporte(cadena_anillo_exterior, cadena_anillo_interior, punto_extremo):
    chains_in_radial_direction = []
    if cadena_anillo_exterior is not None:
        chains_in_radial_direction.append(cadena_anillo_exterior)
    if cadena_anillo_interior is not None:
        chains_in_radial_direction.append(cadena_anillo_interior)

    dot_list_in_radial_direction = ch.get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
        chains_in_radial_direction, punto_extremo.angulo)

    distancia = [ch.distancia_entre_puntos(punto_extremo, punto_cadena_completa) for punto_cadena_completa in
                 dot_list_in_radial_direction]
    if len(distancia) < 2:
        cadena_soporte = cadena_anillo_exterior if cadena_anillo_exterior is not None else cadena_anillo_interior

    else:
        # cadena_soporte = ch.getChain(dot_list_in_radial_direction[np.argmin(distancia)].cadenaId,
        #                              chains_in_radial_direction)
        cadena_soporte = [cad for cad in chains_in_radial_direction if cad.id == dot_list_in_radial_direction[np.argmin(distancia)].cadenaId][0]
    return cadena_soporte



