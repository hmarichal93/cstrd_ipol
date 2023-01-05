import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy

from lib.dibujar import Color, Dibujar
from lib.interpolacion import pegar_dos_cadenas_interpolando_via_cadena_soporte, generar_lista_puntos_entre_dos_distancias_radiales
import lib.chain_v4 as ch
from lib.utils import write_log
from lib.objetos import Interseccion
from lib.celdas import Celda, ROJO
from lib.interpolacion import interpolar_en_domino
MODULE_NAME = 'propiedades_fundamentales'


def dibujar_segmentoo_entre_puntos(pto1, pto2, img, color=(0,255,0),thickness = 2):
    pts = np.array([[pto1.y, pto1.x],[pto2.y, pto2.x]], dtype=int)
    isClosed=False
    img = cv2.polylines(img, [pts],
                          isClosed, color, thickness)

    return img

class InfoBandaVirtual:

    ABAJO = 0
    DENTRO = 1
    ARRIBA = 2
    def __init__(self, ptos_virtuales,cadena_origen, cadena_cand, extremo, cad_soporte=None, inf_orig=None,sup_orig=None,
                 inf_cand=None, sup_cand=None,ancho_banda = 0.1):
        self.ptos_vituales = ptos_virtuales
        self.cadena_origen = cadena_origen
        self.cadena_cand = cadena_cand
        self.extremo = extremo
        self.cad_soporte = cad_soporte
        self.centro = ch.Punto(x=self.cadena_origen.centro[0],y = self.cadena_origen.centro[1], radio=0,
                               angulo = 0, gradFase = -1, cadenaId=-1,Nr=self.cadena_origen.Nr)

        if ancho_banda:
            ext1 = self.cadena_origen.extB if extremo in 'B' else self.cadena_origen.extA
            ext1_soporte = self.cad_soporte.getDotByAngle(ext1.angulo)[0] if self.cad_soporte is not None else self.centro
            ext2 = self.cadena_cand.extB if extremo in 'A' else self.cadena_cand.extA
            ext2_soporte = self.cad_soporte.getDotByAngle(ext2.angulo)[0] if self.cad_soporte is not None else self.centro
            delta_r1 = ch.distancia_entre_puntos(ext1, ext1_soporte)
            delta_r2 = ch.distancia_entre_puntos(ext2, ext2_soporte)
            self.inf_cand = delta_r2 * (1 - ancho_banda)
            self.sup_cand = delta_r2 * (1 + ancho_banda)
            self.inf_orig = delta_r1 * (1 - ancho_banda)
            self.sup_orig = delta_r1 * (1 + ancho_banda)

        else:
            self.inf_orig = inf_orig
            self.inf_cand = inf_cand
            self.sup_orig = sup_orig
            self.sup_cand = sup_cand

        self.generate_band()

    def generate_banda_limit(self,r2,r1,total_de_puntos):
        dominio_de_interpolacion = [punto.angulo for punto in self.ptos_vituales]
        extremo_cad2 = self.ptos_vituales[-1]
        soporte_pto2 = self.cad_soporte.getDotByAngle(extremo_cad2.angulo)[0] if self.cad_soporte is not None else self.centro
        signo = -1 if soporte_pto2.radio > extremo_cad2.radio else +1
        lista_puntos_generados = generar_lista_puntos_entre_dos_distancias_radiales(r2, r1, total_de_puntos,
                                                                                    dominio_de_interpolacion, self.cadena_cand.centro,
                                                                                    signo,
                                                                                    self.cad_soporte, self.cadena_cand)
        self.dominio_de_interpolacion = dominio_de_interpolacion
        return lista_puntos_generados

    def generate_band(self):
        total_de_puntos = len(self.ptos_vituales)
        r1 = self.sup_orig
        r2 = self.sup_cand
        self.banda_sup = self.generate_banda_limit(r2,r1,total_de_puntos)
        r1 = self.inf_orig
        r2 = self.inf_cand
        self.banda_inf = self.generate_banda_limit(r2, r1, total_de_puntos)

        return

    @staticmethod
    def calcular_radio_medio_puntos(lista_puntos):
        return np.mean([dot.radio for dot in lista_puntos])
    def is_dot_in_band(self,punto):
        radio_medio_inf = self.calcular_radio_medio_puntos(self.banda_inf)
        radio_medio_sup = self.calcular_radio_medio_puntos(self.banda_sup)
        banda_interior = self.banda_inf if radio_medio_inf < radio_medio_sup else self.banda_sup
        banda_exterior = self.banda_sup if radio_medio_inf < radio_medio_sup else self.banda_inf
        infimo = [inf for inf in banda_interior if inf.angulo == punto.angulo][0]
        supremo = [inf for inf in banda_exterior if inf.angulo == punto.angulo][0]

        if punto.radio <= infimo.radio:
            posicion_relativa_a_banda = InfoBandaVirtual.ABAJO

        elif supremo.radio >= punto.radio >= infimo.radio:
            posicion_relativa_a_banda = InfoBandaVirtual.DENTRO

        else:
            posicion_relativa_a_banda = InfoBandaVirtual.ARRIBA

        return posicion_relativa_a_banda

    def is_chain_in_band(self,chain):
        label='is_chain_in_band'
        puntos_cadena_en_intervalo = [punto for punto in chain.lista if punto.angulo in self.dominio_de_interpolacion]
        res = False
        prev_status = None
        for punto in puntos_cadena_en_intervalo:
            res = self.is_dot_in_band(punto)
            if res == InfoBandaVirtual.DENTRO:
                write_log(MODULE_NAME, label,
                          f"chain {chain.id} dentro de banda")
                break

            #validar que no hay cruzamiento por la banda-de arriba abajo o de abajo arriba
            if prev_status and prev_status != res:
                write_log(MODULE_NAME, label,
                          f"chain {chain.id} cruza banda")
                break

            prev_status = res
            res = False


        return res

    def crear_cadena_desde_lista_puntos(self, lista_puntos):
        cadena = ch.Cadena(lista_puntos[0].cadenaId, self.cadena_origen.centro, self.cadena_origen.M, self.cadena_origen.N, Nr=lista_puntos[0].Nr)
        cadena.add_lista_puntos(lista_puntos)

        return cadena

    def dibujar_bandas(self, img, cadenas_superpuestas):
        img = Dibujar.cadena(self.crear_cadena_desde_lista_puntos(self.banda_inf),img, color = Color.orange)
        img = Dibujar.cadena(self.crear_cadena_desde_lista_puntos(self.banda_sup), img, color = Color.maroon)
        img = Dibujar.cadena(self.cadena_origen, img, color = Color.blue)
        img = Dibujar.cadena(self.cadena_cand, img, color = Color.yellow)
        if self.cad_soporte is not None:
            img = Dibujar.cadena(self.cad_soporte, img, color = Color.red)

        for cadena in cadenas_superpuestas:
            img = Dibujar.cadena(cadena, img, color=Color.purple)

        return img


def derivada( fxi_menos_1, fxi_mas_1, Nr,paso=2):
    dtheta = paso * 2 * np.pi / Nr
    dtheta = paso
    return np.abs((fxi_mas_1 - fxi_menos_1) / dtheta)

def derivada_vector(f , Nr,paso=2):
    return np.gradient(f)

def diferencia_radial(f1,f2):
    return np.abs(f1-f2)
def diferencias_radiales_vector(f):
    der = np.zeros(len(f))
    for idx in range(len(f)):
        if idx == 0 and len(f) > 1:
            der[idx] = diferencia_radial(f[idx], f[idx+1])
        elif 0 < idx < len(f)-1:
            der[idx] = diferencia_radial(f[idx], f[idx + 1])
        else:
            der[idx] = diferencia_radial( f[idx - 1], f[idx] )

    return der

def derivada_maxima( cad_1, cad_2,extremo,puntos_virtuales, umbral=1,N=20,paso=2):
    sentido_1 = 'anti' if extremo in 'A' else 'horario'
    sentido_2 = 'anti' if extremo in 'B' else 'horario'
    lista_puntos_totales = []
    lista_puntos_ordenados_1 = cad_1.sort_dots(sentido_1)[:N]
    #lista_puntos_totales += lista_puntos_ordenados
    lista_puntos_totales += puntos_virtuales
    radios_1 = [punto.radio for punto in lista_puntos_ordenados_1]
    lista_puntos_ordenados_2 = cad_2.sort_dots(sentido_2)[:N]

    radios_2 = [punto.radio for punto in lista_puntos_ordenados_2]
    if extremo in 'B':
        lista_puntos_totales = lista_puntos_ordenados_1[::-1] + puntos_virtuales[1:-1] + lista_puntos_ordenados_2
    else:
        lista_puntos_totales = lista_puntos_ordenados_2[::-1] + puntos_virtuales[1:-1][::-1] + lista_puntos_ordenados_1
    radios_virtuales = [punto.radio for punto in lista_puntos_totales]
    # dtheta = 2*np.pi/Nr
    # derivada_maxima = np.maximum(np.abs(np.gradient(radios_1)).max(), np.abs(np.gradient(radios_2)).max())
    abs_der_1 = np.abs(derivada_vector(radios_1,cad_2.Nr))
    abs_der_2 = np.abs(derivada_vector(radios_2,cad_2.Nr))
    abs_der_3 = np.abs(derivada_vector(radios_virtuales,cad_2.Nr))
    derivada_maxima_var = np.maximum(abs_der_1.max(), abs_der_2.max())
    # umbral = 15
    # derivada_maxima = np.maximum(abs_der_2.mean() + abs_der_2.std()*umbral,abs_der_1.mean() + abs_der_1.std()*umbral)
    # ext1 = cad_1.extA if extremo in 'A' else cad_1.extB
    # ext2 = cad_2.extA if extremo in 'B' else cad_2.extB
    # salto = np.abs(ext1.radio - ext2.radio)
    # derivada_ext1 = derivada(cad_1.sort_dots(sentido=sentido_1)[1].radio, ext2.radio, paso)
    # derivada_ext2 = derivada(cad_2.sort_dots(sentido=sentido_2)[1].radio, ext1.radio, paso)
    # salto = np.maximum(derivada_ext1, derivada_ext2)
    salto = np.max(abs_der_3)
    coeficiente = umbral
    res = salto <= coeficiente * derivada_maxima_var

    return res, abs_der_1, abs_der_2,abs_der_3, salto, radios_1, radios_2,radios_virtuales, coeficiente, derivada_maxima_var
def diferencia_maxima( cad_1, cad_2,extremo, umbral=1,N=20,paso=2):
    sentido_1 = 'anti' if extremo in 'A' else 'horario'
    sentido_2 = 'anti' if extremo in 'B' else 'horario'
    lista_puntos_ordenados = cad_1.sort_dots(sentido_1)[:N]
    radios_1 = [punto.radio for punto in lista_puntos_ordenados]
    lista_puntos_ordenados = cad_2.sort_dots(sentido_2)[:N]
    radios_2 = [punto.radio for punto in lista_puntos_ordenados]
    # dtheta = 2*np.pi/Nr
    # derivada_maxima = np.maximum(np.abs(np.gradient(radios_1)).max(), np.abs(np.gradient(radios_2)).max())
    abs_der_1 = np.abs(diferencias_radiales_vector(radios_1))
    abs_der_2 = np.abs(diferencias_radiales_vector(radios_2))
    derivada_maxima_var = np.maximum(abs_der_1.max(), abs_der_2.max())
    # umbral = 15
    # derivada_maxima = np.maximum(abs_der_2.mean() + abs_der_2.std()*umbral,abs_der_1.mean() + abs_der_1.std()*umbral)
    ext1 = cad_1.extA if extremo in 'A' else cad_1.extB
    ext2 = cad_2.extA if extremo in 'B' else cad_2.extB
    # salto = np.abs(ext1.radio - ext2.radio)
    derivada_ext1 = diferencia_radial(cad_1.sort_dots(sentido=sentido_1)[0].radio, ext2.radio)#derivada(cad_1.sort_dots(sentido=sentido_1)[1].radio, ext2.radio, paso)
    derivada_ext2 = diferencia_radial(cad_2.sort_dots(sentido=sentido_2)[0].radio, ext1.radio)
    salto = np.maximum(derivada_ext1, derivada_ext2)
    coeficiente = umbral
    res = salto <= coeficiente * derivada_maxima_var
    return res, abs_der_1, abs_der_2, salto, radios_1, radios_2, coeficiente, derivada_maxima_var

def generar_puntos_virtuales_sin_cadena_soporte(cad_1, cad_2,extremo):
    extremo_cad1 = cad_1.extA if extremo in 'A' else cad_1.extB
    extremo_cad2 = cad_2.extB if extremo in 'A' else cad_2.extA
    ###check part############################################
    dominio_angular_cad1 = cad_1._completar_dominio_angular(cad_1)
    assert len(dominio_angular_cad1) == cad_1.size
    ##########################################################
    #1.0 Generar puntos de interpolacion
    puntos_virtuales = []
    cadena_anillo_soporte = None
    interpolar_en_domino(cadena_anillo_soporte, extremo_cad1, extremo_cad2, extremo, cad_1, puntos_virtuales)
    return puntos_virtuales

def criterio_derivada_maxima_no_debbugging(cad_1, cad_2,extremo,puntos_virtuales, umbral=None,N=20,paso=2):
    puntos_virtuales = generar_puntos_virtuales_sin_cadena_soporte(cad_1, cad_2,extremo)

    res, abs_der_1, abs_der_2,abs_der_3, salto, radios_1, radios_2,radios_virtuales, coeficiente, derivada_maxima_var =\
        derivada_maxima( cad_1, cad_2, extremo, puntos_virtuales,umbral, N, paso)
def criterio_derivada_maxima(state, cad_1, cad_2,extremo,puntos_virtuales, umbral=None,N=20,paso=2):
    label = 'criterio_derivada_maxima'

    if state.desde_el_centro:
        puntos_virtuales = generar_puntos_virtuales_sin_cadena_soporte(cad_1, cad_2,extremo)

    res, abs_der_1, abs_der_2,abs_der_3, salto, radios_1, radios_2,radios_virtuales, coeficiente, derivada_maxima_var =\
        derivada_maxima( cad_1, cad_2, extremo, puntos_virtuales,umbral, N, paso)

    if state.debug:
        # Create just a figure and only one subplot
        # Create two subplots and unpack the output array immediately
        f, (ax2, ax1) = plt.subplots(2, 1)
        ax1.plot(abs_der_3)
        if extremo in 'A':
            ax1.plot(np.arange(0,len(abs_der_2)),abs_der_2[::-1])
            ax1.plot(np.arange(len(radios_virtuales)-len(radios_1),len(radios_virtuales)),abs_der_1)
        else:
            ax1.plot(np.arange(0,len(abs_der_1)),abs_der_1[::-1])
            ax1.plot(np.arange(len(radios_virtuales)-len(radios_2),len(radios_virtuales)),abs_der_2)
        ax1.hlines(y=salto, xmin=0, xmax=np.maximum(len(radios_2), len(radios_1)), label='Salto')
        ax1.hlines(y=coeficiente*derivada_maxima_var, xmin=0, xmax=np.maximum(len(radios_2), len(radios_1)),colors='r', label='umbral')
        #ax1.title(f"{res} der_max:{derivada_maxima_var}")
        ax1.legend()

        ax2.plot(radios_virtuales)
        if extremo in 'A':
            ax2.plot(np.arange(0,len(abs_der_2)),radios_2[::-1],'r')
            ax2.plot(np.arange(len(radios_virtuales)-len(radios_1),len(radios_virtuales)),radios_1)
        else:
            ax2.plot(np.arange(0, len(abs_der_1)), radios_1[::-1], 'r')
            ax2.plot(np.arange(len(radios_virtuales) - len(radios_2), len(radios_virtuales)), radios_2)

        #ax2.plot(radios_2)
        #plt.show()

        plt.savefig(f'{str(state.path)}/{state.iteracion}_derivada_{res}.png')
        plt.close()
        state.iteracion += 1

    return res


def criterio_diferencia_radial_maxima(state, cad_1, cad_2,extremo, umbral=None,N=20,paso=2):
    label = 'criterio_derivada_maxima'
    res, abs_der_1, abs_der_2, salto, radios_1, radios_2,radios_virtuales, coeficiente, derivada_maxima_var =\
        diferencia_maxima( cad_1, cad_2, extremo, umbral, N, paso)

    if state.debug:
        plt.figure()
        plt.plot(abs_der_1)
        plt.plot(abs_der_2)
        plt.plot
        plt.hlines(y=salto, xmin=0, xmax=np.maximum(len(radios_2), len(radios_1)), label='Salto')
        plt.hlines(y=coeficiente*derivada_maxima_var, xmin=0, xmax=np.maximum(len(radios_2), len(radios_1)),colors='r', label='umbral')
        plt.title(f"{res} der_max:{derivada_maxima_var}")
        plt.legend()
        plt.savefig(f'{str(state.path)}/{state.iteracion}_derivada_{res}.png')
        plt.close()
        state.iteracion += 1

    return res


def calcular_puntos_virtuales_entre_dos_cadenas(cadena_origen, cadena_cand, cadena_soporte, extremo,add=True):
    puntos_virtuales = []
    cadena_origen_auxiliar = ch.copiar_cadena(cadena_origen)
    cadena_cand_auxiliar = ch.copiar_cadena(cadena_cand)

    pegar_dos_cadenas_interpolando_via_cadena_soporte(cadena_soporte, cadena_origen_auxiliar, cadena_cand_auxiliar,
                                                                 puntos_virtuales, extremo, add=add)
    return puntos_virtuales


def criterio_umbral_radial(umbral, radios_extremos):
    radio_inicio = radios_extremos[0]
    radio_fin = radios_extremos[1]
    radio_origen_infimo = radio_inicio * ( 1 - umbral)
    radio_origen_supremo = radio_inicio * ( 1 + umbral)
    radio_fin_infimo = radio_fin * (1 - umbral)
    radio_fin_supremo = radio_fin * (1 + umbral)
    #criterio = not (radio_fin_infimo> radio_origen_supremo or radio_fin_supremo < radio_origen_infimo)

    #criterio = radio_fin_supremo >= radio_origen_infimo and radio_fin_infimo <= radio_origen_supremo
    criterio = radio_origen_infimo <= radio_fin <= radio_origen_supremo
    return criterio, radio_inicio, radio_fin, radio_origen_infimo, radio_origen_supremo

def criterios_radiales(state, chain, cadena_origen, cadena_cand, extremo, N=20):
    label = 'criterios_radiales'
    control = True
    if control:
        puntos = [punto for punto in cadena_cand.lista if punto.cadenaId != cadena_cand.id]
        if len(puntos) > 0:
            raise
    ######## calcular radios virtuales
    puntos_virtuales = []
    cadena_origen_auxiliar = ch.copiar_cadena(cadena_origen)
    cadena_cand_auxiliar = ch.copiar_cadena(cadena_cand)

    pegar_dos_cadenas_interpolando_via_cadena_soporte(chain, cadena_origen_auxiliar, cadena_cand_auxiliar,
                                                      puntos_virtuales, extremo)
    control = True
    if control:
        puntos = [punto for punto in cadena_cand.lista if punto.cadenaId != cadena_cand.id]
        if len(puntos) > 0:
            raise
    img_radios = state.img.copy() if state.debug else None

    extremo_inicio = cadena_origen.extB if extremo in 'B' else cadena_origen.extA
    extremo_fin = cadena_cand.extA if extremo in 'B' else cadena_cand.extB
    ptos_extremos_radios = [extremo_inicio] + puntos_virtuales + [extremo_fin]

    radios_extremos = contruir_radios([extremo_inicio, extremo_fin], chain, img_radios)
    radios_virtuales = contruir_radios(puntos_virtuales, chain, img_radios)
    ############# calcular radios origen
    Nmax = 20
    Nsize = np.minimum(cadena_origen.size, cadena_cand.size)
    N = Nmax  # if Nmax<Nsize else Nsize
    ptos_extremo_radios_origen = cadena_origen.sort_dots(sentido='anti') if extremo in 'A' \
        else cadena_origen.sort_dots(sentido='horario')

    radios_origen = contruir_radios(ptos_extremo_radios_origen[:N], chain, img_radios)
    ############## calcular radio destino
    ptos_extremo_radios_cand = cadena_cand.sort_dots(sentido='horario') if extremo in 'A' \
        else cadena_cand.sort_dots(sentido='anti')
    radios_cand = contruir_radios(ptos_extremo_radios_cand[:N], chain, img_radios)

    if state.debug:
        ch.visualizarCadenasSobreDiscoTodas([chain, cadena_origen, cadena_cand], img_radios, [],
                                            f'{state.iteracion}_orig_{cadena_origen.label_id}_cand_{cadena_cand.label_id}',
                                            labels=False, save=str(state.path))
        write_log(MODULE_NAME, label, f'{state.iteracion}_orig_{cadena_origen.label_id}_cand_{cadena_cand.label_id}')
        state.iteracion += 1

    if len(radios_origen) == 0 or len(radios_cand) == 0:
        return False, -1, None

    media_origen = np.mean(radios_origen)
    std_origen = np.std(radios_origen)
    ancho_std = state.ancho_std
    inf_limit = media_origen - ancho_std * std_origen
    sup_limit = media_origen + ancho_std * std_origen

    media_estadistico_cand = np.mean(radios_cand)
    inf_cand = media_estadistico_cand - ancho_std * np.std(radios_cand)
    sup_cand = media_estadistico_cand + ancho_std * np.std(radios_cand)
    criterio_distribucion = inf_cand <= sup_limit and inf_limit <= sup_cand
    criterio_umbral, radio_inicio, radio_fin, radio_origen_infimo, radio_origen_supremo = criterio_umbral_radial(state.radio_limit,
                                                                                                    radios_extremos)
    criterio = criterio_distribucion or criterio_umbral
    # radio_inicio = radios_extremos[0]
    # radio_fin = radios_extremos[1]
    #
    # #criterio = radio_inicio*( 1 + state.radio_limit ) >= radio_fin >= radio_inicio*( 1 - state.radio_limit )

    write_log(MODULE_NAME, label,
              f"iter:{state.iteracion} params: chain: {chain.label_id} cadena_origen {cadena_origen.label_id} cadena_cand {cadena_cand.label_id} ext {extremo}")

    write_log(MODULE_NAME, label,
              f"iter: {state.iteracion} inf_limit {inf_limit:0.3f}  estadistico media  {media_estadistico_cand:0.3f}  sup_limit {sup_limit:0.3f}, "
              f"inf_cand {inf_cand:0.3f} sup_cand {sup_cand:.3f}  criterio {criterio}")

    if state.debug:
        plt.figure()
        plt.hist(radios_virtuales, bins=10, alpha=0.3, color='r', label='virtuales')
        plt.hist(radios_origen, bins=10, alpha=0.3, color='b', label='origen')
        plt.axvline(x=media_origen, color='b', label=f'media_origen:{media_origen:.1f}')
        plt.axvline(x=inf_limit, color='b', label=f'inf_origen:{inf_limit:.1f}')
        plt.axvline(x=sup_limit, color='b', label=f'sup_origen:{sup_limit:.1f}')
        plt.hist(radios_cand, bins=10, alpha=0.3, color='k', label='cand')
        plt.axvline(x=media_estadistico_cand, color='k', label=f'media_cand:{media_estadistico_cand:.1f}')
        plt.axvline(x=inf_cand, color='k', label=f'inf_cand:{inf_cand:.1f}')
        plt.axvline(x=sup_cand, color='k', label=f'sup_cand:{sup_cand:.1f}')
        plt.legend()
        plt.title(f"{criterio}: largo_virtuales: {len(radios_virtuales)}. largo_origen: {len(radios_origen)}.\n"
                  f" largo_cand: {len(radios_cand)}.")
        plt.savefig(f'{str(state.path)}/{state.iteracion}_histogramas_.png')
        plt.close()
        state.iteracion += 1



    if criterio_distribucion or criterio_umbral:
        #valida_derivada = criterio_derivada_maxima(state, cadena_origen, cadena_cand, extremo, puntos_virtuales, umbral=2)
        return True, np.abs(media_origen-media_estadistico_cand), InfoBandaVirtual(ptos_extremos_radios, cadena_origen,
                                        cadena_cand, extremo, chain,inf_orig=radios_origen[0]-ancho_std*std_origen,
                                    sup_orig=radios_origen[0]+ancho_std*std_origen, inf_cand=radios_cand[0] - ancho_std*np.std(radios_cand),
                                    sup_cand=radios_cand[0]+ancho_std*np.std(radios_cand),ancho_banda=0.05 if chain.is_center else 0.1)
    return False, -1, None



def criterio_distancia_radial(state,chain, cadena_origen, cadena_cand, extremo):
    label = 'criterio_distancia_radial'
    ######## calcular radios virtuales
    puntos_virtuales = calcular_puntos_virtuales_entre_dos_cadenas(cadena_origen, cadena_cand, chain, extremo)

    img_radios = state.img.copy() if state.debug else None

    extremo_inicio = cadena_origen.extB if extremo in 'B' else cadena_origen.extA
    extremo_fin = cadena_cand.extA if extremo in 'B' else cadena_cand.extB
    ptos_extremos_radios = [extremo_inicio] + puntos_virtuales + [extremo_fin]


    radios_extremos = contruir_radios([extremo_inicio, extremo_fin],chain,img_radios)
    radios_virtuales = contruir_radios(puntos_virtuales, chain, img_radios)
    ############# calcular radios origen
    Nmax = 20
    Nsize = np.minimum(cadena_origen.size, cadena_cand.size)
    N = Nmax #if Nmax<Nsize else Nsize
    ptos_extremo_radios_origen = cadena_origen.sort_dots(sentido='anti') if extremo in 'A' \
        else cadena_origen.sort_dots(sentido='horario')
    ptos_extremo_radios_cand = cadena_cand.sort_dots(sentido='horario') if extremo in 'A' \
        else cadena_cand.sort_dots(sentido='anti')

    if state.debug:
        ch.visualizarCadenasSobreDiscoTodas([chain, cadena_origen, cadena_cand], img_radios,[],
                        f'{state.iteracion}_orig_{cadena_origen.label_id}_cand_{cadena_cand.label_id}_distancia_radial',  labels=False, save=str(state.path))
        write_log(MODULE_NAME, label, f'{state.iteracion}_orig_{cadena_origen.label_id}_cand_{cadena_cand.label_id}')
        state.iteracion += 1


    ############## calcular radio destino
    radios_origen = contruir_radios(ptos_extremo_radios_origen[:N], chain, img_radios)
    radios_cand = contruir_radios(ptos_extremo_radios_cand[:N], chain, img_radios)

    if len(radios_origen) == 0 or len(radios_cand) == 0:
        return False, -1, None

    media_origen = np.mean(radios_origen)
    std_origen = np.std(radios_origen)
    ancho_std = state.radio_limit
    inf_limit = media_origen - ancho_std*std_origen
    sup_limit = media_origen + ancho_std*std_origen

    media_estadistico_cand = np.mean(radios_cand)
    inf_cand = media_estadistico_cand - ancho_std*np.std(radios_cand)
    sup_cand = media_estadistico_cand + ancho_std*np.std(radios_cand)


    ########################################## Esta parte es la que define realmente el criterio########################
    criterio, radio_inicio, radio_fin, radio_origen_infimo, radio_origen_supremo = criterio_umbral_radial(state.radio_limit,
                                                                                                          radios_extremos)
    info_band = InfoBandaVirtual(ptos_extremos_radios, cadena_origen,
                     cadena_cand, extremo, chain, inf_orig=radios_origen[0] - ancho_std * std_origen,
                     sup_orig=radios_origen[0] + ancho_std * std_origen,
                     inf_cand=radios_cand[0] - ancho_std * np.std(radios_cand),
                     sup_cand=radios_cand[0] + ancho_std * np.std(radios_cand),ancho_banda=0.05 if chain.is_center else 0.1)
    ####################################################################################################################
    if state.debug:
        write_log(MODULE_NAME, label,
                  f"iter:{state.iteracion} params: chain: {chain.label_id} cadena_origen {cadena_origen.label_id} cadena_cand {cadena_cand.label_id} ext {extremo}")

        write_log(MODULE_NAME, label ,
                      f"iter: {state.iteracion} inf_limit {inf_limit:0.3f}  estadistico media  {media_estadistico_cand:0.3f}  sup_limit {sup_limit:0.3f}, "
                      f"inf_cand {inf_cand:0.3f} sup_cand {sup_cand:.3f}  criterio {criterio}")

    if state.debug:
        plt.figure()
        plt.hist(radios_virtuales, bins=10, alpha=0.3,color='r', label='virtuales')
        plt.hist(radios_origen, bins=10, alpha=0.3,color='b', label='origen')
        plt.axvline(x=radio_inicio, color='b', label=f'media_origen:{radio_inicio:.1f}')
        plt.axvline(x=radio_origen_infimo, color='b', label=f'inf_origen:{radio_origen_infimo:.1f}')
        plt.axvline(x=radio_origen_supremo, color='b', label=f'sup_origen:{radio_origen_supremo:.1f}')
        plt.hist(radios_cand, bins=10, alpha=0.3, color='k', label='cand')
        plt.axvline(x=radio_fin, color='k', label=f'media_cand:{radio_fin:.1f}')
        plt.legend()
        plt.title(f"{criterio}: Umbral {state.radio_limit}.")
        plt.savefig(f'{str(state.path)}/{state.iteracion}_histogramas_distancia_radial.png')
        plt.close()
        state.iteracion += 1

    return criterio, np.abs(media_origen-media_estadistico_cand), info_band

def criterio_distancia_radial_no_debugging(umbral,chain, cadena_origen, cadena_cand, extremo):
    label = 'criterio_distancia_radial'
    ######## calcular radios virtuales
    puntos_virtuales = calcular_puntos_virtuales_entre_dos_cadenas(cadena_origen, cadena_cand, chain, extremo,add=False)


    extremo_inicio = cadena_origen.extB if extremo in 'B' else cadena_origen.extA
    extremo_fin = cadena_cand.extA if extremo in 'B' else cadena_cand.extB
    ptos_extremos_radios = [extremo_inicio] + puntos_virtuales + [extremo_fin]


    radios_extremos = contruir_radios([extremo_inicio, extremo_fin],chain)
    ############# calcular radios origen
    Nmax = 20
    Nsize = np.minimum(cadena_origen.size, cadena_cand.size)
    N = Nmax #if Nmax<Nsize else Nsize
    ptos_extremo_radios_origen = cadena_origen.sort_dots(sentido='anti') if extremo in 'A' \
        else cadena_origen.sort_dots(sentido='horario')
    ptos_extremo_radios_cand = cadena_cand.sort_dots(sentido='horario') if extremo in 'A' \
        else cadena_cand.sort_dots(sentido='anti')

    if len(ptos_extremo_radios_origen) == 0 or len(ptos_extremo_radios_cand) == 0:
        return False, -1, None

    ############## calcular radio destino
    radios_origen = contruir_radios(ptos_extremo_radios_origen[:N], chain)
    radios_cand = contruir_radios(ptos_extremo_radios_cand[:N], chain)


    media_origen = np.mean(radios_origen)
    std_origen = np.std(radios_origen)
    ancho_std = umbral
    inf_limit = media_origen - ancho_std*std_origen
    sup_limit = media_origen + ancho_std*std_origen

    media_estadistico_cand = np.mean(radios_cand)
    inf_cand = media_estadistico_cand - ancho_std*np.std(radios_cand)
    sup_cand = media_estadistico_cand + ancho_std*np.std(radios_cand)
    #criterio = sup_cand>=inf_limit and inf_cand<=sup_limit

    ########################################## Esta parte es la que define realmente el criterio########################
    criterio, radio_inicio, radio_fin, radio_origen_infimo, radio_origen_supremo = criterio_umbral_radial(umbral,
                                                                                                          radios_extremos)
    info_band = InfoBandaVirtual(ptos_extremos_radios, cadena_origen,
                     cadena_cand, extremo, chain, inf_orig=radios_origen[0] - ancho_std * std_origen,
                     sup_orig=radios_origen[0] + ancho_std * std_origen,
                     inf_cand=radios_cand[0] - ancho_std * np.std(radios_cand),
                     sup_cand=radios_cand[0] + ancho_std * np.std(radios_cand))
    ####################################################################################################################


    return criterio, np.abs(media_origen-media_estadistico_cand), info_band

from lib.devernayEdgeDetector import devernayEdgeDetector

def angulo_entre_extremos_cadenas(ch_up, next_chain, border):
    extremo1 = ch_up.extA if border in 'A' else ch_up.extB
    extremo2 = next_chain.extB if border in 'A' else next_chain.extA

    i_int, j_int = int(extremo1.x), int(extremo1.y)
    i_c = ch_up.centro[0]
    j_c = ch_up.centro[1]
    v1 = devernayEdgeDetector.calculate_vector_refered_to_center(i_int, j_int, i_c, j_c)
    v2 = devernayEdgeDetector.calculate_vector_refered_to_center(i_int, j_int, int(extremo2.x), int(extremo2.y))
    angle_between = edges.angle_between(v1, v2) * 180 / np.pi
    return angle_between
def criterio_distribucion_radial(state,chain, cadena_origen, cadena_cand, extremo):
    label = 'criterio_distancia_radial'
    control = True
    if control:
        puntos = [punto for punto in cadena_cand.lista if punto.cadenaId != cadena_cand.id]
        if len(puntos) > 0:
            raise
    ######## calcular radios virtuales
    puntos_virtuales = []
    cadena_origen_auxiliar = ch.copiar_cadena(cadena_origen)
    cadena_cand_auxiliar = ch.copiar_cadena(cadena_cand)

    pegar_dos_cadenas_interpolando_via_cadena_soporte(chain, cadena_origen_auxiliar, cadena_cand_auxiliar,
                                                                 puntos_virtuales, extremo)
    control = True
    if control:
        puntos = [punto for punto in cadena_cand.lista if punto.cadenaId != cadena_cand.id]
        if len(puntos) > 0:
            raise
    img_radios = state.img.copy() if state.debug else None

    extremo_inicio = cadena_origen.extB if extremo in 'B' else cadena_origen.extA
    extremo_fin = cadena_cand.extA if extremo in 'B' else cadena_cand.extB
    ptos_extremos_radios = [extremo_inicio] + puntos_virtuales + [extremo_fin]


    radios_extremos = contruir_radios([extremo_inicio, extremo_fin],chain,img_radios)
    radios_virtuales = contruir_radios(puntos_virtuales, chain, img_radios)
    ############# calcular radios origen
    Nmax = 20
    Nsize = np.minimum(cadena_origen.size, cadena_cand.size)
    N = Nmax #if Nmax<Nsize else Nsize
    ptos_extremo_radios_origen = cadena_origen.sort_dots(sentido='anti') if extremo in 'A' \
        else cadena_origen.sort_dots(sentido='horario')

    radios_origen = contruir_radios(ptos_extremo_radios_origen[:N], chain, img_radios)
    ############## calcular radio destino
    ptos_extremo_radios_cand = cadena_cand.sort_dots(sentido='horario') if extremo in 'A' \
        else cadena_cand.sort_dots(sentido='anti')
    radios_cand = contruir_radios(ptos_extremo_radios_cand[:N], chain, img_radios)

    if state.debug:
        ch.visualizarCadenasSobreDiscoTodas([chain, cadena_origen, cadena_cand], img_radios,[],
                        f'{state.iteracion}_orig_{cadena_origen.label_id}_cand_{cadena_cand.label_id}',  labels=False, save=str(state.path))
        write_log(MODULE_NAME, label, f'{state.iteracion}_orig_{cadena_origen.label_id}_cand_{cadena_cand.label_id}')
        state.iteracion += 1

    if len(radios_origen) == 0 or len(radios_cand)==0:
        return False,-1,None

    media_origen = np.mean(radios_origen)
    std_origen = np.std(radios_origen)
    ancho_std = state.ancho_std
    inf_limit = media_origen - ancho_std*std_origen
    sup_limit = media_origen + ancho_std*std_origen

    media_estadistico_cand = np.mean(radios_cand)
    inf_cand = media_estadistico_cand - ancho_std*np.std(radios_cand)
    sup_cand = media_estadistico_cand + ancho_std*np.std(radios_cand)
    criterio = sup_cand>=inf_limit and inf_cand<=sup_limit
    criterio = inf_cand<=sup_limit and inf_limit<=sup_cand

    # radio_inicio = radios_extremos[0]
    # radio_fin = radios_extremos[1]
    #
    # #criterio = radio_inicio*( 1 + state.radio_limit ) >= radio_fin >= radio_inicio*( 1 - state.radio_limit )

    write_log(MODULE_NAME, label,
              f"iter:{state.iteracion} params: chain: {chain.label_id} cadena_origen {cadena_origen.label_id} cadena_cand {cadena_cand.label_id} ext {extremo}")

    write_log(MODULE_NAME, label ,
                  f"iter: {state.iteracion} inf_limit {inf_limit:0.3f}  estadistico media  {media_estadistico_cand:0.3f}  sup_limit {sup_limit:0.3f}, "
                  f"inf_cand {inf_cand:0.3f} sup_cand {sup_cand:.3f}  criterio {criterio}")

    if state.debug:
        plt.figure()
        plt.hist(radios_virtuales, bins=10, alpha=0.3,color='r', label='virtuales')
        plt.hist(radios_origen, bins=10, alpha=0.3,color='b', label='origen')
        plt.axvline(x=media_origen, color='b', label=f'media_origen:{media_origen:.1f}')
        plt.axvline(x=inf_limit, color='b', label=f'inf_origen:{inf_limit:.1f}')
        plt.axvline(x=sup_limit, color='b', label=f'sup_origen:{sup_limit:.1f}')
        plt.hist(radios_cand, bins=10, alpha=0.3, color='k', label='cand')
        plt.axvline(x=media_estadistico_cand, color='k', label=f'media_cand:{media_estadistico_cand:.1f}')
        plt.axvline(x=inf_cand, color='k', label=f'inf_cand:{inf_cand:.1f}')
        plt.axvline(x=sup_cand, color='k', label=f'sup_cand:{sup_cand:.1f}')
        plt.legend()
        plt.title(f"{criterio}: largo_virtuales: {len(radios_virtuales)}. largo_origen: {len(radios_origen)}.\n"
                  f" largo_cand: {len(radios_cand)}.")
        plt.savefig(f'{str(state.path)}/{state.iteracion}_histogramas_distribucion_radial.png')
        plt.close()
        state.iteracion += 1

    return criterio, np.abs(media_origen-media_estadistico_cand), InfoBandaVirtual(ptos_extremos_radios, cadena_origen,
                                        cadena_cand, extremo, chain,inf_orig=radios_origen[0]-ancho_std*std_origen,
                                    sup_orig=radios_origen[0]+ancho_std*std_origen, inf_cand=radios_cand[0] - ancho_std*np.std(radios_cand),
                                                                            sup_cand=radios_cand[0]+ancho_std*np.std(radios_cand),ancho_banda=0.05 if chain.is_center else 0.1)



def contruir_celdas(ptos_extremos_celdas, chain):
    contador_celdas = 0
    celdas_virtuales = []
    idx_pto = 0
    dominio_angular = chain._completar_dominio_angular(chain)
    while True:
        rayo_1_ext1 = ptos_extremos_celdas[idx_pto]
        if rayo_1_ext1.angulo not in dominio_angular:
            break
        try:
            rayo_1_ext2 = chain.getDotByAngle(rayo_1_ext1.angulo)[0]

        except Exception as e:
            print(f"Exp: {e}. rayo1 angulo {rayo_1_ext1.angulo}")
            raise

        rayo_2_ext1 = ptos_extremos_celdas[idx_pto+1]
        if rayo_2_ext1.angulo not in dominio_angular:
            break

        try:
            rayo_2_ext2 = chain.getDotByAngle(rayo_2_ext1.angulo)[0]

        except Exception as e:
            print(f"Exp: {e}. rayo2 angulo {rayo_2_ext1.angulo}")
            raise

        if rayo_1_ext2.radio > rayo_1_ext1.radio:
            o_inter = Interseccion(rayo_1_ext1.y, rayo_1_ext1.x, rayo_id = rayo_1_ext2.angulo)
            o_inter_siguiente = Interseccion(rayo_1_ext2.y, rayo_1_ext2.x, rayo_id = rayo_1_ext2.angulo)

        else:
            o_inter_siguiente = Interseccion(rayo_1_ext1.y, rayo_1_ext1.x, rayo_id = rayo_1_ext2.angulo)
            o_inter = Interseccion(rayo_1_ext2.y, rayo_1_ext2.x, rayo_id = rayo_1_ext2.angulo)

        if rayo_2_ext2.radio > rayo_2_ext1.radio:
            sig_inter = Interseccion(rayo_2_ext1.y, rayo_2_ext1.x, rayo_id=rayo_2_ext2.angulo)
            sig_inter_siguiente = Interseccion(rayo_2_ext2.y, rayo_2_ext2.x, rayo_id=rayo_2_ext2.angulo)

        else:
            sig_inter_siguiente = Interseccion(rayo_2_ext1.y, rayo_2_ext1.x, rayo_id=rayo_2_ext2.angulo)
            sig_inter = Interseccion(rayo_2_ext2.y, rayo_2_ext2.x, rayo_id=rayo_2_ext2.angulo)

        celda = Celda(contador_celdas, o_inter, sig_inter, sig_inter_siguiente, o_inter_siguiente,
                      rayo_1_ext2.angulo, (rayo_1_ext2.angulo + 1) % 360)
        celdas_virtuales.append(celda)

        idx_pto += 1
        if idx_pto == len(ptos_extremos_celdas)-1:
            break


    return celdas_virtuales





def criterio_celdas(state,chain, cadena_origen, cadena_cand, extremo):
    label = 'criterio_celda'
    puntos_virtuales = []
    cadena_origen_auxiliar = copy.deepcopy(cadena_origen)
    pegar_dos_cadenas_interpolando_via_cadena_soporte(chain, cadena_origen_auxiliar, cadena_cand,
                                                                 puntos_virtuales, extremo)
    # if state.debug:
    #     ch.visualizarCadenasSobreDisco([chain, cadena_origen_auxiliar, cadena_origen, cadena_cand], state.img.copy(),
    #                                    f'{state.iteracion}_subconjunto_cadenas_inicio_distancia_maxima', labels=True,
    #                                    save=str(state.path))
    #     state.iteracion += 1

    # contruir celdas virtuales
    extremo_inicio = cadena_origen.extB if extremo in 'B' else cadena_origen.extA
    extremo_fin = cadena_cand.extA if extremo in 'B' else cadena_cand.extB

    ptos_extremos_celdas = [extremo_inicio] + puntos_virtuales + [extremo_fin]
    celdas_virtuales = contruir_celdas(ptos_extremos_celdas, chain)

    Nmax = 20
    Nsize = np.minimum(cadena_origen.size, cadena_cand.size)
    N = Nmax if Nmax<Nsize else Nsize
    ptos_extremo_celdas_origen = cadena_origen.sort_dots(sentido='anti') if extremo in 'A' \
        else cadena_origen.sort_dots(sentido='horario')

    celdas_origen = contruir_celdas(ptos_extremo_celdas_origen[:N], chain)

    ptos_extremo_celdas_cand = cadena_cand.sort_dots(sentido='horario') if extremo in 'A' \
        else cadena_cand.sort_dots(sentido='anti')
    celdas_cand = contruir_celdas(ptos_extremo_celdas_cand[:N], chain)

    if state.debug:
        img_celdas = state.img.copy()
        [Dibujar.contorno(celda, img_celdas) for celda in celdas_virtuales]
        [Dibujar.contorno(celda, img_celdas, color=ROJO) for celda in celdas_origen]
        [Dibujar.contorno(celda, img_celdas, color=ROJO) for celda in celdas_cand]
        # ch.visualizarCadenasSobreDisco([chain, cadena_origen_auxiliar, cadena_origen, cadena_cand],
        #                                img_celdas, f'{state.iteracion}_orig_{cadena_origen.label_id}_cand_{cadena_cand.label_id}',
        #                                labels=True, save=str(state.path))
        ch.visualizarCadenasSobreDiscoTodas([chain, cadena_origen, cadena_cand], img_celdas,[],
                        f'{state.iteracion}_orig_{cadena_origen.label_id}_cand_{cadena_cand.label_id}',  labels=False, save=str(state.path))
        write_log(MODULE_NAME, label, f'{state.iteracion}_orig_{cadena_origen.label_id}_cand_{cadena_cand.label_id}')
        state.iteracion += 1


    estadistico_celdas_virtuales = [celda.calcular_estadistico() for celda in celdas_virtuales]
    estadistico_celdas_origen = [celda.calcular_estadistico() for celda in celdas_origen]
    estadistico_celdas_cand = [celda.calcular_estadistico() for celda in celdas_cand]

    if len(estadistico_celdas_cand) ==0 or len(estadistico_celdas_origen)==0:
        return False

    media_origen = np.mean(estadistico_celdas_origen)
    std_origen = np.std(estadistico_celdas_origen)
    ancho_std = state.radio_limit
    inf_limit = media_origen - ancho_std*std_origen
    sup_limit = media_origen + ancho_std*std_origen
    media_estadistico_cand = np.mean(estadistico_celdas_cand)

    inf_cand = media_estadistico_cand - ancho_std*np.std(estadistico_celdas_cand)
    sup_cand = media_estadistico_cand + ancho_std*np.std(estadistico_celdas_cand)
    criterio = sup_cand>=inf_limit and inf_cand<=sup_limit


    write_log(MODULE_NAME, label,
              f"iter:{state.iteracion} params: chain: {chain.label_id} cadena_origen {cadena_origen.label_id} cadena_cand {cadena_cand.label_id} ext {extremo}")

    write_log(MODULE_NAME, label ,
                  f"iter: {state.iteracion} inf_limit {inf_limit:0.3f}  estadistico media  {media_estadistico_cand:0.3f}  sup_limit {sup_limit:0.3f}, "
                  f"inf_cand {inf_cand:0.3f} sup_cand {sup_cand:.3f}  criterio {criterio}")

    if state.debug:
        plt.figure()
        plt.hist(estadistico_celdas_virtuales, bins=10, alpha=0.3,color='r', label='virtuales')
        plt.hist(estadistico_celdas_origen, bins=10, alpha=0.3,color='b', label='origen')
        plt.axvline(x=media_origen, color='b', label='media_origen')
        plt.axvline(x=inf_limit, color='b', label='inf_origen')
        plt.axvline(x=sup_limit, color='b', label='sup_origen')
        plt.hist(estadistico_celdas_cand, bins=10, alpha=0.3, color='k', label='cand')
        plt.axvline(x=media_estadistico_cand, color='k', label='media_cand')
        plt.axvline(x=inf_cand, color='k', label='inf_cand')
        plt.axvline(x=sup_cand, color='k', label='sup_cand')
        plt.legend()
        plt.title(f"{criterio}")
        plt.savefig(f'{str(state.path)}/{state.iteracion}_histogramas.png')
        plt.close()
        state.iteracion += 1

    return criterio

def contruir_radios(ptos_extremos_celdas, chain, img = None):
    if len(ptos_extremos_celdas)==0:
        return []
    distancias_radiales = []
    idx_pto = 0
    dominio_angular = chain._completar_dominio_angular(chain)
    while True:

        rayo_1_ext1 = ptos_extremos_celdas[idx_pto]
        if rayo_1_ext1.angulo not in dominio_angular:
            break
        try:
            rayo_1_ext2 = chain.getDotByAngle(rayo_1_ext1.angulo)[0]

        except Exception as e:
            print(f"Exp: {e}. rayo1 angulo {rayo_1_ext1.angulo}")
            raise


        distancias_radiales.append(ch.distancia_entre_puntos(rayo_1_ext2, rayo_1_ext1))
        if img is not None:
            dibujar_segmentoo_entre_puntos(rayo_1_ext2, rayo_1_ext1, img, color= Color.red)

        idx_pto += 1
        if idx_pto >= len(ptos_extremos_celdas):
            break

    return distancias_radiales

def hay_cadenas_superpuestas_en_banda(lista_cadenas, info_banda):
    lista_cadenas_de_interes = [info_banda.cadena_cand, info_banda.cadena_origen]
    if info_banda.cad_soporte is not None:
        lista_cadenas_de_interes.append(info_banda.cad_soporte)

    cadenas_superpuestas_o_cruzadas = [cad for cad in lista_cadenas if
                            cad not in lista_cadenas_de_interes and info_banda.is_chain_in_band(cad) ]


    return cadenas_superpuestas_o_cruzadas

def hay_cadenas_superpuestas(state, info_banda):
    label='hay_cadenas_superpuestas'

    cadenas_superpuestas_o_cruzadas = hay_cadenas_superpuestas_en_banda(state.lista_cadenas, info_banda)
    hay_cadena_intersectante = len(cadenas_superpuestas_o_cruzadas) > 0

    if state.debug:
        img_debug = Dibujar.lista_cadenas(state.lista_cadenas,state.img.copy())
        img_debug = info_banda.dibujar_bandas(img_debug, cadenas_superpuestas_o_cruzadas)
        cv2.imwrite(f'{str(state.path)}/{state.iteracion}_orig_{info_banda.cadena_origen.label_id}_cand_{info_banda.cadena_cand.label_id}_superpuestas_{hay_cadena_intersectante}.png',
                    img_debug)

        state.iteracion += 1
        write_log(MODULE_NAME, label,
                  f"cad.id {info_banda.cadena_origen.label_id} border {info_banda.extremo} cad.id {info_banda.cadena_cand.label_id} "
                  f"HayCadenaIntersectante {hay_cadena_intersectante}")

    return hay_cadena_intersectante

