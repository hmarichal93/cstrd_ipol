import numpy as np
from shapely.geometry import Polygon
import cv2
import copy




VERDE = 2
ROJO = 1
AZUL = 0
AMARILLO = 3
NEGRO = 4
NARANJA = 5
def distancia_entre_pixeles(i1, j1, i2, j2):
    return np.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2)



class OperacionesCelda:
    def __init__(self,centro):
        self.centro = centro

    def interseccion_en_lateral(self,perimetrales_rayo,inter):
        if inter not in perimetrales_rayo:
            perimetrales_rayo.append(inter)
            perimetrales_rayo.sort(key=lambda x: distancia_entre_pixeles(self.centro[0], self.centro[1], x.y, x.x))

        index_inter = perimetrales_rayo.index(inter)
        return index_inter

    def dividir(self,inter1,inter2,lista_celdas,celda,extremo,lista_intersecciones):
        # 1.0 determinar perimetro de intersecciones de cada celda
        perimetrales_rayo_1 = celda.get_inter_over_lateral(self.centro,rayo='1')
        perimetrales_rayo_2 = celda.get_inter_over_lateral(self.centro, rayo='2')

        index_inter1 = self.interseccion_en_lateral(perimetrales_rayo_1,inter1)
        index_inter2 = self.interseccion_en_lateral(perimetrales_rayo_2, inter2)

        inferior_perimetrales = [inter for idx,inter in enumerate(perimetrales_rayo_1) if idx <= index_inter1]
        inferior_perimetrales += [inter for idx, inter in enumerate(perimetrales_rayo_2) if idx <= index_inter2]
        superior_perimetrales = [inter for idx, inter in enumerate(perimetrales_rayo_1) if idx >= index_inter1]
        superior_perimetrales += [inter for idx, inter in enumerate(perimetrales_rayo_2) if idx >= index_inter2]

        # 2.0 crear celda inferior y superior
        inter11,inter12,inter22,inter21 = celda.lista_intersecciones_vertice
        if extremo in '2':
            celda_inferior = Celda(celda.celda_id,inter11,inter12,inter1,inter2,rayo1=celda.rayo_1,rayo2=celda.rayo_2,lista_inter_perimetrales=inferior_perimetrales)
            celda_superior = Celda(len(lista_celdas), inter2, inter1, inter22, inter21, rayo1=celda.rayo_1, rayo2=celda.rayo_2,lista_inter_perimetrales=superior_perimetrales)
        else:
            celda_inferior = Celda(celda.celda_id, inter11, inter12, inter2, inter1, rayo1=celda.rayo_1,
                                   rayo2=celda.rayo_2, lista_inter_perimetrales=inferior_perimetrales)
            celda_superior = Celda(len(lista_celdas), inter1, inter2, inter22, inter21, rayo1=celda.rayo_1,
                                   rayo2=celda.rayo_2, lista_inter_perimetrales=superior_perimetrales)

        # 2.1 Agregar celdas laterales
        celdas_rayo_1 = [c for c in lista_celdas if c.rayo_2 == celda.rayo_1]
        celdas_rayo_2 = [c for c in lista_celdas if c.rayo_1 == celda.rayo_2]
        for celda_aux in [celda_superior,celda_inferior]:
            celda_aux.set_vecindad(celdas_rayo_1, celdas_rayo_2,self.centro)



        # 3.0 actualizar lista de celdas
        lista_celdas.append(celda_inferior)
        lista_celdas.append(celda_superior)
        lista_celdas.remove(celda)

        # 3.1 actualizar celdas vecinas interseccion nueva
        if inter1 not in lista_intersecciones:
            lista_intersecciones.append(inter1)
            if inter1.rayo_id == celda.rayo_1:
                rayo_id = '2'
                celdas_vecinas = celda.vecindad_1
            else:
                rayo_id = '1'
                celdas_vecinas = celda.vecindad_2

            for vec in celdas_vecinas:
                perimetrales_rayo = vec.get_inter_over_lateral(self.centro, rayo=rayo_id)
                index_inter1 = self.interseccion_en_lateral(perimetrales_rayo, inter1)
                if 0<index_inter1<len(perimetrales_rayo)-1:
                    vec.lista_inter_parimetrales .append(inter1)
                    break



        # 3.2 actualizar celdas vecinas
        celdas_rayo = [c for c in lista_celdas if c.rayo_1 == celda.rayo_1]
        for vec in celdas_rayo_1:
            if vec.celda_lateral_2 == celda:
                vec.celda_lateral_2 = None
            vec.set_vecindad([],celdas_rayo,self.centro)

        for vec in celdas_rayo_2:
            if vec.celda_lateral_1 == celda:
                vec.celda_lateral_1 = None
            vec.set_vecindad( celdas_rayo,[], self.centro)

        return celda_superior,celda_inferior

    def dividir_celda_en_mas_de_2_subceldas(self, celda_a_dividir,  lista_celdas, lista_intersecciones,calcular_ratio_entre_diagonales,ratio_celda_es_valido):
        #En lugar de mirar intersecciones perimetrales mirar celdas vecinas. Tiene que ser la misma cantidad de ambos lados.
        if not( len(celda_a_dividir.vecindad_1) == len(celda_a_dividir.vecindad_2) ) or len(celda_a_dividir.vecindad_1)==0 or len(celda_a_dividir.vecindad_2)==0:
            return

        while True:
            celda_a_dividir.vecindad_1.sort(
                key=lambda x: distancia_entre_pixeles(self.centro[0], self.centro[1], x.centroid.y, x.centroid.x))
            celda_a_dividir.vecindad_2.sort(
                key=lambda x: distancia_entre_pixeles(self.centro[0], self.centro[1], x.centroid.y, x.centroid.x))

            _, int11, int21, _ = celda_a_dividir.vecindad_1[0].lista_intersecciones_vertice
            int12, _, _, int22 = celda_a_dividir.vecindad_2[0].lista_intersecciones_vertice
            ratio_s = calcular_ratio_entre_diagonales(int11, int12, int22, int21)
            validacion = True#ratio_celda_es_valido(ratio_s)
            if validacion:
                inter_partida = int21
                inter_llegada = int22
                celda_superior, celda_inferior = self.dividir(inter_partida, inter_llegada, lista_celdas,
                                                              celda_a_dividir, '1', lista_intersecciones)
                celda_a_dividir = celda_superior

            else:
                break

            if len(celda_a_dividir.vecindad_1)<2 or len(celda_a_dividir.vecindad_2)<2:
                break

        # perimetrales_rayo_1 = celda_a_dividir.get_inter_over_lateral(self.centro, rayo='1')
        # perimetrales_rayo_2 = celda_a_dividir.get_inter_over_lateral(self.centro, rayo='2')
        # if len(perimetrales_rayo_2) == len(perimetrales_rayo_2):
        #     # Misma cantidad de intersecciones de ambos lados.
        #     index = 0
        #     while True:
        #         if len(perimetrales_rayo_2) <= 2:
        #             break
        #         # for index in range(len(perimetrales_rayo_1)-1):
        #         int11, int12, int22, int21 = perimetrales_rayo_1[index], perimetrales_rayo_2[index], \
        #                                      perimetrales_rayo_2[index + 1], perimetrales_rayo_1[index + 1]
        #         ratio_s = calcular_ratio_entre_diagonales(int11, int12, int22, int21)
        #         # ratio_c = self.calcular_ratio_entre_diagonales(*celda.lista_intersecciones_vertice)
        #         validacion = ratio_celda_es_valido(ratio_s)
        #         if validacion:
        #             inter_partida = int21
        #             inter_llegada = int22
        #             celda_superior, celda_inferior = self.dividir(inter_partida, inter_llegada, lista_celdas,
        #                                                           celda_a_dividir, '1', lista_intersecciones)
        #             celda_a_dividir = celda_superior
        #             perimetrales_rayo_1 = celda_a_dividir.get_inter_over_lateral(self.centro, rayo='1')
        #             perimetrales_rayo_2 = celda_a_dividir.get_inter_over_lateral(self.centro, rayo='2')
        #         else:
        #             break

    def unir(self,celda_interior,celda_superior,lista_celdas):
        if celda_interior.tipo == VERDE:
            return "ERROR"
        inter11,inter12,_,_ = celda_interior.lista_intersecciones_vertice
        _,_,inter22,inter21 = celda_superior.lista_intersecciones_vertice
        perimetro = []
        for celda in [celda_interior,celda_superior]:
            perimetrales_rayo_1 = celda.get_inter_over_lateral(self.centro, rayo='1')
            perimetrales_rayo_2 = celda.get_inter_over_lateral(self.centro, rayo='2')
            for inter in perimetrales_rayo_1 + perimetrales_rayo_2:
                if inter not in perimetro:
                    perimetro.append(inter)

        celda_union = Celda(celda_interior.celda_id,inter11,inter12,inter22,inter21,rayo1=celda_interior.rayo_1,rayo2=celda_interior.rayo_2,
                               lista_inter_perimetrales=perimetro)

        # 2.1 Agregar celdas laterales
        celdas_rayo_1 = celda_interior.vecindad_1 + celda_superior.vecindad_1
        celdas_rayo_2 = celda_interior.vecindad_2 + celda_superior.vecindad_2
        celda_union.set_vecindad(celdas_rayo_1, celdas_rayo_2,self.centro)


        #3.0 actualizar entorno
        lista_celdas.remove(celda_interior)
        lista_celdas.remove(celda_superior)
        lista_celdas.append(celda_union)

        # 3.1 actualizar celdas vecinas
        #TODO

        return celda_union




class Celda(Polygon):
    def __init__(self, id_celda, inter11, inter12, inter22, inter21, rayo1, rayo2, tipo=ROJO,
                 lista_inter_perimetrales=[]):
        self.celda_id = id_celda

        self.rayo_1 = rayo1
        self.rayo_2 = rayo2
        self.lista_intersecciones_vertice = [inter11, inter12, inter22, inter21]
        if len(lista_inter_perimetrales) > 0:
            self.lista_inter_parimetrales = lista_inter_perimetrales
        else:
            self.lista_inter_parimetrales = [inter for inter in self.lista_intersecciones_vertice]

        if None in self.lista_intersecciones_vertice:
            self.lista_intersecciones_vertice.remove(None)
        self.tipo = self.tipo_celda()
        self.celda_lateral_1 = None
        self.celda_lateral_2 = None
        self.vecindad_1 = []
        self.vecindad_2 = []
        self.recorrida_en_iteracion = False



        super(self.__class__, self).__init__(self.lista_intersecciones_vertice)



    def __repr__(self):
        return (f'({self.celda_id},{self.tipo})\n')

    def __str__(self):
        return (f'({self.celda_id},{self.tipo})\n')

    def __eq__(self, other):
        if other is None:
            return False
        if self.tipo != other.tipo:
            return False
        if self.tipo == VERDE:
            _, o_21, o_22 = other.lista_intersecciones_vertice
            _, self_21, self_22 = self.lista_intersecciones_vertice
            return self_22 == o_22 and self_21 == o_21
        o_11,o_12,o_22,o_21 = other.lista_intersecciones_vertice
        self_11,self_12,self_22,self_21 = self.lista_intersecciones_vertice
        return self_11 == o_11 and self_12 == o_12 and self_22 == o_22 and self_21 == o_21


    def calcular_estadistico(self,tipo=1):
        """
        tipo 1: ratio dM*dm/Area
        tipo 2: Area
        """
        n_11, n_12, n_22, n_21 = self.lista_intersecciones_vertice
        d1 = n_11.distance(n_22)
        d2 = n_12.distance(n_21)
        d_mayor = np.maximum(d1,d2)
        d_menor = np.minimum(d1,d2)
        if tipo == 1:
            ratio = d_menor * d_mayor / self.area
        else:
            ratio = self.area

        return ratio

    def get_intersecciones_anomalas(self,lateral):
        inter_anomalas = [inter for inter in self.lista_inter_parimetrales if inter not in self.lista_intersecciones_vertice]
        if lateral in '1':
            return [inter for inter in inter_anomalas if inter.rayo_id == self.rayo_1]
        else:
            return [inter for inter in inter_anomalas if inter.rayo_id == self.rayo_2]


    def get_celdas_dato_inter(self,inter1,inter2,lista_celdas):
        celda_vecina = [c for c in lista_celdas if not (
                c == self) and inter1 in c.lista_intersecciones_vertice and inter2 in c.lista_intersecciones_vertice]
        return celda_vecina

    def celdas_vecinas_lateral(self,  lista_celdas,centro,lateral='1'):
        lateral = self.rayo_1 if lateral in '1' else self.rayo_2
        intersecciones = [inter for inter in self.lista_inter_parimetrales if inter.rayo_id == lateral]
        intersecciones.sort(key=lambda x: distancia_entre_pixeles(centro[0], centro[1], x.y, x.x))
        vecindad = []
        for i in range(0, len(intersecciones) - 1):
            inter1 = intersecciones[i]
            inter2 = intersecciones[i + 1]
            celda_vecina = self.get_celdas_dato_inter(inter1, inter2, lista_celdas)
            vecindad += celda_vecina

        return vecindad

    def obtener_celdas_vecinas(self,centro, celdas_rayo_continuo,lateral='1'):
        int11, int12, int22, int21 = self.lista_intersecciones_vertice
        if lateral in '1':
            int_inf = int11
            int_sup = int21
        else:
            int_inf = int12
            int_sup = int22

        #celdas_con_intersecciones_en_comun = [c for c in celdas_rayo_continuo if not(c.tipo == VERDE) and \
        #                                      (int_inf in c.lista_inter_parimetrales or int_sup in c.lista_inter_parimetrales)]
        # celdas_vecinas = []
        # for c in celdas_con_intersecciones_en_comun:
        #     if int_inf not in c.lista_intersecciones_vertice and int_sup not in c.lista_intersecciones_vertice:
        #         celdas_vecinas.append(c)
        #     else:
        #         if int_inf in c.lista_intersecciones_vertice:
        #             index_inf = c.lista_intersecciones_vertice.index(int_inf)
        #             if lateral in '1':
        #                 if index_inf == 1:
        #                     celdas_vecinas.append(c)
        #             elif index_inf == 0:
        #                 celdas_vecinas.append(c)
        #         else:
        #             index_sup = c.lista_intersecciones_vertice.index(int_sup)
        #             if lateral in '1':
        #                 if index_sup == 2:
        #                     celdas_vecinas.append(c)
        #             elif index_sup == 3:
        #                 celdas_vecinas.append(c)

        celdas_con_intersecciones_en_comun = []
        for c in celdas_rayo_continuo:
            if (c.tipo == VERDE):
                continue
            int11, int12, int22, int21 = c.lista_intersecciones_vertice
            if lateral in '1':
                local_intersecciones = [int12,int22] #c.get_inter_over_lateral(centro,'2') if lateral in '1' else c.get_inter_over_lateral(centro,'1')
            else:
                local_intersecciones = [int11,int21]

            if int_inf not in local_intersecciones:
                local_intersecciones.append(int_inf)
            if int_sup not in local_intersecciones:
                local_intersecciones.append(int_sup)

            local_intersecciones.sort(key=lambda x: distancia_entre_pixeles(centro[0], centro[1], x.y, x.x))
            index_1 = local_intersecciones.index(int11) if lateral in '2' else local_intersecciones.index(int12)
            index_2 = local_intersecciones.index(int21) if lateral in '2' else local_intersecciones.index(int22)
            index_inf = local_intersecciones.index(int_inf)
            index_sup = local_intersecciones.index(int_sup)
            if not( index_sup <= index_1 or index_inf>=index_2):
                celdas_con_intersecciones_en_comun.append(c)
            # if  ((index_inf < (len(local_intersecciones)-2)) and index_sup > 1) or len(local_intersecciones)==2:
            #     celdas_con_intersecciones_en_comun.append(c)



        return celdas_con_intersecciones_en_comun




    def set_vecindad(self,lista_celdas_1,lista_celdas_2,centro):
        if self.tipo == VERDE:
            #TODO
            return
        else:
            int11, int12, int22, int21 = self.lista_intersecciones_vertice
            if len(lista_celdas_1) > 0:
                extremo = '1'
                celda_vecinas = self.obtener_celdas_vecinas(centro,lista_celdas_1,extremo)
                if 1>=len(celda_vecinas) > 0:
                    vertices_vecina = celda_vecinas[0].lista_intersecciones_vertice
                    if int11 in vertices_vecina and int21 in vertices_vecina:
                        self.celda_lateral_1 = celda_vecinas[0]
                self.vecindad_1 = celda_vecinas

            if len(lista_celdas_2)> 0:
                extremo = '2'
                celda_vecinas = self.obtener_celdas_vecinas(centro,lista_celdas_2,extremo)
                if 1>=len(celda_vecinas) > 0:
                    vertices_vecina = celda_vecinas[0].lista_intersecciones_vertice
                    if int12 in vertices_vecina and int22 in vertices_vecina:
                        self.celda_lateral_2 = celda_vecinas[0]
                self.vecindad_2 = celda_vecinas

        assert not ((self.celda_lateral_2 == self) or (self.celda_lateral_1 == self))

    def buscar_celdas_pegadas_lateralmente(self,root,lado_en_comun_anterior,lista_celdas_laterales):
        root.recorrida_en_iteracion = True
        celda_vecina = root.celda_lateral_2 if lado_en_comun_anterior in '2' else root.celda_lateral_1
        if celda_vecina is None or celda_vecina in lista_celdas_laterales:
            return None

        lista_celdas_laterales.append(celda_vecina)
        self.buscar_celdas_pegadas_lateralmente(celda_vecina,lado_en_comun_anterior,lista_celdas_laterales)

        return 0

    def obtener_soporte_lateral(self,lateral='1'):
        celda_vecina = self.celda_lateral_1 if lateral in '1' else self.celda_lateral_2
        if celda_vecina is None:
            return [self]
        lista_celdas_laterales = [self,celda_vecina]
        self.buscar_celdas_pegadas_lateralmente(celda_vecina, lateral, lista_celdas_laterales)
        return lista_celdas_laterales

    def obtener_soporte_completo(self):
        lista_celdas = self.obtener_soporte_lateral('1')
        lista_celdas += self.obtener_soporte_lateral('2')
        return lista_celdas

    def calcular_soporte_lateral(self,lateral='1'):
        """Se calcula el soporte sobre uno de los laterales en GRADOS. Es decir cuantas celdas coinciden lateralmente por segmento entre
        intersecciones"""
        lista_celdas_laterales = self.obtener_soporte_lateral(lateral)
        if len(lista_celdas_laterales)==0:
            return 0

        ext_angulo = lista_celdas_laterales[-1].rayo_1

        return np.minimum((self.rayo_1-ext_angulo)%360,(ext_angulo-self.rayo_1)%360)


    def tipo_celda(self):
        if len(self.lista_intersecciones_vertice) == 3:
            inferior_tipo = VERDE
        elif len(self.lista_inter_parimetrales) == 4:
            inferior_tipo = ROJO
        elif len(self.lista_inter_parimetrales) == 5:
            inferior_tipo = AMARILLO
        else:
            inferior_tipo = AZUL
        return inferior_tipo

    def get_inter_over_lateral(self, centro, rayo='1'):
        if rayo in '1':
            perimetrales_rayo = [inter for inter in self.lista_inter_parimetrales if inter.rayo_id == self.rayo_1]
        else:
            perimetrales_rayo = [inter for inter in self.lista_inter_parimetrales if
                                 inter.rayo_id == self.rayo_2]
        perimetrales_rayo.sort(key=lambda x: distancia_entre_pixeles(centro[0], centro[1], x.y, x.x))
        return perimetrales_rayo
