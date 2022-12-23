import numpy as np
from shapely.geometry.point import Point
from shapely.geometry.linestring import LineString


from lib.celdas import distancia_entre_pixeles
from lib.io import Nr
from lib.dibujar import Dibujar

def from_polar_to_cartesian(r,angulo,centro):
    y = centro[0] + r * np.cos(angulo * np.pi / 180)
    x = centro[1] + r * np.sin(angulo * np.pi / 180)
    return (y,x)

class Curva(LineString):

    def __init__(self, lista_pixeles, name):
        self.id = name
        super().__init__(lista_pixeles)
    def __setattr__(self, name, value) -> None:
        object.__setattr__(self, name, value)


class Rayo(LineString):
    def __init__(self, direccion, origen, M, N):
        self.direccion = direccion
        self.centro = origen
        self.extremo = self._calcular_extremo(direccion,origen,M,N)
        super().__init__([origen[::-1],self.extremo[::-1]])


    @staticmethod
    def _calcular_extremo(theta,origen,M,N):
        cte_grados_a_radianes = np.pi/180
        theta = theta % 360
        yc,xc = origen
        if 0<=theta<45:
            ye = M-1
            xe = np.tan(theta*cte_grados_a_radianes)* (M-1-yc) + xc
        elif 45<= theta < 90:
            xe = N-1
            ye = np.tan((90-theta)*cte_grados_a_radianes)*(N-1-xc) + yc
        elif 90<= theta < 135:
            xe = N-1
            ye = yc - np.tan((theta-90)*cte_grados_a_radianes)*(xe-xc)

        elif 135<= theta < 180:
            ye = 0
            xe = np.tan((180-theta)*cte_grados_a_radianes)*(yc) + xc

        elif 180<= theta < 225:
            ye =0
            xe = xc- np.tan((theta-180)*cte_grados_a_radianes)*(yc)

        elif 225 <= theta < 270:
            xe = 0
            ye = yc - np.tan((270-theta)*cte_grados_a_radianes)*(xc)

        elif 270 <= theta < 315:
            xe = 0
            ye = np.tan((theta-270) * cte_grados_a_radianes) * (xc) + yc

        elif 315 <= theta < 360:
            ye = M-1
            xe = xc - np.tan((360-theta) * cte_grados_a_radianes) * (ye - yc)

        else:
            raise 'Error'

        return (ye,xe)

class Interseccion(Point):
    def __init__(self,x,y,rayo_id,curva_id=-1):
        self.curva_id = curva_id
        self.rayo_id = rayo_id
        super().__init__([x,y])

    def __repr__(self):
        return (f'({self.y:.2f},{self.x:.2f},{self.curva_id},{self.rayo_id})\n')

    def __eq__(self, other):
        if other is None:
            return False
        return self.x == other.x and self.y == other.y

    def radio(self,centro):
        return distancia_entre_pixeles(self.y,self.x,centro[0],centro[1])


class Distribucion(Point):
    def __new__(self,x,y,curva_id,rayo_id,mean,std):
        self.mean = mean
        self.std = std
        self.curva_id = curva_id
        self.rayo_id = rayo_id
        self.delante = None
        self.atras = None

        super(self.__class__, self).__init__([x,y])

    def __repr__(self):
        return (f'({self.y:.2f},{self.x:.2f},{self.mean:.2f},{self.std:.2f})\n')


    def __eq__(self, other):
        if other is None:
            return False
        return self.x == other.x and self.y == other.y

    def radio(self,centro):
        return distancia_entre_pixeles(self.y,self.x,centro[0],centro[1])

class Segmento:
    AMARILLO = (23, 208, 253)
    BLANCO = (255, 255, 255)
    ROJO = (0, 0, 255)
    AZUL = (255, 0, 0)

    step = 360 / Nr
    def __init__(self,O:Point,P:Point,centro,identificador):
        self.centro = centro
        self.id = identificador
        self.m, self.n,self.a, self.b,self.dominio = self.generar_representacion_polar(O, P)


    def generar_representacion_polar(self,O,P):
        self.src = O
        self.dst = P

        y1 = distancia_entre_pixeles(self.centro[0], self.centro[1],self.src.y,self.src.x)
        x1 = self.src.rayo_id
        y2 = distancia_entre_pixeles(self.centro[0], self.centro[1], self.dst.y, self.dst.x)
        x2 = self.dst.rayo_id
        if x2 < x1:
            x2+=360

        dominio = np.arange(x1, x2 + Segmento.step, Segmento.step)
        m = (y2 - y1) / (x2 - x1)
        n = y1 - m * x1

        return m, n, x1, x2,dominio



    def get_radio(self,x):
        """
        @param x: vector o escalar en grados
        @return:  vector o escalar que represeanta el radio
        """
        x_normalizado = x % 360
        b_normalizado = self.b % 360
        a_normalizado = self.a % 360

        if a_normalizado <= np.min(x_normalizado) <= np.max(x_normalizado) <= b_normalizado:
            return self.m*x_normalizado+self.n
        if b_normalizado<a_normalizado:
            if b_normalizado<np.min(x_normalizado)<=np.max(x_normalizado)<a_normalizado:
                #angulo fuera de domino
                raise (f"angulo fuera del dominio del segmento")
                return -1

            if np.min(x_normalizado) <= np.max(x_normalizado) <=b_normalizado:
                return self.m*(x+360) + self.n
            else:
                return self.m * (x) + self.n

        raise "caso no considerado"

    @staticmethod
    def from_polar_to_cartesian(radios, angulos, centro):
        y = centro[0] + radios * np.cos(angulos * np.pi / 180)
        x = centro[1] + radios * np.sin(angulos * np.pi / 180)
        return (y, x)

    @staticmethod
    def _ordenar_intersecciones(init, bolsa_intersecciones):
        intersecciones_nuevo_orden = []
        idx_init = [idx for idx, _ in enumerate(bolsa_intersecciones) if _.rayo_id > init.rayo_id][0]
        intersecciones_nuevo_orden += bolsa_intersecciones[idx_init:]
        intersecciones_nuevo_orden += bolsa_intersecciones[:idx_init]
        return intersecciones_nuevo_orden

    def generar_curva_muestreada(self):
        radios = self.get_radio(self.dominio)
        yy, xx = from_polar_to_cartesian(radios, self.dominio, self.centro)
        muestras_cartesianas = [Interseccion(x=xx[pos], y=yy[pos], curva_id=-1,rayo_id=self.dominio[pos]%360) for pos in range(len(self.dominio))]
        muestras_cartesianas = muestras_cartesianas if ((self.a % 360) < (self.b % 360)) else self._ordenar_intersecciones(self.dst, muestras_cartesianas)

        segmento = LineString(coordinates=[(inter.x, inter.y) for inter in muestras_cartesianas])
        return segmento

    def dibujar(self,img_dibujo,color=None,texto=False):
        if color == None:
            color = Segmento.AMARILLO
        segmento = self.generar_curva_muestreada()
        img_dibujo = Dibujar.curva(segmento, img_dibujo,color= color)
        M,N,_ = img_dibujo.shape

        dst_y = M-1 if self.dst.y > M-1 else int(self.dst.y)
        dst_x = N-1 if self.dst.x > N-1 else int(self.dst.x)
        img_dibujo[dst_y,dst_x] = color
        src_y = M-1 if self.src.y > M-1 else int(self.src.y)
        src_x = N-1 if self.src.x > N-1 else int(self.src.x)
        img_dibujo[src_y, src_x] = color

        return img_dibujo


    def dominio_normalizado(self):
        dominio_normalizado = [angulo%360 for angulo in self.dominio]
        return dominio_normalizado
    def seleccionar_conjunto_de_puntos_de_curvas_cercana(self,intersecciones_sector):
        lista_intersecciones_vecinas = []
        distancias_minimas = []
        #2.0 buscar cadenas vecinas
        for idx,rayo_id in enumerate(self.dominio_normalizado()):
            radio_segmento = self.get_radio(rayo_id)
            lista_inter_direccion_radial =  [inter for inter in intersecciones_sector if inter.rayo_id == rayo_id]
            if len(lista_inter_direccion_radial)==0:
                continue
            distances = [np.abs(radio_segmento-inter.radio(self.centro)) for inter in lista_inter_direccion_radial]
            inter_curva_rayo = lista_inter_direccion_radial[np.argmin(distances)]

            distancias_minimas.append(np.min(distances))
            lista_intersecciones_vecinas.append(inter_curva_rayo)

        return lista_intersecciones_vecinas,distancias_minimas

