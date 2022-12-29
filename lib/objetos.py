import numpy as np
from shapely.geometry.point import Point
from shapely.geometry.linestring import LineString


from lib.celdas import distancia_entre_pixeles
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

