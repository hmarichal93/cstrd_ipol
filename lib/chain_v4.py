#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 16:48:15 2021

@author: henry
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt 
from copy import deepcopy
from sklearn.metrics.pairwise import euclidean_distances
from scipy.interpolate import CubicSpline
import logging 

#from dentro_utils.io import write_json,load_json, load_data, save_dots, Nr
import lib.utils

Nr = 360

class Punto:
    def __init__(self,**params):
        #self.x = np.uint32(params['x'])
        #self.y = np.uint32(params['y'])
        self.x = float(params['x'])
        self.y = float(params['y'])
        self.radio = params['radio']
        self.angulo = params['angulo'] if params['angulo']<Nr else 0 
        self.gradFase = params['gradFase']
        self.cadenaId = params['cadenaId']
        #self.gradModulo = params['gradmodulo']
        
    def __repr__(self):
        return (f'({self.x},{self.y}) ang:{self.angulo} radio:{self.radio:0.2f} cad.id {self.cadenaId}\n')

    def __str__(self):
        return (f'({self.x},{self.y}) ang:{self.angulo} radio:{self.radio:0.2f} id {self.cadenaId}')

    def __eq__(self,other):
        return self.x == other.x and self.y == other.y and self.angulo == other.angulo

def dict2matriz(puntos,centro):
    matriz = np.array([[-1,-1,-1,-1,-1]])
    
    for angle in puntos:
        for dot in puntos[angle]:
            #radio = np.sqrt((dot[0]-centro[1])**2+(dot[1]-centro[0])**2) 
            radio = getRadialFromCoordinates(dot[0],dot[1], centro)                           
            row = np.array([dot[0],dot[1],angle,radio,dot[2]],dtype=float)
            matriz = np.vstack((matriz,row))

    matriz = np.delete(matriz,0,axis=0)    
    
    return matriz

def convertirAlistaPuntos(perfiles,labels,centro):
    matriz = dict2matriz(perfiles,centro)
    listaPuntos = []
    for i,j,angulo,radio,fase in matriz:
        params={'x':np.uint16(i),'y':np.uint16(j),'angulo':angulo,'radio':radio,'gradFase':fase,'cadenaId':labels[np.uint16(i),np.uint16(j)]}
        punto = Punto(**params)
        if punto not in listaPuntos:
            listaPuntos.append(punto)
    return listaPuntos
def plotVector(origin,angle,c='k'):
    U,V = np.cos(angle),np.sin(angle)
    X,Y = origin[:,0],origin[:,1]
    #print(f" X {X.shape} Y {Y.shape} U {U.shape} V {V.shape}")
    plt.quiver(X,Y,U,V,color=c,scale=5,scale_units='inches',angles='xy',headwidth=1 ,headlength=3,width =0.005)
    #plt.quiver(*origin,vx,vy,angles='xy', scale_units='xy',units='inches')  


def extraerPixelesPertenecientesAlPerfil(angle,centro,M,N):
    """
        angulo =  {0,pi/4,pi/2,3pi/4,pi,5pi/4,6pi/4,7pi/4}
        ptosCard= {S, SE , E  , NE  , N, NW  , W   , SW   }
         | 
        ----------->x
         |
         | IMAGEN
         |
         y
         
    """
    i = 0
    y_pix =[]
    x_pix = []
    angle_rad = angle * np.pi / 180 
    ctrl = True      
    while ctrl:
        x = centro[1] + i*np.sin(angle_rad)
        y = centro[0] + i*np.cos(angle_rad)
        x = x.astype(int)
        y = y.astype(int)
         
        #print(f'y={y} x={x}')

        if i==0 or not (x==x_pix[-1] and y==y_pix[-1]):
            y_pix.append(y)
            x_pix.append(x)
        if y>=M-1 or y<=1 or x>=N-1 or x<= 1 :
            ctrl = False
        
        i +=1


    return np.array(y_pix),np.array(x_pix)
##interseccion rectas
## sistema lineal
##A1x+B1y=C1
##A2x+B2y=C2

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    #Regla de Cramer
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False
def buildMatrizEtiquetas(M, N, listaPuntos):
    MatrizEtiquetas = -1 * np.ones((M, N))
    for dot in listaPuntos:
        MatrizEtiquetas[int(dot.x), int(dot.y)] = dot.cadenaId
    return MatrizEtiquetas
def llenar_angulos(extA,extB):
    
    if extA.angulo<extB.angulo:
        rango = np.arange(extA.angulo,extB.angulo+1)
    else:
        rango1 = np.arange(extA.angulo,360)
        rango2 = np.arange(0,extB.angulo+1)
        rango = np.hstack((rango2,rango1))
    #print(rango)
    return rango.astype(int)




class Cadena:
    def __init__(self,cadenaId: int,centro,M,N,A_up=None,A_down=None,B_up=None,B_down=None,is_center = False):
        self.lista = []
        self.id = cadenaId
        self.label_id = cadenaId
        self.size = 0
        self.centro = centro
        self.M = M
        self.N = N
        self.A_up = A_up
        self.A_down = A_down
        self.B_up = B_up
        self.B_down = B_down
        self.is_center = is_center

    def get_borders_chain(self, extremo):
        if extremo in 'A':
            up_chain = self.A_up
            down_chain = self.A_down
            dot_border = self.extA
        else:
            up_chain = self.B_up
            down_chain = self.B_down
            dot_border = self.extB
        return down_chain, up_chain, dot_border

    def _completar_dominio_angular(self,cadena):
        if cadena is None:
            extA = self.extA.angulo
            extB = self.extB.angulo
        else:
            extA = cadena.extA.angulo
            extB = cadena.extB.angulo
        paso = 360/Nr
        if extA<= extB:
            dominio_angular = list(np.arange(extA,extB+paso,paso))
        else:
            dominio_angular = list(np.arange(extA,360,paso))
            dominio_angular+= list(np.arange(0,extB+paso,paso))

        return dominio_angular

    def posicion_relativa(self,o):
        up = 1
        down = 0
        angulos_cad = self._completar_dominio_angular(cadena=None)
        angulos_o = self._completar_dominio_angular(o)
        intersection = np.intersect1d(angulos_o,angulos_cad)
        if len(intersection)<1:
            return -1
        angulo_comun = intersection[0]
        dot_o = get_closest_chain_dot_to_angle(o,angulo_comun)
        dot_cad = get_closest_chain_dot_to_angle(self,angulo_comun)
        if dot_o.radio > dot_cad.radio:
            return up
        else:
            return down


    def __eq__(self,other):
        if other is None:
             return False

        return self.id == other.id and self.size == other.size #and counter == self.size
    
    def esta_completa(self,regiones=16):
        if self.size<2:
            return False
        dominio_angular = self._completar_dominio_angular(self)
        if len(dominio_angular)>= (regiones - 1)*360/regiones:
            return True
        else:
            return False

        angulos = llenar_angulos(self.extA,self.extB)
        #print(angulos)
        #angulos = cadena.getDotsAngles()
        step = 360/regiones
        bins = np.arange(0,360,step)
        hist,_ = np.histogram(angulos,bins)
        empties = np.where(hist==0)[0]
        #print(empties)
        if len(empties)==0:
            return True
        else:
            return False

    def sort_dots(self,sentido='horario'):
        if sentido in 'horario':
            return self.puntos_ordenados_horario
        else:
            return self.puntos_ordenados_horario[::-1]

    def _sort_dots(self,sentido='horario'):
        puntos_horario = []
        if sentido in 'horario':
            angle_k = self.extB.angulo
            k = 0
            while len(puntos_horario) < self.size:
                try:
                    dot = self.getDotByAngle(angle_k)[0]
                    dot.cadenaId = self.id
                    puntos_horario.append(dot)
                except:
                    pass
                    #print(angle_k)
                    #continue
                    
                angle_k = (angle_k-1) % 360
                k+=1
        
        else:
            
            angle_k = self.extA.angulo
            k = 0
            while len(puntos_horario) < self.size:
                try:
                    dot = self.getDotByAngle(angle_k)[0]
                    dot.cadenaId = self.id
                    puntos_horario.append(dot)
                except:
                    pass
                    #print(angle_k)
                    #continue
                angle_k = (angle_k+1) % 360
                k+=1
                
        return puntos_horario


    def __repr__(self):
        return (f'(id_l:{self.label_id},id:{self.id}, size {self.size}')





    def __encontrarExtremos(self):
        diff = np.zeros(self.size)
        #lista tiene que estar ordenada en orden creciente
        self.lista.sort(key=lambda x: x.angulo, reverse=False)
        diff[0] = (self.lista[0].angulo+Nr-self.lista[-1].angulo) % Nr

        for i in range(1,self.size):
            diff[i] = (self.lista[i].angulo-self.lista[i-1].angulo) #% 2*np.pi

        if self.size>1:
            extremo1 = diff.argmax()
            if extremo1 == 0:
                #caso1: intervalo conectado
                extremo2 = diff.shape[0]-1
            else:
                #caso2: intervalo partido a la mitad
                extremo2 = extremo1-1

        else:
            extremo1 = extremo2 = 0
        self.extAind = extremo1
        self.extBind = extremo2

        self.extA = self.lista[extremo1]
        self.extB = self.lista[extremo2]

    def add_lista_puntos(self,lista_puntos):
        assert len([punto for punto in lista_puntos if punto.cadenaId != self.id]) ==  0
        self.lista += lista_puntos
        self.update()

    def add_punto(self, punto: Punto):
        se_pego = False
        if punto.cadenaId == self.id:
            if punto not in self.lista:
                self.lista.append(punto)
                se_pego = True
            else:
                raise
        else:
            raise
        return se_pego

            
    def update(self):
        self.size = len(self.lista)
        if self.size>1:
            self.__encontrarExtremos()
            self.puntos_ordenados_horario = self._sort_dots(sentido='horario')
        else:
            raise

    
    def completar_cadena(self):
        ordenados = self.sort_dots(sentido='antihorario')
        angulo_minimo = ordenados[0].angulo
        x = [(dot.angulo-angulo_minimo) % 360 for dot in ordenados]
        y = [dot.radio for dot in ordenados]
        # 3.0 Calcular splines
        #y tiene que ser periodica. Correcciones para la funcion CubicSpline con bc_type periodic. 
        cs = CubicSpline(x, y)
        missingAngles = np.arange(x[-1]+360/Nr,360)
        ySpline = cs(missingAngles)
        
        missingAngles = [(angle+angulo_minimo) % 360 for angle in missingAngles]
        
        return ySpline,missingAngles
        
                  
    def getDotsCoordinates(self):
         x = [dot.x for dot in self.lista]
         y = [dot.y for dot in self.lista]
         x_rot = np.roll(x,-self.extAind)
         y_rot = np.roll(y,-self.extAind)
         return x_rot,y_rot
     
    def getMissingAnglesInterval(self,ext1,ext2):
        intervalo = np.arange(ext1,ext2+1,1)
        intervalo_cad = [dot.angulo for dot in self.lista if dot.angulo>=ext1 and dot.angulo<=ext2 ]
        intervalo_missed = [angulo for angulo in intervalo if angulo not in intervalo_cad ]
        return intervalo_missed    
    
    def getMissingAngles(self):
        if self.extA.angulo>self.extB.angulo:
            intervalo0 = self.getMissingAnglesInterval(0, self.extB.angulo)
            intervalo1 = self.getMissingAnglesInterval(self.extA.angulo,Nr-2)
            anglesMissed = intervalo0+intervalo1
        else:
            anglesMissed = self.getMissingAnglesInterval(self.extA.angulo, self.extB.angulo)

        return anglesMissed
    
    def getDotsAngles(self):
        angles = [dot.angulo for dot in self.lista]
        angles = np.array(angles,dtype=np.int16)
        angles = np.where(angles==360,0,angles)
        return angles

    def getDotByAngle(self,angulo):
        dots = [dot for dot in self.lista if dot.angulo==angulo]
        return list(dots)
    
    def changeId(self,index):
        for dot in self.lista:
            dot.cadenaId = index
        self.id = index
    
    def pop_dot(self,dot):
        if dot in self.lista:
            dot_idx = self.lista.index(dot)
            self.lista.pop(dot_idx)
            self.update()
            
    def formato_array(self):
        x1,y1 = self.getDotsCoordinates()
        puntos1 = np.vstack((x1,y1)).T
        
        #print(puntos1.shape)
        c1a = np.array([self.extA.x,self.extA.y],dtype=float)
        c1b = np.array([self.extB.x,self.extB.y],dtype=float)
        return puntos1.astype(float),c1a,c1b

    def distancia_media_entre_puntos_cadena(self):
        puntos,_,_ = self.formato_array()
        distances = euclidean_distances(puntos, puntos)
        rango = np.arange(0,distances.shape[0])
        maxima=0
        for dot in range(distances.shape[0]):
            rango_clean = np.delete(rango,dot)
            media_dot = distances[dot,rango_clean].mean()
            if media_dot>maxima:
                maxima = media_dot
    
        return maxima    

    def distribucion_angulos(self):
        x,y = [],[]
        for dot in self.sort_dots():
            x.append(dot.angulo)
            y.append(dot.gradFase% 2*np.pi)
        plt.figure()
        plt.plot(y)
        plt.title('Distribucion angulos gradiente')




def get_closest_chain_dot_to_angle(chain,angle):
    label='get_closest_chain_dot_to_angle'
    chain_dots = chain.sort_dots(sentido='antihorario')
    A = chain.extA.angulo
    B = chain.extB.angulo
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
def intersecciones_cadenas(cad,equivalencias,listaCadenas):
    tabla  = np.zeros(Nr)
    cadena = getChain(cad, listaCadenas)
    dominioAngular = cadena.getDotsAngles()
    for angulo in dominioAngular:
        tabla[angulo] = 1
    
    for other_cad_id in equivalencias:
        other_cad = getChain(other_cad_id, listaCadenas)
        dominioAngular = other_cad.getDotsAngles()
        pos_intersect = np.where(tabla[dominioAngular]>0)[0]
        if pos_intersect.shape[0]>0:
            return True
    
    return False

def existeCadena(listaCadenas,cadenaId):
    for index,c in enumerate(listaCadenas):
        if cadenaId == c.id:
            return index
    return -1
def buildMatrizEtiquetas(M, N, listaPuntos):
    MatrizEtiquetas = -1 * np.ones((M, N))
    for dot in listaPuntos:
        MatrizEtiquetas[np.floor(dot.x).astype(int), np.floor(dot.y).astype(int)] = dot.cadenaId
    return MatrizEtiquetas

def verificacion_complitud(listaCadenas):
    for cadena in listaCadenas:
        dominio_angular = cadena._completar_dominio_angular(cadena)
        if not (len(dominio_angular) == cadena.size):
            print(f"cad.id {cadena.label_id} size {cadena.size} dominio_angular {len(dominio_angular)} ")
            raise


def copiar_cadena(cadena):
    cadena_aux = Cadena(cadena.id, cadena.centro, cadena.M, cadena.N)
    lista_cadena_aux = [Punto(**{'x':punto.x,'y':punto.y,'angulo':punto.angulo,'radio':punto.radio,'gradFase':punto.gradFase,'cadenaId':cadena.id})
                    for punto in cadena.lista]
    cadena_aux.lista = lista_cadena_aux
    cadena_aux.extB = cadena.extB
    cadena_aux.extA = cadena.extA
    cadena_aux.size = cadena.size
    #cadena_aux.add_lista_puntos(lista_cadena_aux)
    assert cadena_aux.size == cadena.size
    return cadena_aux

def asignarCadenas(listaPuntos,centro,M,N,centro_id=None):
    listaCadenas= []
    cadenas_ids = set([punto.cadenaId for punto in listaPuntos])
    for cad_id in cadenas_ids:
        puntos_cadena = [punto for punto in listaPuntos if punto.cadenaId == cad_id]
        if len(puntos_cadena) <= 1:
            for punto in puntos_cadena:
                listaPuntos.remove(punto)
            continue
        if centro_id is not None and cad_id == centro_id:
            is_center = True
        else:
            is_center = False
        cadena = Cadena(cad_id, centro, M, N, is_center=is_center)
        cadena.add_lista_puntos(puntos_cadena)
        listaCadenas.append(cadena)

    return listaCadenas

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    #labeled_imS = cv2.resize(labeled_img, (960, 540))
    cv2.imshow('labeled.png', labeled_img)
    cv2.waitKey()
    
def visualizarCadenas(listaCadenas,img_b):
    matriz = np.zeros_like(img_b,dtype=np.uint16)
    for cadena in listaCadenas:
        for punto in cadena.lista:
            matriz[punto.x,punto.y] = cadena.id
    imshow_components(matriz)

def distancia_entre_puntos(d1,d2):
    v1 = np.array([d1.x,d1.y],dtype=float)
    v2 = np.array([d2.x,d2.y],dtype=float)
    
    return np.sqrt((v1[0]-v2[0])**2+(v1[1]-v2[1])**2)

def formato_array(c1):
    x1,y1 = c1.getDotsCoordinates()
    puntos1 = np.vstack((x1,y1)).T
    
    #print(puntos1.shape)
    c1a = np.array([c1.extA.x,c1.extA.y],dtype=float)
    c1b = np.array([c1.extB.x,c1.extB.y],dtype=float)
    return puntos1.astype(float),c1a,c1b
from sklearn.metrics.pairwise import euclidean_distances
def distancia_modal_entre_puntos_cadena(c1):
    puntos,_,_ = formato_array(c1)
    distances = euclidean_distances(puntos, puntos)
    rango = np.arange(0,distances.shape[0])
    maxima=0
    for dot in range(distances.shape[0]):
        rango_clean = np.delete(rango,dot)
        media_dot = distances[dot,rango_clean].mean()
        if media_dot>maxima:
            maxima = media_dot

    return maxima

def dist_extremo(ext,matriz):
    distances = np.sqrt(np.sum((matriz-ext)**2,axis=1))
    return np.min(distances)

def distancia_minima_entre_cadenas(c1,c2):
    puntos1,c1a,c1b = formato_array(c1)
    puntos2,c2a,c2b = formato_array(c2)
    c2a_min = dist_extremo(puntos1,c2a)
    c2b_min = dist_extremo(puntos1,c2b)
    c1a_min = dist_extremo(puntos2,c1a)
    #print(c1b)
    #print(puntos2.astype(float)-c1b.astype(float))
    c1b_min = dist_extremo(puntos2,c1b)
    #print([c2a_min,c2b_min,c1a_min,c1b_min])
    return np.min([c2a_min,c2b_min,c1a_min,c1b_min])

def visualizarCadenasSobreDisco(listaCadenas,img,titulo,labels = False,flechas=False,color=None,hist=None,save=None, gris = False, display=False):
    cadenasSize = []
    figsize = (30,15)
    #figsize=(10,10)
    if gris:
        plt.figure(figsize=figsize)
        imageGray = utils.rgbToluminance(img)
        plt.imshow(imageGray,cmap='gray')
        contador = 0
        for cadena in listaCadenas:
            x,y = cadena.getDotsCoordinates()
            #axs[0].plot(y,x,'-bo', markersize=2,linewidth=1)
            if cadena.esta_completa():
                plt.plot(y,x,'b',linewidth=1)
                contador +=1
            else:
                plt.plot(y,x,'r',linewidth=1)
            if labels:
                plt.annotate(str(cadena.label_id), (y[0], x[0]),c='b')

            if cadena.is_center:
                plt.scatter(y, x,s=2, zorder=10,c='r')

            cadenasSize.append(cadena.size)
        #plt.title(titulo)
        plt.axis('off')
        if save:
            plt.savefig(f"{titulo}")
        
    else:
        plt.figure(figsize=figsize)
        plt.imshow(img)
        for cadena in listaCadenas:
                if cadena.size==0:
                    continue
                x,y = cadena.getDotsCoordinates()
                #axs[0].plot(y,x,'-bo', markersize=2,linewidth=1)
                if cadena.is_center:
                    plt.scatter(y, x,s=2, zorder=10,c='r')
                if not color:
                    if cadena.size > 2:
                        plt.plot(y,x,linewidth=2)
                    else:
                        plt.scatter(y,x,s=2, zorder=10)                        
                else:
                    if cadena.size > 2:
                        plt.plot(y,x,'r',linewidth=2)
                    else:
                        plt.scatter(y,x,s=2, zorder=10,c='r')                        

                if labels:
                    plt.annotate(str(cadena.label_id), (y[0], x[0]),c='b')
                cadenasSize.append(cadena.size)
        #plt.title(titulo)
        plt.axis('off')
        if save:
            print(f"{save}/{titulo}.png")
            plt.savefig(f"{save}/{titulo}.png")

    plt.savefig(f"{titulo}")
    if display: 
        plt.show()
    else:
        plt.close()

from lib.dibujar import Dibujar, Color


def dibujar_cadenas_en_imagen(listaCadenas, img, color=None,labels=False):
    M, N, _ = img.shape
    colors_length = 20
    np.random.seed(10)
    #colors = np.random.randint(low=100, high=255, size=(colors_length, 3), dtype=np.uint8)
    colors = Color()
    color_idx = 0
    for cadena in listaCadenas:
        y, x = cadena.getDotsCoordinates()

        pts = np.vstack((x, y)).T.astype(int)
        isClosed = False
        thickness = 5
        if color is None:
            #b,g,r = 0,255,0
            b,g,r = Color.green
        else:
            #b, g, r = colors[color_idx]
            b, g, r = colors.get_next_color()
            #r=255
        img = cv2.polylines(img, [pts],
                            isClosed, (int(b), int(g), int(r)), thickness)
        color_idx = (color_idx + 1) % colors_length


    if labels:
        for cadena in listaCadenas:
            org = cadena.extA
            img = Dibujar.put_text(str(cadena.label_id), img, (int(org.y), int(org.x)),fontScale=1.5)



    return img

def visualizarCadenasSobreDiscoTodas(listaCadenas, img,lista_cadenas_todas, titulo, labels=False, flechas=False, color=None, hist=None,
                                save=None, gris=False, display=False):

    #img_curvas = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
    img_curvas = np.zeros_like(img)
    for idx in range(3):
        img_curvas[:,:,idx] = img[:,:,1].copy()
    img_curvas = dibujar_cadenas_en_imagen(lista_cadenas_todas, img_curvas, color = color)
    img_curvas = dibujar_cadenas_en_imagen(listaCadenas,img_curvas,labels=True,color= True)


    print(f"{save}/{titulo}.png")
    cv2.imwrite(f"{save}/{titulo}.png",img_curvas)
    return

def visualizarCadenasSobreDiscoTodas_old(listaCadenas, img,lista_cadenas_todas, titulo, labels=False, flechas=False, color=None, hist=None,
                                save=None, gris=False, display=False):
    cadenasSize = []
    figsize = (10,7)
    if hist:
        fig, axs = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [3, 1]})
        axs[0].imshow(img, cmap='gray')
        for cadena in listaCadenas:
            x, y = cadena.getDotsCoordinates()
            # axs[0].plot(y,x,'-bo', markersize=2,linewidth=1)
            if not color:
                axs[0].plot(y, x, linewidth=2)
            else:
                axs[0].plot(y, x, '.r', linewidth=2)
            if labels:
                axs[0].annotate(str(cadena.label_id), (y[0], x[0]), c='b')

            cadenasSize.append(cadena.size)
        axs[0].set_title(titulo)
        axs[0].axis('off')

        axs[1].hist(cadenasSize)
        axs[1].set_title(f'Histograma tamaño cadenas. \n Cantidad cadenas {len(listaCadenas)}')
        axs[1].grid(True)

    elif gris:
        plt.figure(figsize=figsize)
        plt.imshow(img[:, :, 1], cmap='gray')
        contador = 0
        for cadena in listaCadenas:
            x, y = cadena.getDotsCoordinates()
            # axs[0].plot(y,x,'-bo', markersize=2,linewidth=1)
            if cadena.esta_completa():
                plt.plot(y, x, 'b', linewidth=1)
                contador += 1
            else:
                plt.plot(y, x, 'r', linewidth=1)
            if labels:
                plt.annotate(str(cadena.label_id), (y[0], x[0]), c='b')

            cadenasSize.append(cadena.size)
        plt.title(titulo)
        plt.axis('off')
        if save:
            plt.savefig(f"{save}/{titulo}.png")

    else:
        plt.figure(figsize=figsize)
        plt.imshow(img)
        for cadena in lista_cadenas_todas:
            if cadena.size == 0:
                continue
            x, y = cadena.getDotsCoordinates()
            if cadena.size > 2:
                plt.plot(y, x, 'w', linewidth=2)
            else:
                plt.scatter(y, x, s=2, zorder=10, c='w')

        for cadena in listaCadenas:
            if cadena.size == 0:
                continue
            x, y = cadena.getDotsCoordinates()
            # axs[0].plot(y,x,'-bo', markersize=2,linewidth=1)
            if not color:
                if cadena.size > 2:
                    plt.plot(y, x, linewidth=2)
                else:
                    plt.scatter(y, x, s=2, zorder=10)
            else:
                if cadena.size > 2:
                    plt.plot(y, x, 'r', linewidth=2)
                else:
                    plt.scatter(y, x, s=2, zorder=10, c='r')

            if labels:
                plt.annotate(str(cadena.label_id), (y[0], x[0]), c='b')
            cadenasSize.append(cadena.size)
        plt.title(titulo)
        plt.axis('off')
        if save:
            plt.savefig(f"{save}/{titulo}.png")

    if display:
        plt.show()
    else:
        plt.close()


def matrix2Puntos(Y,cadId):
    puntos = []
    for x,y,angulo,radio,fase in Y:
        params={'x':x,'y':y,'angulo':angulo,'radio':radio,'gradFase':fase,'cadenaId':0}
        dot = Punto(**params)
        puntos.append(dot)
    return puntos

def saveCadena2txt(cadena,file="test_chain_cadena3.csv"):
    X = np.array([])
    for dot in cadena.lista:
            coord = np.array([dot.x,dot.y,dot.angulo,dot.radio,dot.gradFase])
            X = np.vstack((coord,X)) if X.shape[0]>0 else coord
    
    X = X.reshape((-1,5))
    np.savetxt(file, X, delimiter=",")

def getChain(chainId,lista):
    for x in lista:
        if x.id == chainId:
            return deepcopy(x)
    return None

def popChain(chainId,lista):
    for index,x in enumerate(lista):
        if x.id == chainId:
            return lista.pop(index)    
def visualizarCadenaSobreDisco(cadenaId,lista,img):
    cadena = getChain(cadenaId,lista)        
    visualizarCadenasSobreDisco([cadena],img,titulo=None,flechas=True)
    
    for dot in cadena.lista:
        fase = dot.gradFase
        x,y = dot.x,dot.y
        plotVector([x,y],[fase])

def get_angle_between_dots(dot_1,dot_2):
    """
    xxx_225_xxx
    270_cen_090
    xxx_000_xxx
    params: 
        - dot_1: is the center.
        - dot_2: is the external point
    """
    centro = [dot_1.x,dot_1.y]
    i = dot_2.x
    j = dot_2.y
    return getAngleFromCoordinates(i,j,centro)

def getAngleFromCoordinates(i,j,centro):
    centro = np.array([float(centro[0]),float(centro[1])])
    vector = np.array([float(i),float(j)])-centro
    radAngle = np.arctan2(vector[1],vector[0])
    radAngle = radAngle if radAngle>0 else radAngle+2*np.pi
    gradAngle = np.round(radAngle*180/np.pi) % 360
    return gradAngle


def test_getAngleFromCoordinates():
    centro = [10,10]
    i = [0,10,20,0,10,20,0,10,20]
    j = [0,0,0,10,10,10,20,20,20]
    for i,j in zip(i,j):
        angle = getAngleFromCoordinates(i,j,centro[::-1])
        print(f"(i,j)=({i},{j}) angle {angle}")
        

def getRadialFromCoordinates(x_pos,y_pos,centro):
    return np.sqrt((x_pos-centro[1])**2+(y_pos-centro[0])**2)

def checkVecindad(MatrizEtiquetas, x, y, cad, ancho=10):
    W = MatrizEtiquetas[x - ancho : x + ancho + 1, y - ancho : y + ancho + 1]
    print(W)
    #plt.figure(figsize=(10, 10))
    #plt.imshow(W)
    #plt.scatter(y_d,x_d,c='r')
    unicos = list(np.unique(W))
    if -1 in unicos:
        unicos.remove(-1)
    return unicos, W


def visualizarMatrizVecindadSobreDisco(
    cadena, extremo, MatrizEtiquetas, nonMaxSup, ancho, gradFase,img,listaCadenas
):
    ext = cadena.extA if extremo in "A" else cadena.extB
    vecinos,W_etiquetas = checkVecindad(MatrizEtiquetas, ext.x, ext.y,cadena,ancho=ancho)
    _, W_max = checkVecindad(nonMaxSup, ext.x, ext.y, cadena, ancho=ancho)
    _, W_img = checkVecindad(img, ext.x, ext.y, cadena, ancho=ancho)
    _, W_fase = checkVecindad(gradFase*180/np.pi, ext.x, ext.y, cadena, ancho=ancho)

    fig, ax = plt.subplots(2, 2)
    fig.suptitle('Matrices relevantes')
    ax[0,0].imshow(W_etiquetas)
    ax[0,0].axis('off')
    ax[0,0].set_title("Etiquetas")
    ax[0,1].imshow(W_max)
    ax[0,1].axis('off')
    ax[0,1].set_title("Maximos")
    ax[1,0].imshow(W_img)
    ax[1,0].axis('off')
    ax[1,0].set_title("Img")
    cadenas = [getChain(cadena_id, listaCadenas) for cadena_id in vecinos]
    for cad in cadenas:
        lista_puntos = cad.lista
        sub_sample = [dot for dot in lista_puntos if (ext.x + ancho + 1 >= dot.x >= ext.x - ancho) and (ext.y + ancho + 1 >= dot.y >= ext.y- ancho)]
        x = [ dot.x - (ext.x - ancho) for dot in sub_sample]
        y = [ dot.y - (ext.y - ancho)  for dot in sub_sample]
        ax[1,0].plot(y,x)
        ax[1,0].annotate(str(cad.id), (y[0], x[0]))
    
    
    
    ax[1,1].imshow(W_fase)
    ax[1,1].axis('off')
    ax[1,1].set_title("Fase")
    
    
    visualizarCadenasSobreDisco(
        cadenas, img, f"visualizar vecinos ancho: {ancho}-extremo {extremo}", labels=True
    )


def check_total_dots(listaCadenas, debug=False):
    ####contar cantidad de puntos
    contador = 0
    for chain_check in listaCadenas:
        contador += chain_check.size
    #if debug:
    #    print(f"Se tienen un total de  {contador} puntos")
    return contador


def renombrarCadenas(listaCadenas, listaPuntos, M,N):
    check_total_dots(listaCadenas, debug=True)
    listaCadenas = sorted(listaCadenas, key=lambda x: x.size, reverse=True)
    for index, chain_fill in enumerate(listaCadenas):
        chain_fill.changeId(index)
    
    MatrizEtiquetas = buildMatrizEtiquetas(M, N, listaPuntos)
    return listaCadenas,listaPuntos, MatrizEtiquetas

def contarPuntosListaCadenas(listaCadenas):
    contadorPuntos = 0
    for cadena in listaCadenas:
        contadorPuntos += cadena.size
    return contadorPuntos


def checkListas(listaCadenas, listaPuntos):
    for cadena in listaCadenas:
        dots = [dot for dot in listaPuntos if dot.cadenaId == cadena.id]
        print(f"cadenaId {cadena.id} size {cadena.size} listaPuntos {len(dots)}")



def checkMatrizEtiquetasCadenas(listaCadenas, MatrizEtiquetas):
    label='checkMatrizEtiquetasCadenas'
    for cadena in listaCadenas:
        len_etiquetas = np.where(MatrizEtiquetas==cadena.id)[0].shape[0]
        if cadena.size != len_etiquetas:
            print(
                f"{label} cadena.id {cadena.id} cadenaSize {cadena.size} MatrizEtiquetas {len_etiquetas}"
            )


def interpolacion_lineal(nueva_cadena,listaPuntos,gradFase,centro,cadenas_lista=False,completar=False):
    ### se probo bastante.
    #centro = nueva_cadena.centro[::-1]
    logging.info(f"[interpolacion_lineal] cad.id {nueva_cadena} completar {completar} lenListaPuntos {len(listaPuntos)} centro {centro} N {nueva_cadena.N} M {nueva_cadena.M}")
    #print(f"Empezando. nueva_c {nueva_cadena.id}")
    puntos_ord = nueva_cadena.sort_dots(sentido='horario')
    #print(f"Empezando. Salid de ordenar. nueva_c {nueva_cadena.id}")
    if nueva_cadena.extA.angulo>nueva_cadena.extB.angulo:
        shift = nueva_cadena.extB.angulo + 360/Nr
    else:
        shift = 0
    intervalos = []
    if not completar:
        for index_dot in range(1,len(puntos_ord)):
            diff = puntos_ord[index_dot-1].angulo - puntos_ord[index_dot].angulo
            if diff > 360/Nr:
                intervalos.append([(puntos_ord[index_dot].angulo-shift) % 360,getRadialFromCoordinates(puntos_ord[index_dot].x,puntos_ord[index_dot].y, centro)\
                                   ,(puntos_ord[index_dot-1].angulo-shift) % 360,getRadialFromCoordinates(puntos_ord[index_dot-1].x,puntos_ord[index_dot-1].y, centro)])
    else:
        shift = nueva_cadena.extA.angulo + 360/Nr
        intervalos.append([(nueva_cadena.extB.angulo-shift) % 360,getRadialFromCoordinates(nueva_cadena.extB.x,nueva_cadena.extB.y, centro)\
                           ,(nueva_cadena.extA.angulo-shift) % 360,getRadialFromCoordinates(nueva_cadena.extA.x,nueva_cadena.extA.y, centro)])    
    #los intervalos son abiertos
    #print(f"Aun no llegue. nueva_c {nueva_cadena.id}")
    angulos, radios = [],[]
    for inter in intervalos:
        #print(f"inter {inter}")
        a,b = inter[0],inter[2]
        ya,yb = inter[1],inter[3]
        if b-a<=0:
            continue
        m,n = (yb-ya) / (b-a), ya - (yb-ya) * a / ( b - a )
        func = lambda x: x*m + n
        x = np.arange(a+360/Nr,b)
        #print(x)
        y = func(x)
        angulos += list(x)
        radios += list(y)
        
    
    #print(f"Estoy ACA nueva_c {nueva_cadena.id}")
    nuevos_angulos_agregados = []
    nuevos_puntos_agregados = []
    for a,r in zip(angulos,radios):
        angle_rad = ((a+shift)%360)*np.pi/180
        x = centro[1] + r * np.cos(angle_rad)
        y = centro[0] + r * np.sin(angle_rad)
        x = np.int(x)
        y = np.int(y)
        if y>= nueva_cadena.N or x>= nueva_cadena.M:
            continue
        #print(f"cad_id {nueva_cadena.id} x {x} y {y} radio {r} angulo {a}")
        try:
            params = {
                "x":x ,
                "y": y,
                "angulo": (a+shift)%360,
                "radio": r,
                "gradFase": gradFase[x,y],
                "cadenaId": nueva_cadena.id,
            }
            #logging.info(f"[interpolacion_lineal] cad.id {nueva_cadena} nuevo dot x,y,angle,radio {x},{y},{a},{r}")
        except Exception as e:
            print(e)
            raise
        punto = Punto(**params)
        if punto.y>= nueva_cadena.N or punto.x>= nueva_cadena.M:
            continue

        if punto not in listaPuntos:
            listaPuntos.append(punto)
            nuevos_puntos_agregados.append(punto)
            nuevos_angulos_agregados.append(punto.angulo)

    nueva_cadena.add_lista_puntos(nuevos_puntos_agregados)
    MatrizEtiquetas = buildMatrizEtiquetas(nueva_cadena.M, nueva_cadena.N, listaPuntos)
    logging.info(f"[interpolacion_lineal] cad.id {nueva_cadena} completar {completar} lenListaPuntos {len(listaPuntos)}")
    #print(f"Me voy  {nueva_cadena.id}")
    return nueva_cadena,listaPuntos, MatrizEtiquetas, nuevos_angulos_agregados

def convertir_a_puntos(datos):
    M,N,bin_sin_pliegos,centro,SAVE_PATH,perfiles = datos['M'], datos['N'], datos['bin_sin_pliegos'], datos['centro'], datos['save_path'] , datos['perfiles']

    num_labels, labels_im = cv2.connectedComponents(bin_sin_pliegos, connectivity=8)

    listaPuntos = convertirAlistaPuntos(perfiles, labels_im, centro)
    MatrizEtiquetas = -1 + np.zeros_like(labels_im)
    for dot in listaPuntos:
        MatrizEtiquetas[dot.x, dot.y] = dot.cadenaId
        
        
    listaCadenas = asignarCadenas(listaPuntos, centro[::-1], M, N)
    datos['listaPuntos'] = listaPuntos
    datos['listaCadenas'] = listaCadenas
    datos['MatrizEtiquetas'] = MatrizEtiquetas
    datos['labels_im'] = labels_im
    return 0

def rellenar_cadenas(datos):
    listaCadenas, listaPuntos, gradFase, M,N,centro = datos['listaCadenas'], datos['listaPuntos'], datos['gradFase'], datos['M'], datos['N'], datos['centro']
    MatrizEtiquetas = datos['MatrizEtiquetas']
    #print("Step 2.4: Rellenar Cadenas")
    # TODO: ver si es posible eliminar este apso. A futuro.
    #print(f"Inicialmente se tienen {len(listaPuntos)} puntos")
    #Todas las cadenas no pueden tener huecos.
    for index,cadena in enumerate(listaCadenas):
        if cadena.size<2:
            continue
        cadena,listaPuntos,MatrizEtiquetas,_ = interpolacion_lineal(cadena,listaPuntos,gradFase,centro,cadenas_lista=listaCadenas)
            
    #listaCadenas,listaPuntos = removeCadenas(listaCadenas, listaPuntos, size_del=1)
    listaCadenas,listaPuntos, MatrizEtiquetas = renombrarCadenas(listaCadenas, listaPuntos, M,N)
    #checkListas(listaCadenas, listaPuntos)
    #MatrizEtiquetas = buildMatrizEtiquetas(M, N, listaPuntos)
    datos['MatrizEtiquetas'] = MatrizEtiquetas 
    
    plt.figure(figsize=(15, 15))
    display, bin_sin_pliegos, perfiles, SAVE_PATH = datos['display'], datos['bin_sin_pliegos'], datos['perfiles'], datos['save_path']
    plt.imshow(bin_sin_pliegos, cmap="gray")
    for key in perfiles.keys():
        dots = perfiles[key]
        plt.scatter(dots[:, 1], dots[:, 0], s=4, zorder=10, c='r')

    plt.title("Puntos")
    plt.axis('off')
    plt.savefig(f"{SAVE_PATH}/puntos.png")
    if display:
        plt.show()
        
    plt.close()


    datos['MatrizEtiquetas'] = MatrizEtiquetas
    datos['listaPuntos'] = listaPuntos
    datos['listaCadenas'] = listaCadenas
    return 0
######################################################################################################################################
######################################################################################################################################
def transformar_a_cadenas(results):
    convertir_a_puntos(results)
    
    ### imprimir resultados
    listaCadenas, listaPuntos, MatrizEtiquetas= results['listaCadenas'], results['listaPuntos'], results['MatrizEtiquetas']
    print(
        f"largoCadenas {len(listaCadenas)} listaPuntos {len(listaPuntos)} MatrizEtiquetas {np.where(MatrizEtiquetas>-1)[0].shape} cadenas {contarPuntosListaCadenas(listaCadenas)}"
    )
    rellenar_cadenas(results)
    ################
    listaCadenas, img, SAVE_PATH, MatrizEtiquetas = results['listaCadenas'],results['img'], results['save_path'], results['MatrizEtiquetas']
    
    
    ### imprimir resultados
    visualizarCadenasSobreDisco(
        listaCadenas, img, "cadenas", labels=False,color='r',save=SAVE_PATH
    )    
    visualizarCadenasSobreDisco(
        listaCadenas, img, "cadenas_color", labels=True,save=SAVE_PATH
    )
    print(
        f"largoCadenas {len(listaCadenas)} listaPuntos {len(listaPuntos)} MatrizEtiquetas {np.where(MatrizEtiquetas>-1)[0].shape} cadenas {contarPuntosListaCadenas(listaCadenas)}"
    )
    return 0

def from_matrix_to_dot_list(matrix,center):
    unique_ids = np.unique(matrix)
    dots_list = []
    for chain_id in unique_ids:
        if chain_id<0:
            continue
        y,x = np.where(matrix==chain_id)
        for i,j in zip(y,x):
            params = {
                "x":i ,
                "y": j,
                "angulo": getAngleFromCoordinates(i,j,center),
                "radio": getRadialFromCoordinates(i, j, center[::-1]),
                "gradFase": -1,
                "cadenaId": chain_id,
            }
            dots_list.append(Punto(**params))

    return dots_list
        
    