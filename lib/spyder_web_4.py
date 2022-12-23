import numpy as np
import cv2
from shapely.geometry import LineString, Point,Polygon
from pathlib import Path
import os
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


from lib.devernayEdgeDetector import devernayEdgeDetector
from lib.io import Nr
from lib.celdas import Celda,distancia_entre_pixeles,OperacionesCelda,AMARILLO,VERDE,ROJO,NEGRO,NARANJA
from lib.dibujar import Dibujar
from lib.utils import write_log
from lib.objetos import Distribucion,Interseccion,Rayo,Curva,Segmento
import lib.chain_v4 as ch

MODULE = "spyder"
def from_polar_to_cartesian(r,angulo,centro):
    y = centro[0] + r * np.cos(angulo * np.pi / 180)
    x = centro[1] + r * np.sin(angulo * np.pi / 180)
    return (y,x)



class SpyderWeb:
    #TODO: pensar los objetos abstrayendome de la representacion.
    #Los objetos (pixeles) tienen que ser flotantes. Definir de una biblioteca que se encarga de dibujar la representacion
    #Rayos: tienen que ser objetos independientementes de la representacion. El radio tiene que tener como objeto unicamente
    # las coordenadas del centro y un angulo
    #Curva: sucesion de puntos
    #Anillo: curva cerrada, los puntos pertenecen a radios distintos.
    #Interseccion: interseccion entre rayo y curva
    #Celda: espacio entre dos curvas consecutivas (en direccion radial) y rayos consecutivos. Los rayos dependiendo de la distancia
    # al centro podran ser consecutivos o no. Una celda esta formada por 4 objetos Interseccion.
    def __init__(self,Nr,img,lista_curvas,centro,save_path,debug=False):
        self.Nr = Nr
        self.contador = 0
        self.path = save_path
        if debug:
            self.debug_path = Path(save_path) / "debug"
            os.system(f"rm -rf {str(self.debug_path)}")
            self.debug_path.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        self.centro = centro
        self.img = img
        M,N,_ = img.shape
        self.lista_curvas = self.convertir_formate_lista_curvas(lista_curvas)
        self.lista_rayos = self.construir_rayos(Nr,M,N,centro)
        self.lista_intersecciones = self.construir_intersecciones(self.lista_rayos,self.lista_curvas)


    def obtener_rayos_con_cantidad_de_intersecciones_igual_a_la_moda_en_rango_angular(self,start, end, moda):
        lista = []
        rango_angular = np.arange(start, end)
        for direccion in rango_angular:
            intersecciones = [inter for inter in self.lista_intersecciones if inter.rayo_id == direccion]
            if len(intersecciones) == moda:
                intersecciones.sort(key=lambda x: distancia_entre_pixeles(self.centro[0], self.centro[1], x.y, x.x))
                distancias = []
                for x in intersecciones:
                    distancias.append(distancia_entre_pixeles(self.centro[0], self.centro[1], x.y, x.x))

                lista.append(distancias)

        return np.array(lista)
    def obtener_rayos_con_cantidad_de_intersecciones_igual_a_la_moda_en_rango_angular_pixeles(self,start, end, moda):
        lista = []
        rango_angular = np.arange(start, end)
        for direccion in rango_angular:
            intersecciones = [inter for inter in self.lista_intersecciones if inter.rayo_id == direccion]
            if len(intersecciones) == moda:
                intersecciones.sort(key=lambda x: distancia_entre_pixeles(self.centro[0], self.centro[1], x.y, x.x))
                distancias = []
                for x in intersecciones:
                    distancias.append([ x.y, x.x])

                lista.append(distancias)

        return np.array(lista)
    @staticmethod
    def generate_plot_data(data,distancia_maxima):
        cantidad_muestras = 360
        media = data[0, :]
        std = data[1, :]
        step = distancia_maxima / cantidad_muestras

        muestras = np.arange(0, distancia_maxima*1.1, step)
        y_plot = np.zeros((media.shape[0], muestras.shape[0]))
        for i in range(media.shape[0]):
            x1 = media[i] - std[i]
            if x1 < 0:
                x1 = 0
            x2 = media[i]
            x3 = media[i] + std[i]
            if x3 > muestras[-1]:
                x3 = muestras[-1]

            x1_idx = [idx for idx, value in enumerate(muestras) if value > x1][0]
            x2_idx = [idx for idx, value in enumerate(muestras) if value > x2][0]
            x3_idx = [idx for idx, value in enumerate(muestras) if value > x3][0]
            if x2_idx-x1_idx<1:
                x1_idx = x2_idx-1
            if x3_idx - x2_idx < 1:
                x3_idx = x2_idx + 1
            valores = np.hstack((np.linspace(0,1,x2_idx-x1_idx+1), np.flip(np.linspace(0,1,x3_idx-x2_idx+1)[:-1])))
            y_plot[i,x1_idx:x3_idx+1] = valores

        return muestras, y_plot

    def dibujar_rayos_sector_y_rayos_utilizados_para_generar_estadistica(self,start,end,filename,moda,debug = False):
        img_dibujo = self.img.copy()
        rango_angular = np.arange(start, end)
        for direccion in rango_angular:
            rayo = [rayo for rayo in self.lista_rayos if rayo.direccion==direccion][0]
            intersecciones = [inter for inter in self.lista_intersecciones if inter.rayo_id == direccion]
            #print(f"{len(intersecciones)} {moda}")
            color = (0,0,255) if len(intersecciones) == moda else (255,0,0)
            img_dibujo = Dibujar.rayo(rayo,img_dibujo,color=color)

        for curva in self.lista_curvas:
            img_dibujo = Dibujar.curva(curva,img_dibujo)

        if debug:
            cv2.imwrite(filename,img_dibujo)

    def dibujar_texto(self,texto,filename):
        img_dibujo = np.zeros_like(self.img)+255
        M,N,_ = img_dibujo.shape
        img_dibujo = Dibujar.put_text(texto,img_dibujo,(0,int(M*0.5)),fontScale=3)

        cv2.imwrite(filename,img_dibujo)

    def estadisticos_por_sector(self, cantidad_sectores, moda):
        self.contador_hist = 0
        sector = {}
        step = int(360 / cantidad_sectores)
        idx_sectores = range(cantidad_sectores)
        distancia_maxima = 0
        for i in idx_sectores:
            start = i*step
            end = (i+1)*step
            data = self.obtener_rayos_con_cantidad_de_intersecciones_igual_a_la_moda_en_rango_angular(start, end, moda)
            if data.shape[0] == 0:
                sector[i] = np.zeros((2,moda))
                continue
            maximo = data.max()
            if maximo > distancia_maxima:
                distancia_maxima = maximo

            sector[i] = np.vstack((data.mean(axis=0), data.std(axis=0)))
        img_sectores = np.zeros_like(self.img)
        fig_all, axs_all = plt.subplots(cantidad_sectores, 1,figsize=(7,21))
        for idx,key in enumerate(idx_sectores):
            img_sectores = self.dibujar_sector_sobre_imagen(sector, key, img_sectores)
            #filename = f"{self.histogramas_path}/{cantidad_sectores}_{self.contador_hist}.png"
            #self.dibujar_texto(f'Sector {idx} de {cantidad_sectores}',filename)
            #self.contador_hist += 1
            filename = f"{self.histogramas_path}/{cantidad_sectores}_{self.contador_hist}.png"
            start = idx*step
            end = (idx+1)*step
            filename = f"{self.histogramas_path}/moda.png"
            self.dibujar_rayos_sector_y_rayos_utilizados_para_generar_estadistica(start,end,filename,moda,debug= True if cantidad_sectores == 1 else False)
            self.contador_hist += 1
            # using the variable axs for multiple Axes

            fig, axs = plt.subplots(3, 1)
            fig.suptitle(f'La moda de intersecciones radiales es {moda}')
            data = sector[key]
            x,y = self.generate_plot_data(data,distancia_maxima)
            for i in range(y.shape[0]):
                axs[1].plot(x,y[i])
                if cantidad_sectores>1:
                    axs_all[idx].plot(x,y[i],'r')
                else:
                    axs_all.plot(x, y[i], 'r')
            axs[1].grid(True)
            axs[1].set_ylabel(f'Sector {idx}')
            if cantidad_sectores > 1:
                axs_all[idx].grid(True)
                axs_all[idx].set_ylabel(f'Sector {idx}')
                axs_all[idx].axis('off')
            else:
                axs_all.grid(True)
                axs_all.set_ylabel(f'Sector {idx}')
                axs_all.axis('off')

            if key == 0:
                data = sector[idx_sectores[-1]]
            else:
                data = sector[idx_sectores[idx-1]]
            x,y = self.generate_plot_data(data,distancia_maxima)
            for i in range(y.shape[0]):
                axs[0].plot(x,y[i])
            axs[0].grid(True)
            axs[0].set_ylabel(f'Sector {(idx-1)%cantidad_sectores}')
            if key == idx_sectores[-1]:
                data = sector[idx_sectores[0]]
            else:
                data = sector[idx_sectores[idx + 1]]
            x, y = self.generate_plot_data(data, distancia_maxima)
            for i in range(y.shape[0]):
                axs[2].plot(x, y[i])
            axs[2].grid(True)
            axs[2].set_ylabel(f'Sector {(idx+1)%cantidad_sectores}')
            #fig.savefig(f"{self.histogramas_path}/{cantidad_sectores}_{self.contador_hist}.png")
            plt.close(fig)
            #self.contador_hist+=1
        #fig_all.savefig(f"{self.histogramas_path}/{cantidad_sectores}_{self.contador_hist}.png")
        fig_all.savefig(f"{self.histogramas_path}/distribuciones.png")
        cv2.imwrite(f'{self.path}/img_dist_{cantidad_sectores}.png', img_sectores)
        plt.close(fig_all)
        self.contador_hist+=1
        return sector

    def estadisticos_por_sector_gaussian_mixture(self, cantidad_sectores, moda):
        from sklearn.mixture import GaussianMixture

        self.contador_hist = 0
        sector = {}
        step = int(360 / cantidad_sectores)
        idx_sectores = range(cantidad_sectores)
        distancia_maxima = 0
        for i in idx_sectores:
            start = i*step
            end = (i+1)*step
            data = self.obtener_rayos_con_cantidad_de_intersecciones_igual_a_la_moda_en_rango_angular(start, end, moda)
            gm = GaussianMixture(n_components=2, random_state=0).fit(data)
            gm.means_
            if data.shape[0] == 0:
                sector[i] = np.zeros((2,moda))
                continue
            maximo = data.max()
            if maximo > distancia_maxima:
                distancia_maxima = maximo

            sector[i] = np.vstack((data.mean(axis=0), data.std(axis=0)))
        img_sectores = np.zeros_like(self.img)
        fig_all, axs_all = plt.subplots(cantidad_sectores, 1,figsize=(7,21))
        for idx,key in enumerate(idx_sectores):
            img_sectores = self.dibujar_sector_sobre_imagen(sector, key, img_sectores)
            #filename = f"{self.histogramas_path}/{cantidad_sectores}_{self.contador_hist}.png"
            #self.dibujar_texto(f'Sector {idx} de {cantidad_sectores}',filename)
            #self.contador_hist += 1
            filename = f"{self.histogramas_path}/{cantidad_sectores}_{self.contador_hist}.png"
            start = idx*step
            end = (idx+1)*step
            filename = f"{self.histogramas_path}/moda.png"
            self.dibujar_rayos_sector_y_rayos_utilizados_para_generar_estadistica(start,end,filename,moda,debug= True if cantidad_sectores == 1 else False)
            self.contador_hist += 1
            # using the variable axs for multiple Axes

            fig, axs = plt.subplots(3, 1)
            fig.suptitle(f'La moda de intersecciones radiales es {moda}')
            data = sector[key]
            x,y = self.generate_plot_data(data,distancia_maxima)
            for i in range(y.shape[0]):
                axs[1].plot(x,y[i])
                if cantidad_sectores>1:
                    axs_all[idx].plot(x,y[i],'r')
                else:
                    axs_all.plot(x, y[i], 'r')
            axs[1].grid(True)
            axs[1].set_ylabel(f'Sector {idx}')
            if cantidad_sectores > 1:
                axs_all[idx].grid(True)
                axs_all[idx].set_ylabel(f'Sector {idx}')
                axs_all[idx].axis('off')
            else:
                axs_all.grid(True)
                axs_all.set_ylabel(f'Sector {idx}')
                axs_all.axis('off')

            if key == 0:
                data = sector[idx_sectores[-1]]
            else:
                data = sector[idx_sectores[idx-1]]
            x,y = self.generate_plot_data(data,distancia_maxima)
            for i in range(y.shape[0]):
                axs[0].plot(x,y[i])
            axs[0].grid(True)
            axs[0].set_ylabel(f'Sector {(idx-1)%cantidad_sectores}')
            if key == idx_sectores[-1]:
                data = sector[idx_sectores[0]]
            else:
                data = sector[idx_sectores[idx + 1]]
            x, y = self.generate_plot_data(data, distancia_maxima)
            for i in range(y.shape[0]):
                axs[2].plot(x, y[i])
            axs[2].grid(True)
            axs[2].set_ylabel(f'Sector {(idx+1)%cantidad_sectores}')
            #fig.savefig(f"{self.histogramas_path}/{cantidad_sectores}_{self.contador_hist}.png")
            plt.close(fig)
            #self.contador_hist+=1
        #fig_all.savefig(f"{self.histogramas_path}/{cantidad_sectores}_{self.contador_hist}.png")
        fig_all.savefig(f"{self.histogramas_path}/distribuciones.png")
        cv2.imwrite(f'{self.path}/img_dist_{cantidad_sectores}.png', img_sectores)
        plt.close(fig_all)
        self.contador_hist+=1
        return sector

    def calcular_spline(self,x,y):
        cs = CubicSpline(x, y,bc_type='periodic')
        angulos_full_range = np.arange(0,360,360/Nr)
        ySpline = cs(angulos_full_range)

        yy = []
        xx = []
        for r,angle in zip(ySpline,angulos_full_range):
            x = self.centro[0] + r * np.cos(angle*np.pi/180)
            y = self.centro[1] + r * np.sin(angle*np.pi/180)
            yy.append(y)
            xx.append(x)

        return np.vstack((np.array(yy),np.array(xx)))

    def agrupar_por_cercania(self,d1,d2,debug=False):
        X = np.abs(d1 - d2.reshape((-1, 1)))
        grafo = np.zeros_like(X)
        for i in range(X.shape[0]):
            j = np.argmin(X[i,:])
            grafo[i,j] = 1
        #postprocesamiento. Solo puede coserse con un anillo
        # plt.imshow(grafo, cmap='autumn', interpolation='nearest')
        # plt.title("2-D Heat Map")
        # plt.show()
        for i in range(X.shape[0]):
            vecino_i = grafo[:,i]
            if vecino_i.sum() <= 1:
                continue
            indices = np.where(vecino_i>0)[0]
            idx_cercano = X[indices,i].argmin()
            grafo[indices,i] = 0
            grafo[indices[idx_cercano],i] = 1

        if debug:
            plt.imshow(grafo, cmap='autumn', interpolation='nearest')
            plt.title("2-D Heat Map")
            plt.show()
        return np.where(grafo>0)

    def estadisticos_por_sector_malla_artificial(self, cantidad_sectores, moda):
        sector = {}
        step = int(360 / cantidad_sectores)
        idx_sectores = range(cantidad_sectores)
        #distancia_maxima = 0
        for i in idx_sectores:
            start = i*step
            end = (i+1)*step
            data = self.obtener_rayos_con_cantidad_de_intersecciones_igual_a_la_moda_en_rango_angular(start, end, moda)
            distancia_radial_media  = data.mean(axis=0)
            angulo_medio = (end + start) * 0.5
            sector[i] = (distancia_radial_media,angulo_medio)

        puntos_malla_artificial = {}
        for anillo_nro in range(moda):
            distancias_anillo = []
            angulos_puntos = []
            for j in sector.keys():
                #Opcion 1, simplemente elejir por orden

                #Opcion 2. Elegir el mas cercano
                d2,angulo = sector[j]
                if j == 0:
                    d1 ,_ = sector[cantidad_sectores-1]
                else:
                    d1,_ = sector[j-1]
                g1,g2 = self.agrupar_por_cercania(d1,d2)
                if np.where(g2==anillo_nro)[0].shape[0]>0:
                    distancias_anillo.append(d2[np.where(g2==anillo_nro)[0][0]])
                    angulos_puntos.append(angulo)
                    #print(f"anillo {anillo_nro} sector {j} elemento {np.where(g2==anillo_nro)[0][0]}")

            puntos_malla_artificial[anillo_nro] = self.calcular_spline(angulos_puntos + [angulos_puntos[0]+360],distancias_anillo + [distancias_anillo[0]])



        plt.figure()
        img_dibujo = self.img.copy()
        for curva in self.lista_curvas:
            img_dibujo = Dibujar.curva(curva,img_dibujo,thickness=2)

        for i in range(moda):
            x = puntos_malla_artificial[i][0]
            y = puntos_malla_artificial[i][1]
            line = LineString([Point(j,i) for i,j in zip(y,x)])
            img_dibujo = Dibujar.curva(line, img_dibujo, thickness=2,color=(0,0,255))

        cv2.imwrite(f"{self.histogramas_path}/{cantidad_sectores}_{self.contador_hist}.png",img_dibujo)
        self.contador_hist += 1


    def hay_rayos_iguales_a_la_moda_en_el_sector(self,sector):
        return sector.sum()>0

    def actualizar_matriz_agrupaciones(self, matriz_agrupaciones, key, key_despues, sector):
        d1, _ = sector[key]
        d2, _ = sector[key_despues]
        g2, g1 = self.agrupar_por_cercania(d1, d2)
        for i, j in zip(g1, g2):
            label_pos_sector_actual = matriz_agrupaciones[key, i]
            label_pos_sector_despues = matriz_agrupaciones[key_despues, j]
            if label_pos_sector_actual == 0 and label_pos_sector_despues == 0:
                # No estan agrupados
                new_label = matriz_agrupaciones.max() + 1
                matriz_agrupaciones[key, i] = new_label
                matriz_agrupaciones[key_despues, j] = new_label

            elif label_pos_sector_despues == 0 and label_pos_sector_actual > 0:
                new_label = matriz_agrupaciones[key, i]
                matriz_agrupaciones[key_despues, j] = new_label

            elif label_pos_sector_despues > 0 and label_pos_sector_actual == 0:
                new_label = matriz_agrupaciones[key_despues, j]
                matriz_agrupaciones[key, i] = new_label

            else:
                colineal_actual = np.where(matriz_agrupaciones[key]==label_pos_sector_despues)[0]
                colineal_despues = np.where(matriz_agrupaciones[key_despues] == label_pos_sector_actual)[0]
                if len(colineal_despues)>0 or len(colineal_actual)>0:
                    continue
                y, x = np.where(matriz_agrupaciones == label_pos_sector_despues )
                matriz_agrupaciones[y, x] = label_pos_sector_actual

        return matriz_agrupaciones

    def agrupar_distribuciones(self, sector, moda,debug=False):
        cantidad = len(sector.keys())
        matriz_agrupaciones = np.zeros((cantidad,moda))
        for key in sector.keys():
            if not self.hay_rayos_iguales_a_la_moda_en_el_sector(sector[key]):
                #No hay rayo con cantidad de intersecciones iguales a la moda
                continue
            if key == cantidad - 1:
                #TODO: se completa la vuelta. Hay que hacerlo con cuidado.
                continue
            key_antes = (key - 1) % cantidad
            key_despues = (key + 1) % cantidad
            # if self.hay_rayos_iguales_a_la_moda_en_el_sector(sector[key_antes]):
            #     matriz_agrupaciones = self.actualizar_matriz_agrupaciones(matriz_agrupaciones,key,key_antes,sector)

            if self.hay_rayos_iguales_a_la_moda_en_el_sector(sector[key_despues]):
                matriz_agrupaciones = self.actualizar_matriz_agrupaciones(matriz_agrupaciones,key,key_despues,sector)


        self.graficar_matriz_agrupaciones(matriz_agrupaciones,sector)
        self.graficar_grafo_de_conexiones(matriz_agrupaciones,sector)
        if debug:
            plt.imshow(matriz_agrupaciones, cmap='autumn', interpolation='nearest')
            plt.title("2-D Heat Map")
            plt.show()
        return 0

    def graficar_grafo_de_conexiones(self,matriz_agrupaciones,sector):
        agrupaciones_id = list(np.unique(matriz_agrupaciones))
        if 0 in agrupaciones_id:
            agrupaciones_id.remove(0)
        cantidad_sectores,moda = matriz_agrupaciones.shape
        distancia_maxima = np.max([np.max(sector[key][0, :]) for key in sector.keys()])
        fig_all, axs_all = plt.subplots(1, 1, figsize=(7, 21))
        for group_id in agrupaciones_id:
            y,x = np.where(matriz_agrupaciones==group_id)
            xx = [sector[i][0,j] for i,j in zip(y,x)]
            axs_all.plot(xx,y,marker='*')

        axs_all.set_xlim([0, distancia_maxima])
        axs_all.set_ylim([0, cantidad_sectores])
        axs_all.invert_yaxis()
        #fig_all.savefig(f"{self.histogramas_path}/{cantidad_sectores}_{self.contador_hist}.png")
        fig_all.savefig(f"{self.histogramas_path}/grafo.png")
        plt.close(fig_all)
        self.contador_hist += 1


    def graficar_matriz_agrupaciones(self,matriz_agrupacines,sector):
        cantidad_sectores, moda = matriz_agrupacines.shape
        #cantidad_sectores = len(sector.keys())
        agrupaciones_id = list(np.unique(matriz_agrupacines))
        total_colores = len(agrupaciones_id)

        from matplotlib.cm import get_cmap

        name = "tab20"
        cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
        color_pallete = cmap.colors
        colors = np.zeros((total_colores,3))
        colores = np.random.randint(len(color_pallete),size=total_colores)
        for idx in range(total_colores):
            colors[idx] = color_pallete[colores[idx]]

        distancia_maxima = np.max([np.max(sector[key][0,:]) for key in sector.keys()])
        fig_all, axs_all = plt.subplots(cantidad_sectores, 1, figsize=(7, 21))
        for key in range(cantidad_sectores):
            data = sector[key]
            if not self.hay_rayos_iguales_a_la_moda_en_el_sector(data):
                axs_all[key].set_ylabel(f'Sector {key}')
                continue
            x, y = self.generate_plot_data(data, distancia_maxima)

            for i in range(y.shape[0]):
                group_id = matriz_agrupacines[key,i]
                if group_id == 0:
                    continue
                color_id = agrupaciones_id.index(group_id)
                if cantidad_sectores > 1:
                    axs_all[key].plot(x, y[i], color=colors[int(color_id)])
                else:
                    axs_all.plot(x, y[i],  color=colors[int(color_id)])

            if cantidad_sectores > 1:
                axs_all[key].grid(True)
                axs_all[key].set_ylabel(f'Sector {key}')
                axs_all[key].axis('off')

            else:
                axs_all.grid(True)
                axs_all.set_ylabel(f'Sector {key}')
                axs_all.axis('off')

        #fig_all.savefig(f"{self.histogramas_path}/{cantidad_sectores}_{self.contador_hist}.png")
        fig_all.savefig(f"{self.histogramas_path}/distribuciones_agrupadas.png")
        plt.close(fig_all)
        self.contador_hist += 1




    def seleccionar_sector_con_dispersion_minima(self,sector):
        minimo_std = np.inf
        minimo_key = -1
        for key in sector.keys():
            mean, std = sector[key]
            if std.mean() < minimo_std:
                minimo_key = key
                minimo_std = std.mean()

        return minimo_key

    def transformar_medias_y_direccion_a_cartesianas(self,direccion,medias):
        yy = []
        xx = []
        for r in medias:
            y = self.centro[0] + r * np.cos(direccion * np.pi / 180)
            x = self.centro[1] + r * np.sin(direccion * np.pi / 180)
            yy.append(y)
            xx.append(x)
        return yy,xx

    def transformar_sector_en_cartesianas(self,sector,key_origen,img_dibujo,dibujar_curva = True):
        cantidad_sectores = len(sector.keys())
        step = 360 / cantidad_sectores
        direccion = step * (key_origen + 1 / 2)
        medias, dispersiones = sector[key_origen]
        yy,xx = self.transformar_medias_y_direccion_a_cartesianas(direccion,medias)
        #graficar
        M,N,_ = img_dibujo.shape
        rayo = Rayo(direccion,self.centro,M,N)
        img_dibujo = Dibujar.rayo(rayo,img_dibujo)
        lista_sectores = [(Interseccion(x=x,y=y,curva_id=-1,rayo_id=direccion),std) for x,y,std in zip(xx,yy,dispersiones)]
        #img_dibujo = Dibujar.intersecciones(lista_sectores,img_dibujo)
        img_dibujo = Dibujar.intersecciones_con_tamaño(lista_sectores, img_dibujo)



        return np.vstack((np.array(yy), np.array(xx))).T,direccion,img_dibujo

    def distantia_minima_intersecciones(self,lista,inter):
        minimo = np.inf
        pos = -1
        for inter_lista in lista:
            d = inter_lista.distance(inter)
            if d<minimo:
                minimo = d
                pos = lista.index(inter_lista)

        return minimo,pos

    def seleccionar_distribucion_dentro_del_sector_con_menor_distancia_a_curva(self,sector,key_origen,img_dibujo,orden):
        pts,direccion,img_dibujo = self.transformar_sector_en_cartesianas(sector,key_origen,img_dibujo)
        bolsa_intersecciones_cercanas = [inter for inter in self.lista_intersecciones if inter.rayo_id in [np.floor(direccion),np.ceil(direccion)]]
        distancias = []
        sector_pts = []
        for pos in range(pts.shape[0]):
            aux_inter = Interseccion(x= pts[pos,1], y= pts[pos,0], curva_id=-1, rayo_id=-1)
            d,_ = self.distantia_minima_intersecciones(bolsa_intersecciones_cercanas,aux_inter)
            distancias.append(d)
            sector_pts.append(aux_inter)
        distancias_orig = distancias.copy()
        for i in range(orden):
            seleccionado_idx =  np.argmin(distancias)
            distancias.pop(seleccionado_idx)
        seleccionado_idx = np.where(np.min(distancias) == distancias_orig)[0][0]
        #print(f"orden={orden} distancia {np.min(distancias)} idx {seleccionado_idx}")
        inter_cercana = sector_pts[seleccionado_idx]
        _,dispersiones = sector[key_origen]
        std = dispersiones[seleccionado_idx]
        img_dibujo = Dibujar.intersecciones_con_tamaño([(inter_cercana,std)],img_dibujo,color=(0,0,0),thickness=-1)
        return seleccionado_idx,img_dibujo

    def dibujar_sector_sobre_imagen(self,sector,key_origen,img_dibujo):
        pts,direccion,img_dibujo = self.transformar_sector_en_cartesianas(sector,key_origen,img_dibujo,dibujar_curva=False)
        return img_dibujo







    def seleccionar_puntos_mas_cercanos_siguiente_sector(self,idx_dist_origen,key_origen,key_siguiente,sector,img_dibujo):
        cantidad_sectores = len(sector.keys())
        step = 360 / cantidad_sectores
        direccion_siguiente = step * (key_siguiente + 1 / 2)
        direccion =  step * (key_origen + 1 / 2)
        d1, _ = sector[key_origen]
        d2, _ = sector[key_siguiente]
        #g2, g1 = self.agrupar_por_cercania(d1, d2)
        idx_siguiente_sector = np.argmin(np.abs(d2 - d1[idx_dist_origen]))#g1[idx_dist_origen]
        vecindad = [(d2[idx_siguiente_sector],idx_siguiente_sector)]

        if idx_siguiente_sector>0:
            vecindad.append((d2[idx_siguiente_sector-1],idx_siguiente_sector-1))

        if idx_siguiente_sector < len(d2)-1:
            vecindad.append((d2[idx_siguiente_sector+1],idx_siguiente_sector+1))

        yy, xx = self.transformar_medias_y_direccion_a_cartesianas(direccion_siguiente,[vec for vec,_ in vecindad])
        y_o, x_o = self.transformar_medias_y_direccion_a_cartesianas(direccion, [d1[idx_dist_origen]])

        lista_segmentos = []
        lista_puntos  = []
        punto_origen = Interseccion(x=x_o[0], y=y_o[0], curva_id=-1, rayo_id=direccion)
        for y,x in zip(yy,xx):
            punto = Interseccion(x=x,y=y,curva_id=-1,rayo_id=direccion_siguiente)
            segmento = LineString(coordinates=[(punto_origen.x,punto_origen.y),(x,y)])
            lista_segmentos.append(segmento)
            img_dibujo = Dibujar.rayo(segmento, img_dibujo,color=(23,208,253))
            lista_puntos.append(punto)

        vecinos = [(pto, vecindad[idx][1]) for idx, pto in enumerate(lista_puntos)]


        return img_dibujo,vecinos , punto_origen
    def buscar_si_hay_puntos_cercanos_a_la_curva_origen(self,idx_dist_origen,key_origen,key_siguiente,sector):
        cantidad_sectores = len(sector.keys())
        step = 360 / cantidad_sectores
        direccion_siguiente = step * (key_siguiente + 1 / 2)
        direccion =  step * (key_origen + 1 / 2)
        d1, _ = sector[key_origen]
        d2, _ = sector[key_siguiente]


        yy, xx = self.transformar_medias_y_direccion_a_cartesianas(direccion_siguiente,d2)
        y_o, x_o = self.transformar_medias_y_direccion_a_cartesianas(direccion, [d1[idx_dist_origen]])


        lista_puntos  = []
        punto_origen = Interseccion(x=x_o[0], y=y_o[0], curva_id=-1, rayo_id=direccion_siguiente)
        for y,x in zip(yy,xx):
            punto = Interseccion(x=x,y=y,curva_id=-1,rayo_id=direccion_siguiente)
            lista_puntos.append(punto)

        vecinos = [(pto, idx) for idx, pto in enumerate(lista_puntos)]
        curva_cercana_orig = self.curva_mas_cercana_a_punto(punto_origen)
        vecinos_nuevo_orden = []
        pos = 0
        for pto,idx in vecinos:
            curva_cercana_otro = self.curva_mas_cercana_a_punto(pto)
            if curva_cercana_otro == curva_cercana_orig:
                vecinos_nuevo_orden.append(pos)
            pos+= 1

        if len(vecinos_nuevo_orden)>0:
            vecinos = [vecinos[vecinos_nuevo_orden[0]]]
        else:
            vecinos = []

        return vecinos , punto_origen
    def curva_mas_cercana_a_punto(self, punto):
        dist = [inter.distance(punto) for inter in self.lista_intersecciones]
        cercano = self.lista_intersecciones[np.argmin(dist)]
        curva_otro = cercano.curva_id
        return curva_otro

    def metrica_distancia_segmento(self,pt1,pt_origen):
        #buscar curvas mas cercana a punto pt1
        dist = [inter.distance(pt1) for inter in self.lista_intersecciones]
        dist_intersecciones_mas_cercana = np.min(dist)
        cercano = self.lista_intersecciones[np.argmin(dist)]
        curva_otro = cercano.curva_id

        #medir distancia entre pto origen y curva de pt1
        dist = [inter.distance(pt_origen) for inter in self.lista_intersecciones if inter.curva_id == curva_otro]
        dist_o = np.min(dist)


        #buscar curvas mas cercana a punto origen
        dist = [inter.distance(pt_origen) for inter in self.lista_intersecciones]
        dist_intersecciones_mas_cercana = np.min(dist)
        cercano = self.lista_intersecciones[np.argmin(dist)]
        curva_origen = cercano.curva_id

        #medir distancai entre pt1 y curva cercana a origen
        dist = [inter.distance(pt1) for inter in self.lista_intersecciones if inter.curva_id == curva_origen]
        dist_1 = np.min(dist)

        return np.maximum(dist_o,dist_1)

    @staticmethod
    def condicion_parada_salto_maximo(idx_dist_origen, key_origen, sector,dist_nuevo_punto,umbral=1):
        cantidad_sectores = len(sector.keys())
        if idx_dist_origen > 0:
            idx_abajo_origen = idx_dist_origen - 1
        else:
            idx_abajo_origen = idx_dist_origen

        abajo = np.abs(sector[key_origen][0][idx_abajo_origen] - sector[key_origen][0][idx_dist_origen])

        if idx_dist_origen < cantidad_sectores - 1:
            idx_arriba_origen = idx_dist_origen + 1
        else:
            idx_arriba_origen = idx_dist_origen
        arriba = np.abs(sector[key_origen][0][idx_arriba_origen] - sector[key_origen][0][idx_dist_origen])

        if abajo == 0:
            separacion_maxima = arriba
        elif arriba == 0:
            separacion_maxima = abajo
        else:
            separacion_maxima = np.minimum(arriba, abajo)

        return dist_nuevo_punto>separacion_maxima*umbral



    def transformar_sector_a_lista(self,sector):
        lista_distribuciones = []
        cantidad_sectores = len(sector.keys())
        step = 360 / cantidad_sectores
        for key in sector.keys():
            direccion = step * (key + 1 / 2)
            mean = sector[key][0]
            std = sector[key][1]
            yy, xx = self.transformar_medias_y_direccion_a_cartesianas(direccion, mean)
            for y, x, m, s in zip(yy, xx, mean, std):
                punto = Interseccion(x=x, y=y, curva_id=-1, rayo_id=direccion)
                curva_id = self.curva_mas_cercana_a_punto(punto)
                distr = Distribucion(x=punto.x, y=punto.y, curva_id=curva_id, rayo_id=punto.rayo_id, mean=m, std=s)
                lista_distribuciones.append(distr)

        return lista_distribuciones

    def determinar_presencia_de_curvas_en_cada_sector(self,sector,lista_distribuciones):
        step = 360 / len(sector.keys())
        rayos_ids = np.unique([dist.rayo_id for dist in lista_distribuciones])
        sector_curvas = {}
        for rayo_id_0 in rayos_ids:
            rayo_id_1 = (step + rayo_id_0 ) % 360
            O = [dist for dist in lista_distribuciones if dist.rayo_id == rayo_id_0]
            if len(O)==0:
                continue
            O = O[0]
            P = [dist for dist in lista_distribuciones if dist.rayo_id == rayo_id_1]
            if len(P)==0:
                continue
            P = P[0]
            segmento_recto = LineString(coordinates=[(O.x, O.y), (P.x, P.y)])
            dominio = [rayo.direccion for rayo in self.lista_rayos if not rayo.intersection(segmento_recto).is_empty]
            lista_intersecciones_en_sector = []
            for rayo_id in dominio:
                lista_intersecciones_en_sector += [inter for inter in self.lista_intersecciones if inter.rayo_id == rayo_id]

            curvas_id_en_sector = np.unique([inter.curva_id for inter in lista_intersecciones_en_sector])
            sector_curvas[rayo_id_0] = []
            for curva_id in curvas_id_en_sector:
                curva = [curva for curva in self.lista_curvas if curva.id ==curva_id][0]
                sector_curvas[rayo_id_0].append(curva)

        return  sector_curvas

    def dibujar_enlaces(self,lista_enlaces,filename):
        img_enlaces = np.zeros_like(self.img)
        for segmento in lista_enlaces:
            img_enlaces = segmento.save_img(img_enlaces)
        cv2.imwrite(f"{self.path}/{filename}",img_enlaces)

    @staticmethod
    def condicion_parada_coser_enlace(origen: Distribucion, lista_distribuciones,
                                      valor_minimo_metrica: float, umbral=0.3):
        distibuciones_mismo_rayo = [dist for dist in lista_distribuciones if dist.rayo_id == origen.rayo_id]
        distibuciones_mismo_rayo.sort(key=lambda x: x.mean)
        idx_dist_origen = distibuciones_mismo_rayo.index(origen)
        if idx_dist_origen > 0:
            idx_abajo_origen = idx_dist_origen - 1
        else:
            idx_abajo_origen = idx_dist_origen

        abajo = np.abs(lista_distribuciones[idx_abajo_origen].mean - origen.mean)

        if idx_dist_origen < len(distibuciones_mismo_rayo) - 1:
            idx_arriba_origen = idx_dist_origen + 1
        else:
            idx_arriba_origen = idx_dist_origen

        arriba = np.abs(lista_distribuciones[idx_arriba_origen].mean - origen.mean)

        if abajo == 0:
            separacion_maxima = arriba
        elif arriba == 0:
            separacion_maxima = abajo
        else:
            separacion_maxima = np.minimum(arriba, abajo)

        return valor_minimo_metrica > separacion_maxima * umbral

    def buscar_curva_mas_cercana_de_cada_distribucion(self,sector):
        nueva_estructura={}
        cantidad_sectores = len(sector.keys())
        step = 360 / cantidad_sectores
        for key in sector.keys():
            direccion =  step * (key + 1 / 2)
            mean = sector[key][0]
            std  = sector[key][1]
            yy, xx = self.transformar_medias_y_direccion_a_cartesianas(direccion, mean)
            lista_distribuciones = []
            for y, x,m,s in zip(yy, xx,mean,std):
                punto = Interseccion(x=x, y=y, curva_id=-1, rayo_id=direccion)
                curva_id = self.curva_mas_cercana_a_punto(punto)
                distr = Distribucion(x=punto.x,y=punto.y,curva_id=curva_id,rayo_id=punto.rayo_id,mean=m,std=s)
                lista_distribuciones.append(distr)

            nueva_estructura[key] = lista_distribuciones

        return nueva_estructura

    def coser_basado_en_cercania_curva(self,sector):
        cantidad_sectores = len(sector.keys())
        for key in range(cantidad_sectores):
            dist_sector = sector[key]
            key_delante = (1 + key) % cantidad_sectores
            dist_sector_delante = sector[key_delante]
            key_atras = (key-1) % cantidad_sectores
            dist_sector_atras = sector[key_atras]
            for dist_actual in dist_sector:
                dist_delante = [dist for dist in dist_sector_delante if dist.curva_id == dist_actual.curva_id]
                if not len(dist_delante)==0:
                    dist_delante = dist_delante[0]
                    dist_actual.delante = dist_delante

                dist_atras = [dist for dist in dist_sector_atras if dist.curva_id == dist_actual.curva_id]
                if not len(dist_atras) == 0:
                    dist_atras = dist_atras[0]
                    dist_actual.delante = dist_atras

        return sector


    def dibujar_conexion_entre_dist(self,sector):
        img_cosida = np.zeros_like(self.img)
        img_dibujo = self.img.copy()
        for curva in self.lista_curvas:
            img_dibujo = Dibujar.curva(curva,img_dibujo)

        cantidad_sectores = len(sector.keys())
        lista_sectores = []
        for key in sector.keys():

            for dist in sector[key]:
                if dist.delante is not None:
                    segmento = LineString(coordinates=[(dist.x, dist.y), (dist.delante.x, dist.delante.y)])
                    img_cosida = Dibujar.rayo(segmento, img_cosida, color=(0, 0, 255))
                    img_dibujo = Dibujar.rayo(segmento, img_dibujo, color=(0, 0, 255),thickness=2)
                if dist.atras is not None:
                    segmento = LineString(coordinates=[(dist.x, dist.y), (dist.atras.x, dist.atras.y)])
                    img_cosida = Dibujar.rayo(segmento, img_cosida, color=(0, 0, 255))
                    img_dibujo = Dibujar.rayo(segmento, img_dibujo, color=(0, 0, 255),thickness=2)
                lista_sectores.append((Interseccion(x=dist.x, y=dist.y, curva_id=-1, rayo_id=dist.rayo_id), dist.std))

            M, N, _ = img_dibujo.shape
            rayo = Rayo(dist.rayo_id, self.centro, M, N)
            img_dibujo = Dibujar.rayo(rayo, img_dibujo)




        img_dibujo = Dibujar.intersecciones_con_tamaño(lista_sectores, img_dibujo)
        cv2.imwrite(f'{self.path}/img_cosida_{cantidad_sectores}.png', img_cosida)
        cv2.imwrite(f'{self.path}/img_cosida_{cantidad_sectores}_superposicion.png', img_dibujo)


    def determinar_curvas_por_sector(self,lista_distribuciones):
        rayos_direcciones =np.unique( [dist.rayo_id for dist in lista_distribuciones])
        self.curvas_agrupadas_por_sector = {}
        for idx,direccion in enumerate(rayos_direcciones):
            direccion_siguiente = rayos_direcciones[idx+1] if idx<len(rayos_direcciones)-1 else rayos_direcciones[0]
            src = [dist for dist in lista_distribuciones if dist.rayo_id == direccion][0]
            dst = [dist for dist in lista_distribuciones if dist.rayo_id == direccion_siguiente][0]
            segmento_recto = LineString(coordinates=[(src.x, src.y),(dst.x, dst.y)])
            dominio = [rayo.direccion for rayo in self.lista_rayos if not rayo.intersection(segmento_recto).is_empty]
            lista_curvas_id = []
            for rayo_id in dominio:
                lista_curvas_id += [inter.curva_id for inter in self.lista_intersecciones if inter.rayo_id == rayo_id]
            lista_curvas_id = np.unique(lista_curvas_id)
            curvas_sector = []
            for curva_id in lista_curvas_id:
                curva = [curva for  curva in self.lista_curvas if curva.id == curva_id][0]
                curvas_sector.append(curva)

            self.curvas_agrupadas_por_sector[idx] = curvas_sector



    def segmentos_en_sector(self,rayo_id_0,rayo_id_P,lista_segmentos):
        return [segmento for segmento in lista_segmentos if segmento.src.rayo_id == rayo_id_0 and segmento.dst.rayo_id == rayo_id_P]

    def coser_distribuciones_segun_vecindad_sectorial(self,lista_segmentos,lista_distribuciones,cantidad_sectores=16):
        step = 360/cantidad_sectores
        for sector_nro in range(cantidad_sectores):
            rayo_id_0 = step*(sector_nro + 1/2)
            dist_O = [dist for dist in lista_distribuciones if dist.rayo_id == rayo_id_0]
            dist_O.sort(key=lambda x: distancia_entre_pixeles(x.y, x.x, self.centro[0], self.centro[1]))
            rayo_id_P = step*(sector_nro + 1 + 1/2) % 360
            dist_P = [dist for dist in lista_distribuciones if dist.rayo_id == rayo_id_P]
            dist_P.sort(key=lambda x: distancia_entre_pixeles(x.y, x.x, self.centro[0], self.centro[1]))
            segmentos_en_sector = self.segmentos_en_sector(rayo_id_0,rayo_id_P,lista_segmentos)
            src_segmentos = [seg.src for seg in lista_segmentos]
            dst_segmentos = [seg.dst for seg in lista_segmentos]
            segmentos_en_sector.sort(key=lambda x: distancia_entre_pixeles(x.src.y, x.src.x, self.centro[0], self.centro[1]))
            for O in dist_O:
                if O in src_segmentos:
                    continue

                #O es una distribucion sin segmento








    def inicializacion(self,sector):
        # 1.0 Se transforman las distribuciones (medias radiales) a  punto cartesianos.
        lista_distribuciones = self.transformar_sector_a_lista(sector)
        lista_distribuciones.sort(key=lambda x: x.std)
        step = 360 / len(sector.keys())
        lista_enlaces = []
        debug_dir = Path(self.path) / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        bolsa_de_distribuciones_no_recorrida = [dist for dist in lista_distribuciones]
        rayo_id_idx=0
        anillo_nro = 0
        data = np.zeros((self.moda,len(sector.keys())))
        while len(bolsa_de_distribuciones_no_recorrida) > 0:
            rayo_id = (step*(rayo_id_idx+1/2))%360
            sector_distribuciones_actual = [dist for dist in lista_distribuciones if dist.rayo_id == rayo_id]
            sector_distribuciones_actual.sort(key=lambda x: distancia_entre_pixeles(x.y,x.x,self.centro[0],self.centro[1]))
            O = sector_distribuciones_actual[anillo_nro]
            rayo_siguiente = (rayo_id + step) % 360
            print(f"{rayo_id} {rayo_siguiente} {anillo_nro}")
            sector_distribuciones_siguiente = [dist for dist in lista_distribuciones if dist.rayo_id == rayo_siguiente]
            sector_distribuciones_siguiente.sort(
                key=lambda x: distancia_entre_pixeles(x.y, x.x, self.centro[0], self.centro[1]))
            P = sector_distribuciones_siguiente[anillo_nro]

            valor, soporte, res = self.calcular_identificadores_enlace(O, P)
            segmento = Segmento(O=O, P=P, identificador=len(lista_enlaces), value=valor,
                                soporte=soporte,
                                centro=self.centro, lista_rayos=self.lista_rayos,
                                umbral=np.floor(step))

            #Compute data
            data[anillo_nro,rayo_id_idx] = segmento.value
            #Update State
            lista_enlaces.append(segmento)
            bolsa_de_distribuciones_no_recorrida.remove(O)
            rayo_id_idx+=1
            if rayo_id_idx>=len(sector.keys()):
                anillo_nro+=1
                rayo_id_idx=0


        #Reporting data
        self.dibujar_enlaces(lista_enlaces,f'init_{len(sector.keys())}.png')
        data = np.hstack((data,np.sum(data,axis=1).reshape(-1,1)))
        np.savetxt(f"{self.path}/data_{len(sector.keys())}.csv",data,header=str([str(key) for key in sector.keys()] + ['suma']),fmt='%1.1f')
        valores = [enlace.value for enlace in lista_enlaces]
        plt.figure()
        plt.hist(valores)
        plt.title(f"Energia {np.sum(valores):.1f}")
        plt.savefig(f"{self.path}/energia_init_{len(sector.keys())}.png")
        plt.close()
        return

    def minimizar_energia(self, malla):
        label = 'minimizar_energia'
        iteracion = 0
        for key_sector in range(malla.cantidad_sectores):
            for posicion in range(self.moda):
                if posicion >= self.moda - 1:
                    iteracion += 1
                    continue

                E0 = malla.energia_global()
                src = (key_sector, posicion)
                dst = (key_sector, posicion + 1)
                anillo = posicion
                malla.mover_vertice(src, dst, anillo)
                Ef = malla.energia_global()
                write_log(MODULE, label, f"{iteracion}:  Anillo: {posicion} E0: {E0} Ef: {Ef}. deltaE: {Ef - E0}.")
                if Ef > E0:
                    malla.mover_vertice(dst, src, anillo)
                    if posicion > 0:
                        iteracion += 1
                        continue
                    E0 = malla.energia_global()
                    src = (key_sector, posicion)
                    dst = (key_sector, posicion - 1)
                    malla.mover_vertice(src, dst, anillo)
                    Ef = malla.energia_global()
                    write_log(MODULE, label, f"{iteracion}: Anillo: {posicion} E0: {E0} Ef: {Ef}. deltaE: {Ef - E0}.")
                    if Ef > E0:
                        malla.mover_vertice(dst, src, posicion)

                malla.dibujar_anillos(f"{iteracion}.png")
                write_log(MODULE, label, f"{iteracion}:Anillo: {posicion} Eglobal {malla.energia_global()}")
                iteracion += 1

    def armar_malla_artificial(self,cantidad_sectores):
        label = 'armar_malla_artificial'
        sector = self.estadisticos_por_sector(cantidad_sectores, self.moda)
        #self.inicializacion(sector)
        lista_distribuciones = self.transformar_sector_a_lista(sector)
        lista_distribuciones.sort(key=lambda x: x.std)
        malla = Malla(lista_distribuciones,[],cantidad_sectores,self.moda,self.lista_rayos,np.zeros_like(self.img),Path(self.path) / "malla",self.centro,self.lista_intersecciones)


        iteracion = 0
        for key_sector in range(cantidad_sectores):
            for posicion in range(self.moda):
                if posicion >= self.moda-1:
                    iteracion += 1
                    continue

                E0 = malla.energia_global()
                src = ( key_sector, posicion)
                dst = ( key_sector, posicion+1)
                anillo = posicion
                malla.mover_vertice( src, dst, anillo)
                Ef = malla.energia_global()
                write_log( MODULE, label, f"{iteracion}:  Anillo: {posicion} E0: {E0} Ef: {Ef}. deltaE: {Ef-E0}.")
                if Ef > E0 :
                    malla.mover_vertice(dst, src, anillo)
                    if posicion>0:
                        iteracion += 1
                        continue
                    E0 = malla.energia_global()
                    src = (key_sector, posicion)
                    dst = (key_sector, posicion - 1)
                    malla.mover_vertice(src, dst, anillo)
                    Ef = malla.energia_global()
                    write_log( MODULE, label, f"{iteracion}: Anillo: {posicion} E0: {E0} Ef: {Ef}. deltaE: {Ef - E0}.")
                    if Ef > E0:
                        malla.mover_vertice(dst, src, posicion)

                malla.dibujar_anillos(f"{iteracion}.png")
                write_log( MODULE, label, f"{iteracion}:Anillo: {posicion} Eglobal {malla.energia_global()}")
                iteracion += 1





    def estadisticos_globales_piramide(self):
        moda = self.calcular_moda_intersecciones_por_rayo(self.lista_intersecciones)
        print(moda)
        self.contador_hist = 0
        self.histogramas_path = Path(self.path) / "malla"
        self.histogramas_path.mkdir(parents=True, exist_ok=True)
        lista_anillos = None
        for cantidad_sectores in [4]:
            if lista_anillos is None:
                sector = self.estadisticos_por_sector(cantidad_sectores, self.moda)
                #sector = self.estadisticos_por_sector_gaussian_mixture(cantidad_sectores, self.moda)
                lista_distribuciones = self.transformar_sector_a_lista(sector)
                lista_distribuciones.sort(key=lambda x: x.std)

            malla = Malla(lista_distribuciones, [], cantidad_sectores, self.moda, self.lista_rayos,
                              np.zeros_like(self.img),self.histogramas_path , self.centro,
                              self.lista_intersecciones,lista_anillos)

            malla.save_results('init')
            malla.minimizar_energia()

            malla.save_results('end')

            lista_anillos,lista_distribuciones = malla.transformar_anillos_siguiente_escala()

        self.build_pdf_malla()

    def estadisticos_globales(self):
        moda = self.calcular_moda_intersecciones_por_rayo(self.lista_intersecciones)
        print(moda)
        self.contador_hist = 0
        self.histogramas_path = Path(self.path) / "histograma"
        self.histogramas_path.mkdir(parents=True, exist_ok=True)
        self.estadisticos_por_sector(1, moda)

        for cantidad_sectores in [8]:
            self.armar_malla_artificial(cantidad_sectores)
            #self.build_pdf_malla()
            # if True:
            #     continue
        #
        #     self.img_cosida = np.zeros_like(self.img)
        #     sector = self.estadisticos_por_sector(cantidad_sectores,  moda)
        #     #self.agrupar_distribuciones( sector, moda)
        #
        #
        #     #key_origen = self.seleccionar_sector_con_dispersion_minima(sector)
        #     for key_origen in range(cantidad_sectores):
        #         for orden in range(moda):
        #             idx_dist_origen,img_debug = self.seleccionar_distribucion_dentro_del_sector_con_menor_distancia_a_curva(sector,key_origen,img_debug,orden)
        #             #cv2.imwrite(f'{self.path}/origen_sector_{orden}.png', img_debug)
        #             #max_iter = self.loop_coser_distribuciones('anti',sector,key_origen,idx_dist_origen,img_debug)
        #             #self.loop_coser_distribuciones('horario',sector,key_origen,idx_dist_origen,img_debug,cantidad_sectores-max_iter)
        #             max_iter = self.loop_coser_distribuciones_cerana('anti', sector, key_origen, idx_dist_origen, img_debug)
        #             max_iter = self.loop_coser_distribuciones_cerana('horario', sector, key_origen, idx_dist_origen, img_debug)
        #
        #         # img_debug = self.dibujar_sector_sobre_imagen(sector,(key_origen - 1) % cantidad_sectores,img_debug)
        #         #
        #         # cv2.imwrite(f'{self.path}/origen_sector_2.png', img_debug)
        #         #self.estadisticos_por_sector_malla_artificial(16,  moda)
        #     cv2.imwrite(f'{self.path}/img_cosida_{cantidad_sectores}.png', self.img_cosida)
        # self.build_histograma_pdf()

    def build_pdf_malla(self):
        from natsort import natsorted
        from fpdf import FPDF
        pdf = FPDF()
        pdf.set_font('Arial', 'B', 16)
        from tqdm import tqdm
        import glob

        #try:
        disco = self.histogramas_path.parts[-2]
        disco = disco.replace("_","")
        pdf.add_page()
        pdf.cell(40, 10, disco)

        x,y = 0,0
        height = 150

        for number in [4,8,16,32]:
            pdf.add_page()
            pdf.cell(40, 10, f"Cantidad Sectores {number}")
            pdf.add_page()
            pdf.image(f"{self.histogramas_path}/superposicion_{number}.png", x, y, h=height)

        pdf.output(f"{self.path}/{disco}_malla.pdf", "F")

    def build_histograma_pdf(self):
        from natsort import natsorted
        from fpdf import FPDF
        pdf = FPDF()
        pdf.set_font('Arial', 'B', 16)
        from tqdm import tqdm
        import glob
        figures = glob.glob(f"{self.histogramas_path}/*.png", recursive=True)
        xx = [inter.x for inter in self.lista_intersecciones ]
        yy = [inter.y for inter in self.lista_intersecciones ]
        plt.figure(figsize=(10,10))
        plt.imshow(self.img)
        plt.scatter(xx,yy,s=2, zorder=10)
        plt.savefig(f"{self.path}/scatter_dots.png")
        plt.close()

        #try:
        disco = self.histogramas_path.parts[-2]
        disco = disco.replace("_","")
        pdf.add_page()
        pdf.cell(40, 10, disco)

        x,y = 0,0
        height = 220

        pdf.add_page()
        pdf.image(f"{self.path}/scatter_dots.png", x, y, h=height)
        x, y = 0, 0
        height = 150
        figuras = []
        for pagina in ['moda.png','distribuciones.png','distribuciones_agrupadas.png','grafo.png']:
            figuras +=[fig for fig in figures if pagina in fig]
        pdf.add_page()
        pdf.image(figuras[0], x, y, h=height)
        figuras.pop(0)
        height = 300
        for fig in tqdm(figuras):
            # if Path(fig).name not in ['moda.png','distribuciones.png','distribuciones_agrupadas.png','grafo.png']:
            #     continue
            pdf.add_page()
            pdf.image(fig, x, y, h=height)


        pdf.output(f"{self.path}/{disco}_histograma_distancias.pdf", "F")

    @staticmethod
    def calcular_moda_intersecciones_por_rayo(lista_intersecciones):
        from scipy import stats

        data = []
        for direccion in np.arange(0,360,360/Nr):
            intersecciones = [inter for inter in lista_intersecciones if inter.rayo_id == direccion]
            data.append(len(intersecciones))

        moda = stats.mode(data).mode[0]

        return moda


    def crop_image_centered_on_cell(self,img_dibujo,celda,region_zoom = 1/10):
        M,N,_ = img_dibujo.shape
        center = celda.centroid
        ymin = int(np.maximum(0,center.y-M*region_zoom))
        ymax = int(np.minimum(M-1, center.y + M * region_zoom))
        xmin = int(np.maximum(0,center.x-N*region_zoom))
        xmax = int(np.minimum(N-1, center.x + N * region_zoom))
        return img_dibujo[ymin:ymax,xmin:xmax]

    def obtener_ratio_diagonal_del_soporte_celda(self,celda):
        soporte = celda.obtener_soporte_completo()
        data = []
        for c in soporte:
            ratio = self.calcular_ratio_entre_diagonales(*c.lista_intersecciones_vertice)
            data.append(ratio)

        return np.array(data)

    def take_snapshot(self,dir_path,celda,celda_nueva,texto):
        self.contador += 1
        dir_path.mkdir(parents=True, exist_ok=True)
        img_dibujo = self.img.copy()
        img_blanco = np.ones_like(img_dibujo,dtype=np.uint8)*255
        img_dibujo = Dibujar.put_text(f"{celda.celda_id}", img_dibujo, (100, 50), fontScale=2)
        color = (255, 0, 0)
        img_dibujo = Dibujar.rellenar(celda, img_dibujo, color)
        color = (0,255,0)
        img_dibujo = Dibujar.rellenar(celda_nueva, img_dibujo, color)
        img_dibujo = self.dibujar_contornos_celdas(self.lista_celdas, img_dibujo)
        cropped_image = self.crop_image_centered_on_cell(img_dibujo,celda)
        color = (255, 0, 0)
        img_blanco = Dibujar.rellenar(celda, img_blanco, color)
        color = (0, 255, 0)
        img_blanco = Dibujar.rellenar(celda_nueva, img_blanco, color)
        img_blanco = self.dibujar_contornos_celdas(self.lista_celdas, img_blanco)
        cropped_image_blanco = self.crop_image_centered_on_cell(img_blanco, celda)
        ratio_celda_nueva = self.calcular_ratio_entre_diagonales(*celda_nueva.lista_intersecciones_vertice)

        filename = f'{dir_path}/{self.contador}.png'
        im_h = cv2.hconcat([cropped_image, cropped_image_blanco])
        M,N,_ = im_h.shape
        label = np.ones((100,N,3),dtype = np.uint8)*255
        im_v = cv2.vconcat([label, im_h])
        im_v = Dibujar.put_text(f"{ratio_celda_nueva:.2f}<{self.ratio_medio-3*self.ratio_std:.2f} . {ratio_celda_nueva<self.ratio_medio-3*self.ratio_std}.",im_v,(10,100),fontScale=1)
        im_v = Dibujar.put_text(f"{texto} {celda.celda_id}",im_v,(10,50),fontScale=1)

        cv2.imwrite(filename, im_v)






    def calcular_estadisticos(self,lista_celdas):
        datos = []
        for celda in tqdm(lista_celdas,'Calculo Estadisticos'):
            if not celda.tipo == VERDE:
                ratio = self.calcular_ratio_entre_diagonales(*celda.lista_intersecciones_vertice)
                datos.append(ratio)
        media,std = np.mean(datos), np.std(datos)
        import matplotlib.pyplot as plt
        # using the variable ax for single a Axes
        filename = f"{str(self.path)}/histograma_ratio_diagonales.png"
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.hist(datos)
        ax.axvline(media, c='r')
        ax.axvline(media + std, c='r')
        ax.axvline(media - std, c='r')
        plt.savefig(filename)
        plt.close(fig)

        return media,std,datos





    def procesar_celdas_verdes(self,lista_celdas,lista_intersecciones):
        #TODO: Cortar celdas verdes porque no se pueden intersectar con celdas rojas.
        # En esta primera etapa se soluciona el problema del origen. Celdas Verdes. Estas celdas, con mas de 3 intersecciones se pueden dividir interpolando. De esta manera se vana tener celdas mas chicas generadas por el ruido.


        return None

    def detectar_celdas_negras(self,lista_celdas,lista_intersecciones,SOPORTE_MINIMO=22):
        """Celdas negras son el soporte de mi teladearaña. Inamovibles"""
        for celda in tqdm(lista_celdas,'Deteccion Celdas Negras'):
            if celda.tipo in [ VERDE,NEGRO]:
                continue
            soporte_celda_1 = celda.calcular_soporte_lateral()
            soporte_celda_2 = celda.calcular_soporte_lateral(lateral='2')
            if soporte_celda_2+soporte_celda_1>=SOPORTE_MINIMO:
                celda.tipo = NEGRO
                self.pintar_todas_las_celdas_del_soporte(celda,NEGRO)


        return None

    def pintar_todas_las_celdas_del_soporte(self,celda,color):
        lista_celdas_soporte = celda.obtener_soporte_completo()
        celda.tipo = color
        for celda_sop in lista_celdas_soporte:
            celda_sop.tipo = color
        return None

    def buscar_extremo_soporte_celda(self,celda,extremo):
        lista_celdas_soporte = celda.obtener_soporte_lateral(lateral=extremo)
        lista_celdas_soporte = [c for c in lista_celdas_soporte if c.tipo == NEGRO]
        if len(lista_celdas_soporte)==0:
            return celda
        return lista_celdas_soporte[-1]

    def calcular_proporcion(self,lateral,int):
        if (lateral[1].x - lateral[0].x) != 0:
            proporcion = (int.x - lateral[0].x) / (lateral[1].x - lateral[0].x)
        elif (lateral[1].y - lateral[0].y) != 0:
            proporcion = (int.y - lateral[0].y) / (lateral[1].y - lateral[0].y)
        else:
            print(f"lateral: {lateral} x {lateral[1].x - lateral[0].x} y {lateral[1].y - lateral[0].y}")
            raise
        return proporcion

    @staticmethod
    def calcular_ratio_entre_diagonales(n_11,n_12,n_22,n_21):
        d1 = n_11.distance(n_22)
        d2 = n_12.distance(n_21)
        d_mayor = np.maximum(d1,d2)
        d_menor = np.minimum(d1,d2)
        ratio = d_menor / d_mayor
        return ratio

    @staticmethod
    def determinar_vertices_nueva_celda(siguiente,int_partida,int_cand,celda):
        c_int11, c_int12, c_int22, c_int21 = siguiente.lista_intersecciones_vertice
        index_partida = celda.lista_intersecciones_vertice.index(int_partida)
        if index_partida == 0:
            n_11, n_12, n_22, n_21 = int_cand, int_partida, c_int22, c_int21

        elif index_partida == 1:
            n_11, n_12, n_22, n_21 = int_partida, int_cand, c_int21, c_int22

        elif index_partida == 2:
            n_11, n_12, n_22, n_21 = c_int11, c_int12, int_cand, int_partida

        else:
            n_11, n_12, n_22, n_21 = c_int11, c_int12, int_partida, int_cand

        return n_11, n_12, n_22, n_21

    def es_siguiente_celda_proporcional_ratio_entre_diagonales(self,celda,siguiente,int_partida,int_cand,umbral=3):
        n_11, n_12, n_22, n_21 = self.determinar_vertices_nueva_celda(siguiente,int_partida,int_cand,celda)
        ratio_s = self.calcular_ratio_entre_diagonales(n_11, n_12, n_22, n_21)
        ratio_c = self.calcular_ratio_entre_diagonales(*celda.lista_intersecciones_vertice)
        sup = self.ratio_medio+umbral*self.ratio_std
        inf = self.ratio_medio-umbral*self.ratio_std
        validacion = sup>=ratio_s>=inf
        if False:
            img_dibujo = self.img.copy()
            img_dibujo = Dibujar.put_text(f"{sup:.2f} {ratio_s:.2f} {inf:.2f}",img_dibujo,(100,50),fontScale=1)
            color = (255, 0, 0) if validacion else (0,255,0)
            img_dibujo = Dibujar.rellenar(celda, img_dibujo, color)
            img_dibujo =self.dibujar_contornos_celdas(self.lista_celdas, img_dibujo)
            lista_nuevas = [inter for inter in celda.lista_intersecciones_vertice]
            for inter in [n_11, n_12, n_22, n_21]:
                if inter not in lista_nuevas:
                    lista_nuevas.append(inter)
            img_dibujo = Dibujar.intersecciones(lista_nuevas, img_dibujo,color=(0,255,0))
            import time
            cv2.imwrite(f'{self.path}/{time.time()}.png', img_dibujo)

        return validacion


    def seleccionar_celda_siguiente_de_vecindad(self,extremo,celda):
        int11, int12, int22, int21 = celda.lista_intersecciones_vertice

        if extremo in '1':
            int_inf = int11
            int_sup = int21
            # lista_celdas_radio = [c for c in lista_celdas if c.rayo_2 == celda.rayo_1]
            lista_candidatas = celda.vecindad_1
        else:
            int_inf = int12
            int_sup = int22
            # lista_celdas_radio = [c for c in lista_celdas if c.rayo_1 == celda.rayo_2]
            lista_candidatas = celda.vecindad_2

        siguiente = None
        for candidata in lista_candidatas:
            c_int11, c_int12, c_int22, c_int21 = candidata.lista_intersecciones_vertice
            if extremo in '1':
                if int_sup == c_int22 or int_inf == c_int12:
                    siguiente = candidata
                    break
            else:
                if int_sup == c_int21 or int_inf == c_int11:
                    siguiente = candidata
                    break
        return siguiente


    def celdas_estan_acopladas(self,celda,lateral,extremo):
        int11, int12, int22, int21 = celda.lista_intersecciones_vertice
        c_int11, c_int12, c_int22, c_int21 = lateral.lista_intersecciones_vertice
        if extremo in '1':
            int_inf = int11
            int_sup = int21
            # matcheo en todos los vertices con celda situiente.Da igual que vertice propago
            if int_sup == c_int22 and int_inf == c_int12:
                return True

        else:
            int_inf = int12
            int_sup = int22
            if int_sup == c_int21 and int_inf == c_int11:
                return True

        return False

    def criterio_mantener_proporcion_lateral(self,lateral_nuevo,int_cand,proporcion_viejo,umbral=1/5):
            proporcion_candidata = self.calcular_proporcion(lateral_nuevo,int_cand)
            return proporcion_viejo+umbral>= proporcion_candidata >=proporcion_viejo-umbral

    @staticmethod
    def crear_nueva_interseccion_proporcional(proporcion,lateral_nuevo,int_partida):
        # crear nueva interseccion
        y_nuevo = proporcion * (lateral_nuevo[1].y - lateral_nuevo[0].y) + lateral_nuevo[0].y
        x_nuevo = proporcion * (lateral_nuevo[1].x - lateral_nuevo[0].x) + lateral_nuevo[0].x
        int_nuevo = Interseccion(x_nuevo, y_nuevo, int_partida.curva_id, lateral_nuevo[0].rayo_id)
        return int_nuevo

    @staticmethod
    def buscar_interseccion_de_partida(celda,siguiente,extremo):
        int11, int12, int22, int21 = celda.lista_intersecciones_vertice
        c_int11, c_int12, c_int22, c_int21 = siguiente.lista_intersecciones_vertice
        if extremo in '1':
            int_inf = int11
            int_sup = int21
        else:
            int_inf = int12
            int_sup = int22

        #buscar interseccion de partida
        if int_sup not in siguiente.lista_intersecciones_vertice:
            int_partida = int_sup
        else:
            int_partida = int_inf

        return int_partida

    def buscar_interseccion_de_propagacion(self,celda,extremo,siguiente):
        int_partida = self.buscar_interseccion_de_partida(celda,siguiente,extremo)

        if extremo in '1':
            lateral_nuevo = [siguiente.lista_intersecciones_vertice[0], siguiente.lista_intersecciones_vertice[-1]]
            lateral = [siguiente.lista_intersecciones_vertice[1], siguiente.lista_intersecciones_vertice[2]]
        else:
            lateral_nuevo = [siguiente.lista_intersecciones_vertice[1], siguiente.lista_intersecciones_vertice[2]]
            lateral = [siguiente.lista_intersecciones_vertice[0], siguiente.lista_intersecciones_vertice[-1]]


        proporcion = self.calcular_proporcion(lateral,int_partida)
        inter_anomalas = siguiente.get_intersecciones_anomalas(extremo)
        if len(inter_anomalas)== 0 :
            #No hay intersecciones libres. Hay que crear nueva.
            int_nuevo = self.crear_nueva_interseccion_proporcional(proporcion,lateral_nuevo,int_partida)
        else:
            #hay intersecciones libres. Que hago? uso una inter libre o creo una nueva interseccion?
            #1.0 Selecciono interseccion libre mas parecida a interseccion de partida
            distancias = []
            for inter in inter_anomalas:
                proporcion_int = self.calcular_proporcion(lateral_nuevo,inter)
                distancias.append(np.abs(proporcion_int-proporcion))
            int_cand = inter_anomalas[np.argmin(distancias)]

            if self.criterio_mantener_proporcion_lateral(lateral_nuevo,int_cand,proporcion):
                validacion_proporcion = self.es_siguiente_celda_proporcional_ratio_entre_diagonales(celda,
                                                                                                    siguiente,
                                                                                                    int_partida,
                                                                                                    int_cand)
                if validacion_proporcion:
                    int_nuevo  = int_cand
                else:
                    #no pasa validacion
                    return int_cand,int_partida,False

            else:
                # crear nueva interseccion
                int_nuevo = self.crear_nueva_interseccion_proporcional(proporcion, lateral_nuevo,int_partida)


        return int_nuevo, int_partida,True







    def propagar_por_extremo_celda(self, celda, extremo,lista_celdas,lista_intersecciones):
        if celda.calcular_soporte_lateral(lateral=extremo) >= 359:
            # Anillo ya esta completo
            self.pintar_todas_las_celdas_del_soporte(celda,NARANJA)
            return
        celda_extremo = self.buscar_extremo_soporte_celda(celda, extremo)
        ctr = -1
        while not (celda_extremo == celda):

            ctr+=1
            if ctr>=360:
                #Evitar quedar infinitamente dando vueltas. Esto es un caso de error
                raise "Error"
                break

            if (extremo in '1' and len(celda_extremo.vecindad_1) > 1) or\
                (extremo in '2' and len(celda_extremo.vecindad_2) > 1):

                break


            celda_vecina = self.seleccionar_celda_siguiente_de_vecindad(extremo,celda_extremo)
            if celda_vecina is None:
                # Tengo celda siguiente que me divide a mi mismo
                #self.take_snapshot(self.debug_path / "union", celda_extremo)
                if self.debug:
                    if ( extremo in '1' and len(celda_extremo.vecindad_1) >0 ) or ( extremo in '2' and len(celda_extremo.vecindad_2) > 0 ):
                        celda_siguiente = celda_extremo.vecindad_1[0] if extremo in '1' else celda_extremo.vecindad_2[0]
                        self.take_snapshot(self.debug_path / "resto", celda_extremo,celda_siguiente,'Flotante')
                    else:
                        self.take_snapshot(self.debug_path / "resto", celda_extremo, celda_extremo, 'Vacio')
                break

            acopladas = self.celdas_estan_acopladas(celda_extremo,celda_vecina,extremo)
            if acopladas:
                if extremo in '2':
                    celda_vecina.celda_lateral_1 = celda_extremo
                    celda_vecina.vecindad_1 = [celda_extremo]
                else:
                    celda_vecina.celda_lateral_2 = celda_extremo
                    celda_vecina.vecindad_2 = [celda_extremo]

                if celda_vecina.tipo ==NEGRO:
                    celda_siguiente = self.buscar_extremo_soporte_celda(celda_vecina, extremo)
                    if celda_siguiente == celda_extremo:
                        #Anillo Completo
                        break
                else:
                    celda_siguiente = celda_vecina
                debug = 'Acopladas'

            else:
                if celda_vecina.tipo == NEGRO:
                    # TODO En principio si la celda es NEGRO, entonces es buena celda.Fin
                    break

                inter1, inter2,valida =  self.buscar_interseccion_de_propagacion(celda_extremo,extremo,celda_vecina)
                if not valida:
                    #Interseccion existente no cumple criterio global
                    if self.debug:
                        debug='No pasa test global'
                        n_11, n_12, n_22, n_21 = self.determinar_vertices_nueva_celda(celda_vecina, inter2, inter1,
                                                                                      celda_extremo)
                        celda_aux = Celda(-1,n_11,n_12,n_22,n_21,-1,-1)
                        self.take_snapshot(self.debug_path / "resto", celda_extremo, celda_aux, debug)

                    break


                else:
                    tamaño_celda_siguiente_equivalente_actual = False
                    if tamaño_celda_siguiente_equivalente_actual:
                        #TODO:
                        #1.0 mover interseccion un poco para abajo
                        #2.0 ir a celda siguiente
                        print("todo")

                    celda_superior, celda_inferior = self.celda_op.dividir(inter1, inter2, lista_celdas, celda_vecina,extremo,lista_intersecciones)
                    if extremo in '1':
                        celda_siguiente = celda_superior if celda_superior in celda_extremo.vecindad_1 else celda_inferior
                    else:
                        celda_siguiente = celda_superior if celda_superior in celda_extremo.vecindad_2 else celda_inferior
                    debug = 'Dividir'



            if self.debug: self.take_snapshot(self.debug_path / "resto", celda_extremo,celda_siguiente,debug)

            celda_extremo.tipo = NEGRO
            celda_extremo = celda_siguiente


        return

    def procesar_celdas(self,lista_celdas,lista_intersecciones,SOPORTE_MINIMO=22):
        """Celdas negras son el soporte de mi teladearaña. Inamovibles"""
        rango_angulos = np.arange(0,360,360/Nr)
        for direccion in tqdm(rango_angulos,'Procesamiento Celdas Negras'):
            celdas_direccion = [celda for celda in lista_celdas if celda.rayo_1 == direccion]
            for celda in celdas_direccion:
                if not (celda.tipo == NEGRO):
                    continue
                if celda.recorrida_en_iteracion:
                    continue
                # Celdas es Negra
                extremo='1'
                self.propagar_por_extremo_celda(celda, extremo,lista_celdas,lista_intersecciones)
                extremo = '2'
                self.propagar_por_extremo_celda(celda, extremo,lista_celdas,lista_intersecciones)


        return None
    def dibujar_celdas(self,lista_celdas,img):
        img_dibujo = img.copy()
        total_colores = 100
        colores = np.random.randint(255, size=(total_colores, 3)).astype(int)
        for idx,celda in enumerate(lista_celdas):
            img_dibujo = Dibujar.rellenar(celda,img_dibujo,tuple(colores[idx%total_colores]))

        cv2.imwrite(f'{self.path}/celdas.png', img_dibujo)


    def dibujar_contornos_celdas(self,lista_celdas,img,nombre=None):
        img_dibujo = img.copy()
        celdas_negras = [celda for celda in lista_celdas if celda.tipo == NEGRO]
        celdas_resto = [celda for celda in lista_celdas if celda.tipo not in [NEGRO]]
        for celda in celdas_resto:
            img_dibujo = Dibujar.contorno(celda,img_dibujo)

        for celda in celdas_negras:
            img_dibujo = Dibujar.contorno(celda,img_dibujo)
        if nombre is not None:
            cv2.imwrite(f'{self.path}/{nombre}', img_dibujo)
        return img_dibujo

    def dibujar_curvas_rayos_e_intersecciones(self,img):
        img_dibujo = img.copy()
        for rayo in self.lista_rayos:
            img_dibujo = Dibujar.rayo(rayo,img_dibujo)

        for curva in self.lista_curvas:
            img_dibujo = Dibujar.curva(curva,img_dibujo)

        img_dibujo = Dibujar.intersecciones(self.lista_intersecciones,img_dibujo)

        cv2.imwrite(f'{self.path}/curvas_rayos_e_intersecciones.png',img_dibujo)

    def convertir_formate_lista_curvas(self,lista_curvas):
        lista_nuevo_formate_curvas = []
        for curve in lista_curvas:
            if curve.get_size()<2:
                continue
            lista_pixeles = [(pix.x,pix.y) for pix in curve.pixels_list]
            lista_nuevo_formate_curvas.append(Curva(lista_pixeles, curve.id))
        return lista_nuevo_formate_curvas

    def obtener_interseccion_por_direccion(self,lista_intersecciones,rayo_id):
        inters = [inter for inter in lista_intersecciones if inter.rayo_id == rayo_id]
        inters.sort(key=lambda x: distancia_entre_pixeles(self.centro[0], self.centro[1],x.y,x.x))
        return inters

    def construir_rayos(self,Nr,M,N,centro):
        """

        @param Nr: cantidad radios
        @param M: altura imagen
        @param N: ancho imagen
        @param centro: (y,x)
        @return: lista rayos
        """
        rango_angulos = np.arange(0, 360, 360 / Nr)
        lista_rayos = [Rayo(direccion,centro,M,N) for direccion in rango_angulos]
        return lista_rayos



    def construir_intersecciones(self,lista_rayos,lista_curvas):
        bolsa_intersecciones = []
        for  rayo in lista_rayos:
            for curva in lista_curvas:
                inter = rayo.intersection(curva)
                if not inter.is_empty:
                    if 'MULTIPOINT' in inter.wkt:
                        inter = inter[0]
                    x,y = inter.xy
                    bolsa_intersecciones.append(Interseccion(x=np.array(x)[0],y=np.array(y)[0],curva_id=int(curva.id),rayo_id=int(rayo.direccion)))

        return bolsa_intersecciones


    def construir_celdas(self,lista_rayos,lista_intersecciones):
        lista_celdas = []
        contador_celdas = 0
        for rayo_origen in tqdm(lista_rayos,'Construyendo Celdas'):
            origen_intersecciones = self.obtener_interseccion_por_direccion(lista_intersecciones,rayo_origen.direccion)
            siguiente_intersecciones = self.obtener_interseccion_por_direccion(lista_intersecciones,(rayo_origen.direccion+1)%self.Nr)
            if len(origen_intersecciones)==0:
                continue
            o_inter = origen_intersecciones[0]
            primera_celda_direccion_radial = True
            while True:
                idx = origen_intersecciones.index(o_inter)
                ultima_interseccion_del_radio = idx>=len(origen_intersecciones)-1
                if ultima_interseccion_del_radio:
                    break

                sig_inter = [inter for inter in siguiente_intersecciones if inter.curva_id == o_inter.curva_id]
                if len(sig_inter) == 0:
                    o_inter = origen_intersecciones[idx+1]
                    continue

                sig_inter = sig_inter[0]

                if primera_celda_direccion_radial:
                    primera_celda_direccion_radial = False
                    centro_interseccion = Interseccion(self.centro[1],self.centro[0],None,None)
                    lista_intersecciones_perimetrales = []
                    radio_2_origen_index = siguiente_intersecciones.index(sig_inter)
                    lista_intersecciones_perimetrales += [inter for aux_idx,inter in enumerate(siguiente_intersecciones) if radio_2_origen_index>=aux_idx]
                    radio_1_origen_index = origen_intersecciones.index(o_inter)
                    lista_intersecciones_perimetrales += [inter for aux_idx,inter in enumerate(origen_intersecciones) if radio_1_origen_index>= aux_idx]
                    celda_triangular = Celda(contador_celdas,centro_interseccion,None,o_inter,sig_inter,rayo_origen.direccion,(rayo_origen.direccion+1)%360,lista_inter_perimetrales=lista_intersecciones_perimetrales)
                    lista_celdas.append(celda_triangular)
                    contador_celdas+=1

                idx_siguiente_borde = 1
                while True:
                    no_quedan_mas_intersecciones_radio_origen = (idx_siguiente_borde + idx)>len(origen_intersecciones) - 1
                    if no_quedan_mas_intersecciones_radio_origen:
                        break
                    o_inter_siguiente = origen_intersecciones[idx+idx_siguiente_borde]
                    sig_inter_siguiente  = [inter for inter in siguiente_intersecciones if inter.curva_id == o_inter_siguiente.curva_id]
                    if len(sig_inter_siguiente) > 0:
                        sig_inter_siguiente = sig_inter_siguiente[0]
                        break
                    idx_siguiente_borde+=1


                if no_quedan_mas_intersecciones_radio_origen:
                    break
                celda_es_normal = siguiente_intersecciones.index(sig_inter_siguiente) == (siguiente_intersecciones.index(sig_inter) + 1 ) and \
                            origen_intersecciones.index(o_inter_siguiente) == (origen_intersecciones.index(o_inter) + 1)
                lista_intersecciones_perimetrales = []
                if not celda_es_normal:
                    radio_2_siguiente_index = siguiente_intersecciones.index(sig_inter_siguiente)
                    radio_2_origen_index = siguiente_intersecciones.index(sig_inter)
                    lista_intersecciones_perimetrales += [inter for aux_idx,inter in enumerate(siguiente_intersecciones) if radio_2_origen_index<= aux_idx <= radio_2_siguiente_index]
                    radio_1_siguiente_index = origen_intersecciones.index(o_inter_siguiente)
                    radio_1_origen_index = origen_intersecciones.index(o_inter)
                    lista_intersecciones_perimetrales += [inter for aux_idx,inter in enumerate(origen_intersecciones) if radio_1_origen_index<= aux_idx <= radio_1_siguiente_index]

                # hay nueva celda
                celda = Celda(contador_celdas,o_inter,sig_inter,sig_inter_siguiente,o_inter_siguiente,rayo_origen.direccion,(rayo_origen.direccion+1)%360,lista_inter_perimetrales = lista_intersecciones_perimetrales)
                lista_celdas.append(celda)
                contador_celdas+=1
                o_inter = o_inter_siguiente

        for rayo_origen in tqdm(lista_rayos,'Configurando Vecindad'):
            celdas_rayo_1 = [celda for celda in lista_celdas if celda.rayo_2 == rayo_origen.direccion]
            celdas_rayo_2 = [celda for celda in lista_celdas if celda.rayo_1 == (rayo_origen.direccion+1) % 360]
            celdas_rayo = [celda for celda in lista_celdas if celda.rayo_1 == rayo_origen.direccion]
            for celda in celdas_rayo:
                celda.set_vecindad(celdas_rayo_1, celdas_rayo_2,self.centro)



        return lista_celdas
    # def gui_spyder_web(self):
    #     #TODO: using click event to mark cells

def main(datos):
    M,N,img,centro,SAVE_PATH,sigma = datos['M'], datos['N'], datos['img'], datos['centro'], datos['save_path'] , datos['sigma']
    lista_curvas = datos['lista_curvas']
    save_path = datos['save_path']
    spyder = SpyderWeb(Nr=Nr, img=img, lista_curvas=lista_curvas, centro=centro[::-1], save_path=save_path)
    listaPuntos = []
    centro_id = np.max(np.unique([inter.curva_id for inter in spyder.lista_intersecciones]))+1
    for inter in spyder.lista_intersecciones:
        i, j, angulo, radio = inter.y , inter.x, inter.rayo_id, inter.radio(centro[::-1])
        #params={'x':np.uint16(i),'y':np.uint16(j),'angulo':angulo,'radio':radio,'gradFase':-1,'cadenaId':inter.curva_id}
        params = {'x': i, 'y': j, 'angulo': angulo, 'radio': radio, 'gradFase': -1,
                  'cadenaId': inter.curva_id}
        punto = ch.Punto(**params)
        if punto not in listaPuntos:
            listaPuntos.append(punto)

    #agregar puntos artificiales pertenecientes al centro
    for angulo in np.arange(0,360,360/Nr):
        params = {'x': centro[1], 'y': centro[0], 'angulo': angulo, 'radio': 0, 'gradFase': -1,
                  'cadenaId': centro_id }
        punto = ch.Punto(**params)
        listaPuntos.append(punto)


    listaCadenas = ch.asignarCadenas(listaPuntos, centro[::-1], M, N,centro_id=centro_id)


    #####
    # for cadena in listaCadenas:
    #     puntos_otro_id = [punto.cadenaId for punto in cadena.lista if punto.cadenaId != cadena.id ]
    #     if len(puntos_otro_id) > 0:
    #         raise
    ###
    listaCadenas, listaPuntos, MatrizEtiquetas = ch.renombrarCadenas(listaCadenas, listaPuntos, M, N)
    # ch.visualizarCadenasSobreDisco(
    #     listaCadenas, img, "cadenas", labels=False, color='r', save=SAVE_PATH
    # )
    # ch.visualizarCadenasSobreDisco(
    #     listaCadenas, img, "cadenas_color", labels=True, save=SAVE_PATH
    # )

    datos['listaCadenas'] = listaCadenas
    datos['listaPuntos'] = listaPuntos

    return 0

if __name__=='__main__':
    from skimage.exposure import  equalize_adapthist

    image_file_name = "/media/data/maestria/datasets/FOTOS_DISCOS_1/segmentadas/F2A_CUT_up.tif"

    #image_file_name = "/media/data/maestria/datasets/artificiales/segmentadas/example_10.tif"
    image = cv2.imread(image_file_name)
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    sigma = 4.5
    high=15
    low=5
    centro = [1204,1264]
    #centro = [500,500]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #img_seg = utils.segmentarImagen(gray, debug=False)
    img_eq = equalize_adapthist(np.uint8(gray), clip_limit=0.03)
    detector = devernayEdgeDetector(10,img_eq,centro[::-1],"./", sigma=sigma,highthreshold=high,lowthreshold=low)
    img_bin,bin_sin_pliegos,thetaMat ,nonMaxImg,gradientMat,Gx,Gy,img_labels,listaCadenas,listaPuntos,lista_curvas = detector.detect()
    spyder = SpyderWeb(Nr=360,img=image,lista_curvas=lista_curvas,centro=centro[::-1],save_path=".")
