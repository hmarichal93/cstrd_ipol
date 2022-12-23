import numpy as np
import cv2



class Color:
    """BGR"""
    yellow = (0, 255, 255)
    red = (0, 0, 255)
    blue = (255, 0, 0)
    dark_yellow = (0, 204, 204)
    cyan = (255, 255, 0)
    orange = (0, 165, 255)
    purple = (255, 0, 255)
    maroon = (34, 34, 178)
    green = (0, 255, 0)

    def __init__(self):
        self.list = [Color.yellow, Color.red,Color.blue, Color.dark_yellow, Color.cyan,Color.orange,Color.purple,Color.maroon]
        self.idx = 0

    def get_next_color(self):
        self.idx = (self.idx + 1 ) % len(self.list)
        return self.list[self.idx]



class Dibujar:
    @staticmethod
    def put_text(text, image, org, color = (0, 0, 0), fontScale = 1 / 4):
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

    @staticmethod
    def intersecciones(lista_intersecciones,img,color=(0,0,255)):
        lista_ptos = [(int(inter.y),int(inter.x)) for inter in lista_intersecciones]
        pts = np.array(lista_ptos)
        img[pts[:,0],pts[:,1],:] = color

        return img

    @staticmethod
    def intersecciones_size(lista_intersecciones,img,color=(0,0,255),size=2):
        lista_ptos = [(int(inter.y),int(inter.x)) for inter in lista_intersecciones]

        for y,x in lista_ptos:
             cv2.circle(img=img, center=(x,y), radius=int(size), color=color, thickness=-1)

        return img
    @staticmethod
    def intersecciones_con_tamaño(lista_intersecciones,img,color=(0,0,255),thickness=1):
        lista_ptos = [(int(inter.y),int(inter.x),size) for inter,size in lista_intersecciones]
        for y,x,size in lista_ptos:
             cv2.circle(img=img, center=(x,y), radius=int(size), color=color, thickness=thickness)
        return img

    @staticmethod
    def distribuciones(lista_distribuciones,img,color=(0,0,255),thickness=1):
        lista_ptos = [(int(inter.y),int(inter.x),inter.std,inter.rayo_id) for inter in lista_distribuciones]
        for y,x,size,rayo_id in lista_ptos:
             cv2.circle(img=img, center=(x,y), radius=thickness, color=color, thickness=-1)
        return img

    @staticmethod
    def curva(curva,img,color=(0,255,0),thickness = 2):
        y, x = curva.xy
        y = np.array(y).astype(int)
        x = np.array(x).astype(int)
        pts = np.vstack((y,x)).T
        isClosed=False
        img = cv2.polylines(img, [pts],
                              isClosed, color, thickness)

        return img

    @staticmethod
    def cadena( cadena, img, color = (0,255,0),thickness = 5):
        y, x = cadena.getDotsCoordinates()
        pts = np.vstack((x, y)).T.astype(int)
        isClosed = False
        img = cv2.polylines(img, [pts],
                            isClosed, color, thickness)

        return img
    @staticmethod
    def lista_cadenas(listaCadenas, img, color=None, labels=False):
        M, N, _ = img.shape
        colors_length = 20
        np.random.seed(10)
        # colors = np.random.randint(low=100, high=255, size=(colors_length, 3), dtype=np.uint8)
        colors = Color()
        color_idx = 0
        thickness = 5
        for cadena in listaCadenas:
            if color is None:
                color_tuple = Color.green
            else:
                color_tuple = colors.get_next_color()

            img = Dibujar.cadena(cadena,img,color_tuple,thickness)

            color_idx = (color_idx + 1) % colors_length

        if labels:
            for cadena in listaCadenas:
                org = cadena.extA
                img = Dibujar.put_text(str(cadena.label_id), img, (int(org.y), int(org.x)), fontScale=1.5)

        return img


    @staticmethod
    def rayo(rayo,img, color=(255, 0, 0),debug=False,thickness=2):
        y, x = rayo.xy
        y = np.array(y).astype(int)
        x = np.array(x).astype(int)
        start_point = (y[0], x[0])
        end_point = (y[1], x[1])
        image = cv2.line(img, start_point, end_point, color, thickness)
        if debug:
            image = self.put_text(str(int(rayo.direccion)),image,((y[1]+y[0])//2,(x[1]+x[0])//2))
        return image
    @staticmethod
    def contorno(celda,img, color=(255, 0, 0)):
        x,y = celda.exterior.coords.xy
        pts = np.vstack((x,y)).T.astype(int)


        img = cv2.polylines(img, [pts],
                         True, color,
                         thickness=1)
        return img

    @staticmethod
    def rellenar(celda,img,color):
        x,y = celda.exterior.coords.xy
        pts = np.vstack((x,y)).T.astype(int)
        color = (int(color[0]), int(color[1]), int(color[2]))
        img = cv2.fillPoly(img,pts=[pts],color=tuple(color))
        return img