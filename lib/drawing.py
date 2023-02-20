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
    white = (255,255,255)
    black = (0,0,0)

    def __init__(self):
        self.list = [Color.yellow, Color.red,Color.blue, Color.dark_yellow, Color.cyan,Color.orange,Color.purple,Color.maroon]
        self.idx = 0

    def get_next_color(self):
        self.idx = (self.idx + 1 ) % len(self.list)
        return self.list[self.idx]



class Drawing:
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
    def intersection(dot, img, color=Color.red):
        img[int(dot.y),int(dot.x),:] = color

        return img





    @staticmethod
    def curve(curva, img, color=(0, 255, 0), thickness = 1):
        y, x = curva.xy
        y = np.array(y).astype(int)
        x = np.array(x).astype(int)
        pts = np.vstack((x,y)).T
        isClosed=False
        img = cv2.polylines(img, [pts],
                              isClosed, color, thickness)

        return img

    @staticmethod
    def chain(chain, img, color=(0, 255, 0), thickness=5):
        y, x = chain.get_nodes_coordinates()
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

            img = Drawing.chain(cadena, img, color_tuple, thickness)

            color_idx = (color_idx + 1) % colors_length

        if labels:
            for cadena in listaCadenas:
                org = cadena.extA
                img = Drawing.put_text(str(cadena.label_id), img, (int(org.y), int(org.x)), fontScale=1.5)

        return img


    @staticmethod
    def radii(rayo, img, color=(255, 0, 0), debug=False, thickness=1):
        y, x = rayo.xy
        y = np.array(y).astype(int)
        x = np.array(x).astype(int)
        start_point = (x[0], y[0])
        end_point = (x[1], y[1])
        image = cv2.line(img, start_point, end_point, color, thickness)

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