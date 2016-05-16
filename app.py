import cv2
import numpy as np
import asyncio
import json
from asyncio import queues, as_completed
from aiohttp import web
from aiohttp.web import Application, Response


class FaceRecognizer:

    def __init__(self, port, address, cascade):
        self._port = port
        self._address = address
        self._cascade = cascade
        self._images = queues.Queue()

    @asyncio.coroutine
    def getlastimage(self):
        ''' returns element from the image queue '''
        image = yield from self._images.get()
        return image

    @asyncio.coroutine
    def _addimage(self, image):
        ''' keeps track of the last three images with recognitzed objects '''
        print("Number of items in the queue: {}".format(self._images.qsize()))
        yield from self._images.put(image)

    def _cascade_detect(self, raw_image):
        ''' use opencv cascades to recognize objects on the incomming images '''
        cascade = cv2.CascadeClassifier(self._cascade)
        image = np.asarray(bytearray(raw_image), dtype="uint8")

        gray_image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        color_image = cv2.imdecode(image, cv2.IMREAD_ANYCOLOR)

        coordinates = cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in coordinates:
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print("({}, {}, {}, {})".format(x, y, w, h))

        return color_image, self._tojson(coordinates)

    @staticmethod
    def _tojson(coordinates):
        ''' creates a valid json output of the returned coordinates '''
        json_rep = []
        for (x, y, w, h) in coordinates:
            json_rep.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)})
        return json.dumps(json_rep)

    @asyncio.coroutine
    def _returnfaces(self, request):
        ''' returnes the processed images with the detected artifacts highlighted '''
        try:

            for f in as_completed([self.getlastimage()], timeout=1):
                image = yield from f

            image_buf = cv2.imencode('.jpg', image)[1]
            image_str = np.array(image_buf).tostring()

        except asyncio.TimeoutError as te:
            return Response(
                text='Timeout error occurred - there is probably no image in the buffer right now',
                status=500,
                content_type='application/json'
            )


        return Response(
            body=image_str,
            status=200,
            content_type='image/jpeg'
        )

    @asyncio.coroutine
    def _detectface(self, request):
        ''' execute the object recognition in a different thread to keep the app responsive '''

        data = yield from request.post()

        input_image = data['img'].file
        image = input_image.read()

        loop = asyncio.get_event_loop()

        #run the face recognition in a different thread
        future = loop.run_in_executor(None, self._cascade_detect, image)

        image, coordinates = yield from future

        yield from self._addimage(image)

        return web.Response(text=coordinates,
                            status=200,
                            content_type='application/json'
                            )

    @asyncio.coroutine
    def init(self, loop):
        app = Application(loop=loop)
        app.router.add_route('GET', '/', self._returnfaces)
        app.router.add_route('POST', '/detectface', self._detectface)


        handler = app.make_handler()
        srv = yield from loop.create_server(handler, self._address, self._port)
        print("Server started on host {} / port {}".format(self._address, self._port))
        return srv, handler


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    detectface = FaceRecognizer(8080, '0.0.0.0', 'haarcascade_frontalface_default.xml')
    srv, handler = loop.run_until_complete(detectface.init(loop))

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        loop.run_until_complete(handler.finish_connections())