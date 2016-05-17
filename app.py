import cv2
import numpy as np
import asyncio
import json
import logging
from asyncio import queues, as_completed
from aiohttp import web
from aiohttp.web import Application, Response


class FaceRecognizer:

    def __init__(self, port, address, cascade):
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

        self._port = port
        self._address = address
        self._cascade = cascade
        self._images = queues.Queue(maxsize=3)


    @asyncio.coroutine
    def _getlastimage(self):
        ''' returns element from the image queue '''

        if not self._images.empty():
            image = yield from self._images.get()
            self._logger.info("getlastimage: Number of items still in the queue: {}".format(self._images.qsize()))
        else:
            # if empty raise QueueEmpty exception
            raise asyncio.QueueEmpty

        return image

    @asyncio.coroutine
    def _addimage(self, image):
        ''' keeps track of the last three images with recognized objects '''

        if not self._images.full():
            self._logger.info("addimage: new image added to the queue")
            yield from self._images.put(image)

        else:
            self._logger.info("addimage: queue is full right now")

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
            self._logger.debug("face recognized at: x: {}, y: {}, w: {}, h: {}".format(x, y, w, h))

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

            image = yield from self._getlastimage()

            image_buf = cv2.imencode('.jpg', image)[1]
            image_str = np.array(image_buf).tostring()

        except asyncio.QueueEmpty as qe:
            msg = 'QueueEmpty exception has been thrown. There is no image ' \
                  'with some recognized artifacts in the queue right now.'
            self._logger.warning(msg)
            return Response(
                text=msg,
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

        # just add a picture to the queue if faces have been recognized, filter for non empty json arrays
        if coordinates != '[]':
            yield from self._addimage(image)

        return web.Response(text=coordinates,
                            status=200,
                            content_type='application/json'
                            )

    @asyncio.coroutine
    def init(self, loop):
        app = Application(loop=loop)

        # configure stream handler for console
        ch = logging.StreamHandler()
        self._logger.addHandler(ch)

        app.router.add_route('GET', '/', self._returnfaces)
        app.router.add_route('POST', '/detectface', self._detectface)


        handler = app.make_handler()
        srv = yield from loop.create_server(handler, self._address, self._port)
        self._logger.info("Server started on host {} / port {}".format(self._address, self._port))
        return srv, handler


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    detectface = FaceRecognizer(8080, '0.0.0.0', 'frontalface.xml')
    srv, handler = loop.run_until_complete(detectface.init(loop))

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        loop.run_until_complete(handler.finish_connections())
