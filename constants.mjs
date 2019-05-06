const constants = {}
constants.CANVAS_WIDTH = 256
constants.CANVAS_HEIGHT = 256
constants.UNIVERSE_WIDTH = Math.floor(constants.CANVAS_WIDTH * Math.sqrt(3.0) / 2.0)
constants.UNIVERSE_HEIGHT = Math.floor(constants.CANVAS_HEIGHT * Math.sqrt(3.0) / 2.0)
constants.UNIVERSE_SIZE = constants.UNIVERSE_WIDTH * constants.UNIVERSE_HEIGHT
constants.UNIVERSE_BYTE_SIZE = constants.UNIVERSE_SIZE * 4

export default constants