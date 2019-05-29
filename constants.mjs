const constants = {}
constants.CANVAS_WIDTH = 2048
constants.CANVAS_HEIGHT = 2048
constants.UNIVERSE_WIDTH = Math.floor(constants.CANVAS_WIDTH / 2)
constants.UNIVERSE_HEIGHT = Math.floor(constants.CANVAS_HEIGHT / 2)
constants.UNIVERSE_SIZE = constants.UNIVERSE_WIDTH * constants.UNIVERSE_HEIGHT
constants.UNIVERSE_BYTE_SIZE = (constants.UNIVERSE_SIZE * 2) / 8 

export default constants