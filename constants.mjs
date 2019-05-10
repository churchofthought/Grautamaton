const constants = {}
constants.CANVAS_WIDTH = 768
constants.CANVAS_HEIGHT = 768
constants.UNIVERSE_WIDTH = Math.floor(constants.CANVAS_WIDTH / 2)
constants.UNIVERSE_HEIGHT = Math.floor(constants.CANVAS_HEIGHT / 2)
constants.UNIVERSE_SIZE = constants.UNIVERSE_WIDTH * constants.UNIVERSE_HEIGHT
constants.UNIVERSE_BYTE_SIZE = constants.UNIVERSE_SIZE * 4

export default constants