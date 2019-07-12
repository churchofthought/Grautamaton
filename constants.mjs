import util from "./util.mjs"

const logb = (n, base) => Math.ceil(Math.log(n) / Math.log(base))

const c = {}

Object.defineProperty(c, "NEIGHBORHOOD", {value: "moore", enumerable: false})

c.CANVAS_WIDTH = 1024
c.CANVAS_HEIGHT = 1024

c.UNIVERSE_WIDTH = c.CANVAS_WIDTH
c.UNIVERSE_HEIGHT = c.CANVAS_HEIGHT


c.UNIVERSE_SIZE = c.UNIVERSE_WIDTH * c.UNIVERSE_HEIGHT

c.NUM_STATES = 4
c.CELL_BITS = logb(c.NUM_STATES, 2)

// make sure it does not go into header file
c.NUM_NEIGHBORS = util.neighborhoods[c.NEIGHBORHOOD]().length

c.UNIVERSE_BIT_SIZE = c.UNIVERSE_SIZE * c.CELL_BITS
c.UNIVERSE_BYTE_SIZE = Math.ceil(c.UNIVERSE_BIT_SIZE / 8)
c.UNIVERSE_INT_SIZE = Math.ceil(c.UNIVERSE_BYTE_SIZE / 4)

c.cDefines = Object.entries(c).map(([x,y]) => `#define ${x} (${JSON.stringify(y)})`).join("\n")

export default c