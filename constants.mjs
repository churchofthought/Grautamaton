const logb = (n, base) => (Math.log(n) / Math.log(base))

const c = {}
c.CANVAS_WIDTH = 2048
c.CANVAS_HEIGHT = 2048

c.UNIVERSE_WIDTH = Math.floor(c.CANVAS_WIDTH / 2)
c.UNIVERSE_HEIGHT = Math.floor(c.CANVAS_HEIGHT / 2)

c.UNIVERSE_SIZE = c.UNIVERSE_WIDTH * c.UNIVERSE_HEIGHT

c.CELL_STATES = 4
c.CELL_BITS = logb(c.CELL_STATES, 2)

c.UNIVERSE_BIT_SIZE = c.UNIVERSE_SIZE * c.CELL_BITS
c.UNIVERSE_BYTE_SIZE = c.UNIVERSE_BIT_SIZE / 8

c.cDefines = Object.entries(c).map(([x,y]) => `#define ${x} (${JSON.stringify(y)})`).join("\n")

export default c