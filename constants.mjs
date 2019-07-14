const logb = (n, base) => Math.ceil(Math.log(n) / Math.log(base))

const c = {}

const neighbors = []
Object.defineProperty(c, "NEIGHBORHOOD_TYPE", { value: "hex", enumerable: false })
Object.defineProperty(c, "NEIGHBORHOOD", { value: neighbors, enumerable: false })
for (var x = -1; x <= 1; ++x) {
	for (var y = -1; y <= 1; ++y) {
		if (x == 0 && y == 0) continue
		if (x == y && c.NEIGHBORHOOD_TYPE == "hex") continue
		neighbors.push([x, y])
	}
}
c.NUM_NEIGHBORS = c.NEIGHBORHOOD.length

const projector = (() => {
	switch (c.NEIGHBORHOOD_TYPE) {
	case "hex":
		return (x,y) => [x,y]
	default:
		return (x,y) => [x,y]
	}
})()
Object.defineProperty(c, "PROJECTOR", { value: projector, enumerable: false })

c.CANVAS_WIDTH = 1024 
c.CANVAS_HEIGHT = 768

c.UNIVERSE_WIDTH = c.CANVAS_WIDTH
c.UNIVERSE_HEIGHT = c.CANVAS_HEIGHT





c.UNIVERSE_SIZE = c.UNIVERSE_WIDTH * c.UNIVERSE_HEIGHT

c.NUM_STATES = 4
c.CELL_BITS = logb(c.NUM_STATES, 2)

c.UNIVERSE_BIT_SIZE = c.UNIVERSE_SIZE * c.CELL_BITS
c.UNIVERSE_BYTE_SIZE = Math.ceil(c.UNIVERSE_BIT_SIZE / 8)
c.UNIVERSE_INT_SIZE = Math.ceil(c.UNIVERSE_BYTE_SIZE / 4)

c.cDefines = Object.entries(c).map(([x, y]) => `#define ${x} (${JSON.stringify(y)})`).join("\n")

export default c