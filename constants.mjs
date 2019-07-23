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
		return (x,y) => [
			x * 2.0/3.0,
			x * -1.0/3.0 + y * 1.0/Math.sqrt(3.0)
		]
	default:
		return (x,y) => [x,y]
	}
})()
Object.defineProperty(c, "PROJECTOR", { value: projector, enumerable: false })


c.CANVAS_WIDTH = 100 
c.CANVAS_HEIGHT = 100


// use projector to scale the universe to minimum size
const t1 = projector(c.CANVAS_WIDTH, 0)
const t2 = projector(0, c.CANVAS_HEIGHT)
c.UNIVERSE_WIDTH = Math.floor(t1[0])
c.UNIVERSE_HEIGHT = Math.floor(t2[1])





c.UNIVERSE_SIZE = c.UNIVERSE_WIDTH * c.UNIVERSE_HEIGHT

c.CELL_BITS = 64

c.UNIVERSE_BIT_SIZE = c.UNIVERSE_SIZE * c.CELL_BITS
c.UNIVERSE_BYTE_SIZE = Math.ceil(c.UNIVERSE_BIT_SIZE / 8)
c.UNIVERSE_INT_SIZE = Math.ceil(c.UNIVERSE_BYTE_SIZE / 4)

c.cDefines = Object.entries(c).map(([x, y]) => `#define ${x} (${JSON.stringify(y)})`).join("\n")

export default c