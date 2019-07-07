#define CANVAS_WIDTH ${constants.CANVAS_WIDTH}
#define CANVAS_HEIGHT ${constants.CANVAS_HEIGHT}
#define UNIVERSE_WIDTH ${constants.UNIVERSE_WIDTH}
#define UNIVERSE_HEIGHT ${constants.UNIVERSE_HEIGHT}
#define UNIVERSE_SIZE ${constants.UNIVERSE_SIZE}
#define UNIVERSE_BYTE_SIZE ${constants.UNIVERSE_BYTE_SIZE}

#define BLACK uint(0)
#define RED uint(1)
#define GREEN uint(2)
#define BLUE uint(3)

#define GET_CELL(dir, index) ((universe[dir].cells[index.x] & (uint(3) << index.y)) >> index.y)
#define SET_CELL(dir, index, val) \
	atomicAnd(universe[dir].cells[index.x], ~(uint(3) << index.y)); \
	atomicOr(universe[dir].cells[index.x], val << index.y);

#define COUNT(neighborhood, val) ( \
  (neighborhood[0] == val ? uint(1) : uint(0)) + \
  (neighborhood[1] == val ? uint(1) : uint(0)) + \
  (neighborhood[2] == val ? uint(1) : uint(0)) + \
  (neighborhood[3] == val ? uint(1) : uint(0)) + \
  (neighborhood[4] == val ? uint(1) : uint(0)) + \
  (neighborhood[5] == val ? uint(1) : uint(0)) \
)




precision highp float;
precision highp int;



layout(std430, binding = 0) coherent restrict ${type == "FRAGMENT" ? "readonly" : ""} buffer UniverseBufferData {
	uint[UNIVERSE_BYTE_SIZE / 4] cells;
} universe[2];



uvec2 idx(uint x, uint y){
	uint index = uint(2) * ((x % uint(UNIVERSE_WIDTH)) * uint(UNIVERSE_HEIGHT) + (y % uint(UNIVERSE_HEIGHT)));

	uint intIndex = index / uint(32);
	uint bitIndex = index - (intIndex * uint(32));

	return uvec2(intIndex, bitIndex);
}



vec4 colors[4] = vec4[](
	vec4(0.0,0.0,0.0,1.0),
	vec4(1.0,0.0,0.0,1.0),
	vec4(0.0,1.0,0.0,1.0),
	vec4(0.0,0.0,1.0,1.0)
);