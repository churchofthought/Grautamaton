${constants.cDefines}

#define BLACK uint(0)
#define RED uint(1)
#define GREEN uint(2)
#define BLUE uint(3)

#define GET_CELL(dir, index) ((universe[dir].cells[index.x] & (uint(NUM_STATES - 1) << index.y)) >> index.y)
#define SET_CELL(dir, index, val) \
	atomicAnd(universe[dir].cells[index.x], ~(uint(NUM_STATES - 1) << index.y)); \
	atomicOr(universe[dir].cells[index.x], val << index.y);

#define COUNT(neighborhood, val) ( \
  (neighborhood[0] == val ? uint(1) : uint(0)) + \
  (neighborhood[1] == val ? uint(1) : uint(0)) + \
  (neighborhood[2] == val ? uint(1) : uint(0)) + \
  (neighborhood[3] == val ? uint(1) : uint(0)) + \
  (neighborhood[4] == val ? uint(1) : uint(0)) + \
  (neighborhood[5] == val ? uint(1) : uint(0)) + \
  (neighborhood[6] == val ? uint(1) : uint(0)) + \
  (neighborhood[7] == val ? uint(1) : uint(0)) \
)




precision highp float;
precision highp int;



layout(std430, binding = 0) coherent restrict ${type == "FRAGMENT" ? "readonly" : ""} buffer UniverseBufferData {
	uint[UNIVERSE_INT_SIZE] cells;
} universe[2];



uvec2 idx(uint x, uint y){
	x = (x + uint(UNIVERSE_WIDTH)) % uint(UNIVERSE_WIDTH);
	y = (y + uint(UNIVERSE_HEIGHT)) % uint(UNIVERSE_HEIGHT);

	uint index = uint(CELL_BITS) * (x * uint(UNIVERSE_HEIGHT) + y);

	uint intIndex = index / uint(32);
	uint bitIndex = index - (intIndex * uint(32));

	return uvec2(intIndex, bitIndex);
}



vec4 colors[NUM_STATES] = vec4[](
	vec4(0.0,0.0,0.0,1.0),
	vec4(1.0,0.0,0.0,1.0),
	vec4(0.0,1.0,0.0,1.0),
	vec4(0.0,0.0,1.0,1.0)
);