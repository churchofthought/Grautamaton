${c.cDefines}

#define BLACK uint(0)
#define RED uint(1)
#define GREEN uint(2)
#define BLUE uint(3)

#define GET_CELL(dir, index) ((universe[dir].cells[index.x] & (uint(NUM_STATES - 1) << index.y)) >> index.y)

#define SET_CELL(dir, index, val) \
	atomicAnd(universe[dir].cells[index.x], ~(uint(NUM_STATES - 1) << index.y)); \
	atomicOr(universe[dir].cells[index.x], val << index.y);

#define COUNT(neighborhood, val) (${u.repeat(c.NUM_NEIGHBORS, i => `(neighborhood[${i}] == val ? uint(1) : uint(0))`, '+')})




precision highp float;
precision highp int;



layout(std430, binding = 0) coherent restrict ${type == "FRAGMENT" ? "readonly" : ""} buffer UniverseBufferData {
	uint[UNIVERSE_INT_SIZE] cells;
} universe[2];

layout(std140) uniform meta {
	uint time;
};

uvec2 idx(uint x, uint y){
	uint index = uint(CELL_BITS) * ((x % uint(UNIVERSE_WIDTH)) * uint(UNIVERSE_HEIGHT) + (y % uint(UNIVERSE_HEIGHT)));

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