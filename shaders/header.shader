${c.cDefines}

#define BLACK 0u
#define RED 1u
#define GREEN 2u
#define BLUE 3u

#define GET_CELL(dir, index) ((universe[dir].cells[index.x] & (uint(NUM_STATES - 1) << index.y)) >> index.y)

#define SET_CELL(dir, index, val) \
	atomicAnd(universe[dir].cells[index.x], ~(uint(NUM_STATES - 1) << index.y)); \
	atomicOr(universe[dir].cells[index.x], val << index.y);

#define COUNT(neighborhood, val) (${u.repeat(c.NUM_NEIGHBORS, i => `(neighborhood[${i}] == val ? 1u : 0u)`, '+')})




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

	uint intIndex = index / 32u;
	uint bitIndex = index - (intIndex * 32u);

	return uvec2(intIndex, bitIndex);
}



vec4 colors[NUM_STATES] = vec4[](
	vec4(1.0,1.0,1.0,1.0),
	vec4(1.0,0.0,0.0,1.0),
	vec4(0.0,1.0,0.0,1.0),
	vec4(0.0,0.0,1.0,1.0)
);