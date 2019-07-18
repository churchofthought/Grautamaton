${c.cDefines}

#define GET_CELL(dir, index) (universe[dir].cells[index])

#define SET_CELL(dir, index, val) universe[dir].cells[index] = val;


precision highp float;
precision highp int;



layout(std430, binding = 0) coherent restrict ${type == "FRAGMENT" ? "readonly" : ""} buffer UniverseBufferData {
	float[UNIVERSE_FLOAT_SIZE] cells;
} universe[2];

layout(std140) uniform meta {
	uint time;
};


uint omod(uint x, int o, uint y){
  return (x + uint(o) + y) % y;
}

uint idx(uint x, uint y){
	return x * uint(UNIVERSE_HEIGHT) + y;
}