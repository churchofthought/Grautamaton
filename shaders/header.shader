${c.cDefines}

#define CELL_TYPE ivec4

#define GET_CELL(dir, index) (universe[dir].cells[index])

#define SET_CELL(dir, index, val) universe[dir].cells[index] = val;


precision highp float;
precision highp int;



layout(std430, binding = 0) coherent restrict ${type == "FRAGMENT" ? "readonly" : ""} buffer UniverseBufferData {
	CELL_TYPE[] cells;
} universe[2];

layout(std140, binding = 2) uniform meta {
	uint time;
};



uint omod(uint x, int o, uint y){
  return (x + uint(o) + y) % y;
}

uint idx(uint x, uint y){
	return x * uint(UNIVERSE_HEIGHT) + y;
}


int gcd(int u, int v) {
	u = abs(u);
	v = abs(v);
	// While loop is not always allowed, use a for loop.
	do {
		if (v == 0) break;
		u = u % v;
		if (u == 0) break;
		v = v % u;
	} while (true);

	return u + v;
}