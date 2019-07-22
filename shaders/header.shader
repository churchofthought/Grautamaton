${c.cDefines}

#define CELL_TYPE vec2

#define GET_CELL(dir, index) (universe[dir].cells[index])

#define SET_CELL(dir, index, val) ${u.cmacro(`
	CELL_TYPE z = val;
	float v = zAbs(z);
	uint flipped = FloatFlip(v); 
	atomicMin(minVal, flipped); 
	atomicMax(maxVal, flipped);
	universe[dir].cells[index] = z;
`)}


precision highp float;
precision highp int;



layout(std430, binding = 0) coherent restrict ${type == "FRAGMENT" ? "readonly" : ""} buffer UniverseBufferData {
	CELL_TYPE[] cells;
} universe[2];

layout(std430, binding = 2) coherent restrict ${type == "FRAGMENT" ? "readonly" : ""} buffer UniverseMetaData {
	uint minVal;
	uint maxVal;
};

layout(std140, binding = 3) uniform meta {
	uint time;
};


uint FloatFlip(float flt)
{
	uint f = floatBitsToUint(flt);
	uint mask = (-(f >> 31u)) | 0x80000000u;
	return f ^ mask;
}

float IFloatFlip(uint f)
{
	uint mask = ((f >> 31u) - 1u) | 0x80000000u;
	return uintBitsToFloat(f ^ mask);
}

uint omod(uint x, int o, uint y){
  return (x + uint(o) + y) % y;
}

uint idx(uint x, uint y){
	return x * uint(UNIVERSE_HEIGHT) + y;
}

#define PI 3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068
#define TWOPI 6.283185307179586476925286766559005768394338798750211641949889184615632812572417997256069650684234136

// complex number functions
float zAbs(vec2 c) {
  return sqrt(c.x * c.x + c.y * c.y);
}

float zArg(vec2 c){
	return 0.5 + atan(c.y, c.x) / TWOPI;
}