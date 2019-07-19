${c.cDefines}

#define GET_CELL(dir, index) (universe[dir].cells[index])

#define SET_CELL(dir, index, val) ${u.cmacro(`
	float v = val;
	uint flipped = FloatFlip(v); 
	atomicMin(minVal, flipped); 
	atomicMax(maxVal, flipped);
	universe[dir].cells[index] = v;
`)}


precision highp float;
precision highp int;



layout(std430, binding = 0) coherent restrict ${type == "FRAGMENT" ? "readonly" : ""} buffer UniverseBufferData {
	float[] cells;
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