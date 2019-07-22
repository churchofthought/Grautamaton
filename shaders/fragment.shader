#version 310 es
${HEADER_INCLUDE("FRAGMENT")}


//layout(origin_upper_left, pixel_center_integer) in vec4 gl_FragCoord;

out vec3 fragColor;

vec3 hsl2rgb(float x, float y, float z)
{
    vec3 rgb = clamp(
			abs(
				mod(3.0 - x * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0
			) - 1.0,
		0.0, 1.0);

    return z + y * (rgb - 0.5) * (1.0 - abs(2.0 * z - 1.0));
}

void main(void) {
	vec2 coords = gl_FragCoord.xy;

		${(() => { 
		switch (c.NEIGHBORHOOD_TYPE){
			case "moore":
				return `
					// do nothing
				`
				break;
			case "hex":
				return `
					coords = mat2(
						2.0/3.0, -1.0/3.0,
						0.0, 1.0/sqrt(3.0)
					) * coords;
				`
		}
	})()}
	
	coords = mod(round(coords), vec2(UNIVERSE_WIDTH, UNIVERSE_HEIGHT));
	uint index = idx(uint(coords.x), uint(coords.y));
	
	CELL_TYPE z;
	if ((time & 1u) == 1u){
		z = GET_CELL(0, index);
	}else{
		z = GET_CELL(1, index);
	}

	float val = zAbs(z);
	float angle = zArg(z);

	float fMinVal = IFloatFlip(minVal);
	float fMaxVal = IFloatFlip(maxVal);
	float scaled = (val - fMinVal) / (fMaxVal - fMinVal);

	fragColor = hsl2rgb(
		angle, 1.0, pow(scaled, 0.25) //pow(0.25, scaled)
	);
}