#version 310 es
${HEADER_INCLUDE("FRAGMENT")}


//layout(origin_upper_left, pixel_center_integer) in vec4 gl_FragCoord;

out vec4 fragColor;

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
	
	float val;
	if ((time & 1u) == 1u){
		val = GET_CELL(1, index);
	}else{
		val = GET_CELL(0, index);
	}

	fragColor = vec4(val, val, val, 1.0);
}