#version 310 es
${HEADER_INCLUDE("FRAGMENT")}


//layout(origin_upper_left, pixel_center_integer) in vec4 gl_FragCoord;

out vec4 fragColor;

void main(void) {
	vec2 coords = gl_FragCoord.xy / vec2(CANVAS_WIDTH, CANVAS_HEIGHT);

		${(() => { 
		switch (c.NEIGHBORHOOD_TYPE){
			case "moore":
				return `
					// do nothing
				`
				break;
			case "hex":
				return `
					// coords = mat2(
					// 	sqrt(3.0)/3.0, -0,
					// 	-1.0/3.0, 2.0/3.0
					// ) * coords;
				`
		}
	})()}
	
	
	coords *= vec2(UNIVERSE_WIDTH, UNIVERSE_HEIGHT);

	uvec2 index = idx(int(coords.x), int(coords.y));
	
	if ((time & 1u) == 1u){
		fragColor = colors[GET_CELL(1, index)];
	}else{
		fragColor = colors[GET_CELL(0, index)];
	}
}