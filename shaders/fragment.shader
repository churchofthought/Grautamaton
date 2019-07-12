#version 310 es
${HEADER_INCLUDE("FRAGMENT")}


//layout(origin_upper_left, pixel_center_integer) in vec4 gl_FragCoord;

out vec4 fragColor;

void main(void) {
		${(() => { 
		switch (c.NEIGHBORHOOD){
			case "moore":
				return `
					vec2 projectedCoords = gl_FragCoord.xy;
				`

			case "hex":
				return `
					vec2 projectedCoords = mat2(
						2.0/3.0, -1.0/3.0,
						0, sqrt(3.0)/3.0
					) * gl_FragCoord.xy;
				`
		}
	})()}
	
	vec2 relativeCoords = projectedCoords / vec2(CANVAS_WIDTH, CANVAS_HEIGHT);
	uvec2 absCoords = uvec2(relativeCoords * vec2(UNIVERSE_WIDTH, UNIVERSE_HEIGHT));
	
	uvec2 index = idx(absCoords.x, absCoords.y);
	
	if ((time & 1u) == 1u){
		fragColor = colors[GET_CELL(1, index)];
	}else{
		fragColor = colors[GET_CELL(0, index)];
	}
}