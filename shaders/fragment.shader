#version 310 es
${HEADER_INCLUDE("FRAGMENT")}



out vec4 fragColor;
void main(void) {
	vec2 centeredCoords = gl_FragCoord.xy - vec2(float(CANVAS_WIDTH) / 2.0, float(CANVAS_HEIGHT) / 2.0);


	uvec2 hexCoords = uvec2(mod(vec2(UNIVERSE_WIDTH, UNIVERSE_HEIGHT) + mat2(
		2.0/3.0, -1.0/3.0,
		0, sqrt(3.0)/3.0
	) * centeredCoords, vec2(UNIVERSE_WIDTH, UNIVERSE_HEIGHT)));

	uvec2 index = idx(hexCoords.x, hexCoords.y);
	
	fragColor = colors[GET_CELL(0, index)];
}