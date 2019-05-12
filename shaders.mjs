import constants from "./constants.mjs"

const HEADER_INCLUDE = (type) => `
#define CANVAS_WIDTH ${constants.CANVAS_WIDTH}
#define CANVAS_HEIGHT ${constants.CANVAS_HEIGHT}
#define UNIVERSE_WIDTH ${constants.UNIVERSE_WIDTH}
#define UNIVERSE_HEIGHT ${constants.UNIVERSE_HEIGHT}
#define UNIVERSE_SIZE ${constants.UNIVERSE_SIZE}

precision highp float;
precision highp int;

layout(std430, binding = 0) coherent restrict ${type == "FRAGMENT" ? "readonly" : ""} buffer UniverseBufferData {
	uint[UNIVERSE_SIZE / 32] cells;
} universe[2];

uvec2 idx(uint x, uint y){
	
	uint index = x * uint(UNIVERSE_HEIGHT) + y;

	uint intIndex = index / uint(32);
	uint bitIndex = index - (intIndex * uint(32));

	return uvec2(intIndex, bitIndex);
}

#define GET_CELL(u, index) ((u[index.x] & (uint(1) << index.y)) >> index.y)

#define SET_CELL(u, index, val) (val ? atomicOr(u[index.x], uint(1) << index.y) : atomicAnd(u[index.x], ~(uint(1) << index.y)));

`

const vertexShaderSource = `#version 310 es
	in vec2 position;
	
	void main(void) {
		gl_Position = vec4(position, 0.0, 1.0);
	}`

const fragmentShaderSource = `#version 310 es

	${HEADER_INCLUDE("FRAGMENT")}



	out vec4 fragColor;
	void main(void) {
		vec2 centeredCoords = gl_FragCoord.xy - vec2(float(CANVAS_WIDTH) / 2.0, float(CANVAS_HEIGHT) / 2.0);


		uvec2 hexCoords = uvec2(mod(vec2(UNIVERSE_WIDTH, UNIVERSE_HEIGHT) + mat2(
			2.0/3.0, -1.0/3.0,
			0, sqrt(3.0)/3.0
		) * centeredCoords, vec2(UNIVERSE_WIDTH, UNIVERSE_HEIGHT)));

		uvec2 index = idx(hexCoords.x, hexCoords.y);

		fragColor = GET_CELL(universe[0].cells, index) == uint(0)
			? vec4(0.0,0.0,0.0,1.0) 
			: vec4(1.0,1.0,1.0,1.0);
	}`

const transitionFunc = direction => {
	return `
			#undef PREV
			#define PREV universe[${direction ^ 0}].cells
			
			#undef NEXT
			#define NEXT universe[${direction ^ 1}].cells

			SET_CELL(NEXT, C, transition(uint[7](
				GET_CELL(PREV, N),
				GET_CELL(PREV, NE),
				GET_CELL(PREV, SE),
				GET_CELL(PREV, S),
				GET_CELL(PREV, SW),
				GET_CELL(PREV, NW),
				GET_CELL(PREV, C)
			)));
	`
}

const computeShaderSource = `#version 310 es

	layout( local_size_x = 1 ) in;

	${HEADER_INCLUDE("COMPUTE")}

	#define True true
	#define False false
	bool[128] rule = bool[](
		True, False, True, False, True, False, True, False, True, False, \
		True, True, True, False, False, True, True, False, False, True, True, \
		True, False, True, True, False, False, True, False, True, True, \
		False, True, False, True, True, False, True, False, True, True, True, \
		True, False, False, True, False, False, True, False, False, True, \
		False, True, False, True, False, True, False, False, True, False, \
		True, False, True, False, True, False, True, True, False, True, \
		False, True, False, True, False, True, True, False, True, True, \
		False, True, True, False, False, False, False, True, False, True, \
		False, False, True, False, True, False, False, True, False, True, \
		True, False, False, True, False, False, False, True, True, False, \
		False, True, True, False, False, False, True, False, True, False, \
		True, False, True, False, True, False
	);

	bool transition(uint[7] neighborhood){
		uint no = (neighborhood[0] << uint(6)) +
							(neighborhood[1] << uint(5)) +
							(neighborhood[2] << uint(4)) +
							(neighborhood[3] << uint(3)) +
							(neighborhood[4] << uint(2)) +
							(neighborhood[5] << uint(1)) +
							(neighborhood[6]);
		return rule[no];
	}

	void main(void){
		uint x = gl_GlobalInvocationID.x;
		uint y = gl_GlobalInvocationID.y;

		uint xm1 = x == uint(0) ? uint(UNIVERSE_WIDTH - 1) : x - uint(1);
		uint xp1 = x == uint(UNIVERSE_WIDTH - 1) ? uint(0) : x + uint(1);
		uint ym1 = y == uint(0) ? uint(UNIVERSE_HEIGHT - 1) : y - uint(1);
		uint yp1 = y == uint(UNIVERSE_HEIGHT - 1) ? uint(0) : y + uint(1);

		uvec2 N = idx(x,ym1);
		uvec2 NE = idx(xp1,ym1);
		uvec2 SE = idx(xp1,y);
		uvec2 S = idx(x,yp1);
		uvec2 SW = idx(xm1,yp1);
		uvec2 NW = idx(xm1,y);
		uvec2 C = idx(x, y);

		
		

		${transitionFunc(0)}
		barrier();
		memoryBarrierBuffer();
		${transitionFunc(1)}
	}`

const lined = (s) => s.split(/[\r\n]+/).map((x, i) => `${i}. ${x}`).join("\n")

// console.log(
// 	lined(fragmentShaderSource) + "\n=====================================\n" + lined(computeShaderSource)
// )
export default gl => {

	const createShader = (program, sourceCode, type) => {
		// Compiles either a shader of type gl.VERTEX_SHADER or gl.FRAGMENT_SHADER
		const shader = gl.createShader(gl[type])
		gl.shaderSource(shader, sourceCode)
		gl.compileShader(shader)

		if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
			throw `Could not compile WebGL ${type} shader. ${gl.getShaderInfoLog(shader)}`

		gl.attachShader(program, shader)

		return shader
	}

	const createProgram = shaders => {
		const program = gl.createProgram()

		for (const [source, type] of shaders)
			createShader(program, source, type)

		gl.linkProgram(program)

		if (!gl.getProgramParameter(program, gl.LINK_STATUS))
			throw `Could not compile WebGL program. ${gl.getProgramInfoLog(program)}`

		return program
	}

	return {
		renderProgram: createProgram([
			[vertexShaderSource, "VERTEX_SHADER"],
			[fragmentShaderSource, "FRAGMENT_SHADER"]
		]),
		computeProgram: createProgram([
			[computeShaderSource, "COMPUTE_SHADER"]
		])
	}
}

