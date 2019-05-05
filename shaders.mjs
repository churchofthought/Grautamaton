import constants from "./constants.mjs"

const HEADER_INCLUDE = `
#define UNIVERSE_WIDTH ${constants.UNIVERSE_WIDTH}
#define UNIVERSE_HEIGHT ${constants.UNIVERSE_HEIGHT}
struct Cell {
	uint x;
};

layout(std430, binding = 0) buffer UniverseBufferData {
	Cell[UNIVERSE_WIDTH][UNIVERSE_HEIGHT] cells;
} universe[2];
`

const vertexShaderSource = `#version 310 es
	in vec2 vertexCoords;
	void main(void) {
		gl_Position = vec4(vertexCoords, 0.0, 1.0);
	}`

const fragmentShaderSource = `#version 310 es

	${HEADER_INCLUDE}
	
	out highp vec4 fragColor;
	void main(void) {
		uint x = uint(gl_FragCoord.x);
		uint y = uint(gl_FragCoord.y);

		highp float col = float(universe[0].cells[x][y].x) / float(10);
		fragColor = vec4(col, col, col, 1.0);
	}`


const transitionFunc = direction => {
	return `
		#undef PREV
		#undef NEXT
		#define PREV universe[${direction ^ 0}].cells
		#define NEXT universe[${direction ^ 1}].cells
		
		// NEXT[x][y].x = (
		// 	C(PREV).x  + 
		// 	N(PREV).x  +
		// 	NE(PREV).x +
		// 	SE(PREV).x +
		// 	S(PREV).x  +
		// 	SW(PREV).x +
		// 	NW(PREV).x
		// ) / uint(6);
	`
}

const computeShaderSource = `#version 310 es

	layout( local_size_x = 1 ) in;

	${HEADER_INCLUDE}

	#define C(u) u[x][y]
	#define N(u) u[x][ym1]
	#define NE(u) u[xp1][ym1]
	#define SE(u) u[xp1][y]
	#define S(u) u[x][yp1]
	#define SW(u) u[xm1][yp1]
	#define NW(u) u[xm1][y]

	void main(void){
		uint x = gl_GlobalInvocationID.x;
		uint y = gl_GlobalInvocationID.y;
		uint xm1 = (x == uint(0) ? uint(UNIVERSE_WIDTH) : x) - uint(1);
		uint xp1 = (x == uint(UNIVERSE_WIDTH - 1) ? uint(-1) : x) + uint(1);
		uint ym1 = (y == uint(0) ? uint(UNIVERSE_HEIGHT) : y) - uint(1);
		uint yp1 = (y == uint(UNIVERSE_HEIGHT - 1) ? uint(-1) : y) + uint(1);
		${transitionFunc(0)}
		memoryBarrier();
		${transitionFunc(1)}
	}`

// console.log(computeShaderSource)
export default gl => {

	const createShader = (program, sourceCode, type) => {
		// Compiles either a shader of type gl.VERTEX_SHADER or gl.FRAGMENT_SHADER
		const shader = gl.createShader(type)
		gl.shaderSource(shader, sourceCode)
		gl.compileShader(shader)

		if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
			throw `Could not compile WebGL shader. ${gl.getShaderInfoLog(shader)}`

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
			[vertexShaderSource, gl.VERTEX_SHADER],
			[fragmentShaderSource, gl.FRAGMENT_SHADER]
		]),
		computeProgram: createProgram([
			[computeShaderSource, gl.COMPUTE_SHADER]
		])
	}
}

