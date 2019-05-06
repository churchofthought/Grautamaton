import constants from "./constants.mjs"

const HEADER_INCLUDE = (type) => `
#define CANVAS_WIDTH ${constants.CANVAS_WIDTH}
#define CANVAS_HEIGHT ${constants.CANVAS_HEIGHT}
#define UNIVERSE_WIDTH ${constants.UNIVERSE_WIDTH}
#define UNIVERSE_HEIGHT ${constants.UNIVERSE_HEIGHT}

precision highp float;
precision highp int;

struct Cell {
	uint x;
};

layout(std430, binding = 0) coherent restrict ${type == "FRAGMENT" ? "readonly" : ""} buffer UniverseBufferData {
	Cell[UNIVERSE_WIDTH][UNIVERSE_HEIGHT] cells;
} universe[2];

layout(std430, binding = 2) restrict ${type == "FRAGMENT" ? "readonly" : "coherent"} buffer UniverseRenderData {
	uint colorMax;
};

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


		vec2 hexCoord = mod(vec2(UNIVERSE_WIDTH, UNIVERSE_HEIGHT) + mat2(
			2.0/3.0, -1.0/3.0,
			0, sqrt(3.0)/3.0
		) * centeredCoords, vec2(UNIVERSE_WIDTH, UNIVERSE_HEIGHT));

		uint v = universe[0].cells[uint(hexCoord.x)][uint(hexCoord.y)].x;

		float col = pow(float(v) / float(colorMax), 0.1);
		fragColor = vec4(col, col, col, 1.0);
	}`


const transitionFunc = direction => {
	return `
		#undef PREV
		#undef NEXT
		#undef VAL
		#define PREV universe[${direction ^ 0}].cells
		#define NEXT universe[${direction ^ 1}].cells
		#define VAL val${direction}

		uint VAL = 
			C(PREV).x - (C(PREV).x >= uint(6) ? uint(6) : uint(0)) +
			(N(PREV).x >= uint(6) ? uint(1) : uint(0)) +
			(NE(PREV).x >= uint(6) ? uint(1) : uint(0)) +
			(SE(PREV).x >= uint(6) ? uint(1) : uint(0)) +
			(S(PREV).x >= uint(6) ? uint(1) : uint(0)) +
			(SW(PREV).x >= uint(6) ? uint(1) : uint(0)) +
			(NW(PREV).x >= uint(6) ? uint(1) : uint(0));

		#if ${direction} == 1
			atomicMax(colorMax, VAL);
		#endif

		C(NEXT).x = VAL;
		
		

	`
}

const computeShaderSource = `#version 310 es

	layout( local_size_x = 1 ) in;

	${HEADER_INCLUDE("COMPUTE")}

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
		if (x == uint(0) && y == uint(0)){
			colorMax = uint(1);
		}
		barrier();

		uint xm1 = x == uint(0) ? uint(UNIVERSE_WIDTH - 1) : x - uint(1);
		uint xp1 = x == uint(UNIVERSE_WIDTH - 1) ? uint(0) : x + uint(1);
		uint ym1 = y == uint(0) ? uint(UNIVERSE_HEIGHT - 1) : y - uint(1);
		uint yp1 = y == uint(UNIVERSE_HEIGHT - 1) ? uint(0) : y + uint(1);
		${transitionFunc(0)}
		barrier();
		memoryBarrierBuffer();
		${transitionFunc(1)}
	}`

const lined = (s) => s.split(/[\r\n]+/).map((x, i) => `${i}. ${x}`).join("\n")

console.log(
	lined(fragmentShaderSource) + "\n=====================================\n" + lined(computeShaderSource)
)
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

