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
	
	uint index = (x % uint(UNIVERSE_WIDTH)) * uint(UNIVERSE_HEIGHT) + (y % uint(UNIVERSE_HEIGHT));

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

			SET_CELL(NEXT, C, transition(GET_CELL(PREV, C), uint[](
				GET_CELL(PREV, idxes[0]),
				GET_CELL(PREV, idxes[1]),
				GET_CELL(PREV, idxes[2]),
				GET_CELL(PREV, idxes[3]),
				GET_CELL(PREV, idxes[4]),
				GET_CELL(PREV, idxes[5])
			)));
	`
}

const computeShaderSource = `#version 310 es

	layout( local_size_x = 1 ) in;

	${HEADER_INCLUDE("COMPUTE")}

	bool[14] rule = bool[](
		false, false,
		false, false,
		false, false,
		true, true,
		false, true,
		false, true,
		false, true
	);

	bool transition(uint center, uint[6] neighborhood){
		uint sum = ( 
			neighborhood[0]
			+ neighborhood[1]
			+ neighborhood[2]
			+ neighborhood[3]
			+ neighborhood[4] 
			+ neighborhood[5]
		);

		return rule[sum * uint(2) + center];
	}

	void main(void){
		uint x = gl_GlobalInvocationID.x;
		uint y = gl_GlobalInvocationID.y;

		uvec2 C = idx(x, y);

		uvec2[6] idxes = uvec2[](
			// 1-away
			// 6 count
			idx(x+uint(0),y-uint(1)),
			idx(x+uint(1),y-uint(1)),
			idx(x+uint(1),y+uint(0)),
			idx(x+uint(0),y+uint(1)),
			idx(x-uint(1),y+uint(1)),
			idx(x-uint(1),y+uint(0))

			// 2-uint(away)
			// 12 count
			// idx(x+uint(0),y-uint(2)),
			// idx(x+uint(1),y-uint(2)),
			// idx(x+uint(2),y-uint(2)),
			// idx(x+uint(2),y-uint(1)),
			// idx(x+uint(2),y+uint(0)),
			// idx(x+uint(1),y+uint(1)),

			// idx(x+uint(0),y+uint(2)),
			// idx(x-uint(1),y+uint(2)),
			// idx(x-uint(2),y+uint(2)),
			// idx(x-uint(2),y+uint(1)),
			// idx(x-uint(2),y+uint(0)),
			// idx(x-uint(1),y-uint(1))
		);
		
		

		${transitionFunc(0)}
		//barrier();
		//memoryBarrierBuffer();
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

