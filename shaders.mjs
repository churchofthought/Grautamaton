const vertexShaderSource = `
  attribute vec2 vertexCoords;
  void main(void) {
      gl_Position = vec4(vertexCoords, 0.0, 1.0);
  }`

const fragmentShaderSource = `
  void main() {
    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
	}`

const createShader = (program, gl, sourceCode, type) => {
	// Compiles either a shader of type gl.VERTEX_SHADER or gl.FRAGMENT_SHADER
	const shader = gl.createShader(type)
	gl.shaderSource(shader, sourceCode)
	gl.compileShader(shader)

	if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
		throw `Could not compile WebGL shader. ${gl.getShaderInfoLog(shader)}`

	gl.attachShader(program, shader)

	return shader
}

export default gl => {
	const program = gl.createProgram()

	createShader(program, gl, vertexShaderSource, gl.VERTEX_SHADER)
	createShader(program, gl, fragmentShaderSource, gl.FRAGMENT_SHADER)
	
	gl.linkProgram(program)

	if (!gl.getProgramParameter(program, gl.LINK_STATUS))
		throw `Could not compile WebGL program. ${gl.getProgramInfoLog(program)}`

	gl.useProgram(program)

	return program
}

