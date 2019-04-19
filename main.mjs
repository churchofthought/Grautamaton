import createProgram from "./shaders.mjs"
import setupBlitting from "./blitting.mjs"

var gl


const render = () => {
	gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
	requestAnimationFrame(render)
}

window.onload = () => {
	const canvas = document.querySelector("#glCanvas")
	gl = canvas.getContext("webgl")

	// Only continue if WebGL is available and working
	if (!gl)
		throw "Unable to initialize WebGL. Your browser or machine may not support it."

	gl.viewport(0, 0, canvas.width, canvas.height)
	gl.clearColor(0, 0.5, 0, 1)
	gl.clear(gl.COLOR_BUFFER_BIT)

	const program = createProgram(gl)
	setupBlitting(gl, program)

	render()
}


