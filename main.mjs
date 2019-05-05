import createPrograms from "./shaders.mjs"
import setupBlitting from "./blitting.mjs"
import constants from "./constants.mjs"



window.onload = () => {
	const canvas = document.querySelector("#glCanvas")
	const gl = canvas.getContext("webgl2-compute")

	// Only continue if WebGL is available and working
	if (!gl)
		throw "Unable to initialize WebGL2 Compute. Your browser or machine may not support it."

	// gl.viewport(0, 0, canvas.width, canvas.height)
	// gl.clearColor(0, 0.5, 0, 1)
	// gl.clear(gl.COLOR_BUFFER_BIT)

	const {renderProgram, computeProgram} = createPrograms(gl)
	
	setupBlitting(gl, renderProgram)

	const bindUniverse = idx => {
		const buffer = gl.createBuffer()
		gl.bindBuffer(gl.SHADER_STORAGE_BUFFER, buffer)
		gl.bufferData(gl.SHADER_STORAGE_BUFFER, constants.UNIVERSE_BYTE_SIZE, gl.DYNAMIC_DRAW)
		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, idx, buffer)
	}

	[1,0].forEach(bindUniverse)
	
	const render = () => {
		gl.useProgram(computeProgram)
		gl.dispatchCompute(constants.UNIVERSE_WIDTH, constants.UNIVERSE_HEIGHT, 1)
		gl.memoryBarrier(gl.SHADER_STORAGE_BARRIER_BIT)
		gl.useProgram(renderProgram)
		gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
		requestAnimationFrame(render)
	}

	render()

	const universe = new Uint32Array(constants.UNIVERSE_SIZE)
	var mouseX = 0, mouseY = 0
	const updateMouse = e => {
		mouseX = e.offsetX
		mouseY = e.offsetY
	}
	const mousedownHandler = () => {
		gl.getBufferSubData(gl.SHADER_STORAGE_BUFFER, 0, universe)
		++universe[mouseX * constants.UNIVERSE_HEIGHT + (constants.UNIVERSE_HEIGHT - mouseY)]
		gl.bufferSubData(gl.SHADER_STORAGE_BUFFER, 0, universe)
	}
	var mousedownInterval = 0
	canvas.onmousemove = updateMouse
	canvas.onmousedown = e => {
		updateMouse(e)
		mousedownInterval = setInterval(mousedownHandler, 10)
	}
	window.onmouseup = () => {
		clearInterval(mousedownInterval)
	}
}


