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


	const safeMod = (x,y) => Math.floor((x + y) % y)
	const mousePosToArr = (x, y) => {

		//console.log(x,y)

		const centeredX = x - constants.CANVAS_WIDTH / 2
		const centeredY = constants.CANVAS_HEIGHT / 2 - y

		const u = safeMod(2.0/3.0 * centeredX, constants.UNIVERSE_WIDTH)
		const v = safeMod(-1.0/3.0 * centeredX + Math.sqrt(3.0)/3.0 * centeredY, constants.UNIVERSE_HEIGHT)


		//console.log(u,v)
		return u * constants.UNIVERSE_HEIGHT + v
	}

	const bindUniverse = idx => {
		const buffer = gl.createBuffer()
		gl.bindBuffer(gl.SHADER_STORAGE_BUFFER, buffer)

		const arr = new Uint32Array(constants.UNIVERSE_SIZE)
		//arr[constants.UNIVERSE_SIZE - 1] = 20000
		arr[0] = 3000000
		arr[mousePosToArr(600, 600)] = 3000000
		arr[mousePosToArr(300, 300)] = 3000000
		//arr[1] = 20000
		gl.bufferData(gl.SHADER_STORAGE_BUFFER, arr, gl.STATIC_DRAW)
		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, idx, buffer)
	}

	const bindRenderMeta = () => {
		const buffer = gl.createBuffer()
		gl.bindBuffer(gl.SHADER_STORAGE_BUFFER, buffer)
		gl.bufferData(gl.SHADER_STORAGE_BUFFER, 4, gl.STATIC_DRAW)
		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 2, buffer)
	}

	
	bindRenderMeta();
	[1,0].forEach(bindUniverse)
	
	const render = () => {
		gl.useProgram(computeProgram)
		for (var i = 128; i--;)
			gl.dispatchCompute(constants.UNIVERSE_WIDTH, constants.UNIVERSE_HEIGHT, 1)
		// gl.memoryBarrier(gl.SHADER_STORAGE_BARRIER_BIT | gl.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
		gl.useProgram(renderProgram)
		gl.drawArrays(gl.TRIANGLE_FAN, 0, 8)
		requestAnimationFrame(render)
	}

	render()

	const universeView = new Uint32Array(1)
	var mouseX = 0, mouseY = 0
	const updateMouse = e => {
		mouseX = e.offsetX
		mouseY = e.offsetY
	}
	const mousedownHandler = () => {
		const offset = 4 * mousePosToArr(mouseX, mouseY)
		gl.getBufferSubData(gl.SHADER_STORAGE_BUFFER, offset, universeView)
		++universeView[0]
		gl.bufferSubData(gl.SHADER_STORAGE_BUFFER, offset, universeView)
	}
	var mousedownInterval = 0
	canvas.onmousemove = updateMouse
	canvas.onmousedown = e => {
		updateMouse(e)
		mousedownInterval = setInterval(mousedownHandler, 0)
	}
	window.onmouseup = () => {
		clearInterval(mousedownInterval)
	}
}


