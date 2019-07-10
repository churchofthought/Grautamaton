import createPrograms from "./shaders.mjs"
import setupBlitting from "./blitting.mjs"
import constants from "./constants.mjs"
import util from "./util.mjs"

// window.onresize = () => {
	
// }

window.onload = async () => {
	
	const container = document.createElement("div")

	const canvas = document.createElement("canvas")
	canvas.width = constants.CANVAS_WIDTH
	canvas.height = constants.CANVAS_HEIGHT
	

	const status = document.createElement("div")
	
	// set styles
	document.documentElement.style = document.body.style = `
		padding: 0;
		margin: 0;
		text-align: center;
	`

	container.style = `
		display:inline-block;
		position: relative;
		margin-left: auto;
		margin-right: auto;
	`

	status.style = `
		position: absolute;
		left: 0;
		top: 0;
		margin: 0.5em;
		z-index: 1;
		color: red;
		font-family: monospace;
		font-weight: bold;
		font-size: 1em;
	`

	container.appendChild(canvas)
	container.appendChild(status)
	document.body.appendChild(container)

	const gl = canvas.getContext("webgl2-compute")
	// Only continue if WebGL is available and working
	if (!gl)
		throw "Unable to initialize WebGL2 Compute. Your browser or machine may not support it."

	// gl.viewport(0, 0, canvas.width, canvas.height)
	// gl.clearColor(0, 0.5, 0, 1)
	// gl.clear(gl.COLOR_BUFFER_BIT)

	const { renderProgram, computeProgram } = await createPrograms(gl)

	setupBlitting(gl, renderProgram)

	const mousePosToArr = (x, y) => {

		const u = util.mod((x / constants.CANVAS_WIDTH), 1.0) * constants.UNIVERSE_WIDTH
		const v = util.mod(1.0 - y / constants.CANVAS_HEIGHT, 1.0) * constants.UNIVERSE_HEIGHT
		
		return u * constants.UNIVERSE_HEIGHT + v
	}

	const bindUniverse = idx => {
		const buffer = gl.createBuffer()
		gl.bindBuffer(gl.SHADER_STORAGE_BUFFER, buffer)

		const arr = new Uint8Array(constants.UNIVERSE_BYTE_SIZE)

		// for (var i = constants.UNIVERSE_BYTE_SIZE; i--;){
		// 	arr[i] = Math.floor(Math.random() * 256)
		// }

		//arr[constants.UNIVERSE_SIZE - 1] = 20000
		// for (var x = 0; x < constants.UNIVERSE_WIDTH; x++)
		// 	for (var y = 0; y < constants.UNIVERSE_HEIGHT; y++)
		// 		arr[x * constants.UNIVERSE_HEIGHT + y] = -1.0

		//arr[0] = 1
		
		// for (var i = Math.floor(constants.UNIVERSE_SIZE / 8 / 10); i--;){
		// 	arr[i] = Math.floor(Math.random() * 256)
		// }

		gl.bufferData(gl.SHADER_STORAGE_BUFFER, arr, gl.DYNAMIC_COPY)
		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, idx, buffer)
	}

	const bindRenderMeta = () => {
		const buffer = gl.createBuffer()
		gl.bindBuffer(gl.UNIFORM_BUFFER, buffer)
		gl.bufferData(gl.UNIFORM_BUFFER, 16, gl.DYNAMIC_DRAW)
		gl.bindBufferBase(gl.UNIFORM_BUFFER, 2, buffer)
		//gl.bindBufferRange(gl.UNIFORM_BUFFER, 2, buffer, 0, 16)
		for (const x of [computeProgram, renderProgram])
			gl.uniformBlockBinding(x, 0, 2)
	}


	bindRenderMeta();
	[1, 0].forEach(bindUniverse)

	var time = new Uint8Array([0])
	var startTime = Date.now()
	const render = () => {
		status.textContent = `${(1000 * ++time[0] / (Date.now() - startTime)).toFixed(2)} fps`
		gl.bufferSubData(gl.UNIFORM_BUFFER, 0, time)
		gl.useProgram(computeProgram)
		gl.dispatchCompute(constants.UNIVERSE_WIDTH, constants.UNIVERSE_HEIGHT, 1)
		// gl.memoryBarrier(gl.SHADER_STORAGE_BARRIER_BIT)
		gl.useProgram(renderProgram)
		gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
		requestAnimationFrame(render)
	}

	render()

	const universeView = new Uint8Array(1)
	var offsetX, offsetY, button
	const updateMouse = event => {
		({offsetX, offsetY} = event)
	}
	const mousedownHandler = () => {
		const offset = constants.CELL_BITS * mousePosToArr(offsetX, offsetY)
		const byteOffset = offset / 8
		const bitOffset = offset % 8
		gl.getBufferSubData(gl.SHADER_STORAGE_BUFFER, byteOffset, universeView)

		const orig = (universeView[0] & ((constants.NUM_STATES - 1) << bitOffset)) >> bitOffset
		// right click
		if (button == 2){
			universeView[0] &= ~(3 << bitOffset)
			universeView[0] |= util.mod(orig - 1, constants.NUM_STATES) << bitOffset
		// left click
		}else if (button == 0){
			universeView[0] &= ~((constants.NUM_STATES - 1) << bitOffset)
			universeView[0] |= util.mod(orig + 1, constants.NUM_STATES) << bitOffset
		}else{
			// middle click
			universeView[0] &= ~((constants.NUM_STATES - 1) << bitOffset)
			universeView[0] |= (constants.NUM_STATES - orig) << bitOffset
		}
		
		gl.bufferSubData(gl.SHADER_STORAGE_BUFFER, byteOffset, universeView)
	}
	var mousedownInterval = 0
	window.onmousemove = updateMouse
	canvas.oncontextmenu = e => e.preventDefault()
	canvas.onmousedown = e => {
		({button, offsetX, offsetY} = e)
		if (mousedownInterval) clearInterval(mousedownInterval)
		mousedownInterval = setInterval(mousedownHandler, 0)
		e.preventDefault()
	}
	window.onmouseup = () => {
		if (mousedownInterval) clearInterval(mousedownInterval)
		mousedownInterval = 0
	}
}


